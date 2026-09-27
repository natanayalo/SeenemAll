from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from api.core.candidate_gen import ann_candidates
from api.core.elasticsearch_search import ElasticsearchSearchError
from api.pipeline.hooks import get_hook
from api.pipeline.intent import float_from_env
from api.pipeline.models import QueryUnderstanding, UserContext
from api.pipeline.retriever.base import BaseRetriever
from api.pipeline.retriever.cold_start import cold_start_candidates
from api.pipeline.retriever.prefilter import (
    filter_excluded_candidate_ids,
    people_only_candidate_ids,
    relax_filters_for_people,
)

logger = logging.getLogger("api.routes.recommend")

_REWRITE_BLEND_ALPHA = 0.5
_REWRITE_BLEND_ALPHA_QUERY = float_from_env("REWRITE_BLEND_ALPHA_QUERY", 0.2)


def _cold_start_kwargs(fn: Any, intent: QueryUnderstanding) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"prefer_top_rated": intent.prefer_top_rated}
    try:
        signature = inspect.signature(fn)
        accepts_filters = "search_filters" in signature.parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    except (TypeError, ValueError):
        accepts_filters = True
    if accepts_filters:
        kwargs["search_filters"] = intent.structured_search_filters
    return kwargs


def _extract_centroid(cluster: Any) -> np.ndarray:
    raw = cluster.centroid if hasattr(cluster, "centroid") else cluster.get("centroid")
    arr = np.array(raw, dtype="float32")
    norm_val = float(np.linalg.norm(arr))
    return (arr / norm_val if norm_val > 0 else arr).astype("float32")


def _fuse_candidate_rankings_rrf(
    candidate_lists: Sequence[Sequence[int]],
    weights: Sequence[float] | None = None,
    limit: int = 20,
    rrf_k: int = 60,
) -> List[int]:
    """Combine multiple ranked candidate lists using Reciprocal Rank Fusion."""
    if not candidate_lists:
        return []
    if len(candidate_lists) == 1:
        return list(candidate_lists[0])[:limit]

    scores: Dict[int, float] = defaultdict(float)
    eff_weights = list(weights) if weights is not None else [1.0] * len(candidate_lists)

    for list_idx, c_list in enumerate(candidate_lists):
        w = eff_weights[list_idx] if list_idx < len(eff_weights) else 1.0
        for rank, item_id in enumerate(c_list, start=1):
            scores[item_id] += w / (rrf_k + rank)

    sorted_ids = sorted(scores.keys(), key=lambda iid: scores[iid], reverse=True)
    return sorted_ids[:limit]


def _run_ann_query(
    ann_candidates_fn: Any,
    db: Session,
    query_vec: np.ndarray,
    exclude: List[int],
    candidate_limit: int,
    allowlist: List[int] | None,
    backend_override: str | None,
    structured_search_filters: Any,
    es_text_query: str | None,
) -> List[int]:
    try:
        return ann_candidates_fn(
            db,
            query_vec,
            exclude,
            limit=candidate_limit,
            allowed_ids=allowlist,
            backend_override=backend_override,
            search_filters=structured_search_filters,
            text_query=es_text_query,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ElasticsearchSearchError as exc:
        logger.error("Elasticsearch retrieval error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service temporarily unavailable.",
        ) from exc


class ANNRetriever(BaseRetriever):
    def retrieve(
        self,
        db: Session,
        context: UserContext,
        intent: QueryUnderstanding,
        allowlist: List[int] | None,
    ) -> Tuple[List[int], bool]:
        ann_candidates_fn = get_hook("ann_candidates", ann_candidates)
        relax_fn = get_hook("_relax_filters_for_people", relax_filters_for_people)
        people_fallback_fn = get_hook(
            "_people_only_candidate_ids", people_only_candidate_ids
        )
        filter_exclude_fn = get_hook(
            "_filter_excluded_candidate_ids", filter_excluded_candidate_ids
        )

        canonical_id = context.canonical_id
        candidate_limit = intent.candidate_limit
        structured_search_filters = intent.structured_search_filters
        has_people_filters = intent.has_people_filters
        es_text_query = intent.es_text_query
        backend_override = intent.backend_override_normalized
        exclude = list(context.exclude_set)

        if context.cold_start:
            rewrite_vec = intent.rewrite_vec
            if rewrite_vec is not None:
                vec_norm = float(np.linalg.norm(rewrite_vec))
                if vec_norm > 0 and np.isfinite(vec_norm):
                    try:
                        ann_ids = ann_candidates_fn(
                            db,
                            rewrite_vec,
                            exclude,
                            limit=candidate_limit,
                            allowed_ids=allowlist,
                            backend_override=backend_override,
                            search_filters=structured_search_filters,
                            text_query=es_text_query,
                        )
                    except ValueError as exc:
                        raise HTTPException(status_code=400, detail=str(exc)) from exc
                    except ElasticsearchSearchError as exc:
                        logger.error("Elasticsearch retrieval error: %s", exc)
                        raise HTTPException(
                            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Search service temporarily unavailable.",
                        ) from exc

                    if not ann_ids and has_people_filters:
                        relaxed_filters = relax_fn(structured_search_filters)
                        if relaxed_filters:
                            intent.structured_search_filters = relaxed_filters
                            structured_search_filters = relaxed_filters
                            try:
                                ann_ids = ann_candidates_fn(
                                    db,
                                    rewrite_vec,
                                    exclude,
                                    limit=candidate_limit,
                                    allowed_ids=None,
                                    backend_override=backend_override,
                                    search_filters=structured_search_filters,
                                    text_query=es_text_query,
                                )
                            except ValueError as exc:
                                raise HTTPException(
                                    status_code=400, detail=str(exc)
                                ) from exc
                            except ElasticsearchSearchError as exc:
                                logger.error("Elasticsearch retrieval error: %s", exc)
                                raise HTTPException(
                                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                    detail="Search service temporarily unavailable.",
                                ) from exc
                            if ann_ids and logger.isEnabledFor(logging.INFO):
                                logger.info(
                                    "Relaxed ANN filters for user %s due to people filters.",
                                    canonical_id,
                                )

                    if ann_ids:
                        logger.info(
                            "Using rewrite ANN candidates for cold-start user %s",
                            canonical_id,
                        )
                        logger.debug(
                            "Rewrite ANN retrieval | candidate_count=%d allowlist_size=%s",
                            len(ann_ids),
                            len(allowlist) if allowlist is not None else None,
                        )
                        return ann_ids, True
                    elif has_people_filters and structured_search_filters:
                        fallback_ids = people_fallback_fn(
                            db,
                            structured_search_filters,
                            limit=candidate_limit,
                        )
                        filtered_fallback = filter_exclude_fn(
                            fallback_ids, context.exclude_set
                        )
                        if filtered_fallback:
                            if logger.isEnabledFor(logging.INFO):
                                logger.info(
                                    "Using catalogue fallback for user %s due to people filters.",
                                    canonical_id,
                                )
                            return filtered_fallback, True

            cold_start_fn = get_hook("_cold_start_candidates", cold_start_candidates)
            logger.info("Using cold-start candidates for user %s", canonical_id)
            ids = cold_start_fn(
                db,
                intent.intent_filters,
                candidate_limit,
                allowlist,
                **_cold_start_kwargs(cold_start_fn, intent),
            )
            return ids, False
        else:
            logger.info("Using ANN candidates for user %s", canonical_id)
            clusters = context.active_taste_clusters
            rewrite_vec = intent.rewrite_vec

            # Multi-interest retrieval when user has >= 2 active taste clusters
            if clusters and len(clusters) > 1:
                if rewrite_vec is not None:
                    # 1. Query mode: select the cluster closest to query intent
                    best_sim = -2.0
                    best_cluster = clusters[0]
                    r_norm = float(np.linalg.norm(rewrite_vec))
                    r_unit = rewrite_vec / r_norm if r_norm > 0 else rewrite_vec

                    for cl in clusters:
                        c_vec = _extract_centroid(cl)
                        sim = float(np.dot(c_vec, r_unit))
                        if sim > best_sim:
                            best_sim = sim
                            best_cluster = cl

                    best_c_vec = _extract_centroid(best_cluster)
                    alpha = (
                        _REWRITE_BLEND_ALPHA_QUERY
                        if intent.query
                        else _REWRITE_BLEND_ALPHA
                    )
                    alpha = max(0.0, min(1.0, alpha))
                    q_vec = (alpha * best_c_vec) + ((1 - alpha) * rewrite_vec)
                    q_norm = float(np.linalg.norm(q_vec))
                    q_vec = q_vec / q_norm if q_norm > 0 else q_vec

                    cl_id = (
                        best_cluster.cluster_id
                        if hasattr(best_cluster, "cluster_id")
                        else (
                            best_cluster.get("cluster_id", 0)
                            if isinstance(best_cluster, dict)
                            else 0
                        )
                    )
                    logger.info(
                        "Selected taste cluster %s (sim=%.3f) for user %s query",
                        cl_id,
                        best_sim,
                        canonical_id,
                    )

                    ids = _run_ann_query(
                        ann_candidates_fn,
                        db,
                        q_vec,
                        exclude,
                        candidate_limit,
                        allowlist,
                        backend_override,
                        structured_search_filters,
                        es_text_query,
                    )
                else:
                    # 2. Browse mode (no query): retrieve across active clusters and fuse via RRF
                    candidate_lists: List[List[int]] = []
                    cluster_weights: List[float] = []

                    for cl in clusters:
                        c_vec = _extract_centroid(cl)
                        c_weight = float(
                            getattr(cl, "weight", cl.get("weight", 1.0))
                            if hasattr(cl, "weight") or isinstance(cl, dict)
                            else 1.0
                        )
                        c_ids = _run_ann_query(
                            ann_candidates_fn,
                            db,
                            c_vec,
                            exclude,
                            candidate_limit,
                            allowlist,
                            backend_override,
                            structured_search_filters,
                            es_text_query,
                        )
                        if c_ids:
                            candidate_lists.append(c_ids)
                            cluster_weights.append(c_weight)

                    if candidate_lists:
                        ids = _fuse_candidate_rankings_rrf(
                            candidate_lists,
                            weights=cluster_weights,
                            limit=candidate_limit,
                        )
                    else:
                        ids = []
            else:
                # Baseline single-vector retrieval
                short_v = context.short_v
                assert short_v is not None
                if rewrite_vec is not None:
                    alpha = (
                        _REWRITE_BLEND_ALPHA_QUERY
                        if intent.query
                        else _REWRITE_BLEND_ALPHA
                    )
                    alpha = max(0.0, min(1.0, alpha))
                    q_vec = (alpha * short_v) + ((1 - alpha) * rewrite_vec)
                    q_norm = float(np.linalg.norm(q_vec))
                    q_vec = q_vec / q_norm if q_norm > 0 else q_vec
                else:
                    q_vec = short_v

                ids = _run_ann_query(
                    ann_candidates_fn,
                    db,
                    q_vec,
                    exclude,
                    candidate_limit,
                    allowlist,
                    backend_override,
                    structured_search_filters,
                    es_text_query,
                )

            if not ids and has_people_filters:
                relaxed_filters = relax_fn(structured_search_filters)
                if relaxed_filters:
                    intent.structured_search_filters = relaxed_filters
                    structured_search_filters = relaxed_filters
                    fallback_vec = (
                        q_vec
                        if "q_vec" in locals()
                        else (
                            _extract_centroid(clusters[0])
                            if clusters
                            else context.short_v
                        )
                    )
                    if fallback_vec is not None:
                        ids = _run_ann_query(
                            ann_candidates_fn,
                            db,
                            fallback_vec,
                            exclude,
                            candidate_limit,
                            None,
                            backend_override,
                            structured_search_filters,
                            es_text_query,
                        )
                        if ids and logger.isEnabledFor(logging.INFO):
                            logger.info(
                                "Relaxed ANN filters for user %s due to people filters.",
                                canonical_id,
                            )

            if not ids and has_people_filters and structured_search_filters:
                fallback_ids = people_fallback_fn(
                    db,
                    structured_search_filters,
                    limit=candidate_limit,
                )
                filtered_fallback = filter_exclude_fn(fallback_ids, context.exclude_set)
                if filtered_fallback:
                    ids = filtered_fallback
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(
                            "Using catalogue fallback for user %s due to people filters.",
                            canonical_id,
                        )
            return ids, False
