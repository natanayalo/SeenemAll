from __future__ import annotations

import logging
from typing import List, Tuple

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
            rewrite_used = False
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

            cold_start_fn = get_hook(
                "_cold_start_candidates", cold_start_candidates
            )
            logger.info("Using cold-start candidates for user %s", canonical_id)
            ids = cold_start_fn(
                db,
                intent.intent_filters,
                candidate_limit,
                allowlist,
                prefer_top_rated=intent.prefer_top_rated,
            )
            return ids, False
        else:
            logger.info("Using ANN candidates for user %s", canonical_id)
            short_v = context.short_v
            assert short_v is not None
            rewrite_vec = intent.rewrite_vec
            if rewrite_vec is not None:
                alpha = (
                    _REWRITE_BLEND_ALPHA_QUERY
                    if intent.query
                    else _REWRITE_BLEND_ALPHA
                )
                alpha = max(0.0, min(1.0, alpha))
                q_vec = (alpha * short_v) + ((1 - alpha) * rewrite_vec)
                q_vec = q_vec / np.linalg.norm(q_vec)
            else:
                q_vec = short_v

            try:
                ids = ann_candidates_fn(
                    db,
                    q_vec,
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

            if not ids and has_people_filters:
                relaxed_filters = relax_fn(structured_search_filters)
                if relaxed_filters:
                    intent.structured_search_filters = relaxed_filters
                    structured_search_filters = relaxed_filters
                    try:
                        ids = ann_candidates_fn(
                            db,
                            q_vec,
                            exclude,
                            limit=candidate_limit,
                            allowed_ids=None,
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
                filtered_fallback = filter_exclude_fn(
                    fallback_ids, context.exclude_set
                )
                if filtered_fallback:
                    ids = filtered_fallback
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(
                            "Using catalogue fallback for user %s due to people filters.",
                            canonical_id,
                        )
            return ids, False
