from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.config import COUNTRY_DEFAULT
from api.core.metrics import timer
from api.db.models import Availability, Item, ItemEmbedding
from api.pipeline.hooks import get_hook
from api.pipeline.models import (
    CandidatePool,
    PrefilterDecision,
    QueryUnderstanding,
    UserContext,
)
from api.pipeline.retriever.ann import ANNRetriever
from api.pipeline.retriever.collaborative import collaborative_candidates
from api.pipeline.retriever.prefilter import prefilter_allowed_ids
from api.pipeline.retriever.trending import trending_prior_candidates

logger = logging.getLogger("api.routes.recommend")


def _supports_parameter(fn: Any, parameter: str) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    return parameter in signature.parameters or any(
        item.kind is inspect.Parameter.VAR_KEYWORD
        for item in signature.parameters.values()
    )


def retrieve_candidates(
    db: Session,
    context: UserContext,
    intent: QueryUnderstanding,
) -> CandidatePool:
    prefilter_fn = get_hook("_prefilter_allowed_ids", prefilter_allowed_ids)
    prefilter_kwargs = dict(intent.prefilter_kwargs)
    if _supports_parameter(prefilter_fn, "search_filters"):
        prefilter_kwargs["search_filters"] = intent.structured_search_filters
    prefilter: PrefilterDecision = prefilter_fn(
        db,
        intent.intent_filters,
        intent.candidate_limit,
        preferred_services=intent.preferred_services,
        prefer_top_rated=intent.prefer_top_rated,
        **prefilter_kwargs,
    )

    allowlist = prefilter.allowed_ids
    raw_collection_ids = list(getattr(intent, "collection_item_ids", ()) or [])
    allow_set: set[int] | None = None
    if allowlist is not None:
        valid_set = set(allowlist)
        allow_set = valid_set
        collection_ids = [cid for cid in raw_collection_ids if cid in valid_set]
        boost_ids = list(collection_ids)
        for bid in prefilter.boost_ids or []:
            if bid not in boost_ids and bid in valid_set:
                boost_ids.append(bid)
    else:
        collection_ids = raw_collection_ids
        boost_ids = list(collection_ids)
        for bid in prefilter.boost_ids or []:
            if bid not in boost_ids:
                boost_ids.append(bid)
    enforce_genres = prefilter.enforce_genres
    candidate_limit = intent.candidate_limit
    exclude = list(context.exclude_set)

    ann_retriever = ANNRetriever()
    ids, rewrite_used = ann_retriever.retrieve(db, context, intent, allowlist)

    collab_fn = get_hook("_collaborative_candidates", collaborative_candidates)
    collab_kwargs: Dict[str, Any] = {"allowed_ids": allowlist}
    if _supports_parameter(collab_fn, "search_filters"):
        collab_kwargs["search_filters"] = intent.structured_search_filters
    collab_neighbors = (
        None if context.cold_start else context.profile_meta.get("neighbors")
    )
    collab_results = collab_fn(
        db,
        collab_neighbors,
        exclude,
        candidate_limit,
        **collab_kwargs,
    )
    collab_scores = {iid: score for iid, score in collab_results}

    merged_scores: Dict[int, Dict[str, float]] = {}
    if not context.cold_start:
        ann_scores = {iid: 1.0 / (1.0 + idx) for idx, iid in enumerate(ids)}
        for iid, score in ann_scores.items():
            merged_scores.setdefault(iid, {})["ann"] = score
    else:
        ann_scores = {}

    if collab_scores:
        max_collab = max(collab_scores.values()) or 1.0
        for iid, score in collab_scores.items():
            merged_scores.setdefault(iid, {})["collab"] = score / max_collab

    trending_results: List[Tuple[int, float]] = []
    if not intent.query:
        if context.cold_start or ann_scores or collab_scores or boost_ids:
            trending_fn = get_hook(
                "_trending_prior_candidates", trending_prior_candidates
            )
            trending_kwargs: Dict[str, Any] = {"allowed_ids": allowlist}
            if _supports_parameter(trending_fn, "search_filters"):
                trending_kwargs["search_filters"] = intent.structured_search_filters
            trending_results = trending_fn(
                db,
                intent.intent_filters,
                exclude,
                candidate_limit,
                **trending_kwargs,
            )
            trending_scores = {iid: score for iid, score in trending_results}
            if trending_scores:
                max_trending = max(trending_scores.values()) or 1.0
                for iid, score in trending_scores.items():
                    merged_scores.setdefault(iid, {})["trending"] = score / max_trending

    collab_ids = [iid for iid, _ in collab_results]
    trending_ids = [iid for iid, _ in trending_results]

    if collab_ids or trending_ids:
        merged: List[int] = []
        seen: set[int] = set()
        sequences = [collab_ids, ids, trending_ids]
        for seq in sequences:
            for iid in seq:
                if iid in seen:
                    continue
                merged.append(iid)
                seen.add(iid)
                if len(merged) >= candidate_limit:
                    break
            if len(merged) >= candidate_limit:
                break
        ids = merged

    if boost_ids:
        priority: List[int] = []
        seen_priority: set[int] = set()
        for candidate in boost_ids:
            if candidate in context.exclude_set or candidate in seen_priority:
                continue
            if allow_set is not None and candidate not in allow_set:
                continue
            priority.append(candidate)
            seen_priority.add(candidate)

        if priority:
            combined: List[int] = list(priority)
            seen_all = set(priority)
            pool_cap = max(candidate_limit, len(priority) + candidate_limit)
            for candidate in ids:
                if candidate in seen_all:
                    continue
                combined.append(candidate)
                seen_all.add(candidate)
                if len(combined) >= pool_cap:
                    break

            ids = combined[:pool_cap]
            for idx, candidate in enumerate(priority):
                if candidate not in ids:
                    continue
                score = 1.0 / (1.0 + idx)
                scores = merged_scores.setdefault(candidate, {})
                current = scores.get("ann")
                if current is None or score > current:
                    scores["ann"] = score

    ids = ids[:candidate_limit]

    negative_items = set(context.profile_meta.get("negative_items") or [])
    if negative_items:
        ids = [i for i in ids if i not in negative_items]

    items_with_data: Dict[int, Tuple[Any, np.ndarray, Any]] = {}
    if ids:
        with timer("recommend.sql_hydration_latency_ms"):
            items_with_data = {
                row.id: (row, vec, watch_options)
                for row, vec, watch_options in db.execute(
                    select(
                        Item,
                        ItemEmbedding.vector,
                        func.json_agg(
                            func.json_build_object(
                                "service",
                                Availability.service,
                                "url",
                                func.coalesce(
                                    Availability.web_url, Availability.deeplink
                                ),
                            )
                        ).label("watch_options"),
                    )
                    .join(ItemEmbedding, Item.id == ItemEmbedding.item_id)
                    .outerjoin(
                        Availability,
                        (Item.id == Availability.item_id)
                        & (Availability.country == COUNTRY_DEFAULT),
                    )
                    .where(Item.id.in_(ids))
                    .group_by(Item.id, ItemEmbedding.vector)
                ).all()
            }

    return CandidatePool(
        ids=ids,
        merged_scores=merged_scores,
        prefilter=prefilter,
        items_with_data=items_with_data,
        boost_ids=boost_ids,
        enforce_genres=enforce_genres,
        structured_search_filters=intent.structured_search_filters,
    )
