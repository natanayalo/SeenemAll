from __future__ import annotations

import logging
import time
from typing import Any, Dict

from fastapi import Request
from sqlalchemy.orm import Session

from api.core.metrics import METRICS, timer
from api.pipeline.context import load_user_context
from api.pipeline.diversity import apply_diversity_policies
from api.pipeline.intent import resolve_query_intent
from api.pipeline.models import ComputeResult, RecommendParams
from api.pipeline.reranker import build_debug_snapshot, rerank_candidates
from api.pipeline.retriever import retrieve_candidates
from api.pipeline.scorer import score_candidates

logger = logging.getLogger(__name__)


class RecommendationPipeline:
    async def run(
        self,
        request: Request,
        params: RecommendParams,
        db: Session,
    ) -> ComputeResult:
        _pipeline_start = time.perf_counter()

        # Stage 1: Context Resolution
        with timer("recommend.context_latency_ms"):
            context = load_user_context(db, params.user_id, params.profile)

        # Stage 2: Query Understanding
        with timer("recommend.intent_latency_ms"):
            intent = await resolve_query_intent(request, params, context, db)

        # Stage 3: Candidate Retrieval
        with timer("recommend.retrieval_latency_ms"):
            pool = retrieve_candidates(db, context, intent)
        METRICS.histogram("recommend.initial_candidate_count").observe(len(pool.ids))
        if not pool.ids:
            METRICS.counter("recommend.pipeline_zero_results").inc()
            return ComputeResult(items=[], debug_context={})

        # Stage 4: Candidate Scoring & Business Rules
        with timer("recommend.scoring_latency_ms"):
            scored = score_candidates(pool, intent, params, context)
        METRICS.histogram("recommend.scored_candidate_count").observe(
            len(scored.ordered)
        )
        if not scored.ordered:
            METRICS.counter("recommend.pipeline_zero_results").inc()
            return ComputeResult(items=[], debug_context={})

        # Stage 5: Diversity & Policy
        with timer("recommend.diversity_latency_ms"):
            diversified = apply_diversity_policies(
                scored.ordered,
                scored.serendipity_context,
                limit=params.limit,
                diversify=params.diversify,
                boost_ids=pool.boost_ids,
                exempt_collection_ids=set(
                    getattr(intent, "matched_collection_ids", ()) or ()
                ),
            )

        # Stage 6: Presentation & Reranking
        METRICS.histogram("recommend.rerank_candidate_count").observe(len(diversified))
        with timer("recommend.rerank_latency_ms"):
            reranked = rerank_candidates(
                diversified,
                intent=intent.intent_filters,
                query=params.query,
                context=context,
                rerank=params.rerank,
                rerank_provider=params.rerank_provider,
            )

        matched_coll_ids = set(getattr(intent, "matched_collection_ids", ()) or ())
        coll_item_ids = set(getattr(intent, "collection_item_ids", ()) or ())
        if matched_coll_ids or coll_item_ids:
            franchise_items = [
                it
                for it in reranked
                if it.get("collection_id") in matched_coll_ids
                or it.get("id") in coll_item_ids
            ]
            other_items = [
                it
                for it in reranked
                if it.get("collection_id") not in matched_coll_ids
                and it.get("id") not in coll_item_ids
            ]
            if getattr(intent, "is_chronological_requested", False):
                franchise_items.sort(
                    key=lambda it: (
                        it.get("release_year") is None,
                        it.get("release_year") or 0,
                    )
                )
            reranked = franchise_items + other_items

        pipeline_ms = (time.perf_counter() - _pipeline_start) * 1000
        METRICS.histogram("recommend.total_latency_ms").observe(pipeline_ms)
        METRICS.histogram("recommend.pipeline_latency_ms").observe(pipeline_ms)
        METRICS.histogram("recommend.returned_item_count").observe(len(reranked))
        if not reranked:
            METRICS.counter("recommend.pipeline_zero_results").inc()

        debug_snapshot: Dict[str, Any] | None = None
        if params.debug:
            debug_snapshot = build_debug_snapshot(
                db=db,
                allowlist=pool.prefilter.allowed_ids,
                boost_ids=pool.boost_ids,
                prefer_top_rated=intent.prefer_top_rated,
                strict_filters=bool(intent.prefilter_kwargs.get("require_all_genres")),
                initial_candidates_count=len(pool.ids),
                post_filter_candidates_count=len(scored.ordered),
                neighbors_count=len(context.profile_meta.get("neighbors") or []),
                cold_start=context.cold_start,
                pipeline_ms=pipeline_ms,
            )

        return ComputeResult(items=reranked, debug_context=debug_snapshot or {})


_pipeline_instance = RecommendationPipeline()


def get_pipeline() -> RecommendationPipeline:
    return _pipeline_instance
