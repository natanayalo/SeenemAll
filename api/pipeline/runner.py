from __future__ import annotations

import logging
import time
from typing import Any, Dict

from fastapi import Request
from sqlalchemy.orm import Session

from api.core.metrics import METRICS
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
        context = load_user_context(db, params.user_id, params.profile)

        # Stage 2: Query Understanding
        intent = await resolve_query_intent(request, params, context, db)

        # Stage 3: Candidate Retrieval
        pool = retrieve_candidates(db, context, intent)
        if not pool.ids:
            return ComputeResult(items=[], debug_context={})

        # Stage 4: Candidate Scoring & Business Rules
        scored = score_candidates(pool, intent, params, context)
        if not scored.ordered:
            return ComputeResult(items=[], debug_context={})

        # Stage 5: Diversity & Policy
        diversified = apply_diversity_policies(
            scored.ordered,
            scored.serendipity_context,
            limit=params.limit,
            diversify=params.diversify,
            boost_ids=pool.boost_ids,
        )

        # Stage 6: Presentation & Reranking
        reranked = rerank_candidates(
            diversified,
            intent=intent.intent_filters,
            query=params.query,
            context=context,
            rerank=params.rerank,
            rerank_provider=params.rerank_provider,
        )

        pipeline_ms = (time.perf_counter() - _pipeline_start) * 1000
        METRICS.histogram("recommend.total_latency_ms").observe(pipeline_ms)

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
