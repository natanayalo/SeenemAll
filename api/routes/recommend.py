from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from api.core import llm_parser as llm_parser
from api.core.user_utils import canonical_profile_id
from api.db.session import get_db
from api.pipeline.context import (
    clear_user_cache as clear_user_cache,
    get_or_compute_recommendations,
)
from api.pipeline.intent import _float_from_env
from api.pipeline.models import (
    ComputeResult,
    PrefilterDecision as PrefilterDecision,
    RecommendParams,
)
from api.pipeline.reranker import (
    _decode_cursor,
    _encode_cursor,
    format_presentation_items,
)
from api.pipeline.retriever import _prefilter_allowed_ids as _prefilter_allowed_ids
from api.pipeline.runner import get_pipeline

router = APIRouter(prefix="/recommend", tags=["recommend"])
logger = logging.getLogger(__name__)

_SERENDIPITY_RATIO_RAW = _float_from_env("SERENDIPITY_RATIO", 0.15)
_SERENDIPITY_RATIO = (
    0.0 if _SERENDIPITY_RATIO_RAW <= 0.0 else max(0.1, min(0.2, _SERENDIPITY_RATIO_RAW))
)


async def _compute_recommendations_async(
    request: Request, params: RecommendParams, db: Session
) -> ComputeResult:
    return await get_pipeline().run(request, params, db)


@router.get("")
async def recommend(
    request: Request,
    params: RecommendParams = Depends(),
    cursor: str | None = Query(None, description="Opaque pagination cursor."),
    db: Session = Depends(get_db),
):
    canonical_id = canonical_profile_id(params.user_id, params.profile)
    reranked, debug_context = await get_or_compute_recommendations(
        request, params, db, canonical_id, _compute_recommendations_async
    )

    start_index = _decode_cursor(cursor)
    if start_index < 0:
        raise HTTPException(status_code=400, detail="Invalid cursor")

    page = reranked[start_index : start_index + params.limit]
    response = format_presentation_items(page)
    payload: Dict[str, Any] = {"items": response}
    if start_index + params.limit < len(reranked):
        payload["next_cursor"] = _encode_cursor(start_index + params.limit)
    if params.debug and debug_context:
        payload["debug"] = debug_context
    return payload


@router.get("/debug")
async def debug_recommend(
    request: Request, params: RecommendParams = Depends(), db: Session = Depends(get_db)
):
    result = await _compute_recommendations_async(request, params, db)
    return {
        "items": format_presentation_items(result.items[: params.limit]),
        "debug": result.debug_context,
    }


_LOOKUP = (
    "api.pipeline.context",
    "api.pipeline.intent",
    "api.pipeline.retriever",
    "api.pipeline.scorer",
    "api.pipeline.diversity",
    "api.pipeline.reranker",
    "api.core.candidate_gen",
    "api.core.business_rules",
    "api.core.filter_matcher",
    "api.core.embeddings",
    "api.core.user_utils",
    "api.core.llm_parser",
)


def __getattr__(name: str) -> Any:
    import importlib

    for mod in _LOOKUP:
        try:
            module = importlib.import_module(mod)
        except ImportError:
            continue
        if hasattr(module, name):
            val = getattr(module, name)
            globals()[name] = val
            return val
    if name == "legacy_parse_intent":
        from api.core.legacy_intent_parser import parse_intent

        globals()["legacy_parse_intent"] = parse_intent
        return parse_intent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
