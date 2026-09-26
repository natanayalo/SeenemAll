from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, List, Sequence

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.core.legacy_intent_parser import IntentFilters
from api.core.metrics import METRICS
from api.core.reranker import rerank_with_explanations
from api.db.models import Item
from api.pipeline.hooks import get_hook
from api.pipeline.models import UserContext

logger = logging.getLogger(__name__)

_DEBUG_ALLOWLIST_LIMIT = 250
_DEBUG_BOOST_LIMIT = 50


def rerank_candidates(
    ordered: List[Dict[str, Any]],
    intent: IntentFilters,
    query: str | None,
    context: UserContext,
) -> List[Dict[str, Any]]:
    rerank_fn = get_hook("rerank_with_explanations", rerank_with_explanations)
    reranked = rerank_fn(
        ordered,
        intent=intent,
        query=query,
        user={
            "user_id": context.canonical_id,
            "base_user_id": context.user_id,
            "profile": context.profile,
            "genre_prefs": context.profile_meta.get("genre_prefs"),
            "neighbors": context.profile_meta.get("neighbors"),
            "negative_items": context.profile_meta.get("negative_items"),
        },
    )
    METRICS.counter("recommend.reranker_used").inc()
    return reranked


def build_debug_snapshot(
    db: Session,
    allowlist: List[int] | None,
    boost_ids: List[int],
    prefer_top_rated: bool,
    strict_filters: bool,
    initial_candidates_count: int,
    post_filter_candidates_count: int,
    neighbors_count: int,
    cold_start: bool,
    pipeline_ms: float,
) -> Dict[str, Any]:
    allow_sample = list((allowlist or [])[:_DEBUG_ALLOWLIST_LIMIT])
    boost_sample = list(boost_ids[:_DEBUG_BOOST_LIMIT])
    lookup_ids = set(allow_sample) | set(boost_sample)
    tmdb_map: Dict[int, int | None] = {}
    if lookup_ids:
        rows = db.execute(
            select(Item.id, Item.tmdb_id).where(Item.id.in_(lookup_ids))
        ).all()
        tmdb_map = {row.id: row.tmdb_id for row in rows}

    return {
        "allowlist_len": len(allowlist or []),
        "allowlist_ids": allow_sample,
        "allowlist_tmdb_ids": [tmdb_map.get(item_id) for item_id in allow_sample],
        "boost_len": len(boost_ids or []),
        "boost_ids": boost_sample,
        "boost_tmdb_ids": [tmdb_map.get(item_id) for item_id in boost_sample],
        "classic_top_rated": prefer_top_rated,
        "strict_filters": strict_filters,
        "metrics": {
            "initial_candidates": initial_candidates_count,
            "post_filter_candidates": post_filter_candidates_count,
            "neighbors_found": neighbors_count,
            "cold_start": cold_start,
            "pipeline_latency_ms": round(pipeline_ms, 2),
        },
    }


def encode_cursor(rank: int) -> str:
    payload = json.dumps({"rank": rank}).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("utf-8").rstrip("=")


def decode_cursor(cursor: str | None) -> int:
    if not cursor:
        return 0
    padding = "=" * (-len(cursor) % 4)
    try:
        decoded = base64.urlsafe_b64decode(cursor + padding)
        payload = json.loads(decoded.decode("utf-8"))
        rank = int(payload["rank"])
        if rank < 0:
            raise ValueError
        return rank
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=400, detail="Invalid cursor") from exc


def format_presentation_items(page: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    response: List[Dict[str, Any]] = []
    for entry in page:
        cleaned = dict(entry)
        cleaned.pop("original_rank", None)
        cleaned.pop("vector", None)
        cleaned.pop("ann_rank", None)
        cleaned.pop("retrieval_score", None)
        cleaned.pop("source_scores", None)
        response.append(cleaned)
    return response


def empty_response() -> Dict[str, Any]:
    return {"items": []}


# Aliases for backwards compatibility with tests
_encode_cursor = encode_cursor
_decode_cursor = decode_cursor
_empty_response = empty_response
