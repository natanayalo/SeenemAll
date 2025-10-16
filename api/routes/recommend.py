from __future__ import annotations

import base64
import json
import os
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from sqlalchemy import select, func, or_, cast
from sqlalchemy.dialects.postgresql import JSONB

from api.db.session import get_db
from api.db.models import Item, ItemEmbedding, Availability
from api.config import COUNTRY_DEFAULT
from api.core.user_utils import load_user_state, canonical_profile_id
from api.core.candidate_gen import ann_candidates
from api.core import llm_parser
from api.core.intent_parser import Intent
from api.core.legacy_intent_parser import (
    parse_intent as legacy_parse_intent,
    item_matches_intent,
    IntentFilters,
)
from api.core.reranker import rerank_with_explanations, diversify_with_mmr
from api.core.business_rules import apply_business_rules
from api.core.entity_linker import EntityLinker
from api.core.llm_parser import rewrite_query
from api.core.embeddings import encode_texts

router = APIRouter(prefix="/recommend", tags=["recommend"])
logger = logging.getLogger(__name__)


def _float_from_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value


_HYBRID_ANN_WEIGHT = _float_from_env("HYBRID_ANN_WEIGHT", 0.5)
_HYBRID_POPULARITY_WEIGHT = _float_from_env("HYBRID_POPULARITY_WEIGHT", 0.25)
_HYBRID_TRENDING_WEIGHT = _float_from_env("HYBRID_TRENDING_WEIGHT", 0.4)
_HYBRID_MIN_ANN_WEIGHT = 0.05
_REWRITE_BLEND_ALPHA = 0.5


def _parse_llm_intent(
    query: str | None,
    user_context: Dict[str, Any],
    linked_entities: Dict[str, Any] | None = None,
) -> Intent:
    if not query:
        return llm_parser.default_intent()
    try:
        return llm_parser.parse_intent(query, user_context, linked_entities)
    except Exception:
        logger.exception("LLM intent parser failed; falling back to default intent.")
        return llm_parser.default_intent()


def _intent_filters_from_llm(query: str | None, llm_intent: Intent) -> IntentFilters:
    genres = list(llm_intent.include_genres or [])
    filters = IntentFilters(
        raw_query=query or "",
        genres=genres,
        moods=[],
        media_types=[],
        min_runtime=llm_intent.runtime_minutes_min,
        max_runtime=llm_intent.runtime_minutes_max,
    )
    return filters


def _merge_with_legacy_filters(
    primary: IntentFilters, fallback: IntentFilters | None
) -> IntentFilters:
    if not fallback:
        return primary

    # Merge genres with order preservation
    seen_genres = {genre for genre in primary.genres}
    for genre in fallback.genres:
        if genre and genre not in seen_genres:
            primary.genres.append(genre)
            seen_genres.add(genre)

    if not primary.moods and fallback.moods:
        primary.moods = list(fallback.moods)

    if not primary.media_types and fallback.media_types:
        primary.media_types = list(fallback.media_types)

    if primary.min_runtime is None and fallback.min_runtime is not None:
        primary.min_runtime = fallback.min_runtime

    if primary.max_runtime is None and fallback.max_runtime is not None:
        primary.max_runtime = fallback.max_runtime

    return primary


@router.get("")
async def recommend(
    request: Request,
    user_id: str = Query(..., description="Seen'emAll user_id (e.g., 'u1')"),
    limit: int = Query(20, ge=1, le=100),
    query: str | None = Query(
        None,
        description="Optional natural-language intent (e.g. 'light sci-fi < 2h')",
    ),
    cursor: str | None = Query(
        None,
        description="Opaque cursor returned by a previous request for pagination.",
    ),
    diversify: bool = Query(True, description="Whether to diversify recommendations."),
    profile: str | None = Query(None, description="Optional profile identifier"),
    db: Session = Depends(get_db),
):
    canonical_id = canonical_profile_id(user_id, profile)
    long_v, short_v, exclude, profile_meta = load_user_state(db, canonical_id)
    cold_start = short_v is None

    tmdb_client = getattr(request.app.state, "tmdb_client", None)
    linked_entities = None
    if query:
        entity_linker = getattr(request.app.state, "entity_linker", None)
        if entity_linker is None and tmdb_client:
            entity_linker = EntityLinker(tmdb_client)
            request.app.state.entity_linker = entity_linker
        if entity_linker:
            linked_entities = await entity_linker.link_entities(query)

    llm_user_context = {"user_id": canonical_id, "profile_id": profile}
    llm_intent = _parse_llm_intent(query, llm_user_context, linked_entities)
    intent = _intent_filters_from_llm(query, llm_intent)
    if query:
        intent = _merge_with_legacy_filters(intent, legacy_parse_intent(query))
    candidate_limit = min(500, max(limit, limit * 3))
    if intent.has_filters():
        candidate_limit = min(500, max(candidate_limit, limit * 5))

    allowlist = _prefilter_allowed_ids(db, intent, candidate_limit)
    if cold_start:
        logger.info("Using cold-start candidates for user %s", canonical_id)
        ids = _cold_start_candidates(db, intent, candidate_limit, allowlist)
    else:
        logger.info("Using ANN candidates for user %s", canonical_id)
        assert short_v is not None
        rewrite = rewrite_query(query or "", llm_intent)
        if rewrite.rewritten_text:
            rewrite_vec = encode_texts([rewrite.rewritten_text])[0]
            alpha = _REWRITE_BLEND_ALPHA
            q_vec = (alpha * short_v) + ((1 - alpha) * rewrite_vec)
            q_vec = q_vec / np.linalg.norm(q_vec)
        else:
            q_vec = short_v

        ids = ann_candidates(
            db, q_vec, exclude, limit=candidate_limit, allowed_ids=allowlist
        )
    negative_items = set(profile_meta.get("negative_items") or [])
    if negative_items:
        ids = [i for i in ids if i not in negative_items]
    if not ids:
        return _empty_response()

    # fetch metadata and streaming links, preserve ANN order
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
                        func.coalesce(Availability.web_url, Availability.deeplink),
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
    ordered: List[Dict[str, Any]] = []
    max_candidates = min(candidate_limit, max(limit * 2, 25))
    for iid in ids:
        it, vec, watch_options = items_with_data.get(iid, (None, None, None))
        if it is None or vec is None or len(vec) == 0:
            continue
        if not item_matches_intent(it, intent):
            continue

        # Clean up watch_options - remove null entries and handle None case
        cleaned_options = []
        if (
            watch_options and watch_options[0] is not None
        ):  # PostgreSQL returns [null] when no matches
            cleaned_options = [
                {"service": opt["service"], "url": opt["url"]}
                for opt in watch_options
                if opt["url"] is not None
            ]

        ordered.append(
            {
                "id": it.id,
                "tmdb_id": it.tmdb_id,
                "media_type": it.media_type,
                "title": it.title,
                "overview": it.overview,
                "poster_url": it.poster_url,
                "runtime": it.runtime,
                "original_language": it.original_language,
                "genres": it.genres,
                "release_year": it.release_year,
                "watch_options": cleaned_options,
                "watch_url": (
                    cleaned_options[0]["url"] if cleaned_options else None
                ),  # For backward compatibility
                "original_rank": len(ordered),
                "ann_rank": len(ordered),
                "vector": vec,
                "popularity": getattr(it, "popularity", None),
                "vote_average": getattr(it, "vote_average", None),
                "vote_count": getattr(it, "vote_count", None),
                "popular_rank": getattr(it, "popular_rank", None),
                "trending_rank": getattr(it, "trending_rank", None),
                "top_rated_rank": getattr(it, "top_rated_rank", None),
                "retrieval_score": None,
            }
        )
        if len(ordered) >= max_candidates:
            break

    if not ordered:
        return _empty_response()

    _apply_hybrid_boost(ordered)
    ordered = apply_business_rules(ordered, intent=intent)
    if not ordered:
        return _empty_response()

    if diversify:
        ordered = diversify_with_mmr(ordered, limit=limit)

    reranked = rerank_with_explanations(
        ordered,
        intent=intent,
        query=query,
        user={
            "user_id": canonical_id,
            "base_user_id": user_id,
            "profile": profile,
            "genre_prefs": profile_meta.get("genre_prefs"),
            "neighbors": profile_meta.get("neighbors"),
            "negative_items": profile_meta.get("negative_items"),
        },
    )

    start_index = _decode_cursor(cursor)
    if start_index < 0:
        raise HTTPException(status_code=400, detail="Invalid cursor")

    page = reranked[start_index : start_index + limit]
    response: List[Dict[str, Any]] = []
    for entry in page:
        cleaned = dict(entry)
        cleaned.pop("original_rank", None)
        cleaned.pop("vector", None)
        cleaned.pop("ann_rank", None)
        cleaned.pop("retrieval_score", None)
        response.append(cleaned)

    next_cursor = None
    if start_index + limit < len(reranked):
        next_cursor = _encode_cursor(start_index + limit)

    payload: Dict[str, Any] = {"items": response}
    if next_cursor:
        payload["next_cursor"] = next_cursor
    return payload


def _apply_hybrid_boost(candidates: List[Dict[str, Any]]) -> None:
    if not candidates:
        return

    ann_weight = max(_HYBRID_MIN_ANN_WEIGHT, _HYBRID_ANN_WEIGHT)
    popularity_weight = max(0.0, _HYBRID_POPULARITY_WEIGHT)
    trending_weight = max(0.0, _HYBRID_TRENDING_WEIGHT)

    max_popularity = max(float(item.get("popularity") or 0.0) for item in candidates)

    for item in candidates:
        ann_rank = float(item.get("ann_rank", item.get("original_rank", 0)))
        ann_score = 1.0 / (1.0 + ann_rank)

        pop_score = 0.0
        if max_popularity > 0:
            pop_score = float(item.get("popularity") or 0.0) / max_popularity

        trending_rank = item.get("trending_rank")
        trending_score = 0.0
        if trending_rank and trending_rank > 0:
            trending_score = 1.0 / float(trending_rank)

        retrieval_score = (
            ann_weight * ann_score
            + popularity_weight * pop_score
            + trending_weight * trending_score
        )
        item["retrieval_score"] = retrieval_score

    candidates.sort(
        key=lambda item: (
            -(item.get("retrieval_score") or 0.0),
            item.get("ann_rank", item.get("original_rank", 0)),
        )
    )
    for idx, item in enumerate(candidates):
        item["original_rank"] = idx


def _prefilter_allowed_ids(
    db: Session, intent: IntentFilters | None, limit: int
) -> List[int] | None:
    if intent is None or not intent.has_filters():
        return None

    media_types = intent.media_types or []
    genres = intent.effective_genres()

    if not media_types and not genres:
        return None

    stmt = select(Item.id)

    if media_types:
        stmt = stmt.where(Item.media_type.in_(media_types))

    if genres:
        filters = []
        for genre in genres:
            if not genre:
                continue
            filters.append(_genre_contains_clause(db, genre))
        if filters:
            stmt = stmt.where(or_(*filters))

    fetch_limit = max(limit * 5, limit)
    rows = db.execute(stmt.limit(fetch_limit)).scalars().all()
    if not rows:
        return []
    # Preserve ordering but ensure uniqueness
    seen: set[int] = set()
    ordered: List[int] = []
    for value in rows:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _genre_contains_clause(db: Session, genre: str):
    bind = getattr(db, "bind", None)
    if bind is None:
        get_bind = getattr(db, "get_bind", None)
        if callable(get_bind):
            try:
                bind = get_bind()
            except Exception:
                bind = None

    dialect_name: Optional[str] = None
    if bind is not None:
        dialect = getattr(bind, "dialect", None)
        if dialect is not None:
            dialect_name = getattr(dialect, "name", None)

    if dialect_name == "postgresql":
        return cast(Item.genres, JSONB).contains([{"name": genre}])

    # Fallback for other dialects used in tests (e.g., SQLite memory stubs)
    return Item.genres.contains([{"name": genre}])


def _cold_start_candidates(
    db: Session,
    intent: IntentFilters,
    limit: int,
    allowlist: List[int] | None,
) -> List[int]:
    stmt = select(Item.id).join(ItemEmbedding, ItemEmbedding.item_id == Item.id)

    if allowlist is not None:
        if not allowlist:
            return []
        stmt = stmt.where(Item.id.in_(allowlist))
    else:
        if intent.media_types:
            stmt = stmt.where(Item.media_type.in_(intent.media_types))
        genres = intent.effective_genres()
        if genres:
            genre_filters = [
                _genre_contains_clause(db, genre) for genre in genres if genre
            ]
            if genre_filters:
                stmt = stmt.where(or_(*genre_filters))

    stmt = stmt.order_by(
        Item.popular_rank.asc().nullslast(),
        Item.trending_rank.asc().nullslast(),
        Item.popularity.desc().nullslast(),
        Item.id.asc(),
    ).limit(limit)

    rows = db.execute(stmt).scalars().all()
    seen: set[int] = set()
    ordered: List[int] = []
    for value in rows:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _encode_cursor(rank: int) -> str:
    payload = json.dumps({"rank": rank}).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("utf-8").rstrip("=")


def _decode_cursor(cursor: str | None) -> int:
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


def _empty_response() -> Dict[str, Any]:
    return {"items": []}
