from __future__ import annotations
from typing import Any, Dict, List
import base64
import json
import os
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, func, or_
from api.db.session import get_db
from api.db.models import Item, ItemEmbedding, Availability
from api.config import COUNTRY_DEFAULT
from api.core.user_utils import load_user_state, canonical_profile_id
from api.core.candidate_gen import ann_candidates
from api.core.legacy_intent_parser import (
    parse_intent,
    item_matches_intent,
    IntentFilters,
)
from api.core.reranker import rerank_with_explanations, diversify_with_mmr
from api.core.business_rules import apply_business_rules

router = APIRouter(prefix="/recommend", tags=["recommend"])


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


@router.get("")
def recommend(
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
    if short_v is None:
        raise HTTPException(
            status_code=400, detail="No user vector. POST /user/history first."
        )

    intent = parse_intent(query)
    candidate_limit = min(500, max(limit, limit * 3))
    if intent.has_filters():
        candidate_limit = min(500, max(candidate_limit, limit * 5))

    allowlist = _prefilter_allowed_ids(db, intent, candidate_limit)
    ids: List[int] = ann_candidates(
        db, short_v, exclude, limit=candidate_limit, allowed_ids=allowlist
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
    if intent is None:
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
            filters.append(Item.genres.contains([{"name": genre}]))
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
