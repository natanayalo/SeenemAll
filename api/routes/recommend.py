from __future__ import annotations
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select
from api.db.session import get_db
from api.db.models import Item
from api.core.user_utils import load_user_state
from api.core.candidate_gen import ann_candidates
from api.core.intent_parser import parse_intent, item_matches_intent

router = APIRouter(prefix="/recommend", tags=["recommend"])


@router.get("")
def recommend(
    user_id: str = Query(..., description="Seen'emAll user_id (e.g., 'u1')"),
    limit: int = Query(20, ge=1, le=100),
    query: str | None = Query(
        None,
        description="Optional natural-language intent (e.g. 'light sci-fi < 2h')",
    ),
    db: Session = Depends(get_db),
):
    long_v, short_v, exclude = load_user_state(db, user_id)
    if short_v is None:
        raise HTTPException(
            status_code=400, detail="No user vector. POST /user/history first."
        )

    intent = parse_intent(query)
    candidate_limit = limit
    if intent.has_filters():
        candidate_limit = min(500, max(limit, limit * 3))

    ids: List[int] = ann_candidates(db, short_v, exclude, limit=candidate_limit)
    if not ids:
        return []

    # fetch metadata and preserve ANN order
    items = {
        row.id: row
        for row in db.execute(select(Item).where(Item.id.in_(ids))).scalars().all()
    }
    ordered = []
    for iid in ids:
        it = items.get(iid)
        if not it:
            continue
        if not item_matches_intent(it, intent):
            continue
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
            }
        )
        if len(ordered) >= limit:
            break
    return ordered
