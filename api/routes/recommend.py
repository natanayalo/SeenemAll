from __future__ import annotations
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select
from api.db.session import get_db
from api.db.models import Item, ItemEmbedding
from api.core.user_utils import load_user_state
from api.core.candidate_gen import ann_candidates
from api.core.intent_parser import parse_intent, item_matches_intent
from api.core.reranker import rerank_with_explanations, diversify_with_mmr

router = APIRouter(prefix="/recommend", tags=["recommend"])


@router.get("")
def recommend(
    user_id: str = Query(..., description="Seen'emAll user_id (e.g., 'u1')"),
    limit: int = Query(20, ge=1, le=100),
    query: str | None = Query(
        None,
        description="Optional natural-language intent (e.g. 'light sci-fi < 2h')",
    ),
    diversify: bool = Query(True, description="Whether to diversify recommendations."),
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
    items_with_vectors = {
        row.id: (row, vec)
        for row, vec in db.execute(
            select(Item, ItemEmbedding.vector)
            .join(ItemEmbedding, Item.id == ItemEmbedding.item_id)
            .where(Item.id.in_(ids))
        ).all()
    }
    ordered: List[Dict[str, Any]] = []
    max_candidates = min(candidate_limit, max(limit * 2, 25))
    for iid in ids:
        it, vec = items_with_vectors.get(iid, (None, None))
        if not it or not vec:
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
                "original_rank": len(ordered),
                "vector": vec,
            }
        )
        if len(ordered) >= max_candidates:
            break

    if not ordered:
        return []

    if diversify:
        ordered = diversify_with_mmr(ordered, limit=limit)

    reranked = rerank_with_explanations(
        ordered,
        intent=intent,
        query=query,
        user={"user_id": user_id},
    )

    response = []
    for entry in reranked[:limit]:
        cleaned = dict(entry)
        cleaned.pop("original_rank", None)
        cleaned.pop("vector", None)
        response.append(cleaned)
    return response
