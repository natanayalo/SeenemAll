from __future__ import annotations

import logging
from typing import List

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from api.core.legacy_intent_parser import IntentFilters
from api.db.models import Item, ItemEmbedding
from api.pipeline.hooks import get_hook
from api.pipeline.models import QueryUnderstanding, UserContext
from api.pipeline.retriever.base import BaseRetriever
from api.pipeline.retriever.prefilter import genre_contains_clause

logger = logging.getLogger("api.routes.recommend")


def cold_start_candidates(
    db: Session,
    intent: IntentFilters,
    limit: int,
    allowlist: List[int] | None,
    prefer_top_rated: bool = False,
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
            clause_fn = get_hook("_genre_contains_clause", genre_contains_clause)
            genre_filters = [clause_fn(db, genre) for genre in genres if genre]
            if genre_filters:
                stmt = stmt.where(or_(*genre_filters))

    if prefer_top_rated:
        stmt = stmt.order_by(
            Item.top_rated_rank.asc().nullslast(),
            Item.vote_average.desc().nullslast(),
            Item.vote_count.desc().nullslast(),
            Item.popular_rank.asc().nullslast(),
            Item.id.asc(),
        )
    else:
        stmt = stmt.order_by(
            Item.popular_rank.asc().nullslast(),
            Item.trending_rank.asc().nullslast(),
            Item.popularity.desc().nullslast(),
            Item.vote_average.desc().nullslast(),
            Item.vote_count.desc().nullslast(),
            Item.id.asc(),
        )

    stmt = stmt.limit(limit)

    rows = db.execute(stmt).scalars().all()
    seen: set[int] = set()
    ordered: List[int] = []
    for value in rows:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


class ColdStartRetriever(BaseRetriever):
    def retrieve(
        self,
        db: Session,
        context: UserContext,
        intent: QueryUnderstanding,
        allowlist: List[int] | None,
    ) -> List[int]:
        cold_start_fn = get_hook("_cold_start_candidates", cold_start_candidates)
        return cold_start_fn(
            db,
            intent.intent_filters,
            intent.candidate_limit,
            allowlist,
            prefer_top_rated=intent.prefer_top_rated,
        )
