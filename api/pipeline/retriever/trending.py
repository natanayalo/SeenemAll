from __future__ import annotations

import logging
from typing import List, Tuple

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from api.core import llm_parser
from api.core.legacy_intent_parser import IntentFilters
from api.db.models import Item
from api.pipeline.hooks import get_hook
from api.pipeline.models import QueryUnderstanding, UserContext
from api.pipeline.retriever.base import BaseRetriever
from api.pipeline.retriever.prefilter import genre_contains_clause

logger = logging.getLogger("api.routes.recommend")


def trending_prior_candidates(
    db: Session,
    intent: IntentFilters | None,
    exclude_ids: List[int],
    limit: int,
    allowed_ids: List[int] | None,
) -> List[Tuple[int, float]]:
    if limit <= 0:
        return []

    stmt = select(
        Item.id,
        Item.trending_rank,
        Item.popular_rank,
        Item.popularity,
        Item.vote_average,
        Item.vote_count,
    )

    filters = []

    if allowed_ids is not None:
        if not allowed_ids:
            return []
        filters.append(Item.id.in_(allowed_ids))

    if exclude_ids:
        filters.append(~Item.id.in_(exclude_ids))

    if intent and intent.media_types:
        filters.append(Item.media_type.in_(intent.media_types))

    if intent:
        genres = intent.effective_genres()
        if genres:
            mapped_genres = []
            for genre in genres:
                normalized = llm_parser._normalize_genre_names([genre])
                if normalized:
                    mapped_genres.extend(normalized)
            genre_filters = []
            if mapped_genres:
                clause_fn = get_hook("_genre_contains_clause", genre_contains_clause)
                genre_filters = [
                    clause_fn(db, genre)
                    for genre in dict.fromkeys(mapped_genres)
                    if genre
                ]
            if genre_filters:
                filters.append(or_(*genre_filters))

    if filters:
        stmt = stmt.where(*filters)

    stmt = stmt.order_by(
        Item.trending_rank.asc().nullslast(),
        Item.popular_rank.asc().nullslast(),
        Item.popularity.desc().nullslast(),
        Item.vote_average.desc().nullslast(),
        Item.vote_count.desc().nullslast(),
        Item.id.asc(),
    ).limit(limit)

    rows = db.execute(stmt).all()
    if not rows:
        return []

    scored: List[Tuple[int, float, float, float]] = []
    for row in rows:
        try:
            (
                item_id,
                trending_rank,
                popular_rank,
                popularity,
                vote_average,
                vote_count,
            ) = row
        except ValueError:
            partial = tuple(row)
            if len(partial) < 4:
                continue
            item_id, trending_rank, popular_rank, popularity = partial[:4]
            vote_average = getattr(row, "vote_average", None)
            vote_count = getattr(row, "vote_count", None)
        if item_id is None:
            continue

        if not isinstance(trending_rank, (int, float)):
            trending_rank = None
        if not isinstance(popular_rank, (int, float)):
            popular_rank = None
        if not isinstance(popularity, (int, float)):
            popularity = None

        vote_avg_val = (
            float(vote_average) if isinstance(vote_average, (int, float)) else 0.0
        )
        vote_count_val = (
            float(vote_count) if isinstance(vote_count, (int, float)) else 0.0
        )

        rank_score = 0.0
        if trending_rank and trending_rank > 0:
            rank_score += 1.0 / (1.0 + float(trending_rank))
        if popular_rank and popular_rank > 0:
            rank_score += 0.5 / (1.0 + float(popular_rank))
        pop_score = float(popularity) if popularity is not None else 0.0
        vote_quality = max((vote_avg_val - 6.5) / 3.5, 0.0)
        vote_volume = min(vote_count_val / 5000.0, 1.0)
        vote_score = vote_quality * (0.5 + 0.5 * vote_volume)
        scored.append((int(item_id), rank_score, pop_score, vote_score))

    max_rank = max((entry[1] for entry in scored), default=0.0)
    max_pop = max((entry[2] for entry in scored), default=0.0)
    max_vote = max((entry[3] for entry in scored), default=0.0)
    if max_rank <= 0:
        max_rank = 1.0
    if max_pop <= 0:
        max_pop = 1.0
    if max_vote <= 0:
        max_vote = 1.0

    results: List[Tuple[int, float]] = []
    for item_id, rank_score, pop_score, vote_score in scored:
        normalized_rank = rank_score / max_rank if max_rank > 0 else 0.0
        normalized_pop = pop_score / max_pop if max_pop > 0 else 0.0
        normalized_vote = vote_score / max_vote if max_vote > 0 else 0.0
        combined = 0.5 * normalized_rank + 0.3 * normalized_pop + 0.2 * normalized_vote
        results.append((item_id, combined))
    if not results:
        return results
    max_combined = max(score for _, score in results)
    if max_combined > 0:
        results = [(item_id, score / max_combined) for item_id, score in results]
    return results


class TrendingPriorRetriever(BaseRetriever):
    def retrieve(
        self,
        db: Session,
        context: UserContext,
        intent: QueryUnderstanding,
        allowlist: List[int] | None,
    ) -> List[Tuple[int, float]]:
        trending_fn = get_hook(
            "_trending_prior_candidates", trending_prior_candidates
        )
        return trending_fn(
            db,
            intent.intent_filters,
            list(context.exclude_set),
            intent.candidate_limit,
            allowed_ids=allowlist,
        )
