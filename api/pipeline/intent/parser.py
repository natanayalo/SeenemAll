from __future__ import annotations

import logging
from typing import Any, Dict, List

from api.core import llm_parser
from api.core.intent_parser import Intent
from api.core.legacy_intent_parser import IntentFilters
from api.core.maturity import rating_level

logger = logging.getLogger("api.routes.recommend")


def parse_llm_intent(
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


def intent_filters_from_llm(query: str | None, llm_intent: Intent) -> IntentFilters:
    genres = llm_parser._normalize_genre_names(list(llm_intent.include_genres or []))
    filters = IntentFilters(
        raw_query=query or "",
        genres=list(genres),
        moods=[],
        media_types=[],
        min_runtime=llm_intent.runtime_minutes_min,
        max_runtime=llm_intent.runtime_minutes_max,
        maturity_rating_max=llm_intent.maturity_rating_max,
        required_genres=[],
    )
    return filters


def merge_maturity_rating(primary: IntentFilters, fallback: IntentFilters) -> None:
    """Merge maturity caps, keeping the stricter (lower) rating."""
    fallback_cap = getattr(fallback, "maturity_rating_max", None)
    if not fallback_cap:
        return

    if not primary.maturity_rating_max:
        primary.maturity_rating_max = fallback_cap
        return

    fallback_level = rating_level(fallback_cap)
    primary_level = rating_level(primary.maturity_rating_max)
    if fallback_level is None:
        return
    if primary_level is None or fallback_level < primary_level:
        primary.maturity_rating_max = fallback_cap


def merge_with_legacy_filters(
    primary: IntentFilters, fallback: IntentFilters | None
) -> IntentFilters:
    if not fallback:
        return primary

    if hasattr(primary, "required_genres"):
        dedup_required: List[str] = []
        seen_required: set[str] = set()
        for genre in primary.required_genres:
            if genre and genre not in seen_required:
                dedup_required.append(genre)
                seen_required.add(genre)
        primary.required_genres = dedup_required

    seen_genres = {genre for genre in primary.genres}
    for genre in llm_parser._normalize_genre_names(fallback.genres):
        if genre and genre not in seen_genres:
            primary.genres.append(genre)
            seen_genres.add(genre)

    primary.genres = llm_parser._normalize_genre_names(primary.genres)

    if not primary.moods and fallback.moods:
        primary.moods = list(fallback.moods)

    if not primary.media_types and fallback.media_types:
        primary.media_types = list(fallback.media_types)

    if primary.min_runtime is None and fallback.min_runtime is not None:
        primary.min_runtime = fallback.min_runtime

    if primary.max_runtime is None and fallback.max_runtime is not None:
        primary.max_runtime = fallback.max_runtime

    if hasattr(primary, "required_genres") and primary.required_genres:
        primary.required_genres = llm_parser._normalize_genre_names(
            primary.required_genres
        )

    merge_maturity_rating(primary, fallback)

    return primary
