from __future__ import annotations

import logging
import re
from typing import List, Sequence, Set

from sqlalchemy.orm import Session

from api.core import llm_parser
from api.core.elasticsearch_search import SearchFilters
from api.core.filter_matcher import QueryFiltersResult
from api.core.legacy_intent_parser import IntentFilters
from api.pipeline.context import get_top_query_keywords, load_media_genres
from api.pipeline.hooks import get_hook

logger = logging.getLogger("api.routes.recommend")


def matches_keywords(text: str | None, keywords: Set[str]) -> bool:
    if not text or not keywords:
        return False
    lowered = text.lower()
    for keyword in keywords:
        if not keyword:
            continue
        pattern = rf"\b{re.escape(keyword.lower().strip())}\b"
        if re.search(pattern, lowered):
            return True
    return False


def _has_people_filters(filters: SearchFilters | None) -> bool:
    if not filters:
        return False
    return any(
        (
            filters.cast,
            filters.directors,
            filters.producers,
            filters.writers,
        )
    )


def strict_required_genres(
    db: Session,
    custom_genres: Sequence[str],
    legacy_filters: IntentFilters | None,
    intent: IntentFilters,
) -> List[str]:
    strict_genres: List[str] = []
    strict_genres.extend(custom_genres)
    if legacy_filters and legacy_filters.genres:
        strict_genres.extend(legacy_filters.genres)
    if not strict_genres:
        strict_genres.extend(intent.genres)
    normalized_required = llm_parser._normalize_genre_names(strict_genres)
    if not normalized_required:
        return []
    media_types: List[str] = []
    if legacy_filters and legacy_filters.media_types:
        media_types.extend(legacy_filters.media_types)
    if intent.media_types:
        media_types.extend(intent.media_types)
    lowered_media = {mt.lower() for mt in media_types if isinstance(mt, str)}
    media_genres_loader = get_hook("_load_media_genres", load_media_genres)
    available_genres = media_genres_loader(db) if lowered_media else {}
    seen: set[str] = set()
    deduped: List[str] = []
    for genre in normalized_required:
        if not genre or genre in seen:
            continue
        if lowered_media:
            if not any(
                genre in available_genres.get(mt, set()) for mt in lowered_media
            ):
                continue
        deduped.append(genre)
        seen.add(genre)
    return deduped


def merge_query_filter_hints(
    intent: IntentFilters,
    query_filters: QueryFiltersResult | None,
    db: Session,
) -> None:
    if not query_filters:
        return

    def _extend_unique(target: List[str], values: Sequence[str]) -> None:
        for value in values:
            if not value:
                continue
            if value not in target:
                target.append(value)

    media_types = [
        mt.strip().lower()
        for mt in (query_filters.media_types or ())
        if isinstance(mt, str) and mt.strip()
    ]
    if not intent.media_types:
        _extend_unique(intent.media_types, media_types)

    normalized_genres = llm_parser._normalize_genre_names(
        list(query_filters.genres or ())
    )
    _extend_unique(intent.genres, normalized_genres)
    normalized_genre_set = {
        genre.lower()
        for genre in intent.genres
        if isinstance(genre, str) and genre.strip()
    }

    keyword_values = [
        keyword.strip().lower()
        for keyword in (query_filters.keywords or ())
        if isinstance(keyword, str) and keyword.strip()
    ]
    filtered_keywords: List[str] = []
    genre_keywords: List[str] = []
    processed_keywords = False
    for keyword in keyword_values:
        processed_keywords = True
        normalized_keyword = llm_parser._normalize_genre_names([keyword])
        lowered_candidates = {
            candidate.lower()
            for candidate in normalized_keyword
            if isinstance(candidate, str)
        }
        if lowered_candidates and any(
            candidate in normalized_genre_set for candidate in lowered_candidates
        ):
            genre_keywords.append(keyword)
            continue
        filtered_keywords.append(keyword)

    _extend_unique(intent.genre_keywords, genre_keywords)

    top_query_fn = get_hook("_get_top_query_keywords", get_top_query_keywords)
    top_query_keywords = top_query_fn(db)
    non_top_keywords = [kw for kw in filtered_keywords if kw not in top_query_keywords]

    fallback_keywords_used = False
    if not non_top_keywords:
        _extend_unique(intent.keywords, filtered_keywords)
        _extend_unique(intent.keywords, genre_keywords)
        fallback_keywords_used = True
    else:
        _extend_unique(intent.keywords, filtered_keywords)

    if processed_keywords:
        setattr(intent, "_query_keywords_merged", True)
    if fallback_keywords_used:
        setattr(intent, "_fallback_keywords_used", True)

    titles = [
        title.strip()
        for title in (query_filters.reference_titles or ())
        if isinstance(title, str) and title.strip()
    ]
    _extend_unique(intent.reference_titles, titles)
