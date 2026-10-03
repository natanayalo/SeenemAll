from __future__ import annotations

import logging
import inspect
from typing import Any, Dict, List, Optional, Sequence, Set

from sqlalchemy import bindparam, cast, func, or_, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session
from sqlalchemy.types import Text

from api.config import COUNTRY_DEFAULT
from api import config
from api.core import llm_parser
from api.core.elasticsearch_search import SearchFilters
from api.core.legacy_intent_parser import IntentFilters
from api.db.models import Availability, Item
from api.pipeline.hooks import get_hook
from api.pipeline.intent import _has_people_filters
from api.pipeline.models import PrefilterDecision

logger = logging.getLogger("api.routes.recommend")

LANGUAGE_TO_ISO: Dict[str, str] = {
    "french": "fr",
    "korean": "ko",
    "japanese": "ja",
    "spanish": "es",
    "german": "de",
    "italian": "it",
    "english": "en",
    "chinese": "zh",
    "mandarin": "zh",
    "hindi": "hi",
    "tamil": "ta",
    "swedish": "sv",
    "danish": "da",
    "norwegian": "no",
    "finnish": "fi",
}


def relax_filters_for_people(filters: SearchFilters | None) -> SearchFilters | None:
    if not filters:
        return None
    return SearchFilters(
        include_item_ids=filters.include_item_ids,
        media_types=filters.media_types,
        providers=filters.providers,
        maturity=filters.maturity,
        languages=filters.languages,
        keywords=filters.keywords,
        cast=filters.cast,
        directors=filters.directors,
        producers=filters.producers,
        writers=filters.writers,
        release_year_gte=filters.release_year_gte,
        release_year_lte=filters.release_year_lte,
        runtime_gte=filters.runtime_gte,
        runtime_lte=filters.runtime_lte,
        exclude_item_ids=filters.exclude_item_ids,
    )


def relax_people_filters(filters: SearchFilters | None) -> SearchFilters | None:
    """Drop people constraints when person names fail to match catalog entities."""
    if not filters:
        return None
    return SearchFilters(
        include_item_ids=filters.include_item_ids,
        media_types=filters.media_types,
        providers=filters.providers,
        maturity=filters.maturity,
        languages=filters.languages,
        keywords=filters.keywords,
        genres=filters.genres,
        cast=(),
        directors=(),
        producers=(),
        writers=(),
        release_year_gte=filters.release_year_gte,
        release_year_lte=filters.release_year_lte,
        runtime_gte=filters.runtime_gte,
        runtime_lte=filters.runtime_lte,
        exclude_item_ids=filters.exclude_item_ids,
        strict_genres=filters.strict_genres,
    )


def people_only_candidate_ids(
    db: Session,
    filters: SearchFilters,
    *,
    limit: int,
) -> List[int]:
    if not _has_people_filters(filters):
        return []

    name_conditions = []
    params: Dict[str, str] = {}
    text_cast_expr = func.lower(cast(Item.cast, Text))
    text_directors_expr = func.lower(cast(Item.directors, Text))
    text_producers_expr = func.lower(cast(Item.producers, Text))
    text_writers_expr = func.lower(cast(Item.writers, Text))

    def _add_conditions(values: Sequence[str], expr, prefix: str) -> None:
        for idx, value in enumerate(values or [], start=1):
            key = f"{prefix}_{idx}"
            params[key] = f"%{value.strip().lower()}%"
            name_conditions.append(expr.like(bindparam(key)))

    _add_conditions(filters.cast, text_cast_expr, "cast")
    _add_conditions(filters.directors, text_directors_expr, "director")
    _add_conditions(filters.producers, text_producers_expr, "producer")
    _add_conditions(filters.writers, text_writers_expr, "writer")

    if not name_conditions:
        return []

    stmt = select(Item.id).where(or_(*name_conditions))
    if filters.media_types:
        stmt = stmt.where(Item.media_type.in_(filters.media_types))
    stmt = stmt.limit(limit)

    rows = db.execute(stmt, params).scalars().all()
    return [int(row) for row in rows]


def filter_excluded_candidate_ids(
    candidates: Sequence[int], exclude_set: Set[int]
) -> List[int]:
    if not candidates:
        return []
    if not exclude_set:
        return list(candidates)
    return [candidate for candidate in candidates if candidate not in exclude_set]


def apply_search_filters(stmt, filters: SearchFilters | None, exclude_ids=()):
    """Apply hard catalog eligibility constraints to an Item-based SQL query."""
    if not filters:
        return stmt
    if filters.media_types:
        stmt = stmt.where(Item.media_type.in_(filters.media_types))
    if filters.genres:
        genre_clauses = [
            cast(Item.genres, Text).ilike(f"%{genre.strip().lower()}%")
            for genre in filters.genres
            if genre and genre.strip()
        ]
        if genre_clauses:
            if filters.strict_genres:
                stmt = stmt.where(*genre_clauses)
            else:
                stmt = stmt.where(or_(*genre_clauses))
    for column, minimum, maximum in (
        (Item.release_year, filters.release_year_gte, filters.release_year_lte),
        (Item.runtime, filters.runtime_gte, filters.runtime_lte),
    ):
        if minimum is not None:
            stmt = stmt.where(column >= minimum)
        if maximum is not None:
            stmt = stmt.where(column <= maximum)
    if filters.maturity:
        stmt = stmt.where(Item.maturity_rating.in_(filters.maturity))
    if filters.languages:
        clauses = [
            or_(
                func.lower(Item.original_language) == language.strip().lower(),
                cast(Item.spoken_languages, Text).ilike(
                    f"%{language.strip().lower()}%"
                ),
            )
            for language in filters.languages
            if language and language.strip()
        ]
        if clauses:
            stmt = stmt.where(or_(*clauses))
    people_clauses: List[Any] = []
    for values, column in (
        (filters.cast, Item.cast),
        (filters.directors, Item.directors),
        (filters.producers, Item.producers),
        (filters.writers, Item.writers),
    ):
        people_clauses.extend(
            cast(column, Text).ilike(f"%{value.strip().lower()}%")
            for value in values
            if value and value.strip()
        )
    if people_clauses:
        stmt = stmt.where(or_(*people_clauses))
    if filters.providers:
        stmt = stmt.where(
            Item.id.in_(
                select(Availability.item_id).where(
                    Availability.country == COUNTRY_DEFAULT,
                    Availability.service.in_(filters.providers),
                )
            )
        )
    exclusions = set(exclude_ids or ()) | {
        int(value) for value in filters.exclude_item_ids if str(value).isdigit()
    }
    if exclusions:
        stmt = stmt.where(~Item.id.in_(exclusions))
    return stmt


def ordered_unique(values: List[int]) -> List[int]:
    seen: set[int] = set()
    ordered: List[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def genre_contains_clause(db: Session, genre: str):
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

    return Item.genres.contains([{"name": genre}])


def keyword_contains_clause(db: Session, keyword: str):
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
        return cast(Item.keywords, JSONB).contains([{"name": keyword}])

    return Item.keywords.contains([{"name": keyword}])


def run_prefilter_query(
    db: Session,
    intent: IntentFilters,
    *,
    fetch_limit: int,
    include_genres: bool,
    required_services: Set[str] | None = None,
    prefer_top_rated: bool = False,
    require_all_genres: bool = False,
    genres_override: Sequence[str] | None = None,
    include_keywords: bool = False,
    require_all_keywords: bool = False,
    keywords_override: Sequence[str] | None = None,
    search_filters: SearchFilters | None = None,
) -> List[int]:
    stmt = select(Item.id)

    if required_services:
        stmt = stmt.where(
            Item.id.in_(
                select(Availability.item_id).where(
                    (Availability.country == COUNTRY_DEFAULT)
                    & Availability.service.in_(required_services)
                )
            )
        )

    media_types = intent.media_types or []
    if media_types:
        stmt = stmt.where(Item.media_type.in_(media_types))

    if include_genres:
        genres_source = (
            list(genres_override)
            if genres_override is not None
            else intent.effective_genres()
        )
        genres = list(genres_source or [])
        if genres:
            mapped_genres: List[str] = []
            if genres_override is None:
                for genre in genres:
                    normalized = llm_parser._normalize_genre_names([genre])
                    if normalized:
                        mapped_genres.extend(normalized)
            else:
                mapped_genres = [genre for genre in genres if genre]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Prefilter genre clauses | include_genres=%s require_all=%s mapped=%s",
                    include_genres,
                    require_all_genres,
                    mapped_genres,
                )
            genre_filters = []
            if mapped_genres:
                clause_fn = get_hook("_genre_contains_clause", genre_contains_clause)
                genre_filters = [
                    clause_fn(db, genre)
                    for genre in dict.fromkeys(mapped_genres)
                    if genre
                ]
            if genre_filters:
                if require_all_genres:
                    for clause in genre_filters:
                        stmt = stmt.where(clause)
                else:
                    stmt = stmt.where(or_(*genre_filters))

    if include_keywords:
        keywords_source = (
            list(keywords_override)
            if keywords_override is not None
            else list(intent.keywords or [])
        )
        keyword_values = [kw for kw in keywords_source if kw]
        keyword_filters = []
        if keyword_values:
            kw_clause_fn = get_hook("_keyword_contains_clause", keyword_contains_clause)
            keyword_filters = [
                kw_clause_fn(db, keyword)
                for keyword in dict.fromkeys(keyword_values)
                if keyword
            ]
        if keyword_filters:
            if require_all_keywords:
                for clause in keyword_filters:
                    stmt = stmt.where(clause)
            else:
                stmt = stmt.where(or_(*keyword_filters))

    if search_filters is not None:
        stmt = apply_search_filters(stmt, search_filters)

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
            Item.trending_rank.asc().nullslast(),
            Item.popular_rank.asc().nullslast(),
            Item.popularity.desc().nullslast(),
            Item.vote_average.desc().nullslast(),
            Item.vote_count.desc().nullslast(),
            Item.id.asc(),
        )

    rows = db.execute(stmt.limit(fetch_limit)).scalars().all()
    dedup_fn = get_hook("_ordered_unique", ordered_unique)
    return dedup_fn(rows)


def prefilter_allowed_ids(
    db: Session,
    intent: IntentFilters | None,
    limit: int,
    preferred_services: Set[str] | None = None,
    prefer_top_rated: bool = False,
    require_all_genres: bool = False,
    search_filters: SearchFilters | None = None,
) -> PrefilterDecision:
    if intent is None or not intent.has_filters():
        return PrefilterDecision(None, [], True, False)

    threshold = max(10, limit // 2)
    boost_cap = max(5, min(limit, 25))
    fetch_limit = (
        boost_cap if config.RETRIEVAL_CONSOLIDATED_FILTERS else max(limit * 5, limit)
    )
    strict_genres = intent.required_genres or intent.effective_genres()

    run_query_fn = get_hook("_run_prefilter_query", run_prefilter_query)
    accepts_search_filters = False
    try:
        signature = inspect.signature(run_query_fn)
        accepts_search_filters = "search_filters" in signature.parameters or any(
            item.kind is inspect.Parameter.VAR_KEYWORD
            for item in signature.parameters.values()
        )
    except (TypeError, ValueError):
        accepts_search_filters = True
    query_filter_kwargs = (
        {"search_filters": search_filters} if accepts_search_filters else {}
    )
    strict_ids = run_query_fn(
        db,
        intent,
        fetch_limit=fetch_limit,
        include_genres=True,
        required_services=preferred_services,
        prefer_top_rated=prefer_top_rated,
        require_all_genres=require_all_genres,
        genres_override=strict_genres,
        **query_filter_kwargs,
    )
    logger.debug(
        "Prefilter strict run | threshold=%d strict_count=%d media_types=%s genres=%s",
        threshold,
        len(strict_ids),
        intent.media_types,
        strict_genres,
    )

    def _unique_slice(values: Sequence[int], cap: int | None = None) -> List[int]:
        seen: set[int] = set()
        result: List[int] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
            if cap is not None and len(result) >= cap:
                break
        return result

    intent_keywords = [
        keyword.strip()
        for keyword in (intent.keywords or [])
        if isinstance(keyword, str) and keyword.strip()
    ]
    if hasattr(intent, "genre_keywords"):
        genre_keywords = [
            keyword.strip()
            for keyword in (intent.genre_keywords or [])
            if isinstance(keyword, str) and keyword.strip()
        ]
        if genre_keywords:
            intent_keywords.extend(genre_keywords)

    keyword_boost_ids: List[int] = []
    keyword_boost_active = False
    if (
        config.RETRIEVAL_PREFILTER_BOOSTS_ENABLED
        and intent_keywords
        and not getattr(intent, "is_vibe", False)
    ):
        keyword_boost_ids.extend(
            run_query_fn(
                db,
                intent,
                fetch_limit=fetch_limit,
                include_genres=True,
                required_services=preferred_services,
                prefer_top_rated=prefer_top_rated,
                require_all_genres=require_all_genres,
                genres_override=strict_genres,
                include_keywords=True,
                keywords_override=intent_keywords,
                **query_filter_kwargs,
            )
        )
        if len(keyword_boost_ids) < boost_cap:
            keyword_boost_ids.extend(
                run_query_fn(
                    db,
                    intent,
                    fetch_limit=fetch_limit,
                    include_genres=False,
                    required_services=preferred_services,
                    prefer_top_rated=prefer_top_rated,
                    require_all_genres=False,
                    include_keywords=True,
                    keywords_override=intent_keywords,
                    **query_filter_kwargs,
                )
            )
        keyword_boost_ids = ordered_unique(keyword_boost_ids)
        if keyword_boost_ids:
            keyword_boost_active = True

    keyword_min = max(1, boost_cap // 2)

    def _resolve_boost_ids(primary: Sequence[int]) -> List[int]:
        if not config.RETRIEVAL_PREFILTER_BOOSTS_ENABLED:
            return []
        if not strict_genres and not keyword_boost_ids:
            return []
        if getattr(intent, "reference_titles", None) or getattr(
            intent, "is_vibe", False
        ):
            return []
        if keyword_boost_ids:
            combined = list(keyword_boost_ids)
            if len(combined) < boost_cap or len(keyword_boost_ids) < keyword_min:
                if strict_genres:
                    combined.extend(primary)
            return _unique_slice(combined, boost_cap)
        return _unique_slice(primary, boost_cap)

    if config.RETRIEVAL_CONSOLIDATED_FILTERS:
        return PrefilterDecision(
            None,
            _resolve_boost_ids(strict_ids),
            bool(strict_genres),
            keyword_boost_active,
        )

    if len(strict_ids) >= threshold or prefer_top_rated:
        logger.debug("Prefilter returning strict allowlist.")
        unique_strict = _unique_slice(strict_ids)
        return PrefilterDecision(
            unique_strict, _resolve_boost_ids(unique_strict), True, keyword_boost_active
        )

    relaxed_ids = run_query_fn(
        db,
        intent,
        fetch_limit=fetch_limit,
        include_genres=False,
        required_services=preferred_services,
        prefer_top_rated=prefer_top_rated,
        require_all_genres=False,
        **query_filter_kwargs,
    )

    if len(relaxed_ids) >= threshold:
        logger.debug(
            "Prefilter relaxed pass selected | relaxed_count=%d strict_count=%d",
            len(relaxed_ids),
            len(strict_ids),
        )
        unique_relaxed = _unique_slice(relaxed_ids)
        return PrefilterDecision(
            unique_relaxed, _resolve_boost_ids(strict_ids), False, keyword_boost_active
        )

    logger.debug(
        "Prefilter falling back to ANN-first | relaxed_count=%d strict_count=%d",
        len(relaxed_ids),
        len(strict_ids),
    )
    enforce_on_fallback = bool(strict_genres)
    return PrefilterDecision(
        None, _resolve_boost_ids(strict_ids), enforce_on_fallback, keyword_boost_active
    )
