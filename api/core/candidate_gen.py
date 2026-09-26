from __future__ import annotations
from typing import List, Sequence, Optional, Dict, Any

import numpy as np
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector

from api import config
from api.core.elasticsearch_search import SearchFilters, knn_search


def _to_string_ids(values: Sequence[int] | None) -> List[str]:
    if not values:
        return []
    return [str(int(v)) for v in values]


_VALID_BACKENDS = {"elasticsearch", "pgvector"}


def _combine_sequences(existing: Sequence[str], additional: Sequence[str]) -> List[str]:
    seen = set(existing)
    result = list(existing)
    for value in additional:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _ann_candidates_elasticsearch(
    user_vec: np.ndarray,
    exclude_ids: Sequence[int],
    limit: int,
    allowed_ids: Sequence[int] | None,
    *,
    search_filters: Optional[SearchFilters],
    text_query: Optional[str],
) -> List[int]:
    base_filters = search_filters or SearchFilters()
    include_ids = _combine_sequences(
        base_filters.include_item_ids, _to_string_ids(allowed_ids)
    )
    exclude_ids_combined = _combine_sequences(
        base_filters.exclude_item_ids, _to_string_ids(exclude_ids)
    )
    filters = SearchFilters(
        include_item_ids=tuple(include_ids),
        genres=base_filters.genres,
        media_types=base_filters.media_types,
        providers=base_filters.providers,
        maturity=base_filters.maturity,
        languages=base_filters.languages,
        keywords=base_filters.keywords,
        cast=base_filters.cast,
        directors=base_filters.directors,
        producers=base_filters.producers,
        writers=base_filters.writers,
        release_year_gte=base_filters.release_year_gte,
        release_year_lte=base_filters.release_year_lte,
        runtime_gte=base_filters.runtime_gte,
        runtime_lte=base_filters.runtime_lte,
        exclude_item_ids=tuple(exclude_ids_combined),
    )

    hits = knn_search(
        list(map(float, user_vec)),
        k=limit,
        filters=filters,
        text_query=text_query or None,
        source_includes=["item_id"],
    )

    results: List[int] = []
    seen: set[int] = set()
    for hit in hits:
        raw_id = hit.get("item_id")
        if raw_id is None:
            continue
        try:
            as_int = int(raw_id)
        except (TypeError, ValueError):
            continue
        if as_int in seen:
            continue
        results.append(as_int)
        seen.add(as_int)
        if len(results) >= limit:
            break
    return results


def _ann_candidates_pgvector(
    db: Session,
    user_vec: np.ndarray,
    exclude_ids: Sequence[int],
    limit: int,
    allowed_ids: Sequence[int] | None,
    *,
    search_filters: Optional[SearchFilters] = None,
    text_query: Optional[str] = None,
) -> List[int]:
    base_filters = search_filters or SearchFilters()
    include_ids = _combine_sequences(
        base_filters.include_item_ids, _to_string_ids(allowed_ids)
    )
    exclude_ids_combined = _combine_sequences(
        base_filters.exclude_item_ids, _to_string_ids(exclude_ids)
    )

    if allowed_ids is not None and len(allowed_ids) == 0:
        return []
    if base_filters.include_item_ids and not include_ids:
        return []

    where_clauses: List[str] = []
    params: Dict[str, Any] = {
        "uvec": list(map(float, user_vec)),
        "lim": limit,
    }

    if exclude_ids_combined:
        clean_exclude = [int(v) for v in exclude_ids_combined if v.isdigit()]
        if clean_exclude:
            where_clauses.append("NOT (e.item_id = ANY(:exclude))")
            params["exclude"] = clean_exclude

    if include_ids:
        clean_include = [int(v) for v in include_ids if v.isdigit()]
        if clean_include:
            where_clauses.append("e.item_id = ANY(:allowed)")
            params["allowed"] = clean_include
        else:
            return []

    needs_item_join = False
    needs_availability_join = False

    if base_filters.media_types:
        needs_item_join = True
        where_clauses.append("i.media_type = ANY(:media_types)")
        params["media_types"] = list(base_filters.media_types)

    if base_filters.release_year_gte is not None:
        needs_item_join = True
        where_clauses.append("i.release_year >= :year_gte")
        params["year_gte"] = int(base_filters.release_year_gte)

    if base_filters.release_year_lte is not None:
        needs_item_join = True
        where_clauses.append("i.release_year <= :year_lte")
        params["year_lte"] = int(base_filters.release_year_lte)

    if base_filters.runtime_gte is not None:
        needs_item_join = True
        where_clauses.append("i.runtime >= :runtime_gte")
        params["runtime_gte"] = int(base_filters.runtime_gte)

    if base_filters.runtime_lte is not None:
        needs_item_join = True
        where_clauses.append("i.runtime <= :runtime_lte")
        params["runtime_lte"] = int(base_filters.runtime_lte)

    # People filters (cast, directors, producers, writers)
    people_conditions: List[str] = []

    def _add_people(vals: Sequence[str], col: str, prefix: str) -> None:
        for idx, val in enumerate(vals or [], start=1):
            key = f"{prefix}_{idx}"
            params[key] = f"%{val.strip().lower()}%"
            people_conditions.append(f"lower(i.{col}::text) LIKE :{key}")

    if base_filters.cast:
        _add_people(base_filters.cast, "cast", "cast")
    if base_filters.directors:
        _add_people(base_filters.directors, "directors", "dir")
    if base_filters.producers:
        _add_people(base_filters.producers, "producers", "prod")
    if base_filters.writers:
        _add_people(base_filters.writers, "writers", "writ")

    if people_conditions:
        needs_item_join = True
        where_clauses.append(f"({' OR '.join(people_conditions)})")

    if base_filters.genres:
        needs_item_join = True
        genre_conditions: List[str] = []
        for idx, g in enumerate(base_filters.genres, start=1):
            key = f"genre_{idx}"
            params[key] = f"%{g.strip().lower()}%"
            genre_conditions.append(f"lower(i.genres::text) LIKE :{key}")
        if genre_conditions:
            where_clauses.append(f"({' OR '.join(genre_conditions)})")

    if base_filters.languages:
        needs_item_join = True
        lang_conditions: List[str] = []
        for idx, lang in enumerate(base_filters.languages, start=1):
            key_exact = f"lang_{idx}"
            key_like = f"lang_like_{idx}"
            params[key_exact] = lang.strip().lower()
            params[key_like] = f"%{lang.strip().lower()}%"
            lang_conditions.append(
                f"(lower(i.original_language) = :{key_exact} OR lower(i.spoken_languages::text) LIKE :{key_like})"
            )
        if lang_conditions:
            where_clauses.append(f"({' OR '.join(lang_conditions)})")

    if base_filters.providers:
        needs_availability_join = True
        where_clauses.append("a.service = ANY(:providers)")
        where_clauses.append("a.country = :country")
        params["providers"] = list(base_filters.providers)
        params["country"] = "US"

    from_clause = "item_embeddings e"
    if needs_item_join:
        from_clause += " JOIN items i ON i.id = e.item_id"
    if needs_availability_join:
        from_clause += " JOIN availability a ON a.item_id = e.item_id"

    where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    query = text(
        f"""
        SELECT e.item_id
        FROM {from_clause}
        {where_str}
        ORDER BY e.vector <=> :uvec
        LIMIT :lim
        """
    ).bindparams(bindparam("uvec", type_=Vector(384)))

    rows = db.execute(query, params).fetchall()
    return [int(row[0]) for row in rows]


def ann_candidates(
    db: Session,
    user_vec: np.ndarray | None,
    exclude_ids: List[int] | Sequence[int],
    limit: int = 300,
    allowed_ids: Sequence[int] | None = None,
    backend_override: str | None = None,
    *,
    search_filters: Optional[SearchFilters] = None,
    text_query: Optional[str] = None,
) -> List[int]:
    """
    Returns item_ids ordered by cosine distance to user_vec.
    Backend is controlled by ANN_BACKEND (elasticsearch | pgvector) and can be
    overridden per call.
    """
    if user_vec is None:
        return []

    exclude_list = list(exclude_ids or [])

    if allowed_ids is not None and len(allowed_ids) == 0:
        return []

    backend = (backend_override or config.ANN_BACKEND or "").strip().lower()
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"Unsupported ANN backend '{backend}'")

    if backend == "pgvector":
        return _ann_candidates_pgvector(
            db,
            user_vec,
            exclude_list,
            limit,
            allowed_ids,
            search_filters=search_filters,
            text_query=text_query,
        )

    # Default to Elasticsearch unless explicitly overridden.
    return _ann_candidates_elasticsearch(
        user_vec,
        exclude_list,
        limit,
        allowed_ids,
        search_filters=search_filters,
        text_query=text_query,
    )
