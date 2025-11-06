from __future__ import annotations
from typing import List, Sequence, Optional

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
) -> List[int]:
    where_clauses = ["NOT (e.item_id = ANY(:exclude))"]
    params = {
        "exclude": list(exclude_ids or []),
        "uvec": list(map(float, user_vec)),
        "lim": limit,
    }

    if allowed_ids is not None:
        if len(allowed_ids) == 0:
            return []
        where_clauses.append("e.item_id = ANY(:allowed)")
        params["allowed"] = list(allowed_ids)

    query = text(
        """
        SELECT e.item_id
        FROM item_embeddings e
        WHERE {where_clause}
        ORDER BY e.vector <-> :uvec
        LIMIT :lim
        """.format(
            where_clause=" AND ".join(where_clauses)
        )
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
