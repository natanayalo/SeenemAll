"""Helpers for resolving human-authored titles to TMDB identifiers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional

try:
    from sqlalchemy import bindparam, create_engine, text
    from sqlalchemy.engine import Engine

    HAVE_SQLALCHEMY = True
except ImportError:  # pragma: no cover - graceful fallback when dependency missing
    create_engine = None  # type: ignore[assignment]
    text = None  # type: ignore[assignment]
    Engine = Any  # type: ignore[assignment]
    HAVE_SQLALCHEMY = False

EXACT_QUERY = """
SELECT tmdb_id
FROM items
WHERE title = :title
  AND (:mtype IS NULL OR media_type = :mtype)
  AND (:year IS NULL OR release_year = :year)
ORDER BY popularity DESC NULLS LAST
LIMIT 1
"""

RELAXED_QUERY = """
SELECT tmdb_id
FROM items
WHERE LOWER(title) LIKE LOWER(:title_like)
  AND (:mtype IS NULL OR media_type = :mtype)
ORDER BY ABS(COALESCE(release_year, 0) - COALESCE(:year, 0)),
         popularity DESC NULLS LAST
LIMIT 1
"""


@lru_cache(maxsize=4)
def _get_engine(dsn: str) -> Engine:
    if not HAVE_SQLALCHEMY:
        raise RuntimeError("SQLAlchemy is required for title resolution.")
    return create_engine(dsn, pool_pre_ping=True)


def resolve_titles_to_tmdb_ids(items: List[Dict[str, Any]], dsn: str) -> List[int]:
    """Resolve a list of title descriptors to TMDB identifiers using the local DB."""
    if not HAVE_SQLALCHEMY:
        print(
            "SQLAlchemy not installed; cannot resolve titles. "
            "Install project requirements to enable title-based evaluation."
        )
        return []
    try:
        engine = _get_engine(dsn)
    except Exception as exc:  # pragma: no cover - connection failures
        print(f"Failed to create engine for DSN {dsn!r}: {exc}")
        return []

    resolved: List[int] = []
    with engine.connect() as conn:
        for item in items:
            if not isinstance(item, dict):
                print(f"Skipping resolver entry with invalid format: {item!r}")
                continue

            title = item.get("title")
            if not title:
                print("Skipping resolver entry without a title:", item)
                continue

            media_type = item.get("media_type")
            year = item.get("year")
            params = {
                "title": title,
                "mtype": media_type,
                "year": year,
            }
            exact_match = _execute_scalar(conn, EXACT_QUERY, params)
            if exact_match is not None:
                resolved.append(exact_match)
                continue

            like_title = f"%{title}%"
            relaxed_match = _execute_scalar(
                conn,
                RELAXED_QUERY,
                {"title_like": like_title, "mtype": media_type, "year": year},
            )
            if relaxed_match is not None:
                resolved.append(relaxed_match)
            else:
                print(
                    "Warning: no TMDB match for title "
                    f"{title!r} ({media_type or 'any'}, {year or 'unknown'})."
                )

    return resolved


def fetch_titles_for_tmdb_ids(tmdb_ids: Iterable[int], dsn: str) -> Dict[int, str]:
    """Return mapping of tmdb_id -> title for the provided ids."""
    if not HAVE_SQLALCHEMY:
        return {}

    unique_ids = {int(tid) for tid in tmdb_ids if tid is not None}
    if not unique_ids:
        return {}

    try:
        engine = _get_engine(dsn)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to connect for title lookup using DSN {dsn!r}: {exc}")
        return {}

    query = text("SELECT tmdb_id, title FROM items WHERE tmdb_id IN :ids").bindparams(
        bindparam("ids", expanding=True)
    )
    titles: Dict[int, str] = {}
    with engine.connect() as conn:
        try:
            result = conn.execute(query, {"ids": list(unique_ids)})
        except Exception as exc:  # pragma: no cover
            print(f"Title lookup query failed: {exc}")
            return {}
        for row in result:
            tmdb_id, title = row
            if tmdb_id is not None and title:
                titles[int(tmdb_id)] = title
    return titles


def _execute_scalar(conn: Any, query: str, params: Dict[str, Any]) -> Optional[int]:
    result = conn.execute(text(query), params)
    row = result.first()
    if row is None:
        return None
    tmdb_id = row[0]
    return int(tmdb_id) if tmdb_id is not None else None
