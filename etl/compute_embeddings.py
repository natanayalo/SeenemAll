from __future__ import annotations

import os
from typing import List, Tuple, Dict, Any

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from api.db.session import get_engine as _get_engine, get_sessionmaker
from api.core.embeddings import encode_texts
from api.db.models import Item, ItemEmbedding
from etl.embedding_templates import format_with_template

# Re-export database engine accessor for tests that monkeypatch this module.
get_engine = _get_engine

MAX_LEN = 800  # limit overview length to keep inputs compact
DEFAULT_EMBED_VERSION = os.getenv("EMBED_VERSION", "v1")


def _build_text(
    item_or_title: Dict[str, Any] | str, overview: str | None = None
) -> str:
    """Build text representation using configured template.

    Can be called in two ways:
    1. _build_text(item_dict) - Formats an item dictionary using template
    2. _build_text(title, overview) - Direct text formatting for testing
    """
    if isinstance(item_or_title, dict):
        item = item_or_title
        template = os.getenv("EMBED_TEMPLATE", "basic")
        overview = item.get("overview", "")
        if overview and len(overview) > MAX_LEN:
            item = dict(item)  # Make a copy to avoid modifying original
            item["overview"] = overview[:MAX_LEN] + "..."
        return format_with_template(template, item)
    else:
        title = item_or_title
        overview = overview or ""
        if overview and len(overview) > MAX_LEN:
            overview = overview[:MAX_LEN] + "..."
        return f"{title} :: {overview}"


def _select_missing_ids(db: Session, limit: int, version: str) -> List[int]:
    """
    Return item ids that do not have an embedding yet.
    """
    # Using raw SQL LEFT JOIN for speed/readability
    q = text(
        """
        SELECT i.id
        FROM items i
        LEFT JOIN item_embeddings e ON e.item_id = i.id AND e.version = :version
        WHERE e.id IS NULL
        ORDER BY i.id
        LIMIT :lim
"""
    )
    rows = db.execute(q, {"lim": limit, "version": version}).fetchall()
    return [r[0] for r in rows]


def _fetch_items(db: Session, ids: List[int]) -> List[Tuple[int, str]]:
    results: List[Tuple[int, str]] = []
    if not ids:
        return results

    # Select all item fields for full context
    stmt = select(Item).where(Item.id.in_(ids))

    for item in db.execute(stmt).scalars():
        # Convert SQLAlchemy model to dict for template processing
        item_dict = {
            "id": item.id,
            "title": item.title,
            "overview": item.overview,
            "genres": item.genres,
            "release_year": item.release_year,
            "media_type": item.media_type,
            "runtime": item.runtime,
        }
        text = _build_text(item_dict)
        results.append((item.id, text))

    return results


def _upsert_vectors(db: Session, ids: List[int], vectors, version: str) -> None:
    # Bulk upsert using SQLAlchemy ORM-friendly approach
    payload = []
    for item_id, vec in zip(ids, vectors):
        payload.append({"item_id": int(item_id), "vector": vec.tolist()})
    # Try insert, then update existing (idempotent behavior)
    # (item_id, version) is unique so this simple pattern is safe:
    for row in payload:
        existing = (
            db.query(ItemEmbedding)
            .filter(
                ItemEmbedding.item_id == row["item_id"],
                ItemEmbedding.version == version,
            )
            .one_or_none()
        )
        if existing:
            existing.vector = row["vector"]
        else:
            db.add(
                ItemEmbedding(
                    item_id=row["item_id"], version=version, vector=row["vector"]
                )
            )


def run(
    batch: int = 256, max_items: int | None = None, version: str | None = None
) -> None:
    version = version or DEFAULT_EMBED_VERSION
    SessionLocal = get_sessionmaker()
    processed = 0

    with SessionLocal() as db:
        while True:
            todo = _select_missing_ids(db, limit=batch, version=version)
            if not todo:
                break
            if max_items is not None:
                remaining = max_items - processed
                if remaining <= 0:
                    break
                if len(todo) > remaining:
                    todo = todo[:remaining]

            items = _fetch_items(db, todo)
            texts = [t for _, t in items]
            vectors = encode_texts(texts)  # (N, 384) float32 normalized
            ids = [i for (i, _) in items]

            _upsert_vectors(db, ids, vectors, version=version)
            db.commit()

            processed += len(ids)
            print(f"Embedded {processed} items for version '{version}'...")

    print(f"Embedding run complete for version '{version}'.")


if __name__ == "__main__":  # pragma: no cover
    run()
