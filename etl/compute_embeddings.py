from __future__ import annotations

from typing import List, Tuple

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from api.db.session import get_engine as _get_engine, get_sessionmaker
from api.core.embeddings import encode_texts
from api.db.models import Item, ItemEmbedding

# Re-export database engine accessor for tests that monkeypatch this module.
get_engine = _get_engine

MAX_LEN = 800  # limit overview length to keep inputs compact


def _build_text(
    title: str | None, overview: str | None, genres: List[str] | None = None
) -> str:
    title = (title or "").strip()
    overview = (overview or "").strip()
    if len(overview) > MAX_LEN:
        overview = overview[:MAX_LEN] + "..."

    # Build genre prefix if available
    genre_text = ""
    if genres:
        genre_text = f"[{', '.join(genres)}] "

    # Combine all components
    if overview:
        return f"{genre_text}{title} :: {overview}"
    return f"{genre_text}{title}"


def _select_missing_ids(db: Session, limit: int) -> List[int]:
    """
    Return item ids that do not have an embedding yet.
    """
    # Using raw SQL LEFT JOIN for speed/readability
    q = text(
        """
        SELECT i.id
        FROM items i
        LEFT JOIN item_embeddings e ON e.item_id = i.id
        WHERE e.id IS NULL
        ORDER BY i.id
        LIMIT :lim
    """
    )
    rows = db.execute(q, {"lim": limit}).fetchall()
    return [r[0] for r in rows]


def _fetch_items(db: Session, ids: List[int]) -> List[Tuple[int, str]]:
    results: List[Tuple[int, str]] = []
    if not ids:
        return results

    stmt = select(Item.id, Item.title, Item.overview, Item.genres).where(
        Item.id.in_(ids)
    )

    for iid, title, overview, genres in db.execute(stmt):
        # Extract genre names from the JSONB array of genre objects
        genre_names = [g.get("name") for g in (genres or []) if g.get("name")]
        text = _build_text(title, overview, genre_names)
        results.append((iid, text))

    return results


def _upsert_vectors(db: Session, ids: List[int], vectors) -> None:
    # Bulk upsert using SQLAlchemy ORM-friendly approach
    payload = []
    for item_id, vec in zip(ids, vectors):
        payload.append({"item_id": int(item_id), "vector": vec.tolist()})
    # Try insert, then update existing (idempotent behavior)
    # Since item_id is unique in item_embeddings, use simple pattern:
    for row in payload:
        existing = (
            db.query(ItemEmbedding)
            .filter(ItemEmbedding.item_id == row["item_id"])
            .one_or_none()
        )
        if existing:
            existing.vector = row["vector"]
        else:
            db.add(ItemEmbedding(item_id=row["item_id"], vector=row["vector"]))


def run(batch: int = 256, max_items: int | None = None) -> None:
    SessionLocal = get_sessionmaker()
    processed = 0

    with SessionLocal() as db:
        while True:
            todo = _select_missing_ids(db, limit=batch)
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

            _upsert_vectors(db, ids, vectors)
            db.commit()

            processed += len(ids)
            print(f"Embedded {processed} items...")

    print("Embedding run complete.")


if __name__ == "__main__":  # pragma: no cover
    run()
