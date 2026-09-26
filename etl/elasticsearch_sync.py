from __future__ import annotations

import argparse
import datetime as dt
import os
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from elasticsearch import helpers
from elasticsearch.exceptions import TransportError
from sqlalchemy import select
from sqlalchemy.orm import Session

from api import config
from api.core.elasticsearch_client import create_elasticsearch_client
from api.db.models import Availability, Item, ItemEmbedding
from api.db.session import get_sessionmaker

DEFAULT_BATCH_SIZE = int(os.getenv("ES_SYNC_BATCH_SIZE", "500"))
DEFAULT_EMBED_VERSION = os.getenv("EMBED_VERSION", "v1")


@dataclass(slots=True)
class ItemSnapshot:
    id: int
    media_type: str
    title: str
    overview: Optional[str]
    runtime: Optional[int]
    release_year: Optional[int]
    maturity_rating: Optional[str]
    genres: Any
    popularity: Optional[float]
    cast: Any
    directors: Any
    producers: Any
    writers: Any
    keywords: Any
    spoken_languages: Any
    updated_at: dt.datetime


def _normalise_genres(genres_payload: Any) -> List[str]:
    if not genres_payload:
        return []
    if isinstance(genres_payload, list):
        names: List[str] = []
        for entry in genres_payload:
            if isinstance(entry, str):
                names.append(entry)
            elif isinstance(entry, Mapping):
                name = entry.get("name")
                if isinstance(name, str):
                    names.append(name)
        return names
    if isinstance(genres_payload, Mapping):
        values = genres_payload.get("genres") or genres_payload.get("names")
        return _normalise_genres(values)
    return []


def _normalise_providers(rows: Iterable[Tuple[int, str]]) -> Dict[int, List[str]]:
    providers: Dict[int, set[str]] = {}
    for item_id, service in rows:
        if not service:
            continue
        providers.setdefault(int(item_id), set()).add(str(service))
    return {item_id: sorted(services) for item_id, services in providers.items()}


def _vector_to_list(vector: Sequence[float] | Any) -> List[float]:
    if vector is None:
        return []
    if hasattr(vector, "tolist"):
        return [float(v) for v in vector.tolist()]
    try:
        return [float(v) for v in vector]
    except TypeError:
        return []


def _isoformat(value: dt.datetime) -> str:
    if value.tzinfo:
        return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    return value.replace(tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _build_document(
    item: ItemSnapshot,
    *,
    embedding: Sequence[float],
    providers: Sequence[str],
) -> Dict[str, Any]:
    def _normalise_person_list(payload: Any) -> List[str]:
        if not payload:
            return []
        names: List[str] = []
        if isinstance(payload, list):
            for entry in payload:
                if isinstance(entry, str):
                    names.append(entry)
                elif isinstance(entry, Mapping):
                    name = entry.get("name")
                    if isinstance(name, str):
                        names.append(name)
        return names

    def _normalise_keyword_list(payload: Any) -> List[str]:
        if not payload:
            return []
        result: List[str] = []
        if isinstance(payload, list):
            for entry in payload:
                if isinstance(entry, str):
                    result.append(entry)
                elif isinstance(entry, Mapping):
                    name = entry.get("name")
                    if isinstance(name, str):
                        result.append(name)
        return result

    def _normalise_language_list(payload: Any) -> List[str]:
        if not payload:
            return []
        result: List[str] = []
        if isinstance(payload, list):
            for entry in payload:
                if isinstance(entry, str):
                    result.append(entry)
                elif isinstance(entry, Mapping):
                    name = entry.get("name") or entry.get("english_name")
                    if isinstance(name, str):
                        result.append(name)
        return result

    return {
        "item_id": str(item.id),
        "title": item.title,
        "overview": item.overview or "",
        "genres": _normalise_genres(item.genres),
        "media_type": item.media_type,
        "runtime": int(item.runtime) if item.runtime is not None else None,
        "release_year": (
            int(item.release_year) if item.release_year is not None else None
        ),
        "maturity": item.maturity_rating,
        "streaming_providers": list(providers),
        "popularity": float(item.popularity) if item.popularity is not None else None,
        "cast": _normalise_person_list(item.cast),
        "directors": _normalise_person_list(item.directors),
        "producers": _normalise_person_list(item.producers),
        "writers": _normalise_person_list(item.writers),
        "keywords": _normalise_keyword_list(item.keywords),
        "spoken_languages": _normalise_language_list(item.spoken_languages),
        "embedding": [float(v) for v in embedding],
        "updated_at": _isoformat(item.updated_at),
    }


def _chunker(
    session: Session,
    *,
    batch_size: int,
    updated_after: Optional[dt.datetime],
) -> Iterator[List[ItemSnapshot]]:
    offset = 0
    while True:
        stmt = select(Item).order_by(Item.id).limit(batch_size).offset(offset)
        if updated_after is not None:
            stmt = stmt.where(Item.updated_at >= updated_after)
        rows = session.execute(stmt).scalars().all()
        if not rows:
            break
        snapshots = [
            ItemSnapshot(
                id=row.id,
                media_type=row.media_type,
                title=row.title,
                overview=row.overview,
                runtime=row.runtime,
                release_year=row.release_year,
                maturity_rating=row.maturity_rating,
                genres=row.genres,
                popularity=row.popularity,
                cast=row.cast,
                directors=row.directors,
                producers=row.producers,
                writers=row.writers,
                keywords=row.keywords,
                spoken_languages=row.spoken_languages,
                updated_at=row.updated_at,
            )
            for row in rows
        ]
        yield snapshots
        offset += batch_size


def _load_embeddings(
    session: Session,
    *,
    item_ids: Sequence[int],
    version: str,
) -> Dict[int, List[float]]:
    if not item_ids:
        return {}
    stmt = select(ItemEmbedding.item_id, ItemEmbedding.vector).where(
        ItemEmbedding.item_id.in_(item_ids),
        ItemEmbedding.version == version,
    )
    rows = session.execute(stmt).all()
    embeddings: Dict[int, List[float]] = {}
    for item_id, vector in rows:
        as_list = _vector_to_list(vector)
        if as_list:
            embeddings[int(item_id)] = as_list
    return embeddings


def _load_providers(
    session: Session, *, item_ids: Sequence[int]
) -> Dict[int, List[str]]:
    if not item_ids:
        return {}
    stmt = select(Availability.item_id, Availability.service).where(
        Availability.item_id.in_(item_ids)
    )
    rows = session.execute(stmt).all()
    return _normalise_providers(rows)


def _generate_actions(
    items: Iterable[ItemSnapshot],
    embeddings: Mapping[int, Sequence[float]],
    providers: Mapping[int, Sequence[str]],
    *,
    index_name: str,
) -> Iterator[Dict[str, Any]]:
    for item in items:
        embedding = embeddings.get(item.id)
        if not embedding:
            continue
        doc = _build_document(
            item,
            embedding=embedding,
            providers=providers.get(item.id, ()),
        )
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": doc["item_id"],
            "_source": doc,
        }


def sync_catalog(
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_items: Optional[int] = None,
    embed_version: str = DEFAULT_EMBED_VERSION,
    updated_after: Optional[dt.datetime] = None,
    refresh: bool = False,
) -> Dict[str, int]:
    SessionLocal = get_sessionmaker()
    client = create_elasticsearch_client()
    total_indexed = 0
    total_missing_embeddings = 0

    try:
        with SessionLocal() as session:
            for chunk in _chunker(
                session, batch_size=batch_size, updated_after=updated_after
            ):
                if max_items is not None and total_indexed >= max_items:
                    break

                item_ids = [item.id for item in chunk]
                embeddings = _load_embeddings(
                    session, item_ids=item_ids, version=embed_version
                )
                providers = _load_providers(session, item_ids=item_ids)

                actions = list(
                    _generate_actions(
                        chunk,
                        embeddings,
                        providers,
                        index_name=config.ELASTICSEARCH_ITEMS_INDEX,
                    )
                )
                total_missing_embeddings += len(chunk) - len(actions)

                if not actions:
                    continue

                if max_items is not None:
                    overflow = (total_indexed + len(actions)) - max_items
                    if overflow > 0:
                        actions = actions[:-overflow]

                if not actions:
                    break

                helpers.bulk(client, actions, refresh=refresh)
                total_indexed += len(actions)

                if max_items is not None and total_indexed >= max_items:
                    break
    finally:
        client.close()

    return {
        "indexed": total_indexed,
        "skipped_missing_embedding": total_missing_embeddings,
    }


def _parse_datetime(value: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime '{value}'; use ISO format like 2024-01-31T12:34:56Z"
        ) from exc


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync catalog items and embeddings into Elasticsearch."
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--embed-version", type=str, default=DEFAULT_EMBED_VERSION)
    parser.add_argument("--since", type=_parse_datetime, default=None)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force index refresh after bulk upserts (slower).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        stats = sync_catalog(
            batch_size=args.batch_size,
            max_items=args.max_items,
            embed_version=args.embed_version,
            updated_after=args.since,
            refresh=args.refresh,
        )
    except TransportError as exc:
        print(f"Elasticsearch sync failed: {exc}")
        return 1

    print(
        f"Indexed {stats['indexed']} items "
        f"(skipped {stats['skipped_missing_embedding']} without embeddings)."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
