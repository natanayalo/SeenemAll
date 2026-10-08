"""Typed catalog hydration shared by qualification and pooled evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evaluation.models import ItemEvidence, TypedId

CATALOG_EVIDENCE_PATH = Path("evaluation/fixtures/catalog_evidence_v2.2.json")
LEGACY_CATALOG_PATH = Path("evaluation/fixtures/catalog_metadata.json")
CATALOG_FIELDS = (
    "tmdb_id",
    "media_type",
    "title",
    "overview",
    "genres",
    "keywords",
    "cast",
    "directors",
    "producers",
    "writers",
    "release_year",
    "runtime",
    "original_language",
    "maturity_rating",
    "collection_id",
    "collection_name",
)


def catalog_metadata_from_item(item: Any) -> dict[str, Any]:
    """Freeze semantic facts only; exclude scores, popularity and rank."""
    return {name: getattr(item, name, None) for name in CATALOG_FIELDS}


def index_catalog_metadata(data: Any) -> dict[str, dict[str, Any]]:
    """Index legacy numeric snapshots and typed snapshots without ID collisions."""
    if not isinstance(data, (dict, list)):
        raise ValueError("Catalog evidence must be a mapping or list")
    entries = data.items() if isinstance(data, dict) else ((None, row) for row in data)
    indexed = {}
    for key, row in entries:
        if not isinstance(row, dict):
            raise ValueError("Catalog evidence rows must be objects")
        raw_id = row.get("tmdb_id") or row.get("id") or key
        media_type = row.get("media_type")
        if isinstance(key, str) and ":" in key:
            tid = TypedId.parse(key)
            if raw_id is not None and TypedId.parse(raw_id, tid.media_type) != tid:
                raise ValueError(f"Catalog key/ID mismatch: {key}")
        else:
            if not media_type:
                raise ValueError("Numeric catalog IDs require an explicit media type")
            tid = TypedId.parse(raw_id, media_type)
        if media_type and media_type != tid.media_type:
            raise ValueError(f"Catalog media type mismatch: {tid}")
        if str(tid) in indexed:
            raise ValueError(f"Duplicate catalog item: {tid}")
        indexed[str(tid)] = {**row, "tmdb_id": tid.id, "media_type": tid.media_type}
    return indexed


def load_catalog_metadata(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Use the frozen corrected snapshot, then legacy data, then a live DB read.

    Explicit or existing corrupt snapshots fail rather than silently changing
    the benchmark source. The legacy snapshot's actual media types are honored.
    """
    candidates = (
        [path] if path is not None else [CATALOG_EVIDENCE_PATH, LEGACY_CATALOG_PATH]
    )
    for candidate in candidates:
        if candidate.exists():
            return index_catalog_metadata(
                json.loads(candidate.read_text(encoding="utf-8"))
            )
    if path is not None:
        raise FileNotFoundError(path)
    from api.db.models import Item
    from api.db.session import get_sessionmaker

    with get_sessionmaker()() as db:
        rows = db.query(Item).order_by(Item.media_type, Item.tmdb_id).all()
        return index_catalog_metadata([catalog_metadata_from_item(row) for row in rows])


def _names(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    names = []
    for value in values:
        name = value.get("name") if isinstance(value, dict) else value
        if isinstance(name, str) and name.strip() and name.strip() not in names:
            names.append(name.strip())
    return names


def build_item_evidence(tid: TypedId, metadata: Mapping[str, Any]) -> ItemEvidence:
    """Preserve available catalog facts and mark absent facts explicitly."""
    media_type = metadata.get("media_type") or tid.media_type
    if media_type != tid.media_type:
        raise ValueError(f"Evidence media type mismatch: {tid} versus {media_type}")
    return ItemEvidence(
        typed_id=tid,
        title=str(metadata.get("title") or metadata.get("name") or ""),
        synopsis=metadata.get("synopsis") or metadata.get("overview"),
        genres=_names(metadata.get("genres")),
        keywords=_names(metadata.get("keywords")),
        cast=_names(metadata.get("cast")),
        directors=_names(metadata.get("directors")),
        crew=_names(metadata.get("crew"))
        + [f"Producer: {name}" for name in _names(metadata.get("producers"))]
        + [f"Writer: {name}" for name in _names(metadata.get("writers"))],
        media_type=media_type,
        release_year=metadata.get("release_year"),
        runtime=metadata.get("runtime"),
        original_language=metadata.get("original_language"),
        maturity_rating=metadata.get("maturity_rating"),
        collection_id=metadata.get("collection_id"),
        collection_name=metadata.get("collection_name"),
    )


def pool_item_evidence(
    tid: TypedId, result: Mapping[str, Any], catalog: Mapping[str, Any]
) -> ItemEvidence:
    """Hydrate thin API results from the same frozen catalog used by the pilot."""
    metadata = dict(result)
    metadata.update(catalog.get(str(tid), {}))
    return build_item_evidence(tid, metadata)
