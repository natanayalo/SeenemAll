"""Resolve benchmark references without treating annotations as catalog facts."""

from typing import Any

from evaluation.models import TypedId


def reconcile_reference(
    reference: Any, catalog: dict[str, dict[str, Any]], default_media: str = "movie"
) -> tuple[Any, str]:
    """Return a unique catalog-backed reference, or quarantine an unresolved one.

    Matching a title verifies identity only, never relevance to its query.
    """
    tid = TypedId.parse(reference, default_media)
    metadata = catalog.get(str(tid))
    title = reference.get("title") if isinstance(reference, dict) else None
    if not title:
        return (reference, "unchanged") if metadata else (None, "missing_catalog")

    def normalize(text: str | None) -> str:
        return " ".join((text or "").casefold().split())

    if metadata and normalize(title) == normalize(metadata.get("title", "")):
        return reference, "unchanged"
    matches = [
        TypedId.parse(key)
        for key, row in catalog.items()
        if normalize(title) == normalize(row.get("title", ""))
    ]
    if len(matches) != 1:
        return None, "ambiguous_title" if matches else "unresolved_title"
    resolved = matches[0]
    return {
        **reference,
        "id": resolved.id,
        "tmdb_id": resolved.id,
        "media_type": resolved.media_type,
        "title": catalog[str(resolved)]["title"],
    }, "resolved_by_unique_title"
