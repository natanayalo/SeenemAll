"""Public dataset loaders for evaluation experiments."""

from __future__ import annotations

from typing import List, TypedDict


class PublicDatasetEntry(TypedDict):
    query: str
    golden_ids: List[int]


def load_public_dataset(name: str) -> List[PublicDatasetEntry]:
    """Return evaluation entries sourced from a public dataset."""
    normalized = name.lower()
    if normalized == "movielens20m":
        raise NotImplementedError(
            "TODO: hook up MovieLens 20M loader. Expected to return a list of "
            "{'query': str, 'golden_ids': list[int]} dictionaries."
        )
    raise NotImplementedError(f"Unknown public dataset '{name}'.")
