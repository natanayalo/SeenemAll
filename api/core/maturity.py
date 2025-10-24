from __future__ import annotations

from typing import Optional

_CANONICAL_ALIASES = {
    "G": "G",
    "E": "G",
    "PG": "PG",
    "PG13": "PG-13",
    "PG-13": "PG-13",
    "PG_13": "PG-13",
    "R": "R",
    "NC17": "NC-17",
    "NC-17": "NC-17",
    "NC_17": "NC-17",
    "NR": "NR",
    "NOTRATED": "NR",
    "NOT-RATED": "NR",
    "NOT RATED": "NR",
    "UNRATED": "NR",
    "TVY": "TV-Y",
    "TV-Y": "TV-Y",
    "TV_Y": "TV-Y",
    "TVG": "TV-G",
    "TV-G": "TV-G",
    "TVG7": "TV-Y7",
    "TVY7": "TV-Y7",
    "TV-Y7": "TV-Y7",
    "TVY7FV": "TV-Y7-FV",
    "TV-Y7-FV": "TV-Y7-FV",
    "TVPG": "TV-PG",
    "TV-PG": "TV-PG",
    "TV14": "TV-14",
    "TV-14": "TV-14",
    "TVMA": "TV-MA",
    "TV-MA": "TV-MA",
    "MA15": "MA15+",
    "MA15+": "MA15+",
    "M": "M",
    "16": "MA15+",
    "18": "NC-17",
    "18+": "NC-17",
    "R18": "NC-17",
    "R18+": "NC-17",
}

_RATING_LEVELS: dict[str, int] = {
    "TV-Y": 0,
    "TV-G": 0,
    "G": 0,
    "TV-Y7": 7,
    "TV-Y7-FV": 7,
    "PG": 10,
    "TV-PG": 10,
    "PG-13": 13,
    "TV-14": 14,
    "MA15+": 15,
    "M": 15,
    "R": 17,
    "TV-MA": 17,
    "NC-17": 18,
    "NR": 99,
}


def normalize_rating(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    token = cleaned.upper().replace("_", "-")
    compact = token.replace(" ", "")

    normalized = _CANONICAL_ALIASES.get(compact)
    if normalized:
        return normalized

    normalized = _CANONICAL_ALIASES.get(token)
    if normalized:
        return normalized

    return token


def rating_level(raw: Optional[str]) -> Optional[int]:
    normalized = normalize_rating(raw)
    if normalized is None:
        return None
    return _RATING_LEVELS.get(normalized)


def is_rating_within(item_rating: Optional[str], max_rating: Optional[str]) -> bool:
    if max_rating is None:
        return True
    max_level = rating_level(max_rating)
    if max_level is None:
        return True
    item_level = rating_level(item_rating)
    if item_level is None:
        return False
    return item_level <= max_level
