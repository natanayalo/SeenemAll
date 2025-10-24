from __future__ import annotations

import functools
import logging
import time
import re
from datetime import timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from api.core.prompt_eval import load_prompt_template
from api.core.maturity import is_rating_within
from api.db.models import Item
from api.db.session import get_sessionmaker


@functools.lru_cache(maxsize=1)
def _get_prompt_config():
    """Load the intent parser prompt configuration."""
    return load_prompt_template("intent_parser")


def _get_genre_mapping():
    """Get genre mapping from prompt config."""
    config = _get_prompt_config()
    return config.get("genre_mapping", {})


# Canonical TMDB genres we care about
_CANONICAL_GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction",
    "TV Movie",
    "Thriller",
    "War",
    "Western",
]

# Map user utterances -> canonical genres
_GENRE_SYNONYMS: dict[str, List[str]] = {g.lower(): [g] for g in _CANONICAL_GENRES}
_GENRE_SYNONYMS.update(
    {
        "sci fi": ["Science Fiction"],
        "sci-fi": ["Science Fiction"],
        "scifi": ["Science Fiction"],
        "romcom": ["Romance", "Comedy"],
        "rom com": ["Romance", "Comedy"],
        "feel good": ["Comedy", "Family"],
        "feel-good": ["Comedy", "Family"],
        "superhero": ["Action", "Science Fiction"],
        "kids": ["Family", "Animation"],
        "animated": ["Animation"],
        "anime": ["Animation"],
        "biopic": ["Drama"],
        "epic": ["Adventure"],
        "period": ["Drama", "History"],
        "historical": ["History", "Drama"],
        "noir": ["Crime", "Mystery"],
        "doc": ["Documentary"],
    }
)

# Moods we detect + keywords
_MOOD_SYNONYMS: dict[str, Sequence[str]] = {
    "light": (
        "light",
        "uplifting",
        "wholesome",
        "feel good",
        "feel-good",
        "heartwarming",
    ),
    "dark": ("dark", "gritty", "bleak", "brooding"),
    "funny": ("funny", "hilarious", "comedic", "witty"),
    "romantic": ("romantic", "date night", "love story", "lovey"),
    "scary": ("scary", "spooky", "terrifying", "creepy"),
    "exciting": (
        "exciting",
        "thrilling",
        "adrenaline",
        "action packed",
        "action-packed",
    ),
}

# Mood -> fallback genres (used when no explicit genre is found)
_MOOD_TO_GENRES: dict[str, Sequence[str]] = {
    "light": ("Comedy", "Family", "Animation"),
    "funny": ("Comedy",),
    "romantic": ("Romance", "Comedy"),
    "scary": ("Horror", "Thriller"),
    "dark": ("Thriller", "Crime", "Mystery"),
    "exciting": ("Action", "Adventure", "Thriller"),
}

_MEDIA_TYPE_KEYWORDS: dict[str, Sequence[str]] = {
    "movie": ("movie", "movies", "film", "films"),
    "tv": ("tv", "series", "show", "shows", "tv show", "tv series"),
}

_RUNTIME_OPERATORS = (
    "<=",
    ">=",
    "<",
    ">",
    "less than",
    "under",
    "at most",
    "no more than",
    "more than",
    "over",
    "longer than",
    "shorter than",
    "at least",
    "minimum",
    "greater than",
)

_RUNTIME_PATTERN = re.compile(
    r"(?P<op>"
    + "|".join(
        re.escape(op) for op in sorted(_RUNTIME_OPERATORS, key=len, reverse=True)
    )
    + r")\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>h(?:ours?)?|hr|hrs|minutes?|mins?|m)\b",
    flags=re.IGNORECASE,
)

_BETWEEN_PATTERN = re.compile(
    r"between\s+(?P<low>\d+(?:\.\d+)?)\s*(?P<unit_low>h(?:ours?)?|hr|hrs|minutes?|mins?|m)"
    r"\s+(?:and|to)\s+(?P<high>\d+(?:\.\d+)?)\s*(?P<unit_high>h(?:ours?)?|hr|hrs|minutes?|mins?|m)",
    flags=re.IGNORECASE,
)


@dataclass
class IntentFilters:
    raw_query: str
    genres: List[str] = field(default_factory=list)
    moods: List[str] = field(default_factory=list)
    media_types: List[str] = field(default_factory=list)
    min_runtime: Optional[int] = None
    max_runtime: Optional[int] = None
    maturity_rating_max: Optional[str] = None

    def effective_genres(self) -> List[str]:
        seen: set[str] = set()
        ordered: List[str] = []

        # Add explicitly mentioned genres first
        for genre in self.genres:
            if genre not in seen:
                ordered.append(genre)
                seen.add(genre)

        # Add genres from mood mappings
        genre_mapping = _get_genre_mapping()
        for mood in self.moods:
            # Check both the legacy mapping and prompt-based mapping
            legacy_genres = _MOOD_TO_GENRES.get(mood, ())
            prompt_genres = genre_mapping.get(mood.lower(), [])

            for mapped in set(legacy_genres) | set(prompt_genres):
                if mapped not in seen:
                    ordered.append(mapped)
                    seen.add(mapped)

        return ordered

    def has_filters(self) -> bool:
        return bool(
            self.media_types
            or self.min_runtime is not None
            or self.max_runtime is not None
            or self.effective_genres()
            or self.maturity_rating_max
        )


def parse_intent(query: str | None) -> IntentFilters:
    if not query:
        return IntentFilters(raw_query=query or "")

    normalized = _normalize(query)
    genres = _extract_genres(normalized)
    moods = _extract_moods(normalized)
    media_types = _extract_media_types(normalized)
    min_runtime, max_runtime = _extract_runtime(query)

    filters = IntentFilters(
        raw_query=query,
        genres=genres,
        moods=moods,
        media_types=media_types,
        min_runtime=min_runtime,
        max_runtime=max_runtime,
    )

    # Validate parser output in debug/test environments
    if logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
        config = _get_prompt_config()
        for example in config.get("examples", []):
            if example.get("input") == query:
                from api.core.prompt_eval import evaluate_intent_parser_output

                output = {
                    "media_types": filters.media_types,
                    "genres": filters.effective_genres(),
                    "min_runtime": filters.min_runtime,
                    "max_runtime": filters.max_runtime,
                }
                score, errors = evaluate_intent_parser_output(
                    output, example["expected_output"], query
                )
                if errors:
                    logging.getLogger(__name__).debug(
                        "Intent parser evaluation found issues: %s", errors
                    )
                logging.getLogger(__name__).debug(
                    "Intent parser evaluation score: %.2f", score
                )
                break

    return filters


def item_matches_intent(
    item: "Item | object",
    filters: IntentFilters | None,
    *,
    enforce_genres: bool = True,
) -> bool:
    if not filters or not filters.has_filters():
        return True

    if filters.media_types:
        media_type = getattr(item, "media_type", None)
        if media_type not in filters.media_types:
            return False

    runtime = getattr(item, "runtime", None)
    if filters.min_runtime is not None:
        if runtime is None or runtime < filters.min_runtime:
            return False
    if filters.max_runtime is not None:
        if runtime is None or runtime > filters.max_runtime:
            return False

    if filters.maturity_rating_max:
        item_rating = getattr(item, "maturity_rating", None)
        if not is_rating_within(item_rating, filters.maturity_rating_max):
            return False

    if enforce_genres:
        effective_genres = {g.lower() for g in filters.effective_genres()}
        if effective_genres:
            item_genres = _item_genre_names(item)
            if not item_genres & effective_genres:
                return False

    return True


_GENRE_CACHE_TTL_SECONDS = timedelta(minutes=30).total_seconds()
_GENRE_CACHE: Dict[tuple[str, ...], tuple[float, List[str]]] = {}
logger = logging.getLogger(__name__)


def canonical_genres(media_types: Sequence[str] | None = None) -> List[str]:
    """
    Return the catalog-aware genre list.

    Attempts to hydrate the set from the database to reflect the latest ingest.
    Falls back to the hard-coded canonical list when the database is unavailable
    or returns no data. Results are cached for a short period to avoid repeated
    catalog scans.
    """
    if media_types:
        normalized = tuple(
            sorted(
                {
                    mt.strip().lower()
                    for mt in media_types
                    if isinstance(mt, str) and mt.strip()
                }
            )
        )
    else:
        normalized = ()
    cache_key = normalized if normalized else ("__all__",)
    now = time.monotonic()
    cached = _GENRE_CACHE.get(cache_key)
    if cached and now - cached[0] < _GENRE_CACHE_TTL_SECONDS:
        return list(cached[1])

    loaded = _load_genres_from_db(list(normalized) or None)
    if loaded:
        if logger.isEnabledFor(logging.DEBUG):
            preview = ", ".join(loaded[: min(len(loaded), 30)])
            tail = f"... (+{len(loaded) - 30} more)" if len(loaded) > 30 else ""
            logger.debug(
                "Loaded %d catalog genres%s: %s%s",
                len(loaded),
                f" for media_types={normalized!r}" if normalized else "",
                preview,
                tail,
            )
        _GENRE_CACHE[cache_key] = (now, loaded)
        return list(loaded)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Falling back to canonical genre list%s (cache_key=%s).",
            f" for media_types={normalized!r}" if normalized else "",
            cache_key,
        )
    return list(_CANONICAL_GENRES)


def _load_genres_from_db(media_types: Sequence[str] | None) -> List[str] | None:
    try:
        SessionLocal = get_sessionmaker()
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug(
            "Unable to obtain sessionmaker for catalog genres: %s", exc, exc_info=True
        )
        return None

    session = None
    result = None
    try:
        session = SessionLocal()
        stmt = select(Item.media_type, Item.genres).where(Item.genres.isnot(None))
        if media_types:
            stmt = stmt.where(Item.media_type.in_(list(media_types)))
        stmt = stmt.execution_options(stream_results=True)
        result = session.execute(stmt)
        genres: Dict[str, str] = {}
        for _media_type, payload in result:
            for name in _iterate_genre_names(payload):
                key = name.lower()
                if key not in genres:
                    genres[key] = name
        if not genres:
            logger.debug(
                "Catalog genre query returned no rows%s.",
                f" for media_types={media_types!r}" if media_types else "",
            )
            return None
        return sorted(genres.values())
    except SQLAlchemyError as exc:
        logger.debug(
            "Failed to load catalog genres from database: %s", exc, exc_info=True
        )
        return None
    finally:
        if result is not None:
            result.close()
        if session is not None:
            session.close()


def _iterate_genre_names(raw: object) -> Iterable[str]:
    if raw is None:
        return
    raw_iter: Iterable[Any] = raw if isinstance(raw, list) else [raw]

    for entry in raw_iter:
        if isinstance(entry, dict):
            name = entry.get("name")
        else:
            name = entry
        if isinstance(name, str):
            cleaned = name.strip()
            if cleaned:
                yield cleaned


def _item_genre_names(item: "Item | object") -> set[str]:
    raw = getattr(item, "genres", None)
    names: set[str] = set()
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                name = entry.get("name")
            else:
                name = entry
            if isinstance(name, str):
                names.add(name.lower())
    elif isinstance(raw, dict):
        name = raw.get("name")
        if isinstance(name, str):
            names.add(name.lower())
    elif isinstance(raw, str):
        names.add(raw.lower())
    return names


def _normalize(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _extract_genres(normalized_text: str) -> List[str]:
    if not normalized_text:
        return []
    padded = f" {normalized_text} "
    found: List[str] = []
    seen: set[str] = set()
    for phrase, canonicals in _GENRE_SYNONYMS.items():
        token = f" {phrase} "
        if token in padded:
            for canonical in canonicals:
                if canonical not in seen:
                    found.append(canonical)
                    seen.add(canonical)
    return found


def _extract_moods(normalized_text: str) -> List[str]:
    if not normalized_text:
        return []
    padded = f" {normalized_text} "
    moods: List[str] = []
    for mood, keywords in _MOOD_SYNONYMS.items():
        for keyword in keywords:
            token = f" {keyword} "
            if token in padded:
                moods.append(mood)
                break
    return moods


def _extract_media_types(normalized_text: str) -> List[str]:
    if not normalized_text:
        return []
    padded = f" {normalized_text} "
    media: List[str] = []
    for media_type, keywords in _MEDIA_TYPE_KEYWORDS.items():
        for keyword in keywords:
            token = f" {keyword} "
            if token in padded:
                media.append(media_type)
                break
    return media


def _extract_runtime(text: str) -> tuple[Optional[int], Optional[int]]:
    lowered = text.lower()
    min_runtime: Optional[int] = None
    max_runtime: Optional[int] = None

    for match in _BETWEEN_PATTERN.finditer(lowered):
        low_minutes = _to_minutes(match.group("low"), match.group("unit_low"))
        high_minutes = _to_minutes(match.group("high"), match.group("unit_high"))
        if low_minutes is not None:
            min_runtime = (
                low_minutes if min_runtime is None else max(min_runtime, low_minutes)
            )
        if high_minutes is not None:
            max_runtime = (
                high_minutes if max_runtime is None else min(max_runtime, high_minutes)
            )

    for match in _RUNTIME_PATTERN.finditer(lowered):
        op = match.group("op").strip().lower()
        minutes = _to_minutes(match.group("value"), match.group("unit"))
        if minutes is None:
            continue
        if op in {
            "<",
            "<=",
            "less than",
            "under",
            "at most",
            "no more than",
            "shorter than",
        }:
            max_runtime = minutes if max_runtime is None else min(max_runtime, minutes)
        elif op in {
            ">",
            ">=",
            "more than",
            "over",
            "longer than",
            "at least",
            "minimum",
            "greater than",
        }:
            min_runtime = minutes if min_runtime is None else max(min_runtime, minutes)

    if (
        min_runtime is not None
        and max_runtime is not None
        and min_runtime > max_runtime
    ):
        # Swap if parsing produced inverted bounds.
        min_runtime, max_runtime = max_runtime, min_runtime

    return min_runtime, max_runtime


def _to_minutes(value: str, unit: str) -> Optional[int]:
    try:
        numeric = float(value)
    except ValueError:
        return None
    unit = unit.lower()
    if unit.startswith("h") or unit.startswith("hr"):
        minutes = int(round(numeric * 60))
    else:
        minutes = int(round(numeric))
    return minutes if minutes > 0 else None
