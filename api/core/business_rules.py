from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Sequence

from api.core.legacy_intent_parser import IntentFilters

logger = logging.getLogger(__name__)
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False

_RULES_CACHE: Dict[str, Any] = {"path": None, "mtime": None, "rules": None}


def _get_rules_path() -> str:
    env_path = os.getenv("BUSINESS_RULES_PATH")
    if env_path:
        return env_path
    return os.path.join(os.getcwd(), "config", "business_rules.json")


def clear_rules_cache() -> None:
    global _RULES_CACHE
    _RULES_CACHE = {"path": None, "mtime": None, "rules": None}


def load_rules() -> Dict[str, Any]:
    path = _get_rules_path()
    global _RULES_CACHE

    try:
        mtime = os.path.getmtime(path)
    except FileNotFoundError:
        if _RULES_CACHE.get("path") != path:
            logger.info("Business rules file not found at %s; using defaults.", path)
        clear_rules_cache()
        _RULES_CACHE["path"] = path
        _RULES_CACHE["rules"] = {}
        return {}

    if (
        _RULES_CACHE.get("path") == path
        and _RULES_CACHE.get("mtime") == mtime
        and _RULES_CACHE.get("rules") is not None
    ):
        return _RULES_CACHE["rules"]  # type: ignore[return-value]

    try:
        with open(path, "r", encoding="utf-8") as handle:
            rules = json.load(handle)
    except json.JSONDecodeError:
        logger.warning("Failed to parse business rules at %s. Ensure valid JSON.", path)
        rules = {}

    _RULES_CACHE = {"path": path, "mtime": mtime, "rules": rules}
    return rules


def apply_business_rules(
    candidates: Sequence[Dict[str, Any]],
    intent: IntentFilters | None = None,
) -> List[Dict[str, Any]]:
    rules = load_rules()
    if not rules:
        return list(candidates)

    filters = rules.get("filters", {})
    boosts = rules.get("boosts", {})
    filtered: List[Dict[str, Any]] = [
        item for item in candidates if not _should_filter(item, filters, intent)
    ]

    if not filtered:
        return []

    if boosts:
        for item in filtered:
            _apply_boosts(item, boosts)
        filtered.sort(
            key=lambda item: -(float(item.get("retrieval_score") or 0.0)),
        )
        for idx, item in enumerate(filtered):
            item["original_rank"] = idx

    return filtered


def _should_filter(
    item: Dict[str, Any], filters: Dict[str, Any], intent: IntentFilters | None
) -> bool:
    if not filters:
        return False

    genre_names = set(_genre_names(item))
    if filters.get("exclude_genres"):
        exclude_genres = {str(name).strip() for name in filters["exclude_genres"]}
        if genre_names & exclude_genres:
            return True

    if filters.get("include_genres"):
        include_genres = {str(name).strip() for name in filters["include_genres"]}
        if include_genres and not (genre_names & include_genres):
            return True

    min_vote_count = _safe_int(filters.get("min_vote_count"))
    if min_vote_count is not None:
        vote_count = item.get("vote_count")
        if vote_count is None or vote_count < min_vote_count:
            return True

    min_year = _safe_int(filters.get("min_release_year"))
    if min_year and (item.get("release_year") or 0) < min_year:
        return True

    max_year = _safe_int(filters.get("max_release_year"))
    if max_year and item.get("release_year") and item["release_year"] > max_year:
        return True

    if filters.get("exclude_media_types"):
        excluded_media = {str(mt).strip() for mt in filters["exclude_media_types"]}
        if item.get("media_type") in excluded_media:
            return True

    if filters.get("exclude_ids"):
        excluded_ids = {
            parsed
            for parsed in (_safe_int(_id) for _id in filters["exclude_ids"])
            if parsed is not None
        }
        if _safe_int(item.get("id")) in excluded_ids:
            return True

    if filters.get("exclude_tmdb_ids"):
        excluded_tmdb = {
            parsed
            for parsed in (_safe_int(_id) for _id in filters["exclude_tmdb_ids"])
            if parsed is not None
        }
        if _safe_int(item.get("tmdb_id")) in excluded_tmdb:
            return True

    if filters.get("require_intent_genres"):
        if intent:
            intent_genres = set(intent.effective_genres())
            if intent_genres and not (genre_names & intent_genres):
                return True

    return False


def _apply_boosts(item: Dict[str, Any], boosts: Dict[str, Any]) -> None:
    base_score = float(item.get("retrieval_score") or 0.0)
    multiplier = 1.0
    bonus = 0.0

    genre_names = set(_genre_names(item))
    genre_multipliers = boosts.get("genre_multipliers") or {}
    for name, factor in genre_multipliers.items():
        if name in genre_names:
            try:
                multiplier *= float(factor)
            except (TypeError, ValueError):
                continue

    id_multipliers = boosts.get("id_multipliers") or {}
    multiplier *= _lookup_multiplier(id_multipliers, item, default=1.0)

    tmdb_multipliers = boosts.get("tmdb_id_multipliers") or {}
    multiplier *= _lookup_multiplier(tmdb_multipliers, item, key="tmdb_id", default=1.0)

    recent_release = boosts.get("recent_release") or {}
    threshold = _safe_int(recent_release.get("year"))
    if threshold and (item.get("release_year") or 0) >= threshold:
        try:
            bonus += float(recent_release.get("bonus", 0.0))
        except (TypeError, ValueError):
            pass

    popularity_boost = boosts.get("popularity_multiplier")
    if popularity_boost:
        pop_threshold = _safe_float(popularity_boost.get("threshold", 0.0))
        pop_factor = _safe_float(popularity_boost.get("factor", 1.0))
        if pop_threshold is not None and pop_factor is not None:
            popularity = float(item.get("popularity") or 0.0)
            if popularity >= pop_threshold:
                multiplier *= pop_factor

    item["retrieval_score"] = base_score * multiplier + bonus


def _lookup_multiplier(
    mapping: Dict[Any, Any],
    item: Dict[str, Any],
    key: str = "id",
    default: float = 1.0,
) -> float:
    if not mapping:
        return 1.0
    value = item.get(key)
    if value is None:
        return 1.0
    str_value = str(value)
    if str_value in mapping:
        try:
            return float(mapping[str_value])
        except (TypeError, ValueError):
            return default
    return 1.0


def _genre_names(item: Dict[str, Any]) -> Iterable[str]:
    genres = item.get("genres") or []
    for genre in genres:
        name = genre.get("name") if isinstance(genre, dict) else None
        if name:
            yield str(name)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None
