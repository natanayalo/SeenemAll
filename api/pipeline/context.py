from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import asdict
from threading import Lock
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from cachetools import TTLCache
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.core.metrics import METRICS
from api.core.user_utils import canonical_profile_id, load_user_state
from api.db.models import CatalogMetadata, Item
from api.pipeline.hooks import get_hook
from api.pipeline.models import ComputeResult, RecommendParams, UserContext

logger = logging.getLogger(__name__)

# --- In-Memory Catalog Metadata Caches ---
_MEDIA_GENRE_CACHE: Dict[str, Set[str]] | None = None
_MEDIA_GENRE_CACHE_TS: float | None = None
_MEDIA_GENRE_CACHE_TTL = 3600.0  # seconds (1 hour)
_CATALOG_METADATA_CACHE: Dict[str, Tuple[float, Any]] = {}
_CATALOG_METADATA_TTL = 3600.0

_DEFAULT_STREAMING_PROVIDER_ALIASES: Dict[str, Set[str]] = {
    "netflix": {"netflix", "nfx"},
    "disney_plus": {"disney_plus", "disney", "dnp"},
    "prime_video": {"prime_video", "primevideo", "amazon", "amz", "amp"},
    "hulu": {"hulu", "hlu"},
    "max": {"max", "hbomax", "hbo", "hbm"},
    "apple_tv_plus": {"apple_tv_plus", "appletvplus", "apple", "atp"},
    "paramount_plus": {"paramount_plus", "paramountplus", "prm", "pmnt", "paramount"},
}

_DEFAULT_TOP_QUERY_KEYWORDS = {
    "best",
    "best of",
    "must watch",
    "must-watch",
    "must see",
    "must-see",
    "essential",
    "top",
    "top-rated",
    "top rated",
    "classic",
    "classics",
    "all-time",
    "all time",
    "epic",
    "greatest",
    "award-winning",
}

# --- Response & Inflight Caching ---
_RECOMMEND_CACHE_TTL_SECONDS = int(os.getenv("RECOMMEND_CACHE_TTL_SECONDS", "300"))
_RECOMMEND_CACHE_MAXSIZE = int(os.getenv("RECOMMEND_CACHE_MAXSIZE", "512"))
_RECOMMEND_CACHE: TTLCache[str, Dict[str, Any]] = TTLCache(
    maxsize=_RECOMMEND_CACHE_MAXSIZE, ttl=_RECOMMEND_CACHE_TTL_SECONDS
)
_RECOMMEND_CACHE_LOCK = Lock()
_RECOMMEND_CACHE_USER_KEYS: Dict[str, Set[str]] = {}
_INFLIGHT_RECOMMEND_TASKS: Dict[Tuple[int, str], asyncio.Task[ComputeResult]] = {}
_INFLIGHT_TASKS_LOCK = Lock()


def get_catalog_metadata(db: Session, key: str) -> Any:
    now = time.time()
    cached = _CATALOG_METADATA_CACHE.get(key)
    if cached and now - cached[0] < _CATALOG_METADATA_TTL:
        return cached[1]
    try:
        result = db.execute(
            select(CatalogMetadata.data).where(CatalogMetadata.key == key)
        )
        if hasattr(result, "scalar_one_or_none"):
            value = result.scalar_one_or_none()
        else:
            rows = result.all() if hasattr(result, "all") else []
            raw = rows[0] if rows else None
            if isinstance(raw, (tuple, list)):
                value = raw[0]
            elif isinstance(raw, Mapping) and "data" in raw:
                value = raw["data"]
            else:
                value = raw
    except Exception:
        value = None
    _CATALOG_METADATA_CACHE[key] = (now, value)
    return value


def get_streaming_alias_map(db: Session) -> Dict[str, Set[str]]:
    data = get_catalog_metadata(db, "streaming_provider_aliases")
    mapping: Dict[str, Set[str]] = {}
    if isinstance(data, dict):
        for canonical, aliases in data.items():
            if not isinstance(canonical, str):
                continue
            alias_set: Set[str] = {canonical.strip().lower()}
            if isinstance(aliases, (list, tuple, set)):
                alias_set.update(
                    alias.strip().lower()
                    for alias in aliases
                    if isinstance(alias, str) and alias.strip()
                )
            elif isinstance(aliases, str) and aliases.strip():
                alias_set.add(aliases.strip().lower())
            mapping[canonical.strip().lower()] = alias_set
    if not mapping:
        mapping = {
            canonical: set(alias_list) | {canonical}
            for canonical, alias_list in _DEFAULT_STREAMING_PROVIDER_ALIASES.items()
        }
    return mapping


def get_top_query_keywords(db: Session) -> Set[str]:
    data = get_catalog_metadata(db, "top_query_keywords")
    keywords: Set[str] = set()
    if isinstance(data, (list, tuple, set)):
        for value in data:
            if isinstance(value, str) and value.strip():
                keywords.add(value.strip().lower())
    if not keywords:
        keywords = {kw.lower() for kw in _DEFAULT_TOP_QUERY_KEYWORDS}
    return keywords


def normalize_streaming_services(
    providers: Sequence[str] | None,
    alias_map: Mapping[str, Set[str]],
) -> Set[str]:
    normalized: Set[str] = set()
    if not providers:
        return normalized

    for provider in providers:
        if not isinstance(provider, str):
            continue
        key = provider.strip().lower()
        if not key:
            continue
        matched = False
        for canonical, variants in alias_map.items():
            if key in variants:
                normalized.update(variants)
                matched = True
                break
        if not matched:
            normalized.add(key)
    return normalized


def load_media_genres(db: Session) -> Dict[str, Set[str]]:
    global _MEDIA_GENRE_CACHE, _MEDIA_GENRE_CACHE_TS
    now = time.time()
    if _MEDIA_GENRE_CACHE and _MEDIA_GENRE_CACHE_TS:
        if now - _MEDIA_GENRE_CACHE_TS < _MEDIA_GENRE_CACHE_TTL:
            return _MEDIA_GENRE_CACHE

    mapping: Dict[str, Set[str]] = defaultdict(set)
    rows = db.execute(
        select(Item.media_type, Item.genres).where(Item.genres.isnot(None))
    ).all()
    for row in rows:
        media_type = getattr(row, "media_type", None)
        genres = getattr(row, "genres", None)
        if media_type is None:
            try:
                media_type = row[0]
            except (IndexError, TypeError, KeyError):
                media_type = None
        if genres is None:
            try:
                genres = row[1]
            except (IndexError, TypeError, KeyError):
                genres = None
        if not media_type or not isinstance(genres, list):
            continue
        bucket = mapping[media_type.lower()]
        for entry in genres:
            if isinstance(entry, Mapping):
                name = entry.get("name")
                if isinstance(name, str) and name:
                    bucket.add(name)

    _MEDIA_GENRE_CACHE = mapping
    _MEDIA_GENRE_CACHE_TS = now
    return mapping


def cache_get(cache_key: str) -> Dict[str, Any] | None:
    with _RECOMMEND_CACHE_LOCK:
        return _RECOMMEND_CACHE.get(cache_key)


def cache_set(
    canonical_id: str,
    cache_key: str,
    items: List[Dict[str, Any]],
    debug_context: Dict[str, Any] | None = None,
) -> None:
    with _RECOMMEND_CACHE_LOCK:
        entry: Dict[str, Any] = {"items": items}
        if debug_context is not None:
            entry["debug_context"] = debug_context
        _RECOMMEND_CACHE[cache_key] = entry
        user_keys = _RECOMMEND_CACHE_USER_KEYS.setdefault(canonical_id, set())
        user_keys.add(cache_key)


def cache_remove_user(canonical_id: str) -> int:
    with _RECOMMEND_CACHE_LOCK:
        keys_to_remove = _RECOMMEND_CACHE_USER_KEYS.pop(canonical_id, set())
        removed = 0
        for cache_key in keys_to_remove:
            if _RECOMMEND_CACHE.pop(cache_key, None) is not None:
                removed += 1
        return removed


def clear_recommend_cache_for_tests() -> None:
    """Reset cache internals for test isolation."""
    with _RECOMMEND_CACHE_LOCK:
        _RECOMMEND_CACHE.clear()
        _RECOMMEND_CACHE_USER_KEYS.clear()
    with _INFLIGHT_TASKS_LOCK:
        _INFLIGHT_RECOMMEND_TASKS.clear()


def clear_user_cache(canonical_id: str) -> None:
    """Clear all cached recommendations for a specific user profile."""
    removed = cache_remove_user(canonical_id)
    logger.debug(
        "Cleared recommendation cache for %s (removed %d entries)",
        canonical_id,
        removed,
    )


def get_cache_key(canonical_id: str, params: RecommendParams) -> str:
    # Serialize params canonically so future fields are automatically reflected in the key.
    if hasattr(params, "__dataclass_fields__"):
        params_dict = asdict(params)
    elif isinstance(params, dict):
        params_dict = dict(params)
    else:
        params_dict = getattr(params, "__dict__", {})
    params_dict.pop("user_id", None)
    params_dict.pop("profile", None)
    hash_payload = json.dumps(params_dict, sort_keys=True, default=str)
    hashed = hashlib.sha256(hash_payload.encode("utf-8")).hexdigest()
    return f"{canonical_id}:{hashed}"


def load_user_context(db: Session, user_id: str, profile: str | None) -> UserContext:
    canonical_id = canonical_profile_id(user_id, profile)
    loader = get_hook("load_user_state", load_user_state)
    long_v, short_v, exclude, profile_meta = loader(db, canonical_id)
    exclude_set: Set[int] = set(exclude or [])
    cold_start = short_v is None
    if cold_start:
        METRICS.counter("recommend.cold_start").inc()

    provider_alias_map = get_streaming_alias_map(db)
    top_query_keywords = get_top_query_keywords(db)

    return UserContext(
        canonical_id=canonical_id,
        user_id=user_id,
        profile=profile,
        long_v=long_v,
        short_v=short_v,
        exclude_set=exclude_set,
        profile_meta=profile_meta or {},
        cold_start=cold_start,
        provider_alias_map=provider_alias_map,
        top_query_keywords=top_query_keywords,
    )


async def get_or_compute_recommendations(
    request: Any,
    params: RecommendParams,
    db: Session,
    canonical_id: str,
    compute_fn: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    cache_key = get_cache_key(canonical_id, params)
    cached = cache_get(cache_key)

    if cached:
        METRICS.counter("recommend.cache_hit").inc()
        return cached["items"], cached.get("debug_context")

    loop_key = (id(asyncio.get_running_loop()), cache_key)
    created_task = False
    with _INFLIGHT_TASKS_LOCK:
        task = _INFLIGHT_RECOMMEND_TASKS.get(loop_key)
        if task is None:
            METRICS.counter("recommend.cache_miss").inc()
            task = asyncio.create_task(compute_fn(request, params, db))
            _INFLIGHT_RECOMMEND_TASKS[loop_key] = task
            created_task = True
        else:
            METRICS.counter("recommend.cache_wait").inc()
    try:
        result = await task
    finally:
        if created_task:
            with _INFLIGHT_TASKS_LOCK:
                if _INFLIGHT_RECOMMEND_TASKS.get(loop_key) is task:
                    _INFLIGHT_RECOMMEND_TASKS.pop(loop_key, None)
    reranked, debug_context = result.items, result.debug_context
    if created_task and reranked:
        cache_set(canonical_id, cache_key, reranked, debug_context=debug_context)
    return reranked, debug_context


# Aliases for backwards compatibility with tests importing private names
_get_catalog_metadata = get_catalog_metadata
_get_streaming_alias_map = get_streaming_alias_map
_get_top_query_keywords = get_top_query_keywords
_normalize_streaming_services = normalize_streaming_services
_load_media_genres = load_media_genres
_cache_get = cache_get
_cache_set = cache_set
_cache_remove_user = cache_remove_user
_clear_recommend_cache_for_tests = clear_recommend_cache_for_tests
_get_cache_key = get_cache_key
