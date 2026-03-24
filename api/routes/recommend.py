from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from cachetools import TTLCache
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select, func, or_, cast
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.db.session import get_db
from api.db.models import Item, ItemEmbedding, Availability, UserHistory, Feedback
from api.config import COUNTRY_DEFAULT, EMBED_VERSION
from api.core.user_utils import load_user_state, canonical_profile_id
from api.core.candidate_gen import ann_candidates, lexical_candidates
from api.core import llm_parser
from api.core.intent_parser import Intent
from api.core.legacy_intent_parser import (
    parse_intent as legacy_parse_intent,
    item_matches_intent,
    IntentFilters,
)
from api.core.reranker import rerank_with_explanations, diversify_with_mmr
from api.core.business_rules import apply_business_rules
from api.core.llm_parser import (
    rewrite_query,
    linked_media_types,
    extract_intent_signals,
)
from api.core.embeddings import encode_texts
from api.core.user_profile import NEGATIVE_EVENT_TYPES, _event_weight
from api.core.maturity import rating_level
from api.core.metrics import METRICS, timer
from api.core.logger import request_id_ctx
from api.core.query_profile import (
    IntentSignals,
    _normalize_query_text,
    build_query_profile,
)

router = APIRouter(prefix="/recommend", tags=["recommend"])
logger = logging.getLogger(__name__)
_UNKNOWN_RANK = 10**9
_PG13_LEVEL = rating_level("PG-13") or 13
_EXPLICIT_MATURITY_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\btv[\s-]?ma\b", flags=re.IGNORECASE), "TV-MA"),
    (re.compile(r"\btv[\s-]?14\b", flags=re.IGNORECASE), "TV-14"),
    (re.compile(r"\btv[\s-]?pg\b", flags=re.IGNORECASE), "TV-PG"),
    (re.compile(r"\bpg[\s-]?13\b", flags=re.IGNORECASE), "PG-13"),
    (re.compile(r"\bnc[\s-]?17\b", flags=re.IGNORECASE), "NC-17"),
    (re.compile(r"\bpg\b", flags=re.IGNORECASE), "PG"),
    (re.compile(r"\br\b", flags=re.IGNORECASE), "R"),
    (re.compile(r"\bg\b", flags=re.IGNORECASE), "G"),
)


def _ann_candidates_with_version(
    db: Session,
    user_vec: np.ndarray,
    exclude_ids: list[int],
    *,
    limit: int,
    allowed_ids: Sequence[int] | None,
) -> list[int]:
    try:
        return ann_candidates(
            db,
            user_vec,
            exclude_ids,
            limit=limit,
            allowed_ids=allowed_ids,
            version=EMBED_VERSION,
        )
    except TypeError as exc:
        if "version" not in str(exc):
            raise
        return ann_candidates(
            db,
            user_vec,
            exclude_ids,
            limit=limit,
            allowed_ids=allowed_ids,
        )


def _remove_values(values: Sequence[str], blocked: Set[str]) -> List[str]:
    return [value for value in values if value not in blocked]


def _rrf_merge(ann_ids: List[int], lexical_ids: List[int], k: int = 60) -> List[int]:
    """
    Combines two ranked lists using Reciprocal Rank Fusion.
    Score = sum(1 / (rank + k))
    """
    scores: Dict[int, float] = {}

    for rank, item_id in enumerate(ann_ids):
        scores[item_id] = scores.get(item_id, 0.0) + 1 / (rank + k)

    for rank, item_id in enumerate(lexical_ids):
        scores[item_id] = scores.get(item_id, 0.0) + 1 / (rank + k)

    # Sort by score descending
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item_id for item_id, score in sorted_items]


# --- Caching ---
_RECOMMEND_CACHE_TTL_SECONDS = int(os.getenv("RECOMMEND_CACHE_TTL_SECONDS", "300"))
_RECOMMEND_CACHE_MAXSIZE = int(os.getenv("RECOMMEND_CACHE_MAXSIZE", "512"))
_RECOMMEND_ALGO_VERSION = os.getenv("RECOMMEND_ALGO_VERSION", "v1")
_RECOMMEND_CACHE: TTLCache[str, Dict[str, Any]] = TTLCache(
    maxsize=_RECOMMEND_CACHE_MAXSIZE, ttl=_RECOMMEND_CACHE_TTL_SECONDS
)
_RECOMMEND_CACHE_LOCK = Lock()
_RECOMMEND_CACHE_USER_KEYS: Dict[str, Set[str]] = {}
_INFLIGHT_RECOMMEND_TASKS: Dict[Tuple[int, str], asyncio.Task[ComputeResult]] = {}
_INFLIGHT_TASKS_LOCK = Lock()


def _cache_get(cache_key: str) -> Dict[str, Any] | None:
    with _RECOMMEND_CACHE_LOCK:
        return _RECOMMEND_CACHE.get(cache_key)


def _cache_set(canonical_id: str, cache_key: str, items: List[Dict[str, Any]]) -> None:
    with _RECOMMEND_CACHE_LOCK:
        _RECOMMEND_CACHE[cache_key] = {"items": items}
        user_keys = _RECOMMEND_CACHE_USER_KEYS.setdefault(canonical_id, set())
        user_keys.add(cache_key)


def _cache_remove_user(canonical_id: str) -> int:
    with _RECOMMEND_CACHE_LOCK:
        keys_to_remove = _RECOMMEND_CACHE_USER_KEYS.pop(canonical_id, set())
        removed = 0
        for cache_key in keys_to_remove:
            if _RECOMMEND_CACHE.pop(cache_key, None) is not None:
                removed += 1
        return removed


def _record_impressions(
    db: Session,
    *,
    canonical_id: str,
    params: RecommendParams,
    page: Sequence[Dict[str, Any]],
    start_index: int,
    request_id: str | None,
) -> None:
    if not page:
        return

    add_fn = getattr(db, "add", None)
    commit_fn = getattr(db, "commit", None)
    if not callable(add_fn) or not callable(commit_fn):
        return

    query_present = bool(params.query and params.query.strip())
    written = 0

    for offset, item in enumerate(page):
        item_id = item.get("id")
        if not isinstance(item_id, int):
            continue
        meta: Dict[str, Any] = {
            "rank": int(start_index + offset),
            "profile": params.profile,
            "query_present": query_present,
            "algo_version": _RECOMMEND_ALGO_VERSION,
        }
        if request_id:
            meta["request_id"] = request_id
        add_fn(
            Feedback(
                user_id=canonical_id,
                item_id=item_id,
                type="impression",
                meta=meta,
            )
        )
        written += 1

    if written == 0:
        return

    try:
        commit_fn()
    except SQLAlchemyError:
        rollback_fn = getattr(db, "rollback", None)
        if callable(rollback_fn):
            rollback_fn()
        logger.warning(
            "Failed to commit recommendation impression events for %s",
            canonical_id,
            exc_info=True,
        )
    except Exception:
        rollback_fn = getattr(db, "rollback", None)
        if callable(rollback_fn):
            rollback_fn()
        logger.warning(
            "Unexpected failure while recording recommendation impressions for %s",
            canonical_id,
            exc_info=True,
        )


def _clear_recommend_cache_for_tests() -> None:
    """Reset cache internals for test isolation."""
    with _RECOMMEND_CACHE_LOCK:
        _RECOMMEND_CACHE.clear()
        _RECOMMEND_CACHE_USER_KEYS.clear()
    with _INFLIGHT_TASKS_LOCK:
        _INFLIGHT_RECOMMEND_TASKS.clear()


def clear_user_cache(canonical_id: str) -> None:
    """Clear all cached recommendations for a specific user profile."""
    removed = _cache_remove_user(canonical_id)
    logger.debug(
        "Cleared recommendation cache for %s (removed %d entries)",
        canonical_id,
        removed,
    )


def _get_cache_key(
    canonical_id: str,
    params: RecommendParams,
) -> str:
    # Serialize params canonically so future fields are automatically reflected in the key.
    params_dict = asdict(params)
    params_dict.pop("user_id", None)
    params_dict.pop("profile", None)
    hash_payload = json.dumps(params_dict, sort_keys=True, default=str)
    hashed = hashlib.sha256(hash_payload.encode("utf-8")).hexdigest()
    return f"{canonical_id}:{hashed}"


_STREAMING_PROVIDER_ALIASES: Dict[str, Set[str]] = {
    "netflix": {"netflix", "nfx"},
    "disney_plus": {"disney_plus", "disney", "dnp"},
    "prime_video": {"prime_video", "primevideo", "amazon", "amz", "amp"},
    "hulu": {"hulu", "hlu"},
    "max": {"max", "hbomax", "hbo", "hbm"},
    "apple_tv_plus": {"apple_tv_plus", "appletvplus", "apple", "atp"},
    "paramount_plus": {"paramount_plus", "paramountplus", "prm", "pmnt", "paramount"},
}


def _normalize_streaming_services(
    providers: Sequence[str] | None,
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
        for canonical, variants in _STREAMING_PROVIDER_ALIASES.items():
            all_aliases = variants.union({canonical})
            if key in all_aliases:
                normalized.update(all_aliases)
                matched = True
                break
        if not matched:
            normalized.add(key)
    return normalized


def _float_from_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no"}


_HYBRID_ANN_WEIGHT = _float_from_env("HYBRID_ANN_WEIGHT", 0.7)
_HYBRID_POPULARITY_WEIGHT = _float_from_env("HYBRID_POPULARITY_WEIGHT", 0.1)
_HYBRID_TRENDING_WEIGHT = _float_from_env("HYBRID_TRENDING_WEIGHT", 0.1)
_HYBRID_MIN_ANN_WEIGHT = 0.05
_QUERY_DISABLE_NON_ANN_SIGNALS = _env_flag(
    "QUERY_DISABLE_NON_ANN_SIGNALS",
    default=False,
)
_REWRITE_BLEND_ALPHA = 0.5
_REWRITE_BLEND_ALPHA_QUERY = _float_from_env("REWRITE_BLEND_ALPHA_QUERY", 0.2)
_COLLAB_HISTORY_LIMIT_MULTIPLIER = 4
_BASE_CANDIDATE_MULTIPLIER = 3
_FILTERED_QUERY_CANDIDATE_MULTIPLIER = 5
_NO_QUERY_CANDIDATE_MULTIPLIER = 10
_MIXER_COLLAB_WEIGHT = _float_from_env("MIXER_COLLAB_WEIGHT", 0.2)
_MIXER_TRENDING_WEIGHT = _float_from_env("MIXER_TRENDING_WEIGHT", 0.1)
_MIXER_CONSTRAINT_PRIOR_WEIGHT = _float_from_env("MIXER_CONSTRAINT_PRIOR_WEIGHT", 0.18)
_MIXER_NOVELTY_WEIGHT = _float_from_env("MIXER_NOVELTY_WEIGHT", 0.1)
_HYBRID_VOTE_WEIGHT = _float_from_env("HYBRID_VOTE_WEIGHT", 0.2)

_SERENDIPITY_RATIO_RAW = _float_from_env("SERENDIPITY_RATIO", 0.15)
if _SERENDIPITY_RATIO_RAW <= 0.0:
    _SERENDIPITY_RATIO = 0.0
else:
    _SERENDIPITY_RATIO = max(0.1, min(0.2, _SERENDIPITY_RATIO_RAW))

_ANN_DESCRIPTION_WEIGHT = _float_from_env("ANN_DESCRIPTION_WEIGHT", 1.2)
_REWRITE_TEXT_WEIGHT = _float_from_env("REWRITE_TEXT_WEIGHT", 1.0)


def _append_weighted_text(
    text: str | None,
    weight_override: float | None,
    default_weight: float,
    texts: List[str],
    weights: List[float],
) -> float:
    normalized = (text or "").strip()
    if not normalized:
        return 0.0
    weight = default_weight if weight_override is None else weight_override
    weight = max(0.0, weight)
    if weight <= 0.0:
        return 0.0
    texts.append(normalized)
    weights.append(weight)
    return weight


@dataclass
class PrefilterDecision:
    allowed_ids: List[int] | None
    boost_ids: List[int]
    enforce_genres: bool
    fetch_limit: int | None = None


def _build_rewrite_vector(
    rewrite_text: str | None,
    ann_description: str | None,
    ann_weight_override: float | None = None,
    rewrite_weight_override: float | None = None,
) -> np.ndarray | None:
    texts: List[str] = []
    weights: List[float] = []

    description_weight = _append_weighted_text(
        ann_description,
        ann_weight_override,
        _ANN_DESCRIPTION_WEIGHT,
        texts,
        weights,
    )
    rewrite_weight = _append_weighted_text(
        rewrite_text,
        rewrite_weight_override,
        _REWRITE_TEXT_WEIGHT,
        texts,
        weights,
    )

    if not texts:
        return None

    vectors = encode_texts(texts)
    if not isinstance(vectors, np.ndarray) or vectors.size == 0:
        return None

    combined = np.zeros(vectors.shape[1], dtype=np.float32)
    total_weight = 0.0
    for vec, weight in zip(vectors, weights):
        if weight <= 0.0:
            continue
        combined += weight * vec
        total_weight += weight

    if total_weight <= 0.0:
        return None

    norm = float(np.linalg.norm(combined))
    if norm == 0.0 or not np.isfinite(norm):
        return None
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Rewrite vector sources | description=%s rewrite=%s desc_weight=%.3f rewrite_weight=%.3f total_weight=%.3f",
            bool(description_weight),
            bool(rewrite_weight),
            description_weight,
            rewrite_weight,
            total_weight,
        )
    return combined / norm


def _normalize_text_signal(text: str | None) -> str:
    return " ".join(str(text or "").lower().replace("-", " ").split())


def _effective_ann_description(
    *,
    raw_query: str | None,
    ann_description: str | None,
    rewrite_text: str | None,
) -> str | None:
    normalized_query = _normalize_text_signal(raw_query)
    normalized_description = _normalize_text_signal(ann_description)
    normalized_rewrite = _normalize_text_signal(rewrite_text)
    if (
        normalized_description
        and normalized_description == normalized_query
        and normalized_rewrite
        and normalized_rewrite != normalized_query
    ):
        return None
    return ann_description


def _parse_llm_intent(
    query: str | None,
    user_context: Dict[str, Any],
    linked_entities: Dict[str, Any] | None = None,
) -> Intent:
    if not query:
        return llm_parser.default_intent()
    try:
        return llm_parser.parse_intent(query, user_context, linked_entities)
    except Exception:
        logger.exception("LLM intent parser failed; falling back to default intent.")
        return llm_parser.default_intent()


def _intent_filters_from_llm(query: str | None, llm_intent: Intent) -> IntentFilters:
    genres = list(llm_intent.include_genres or [])
    media_types = [
        media_type
        for media_type in (llm_intent.media_types or [])
        if media_type in {"movie", "tv"}
    ]
    filters = IntentFilters(
        raw_query=query or "",
        genres=genres,
        moods=[],
        media_types=media_types,
        year_min=llm_intent.year_min,
        year_max=llm_intent.year_max,
        min_runtime=llm_intent.runtime_minutes_min,
        max_runtime=llm_intent.runtime_minutes_max,
        maturity_rating_max=llm_intent.maturity_rating_max,
    )
    return filters


def _merge_with_legacy_filters(
    primary: IntentFilters, fallback: IntentFilters | None
) -> IntentFilters:
    if not fallback:
        return primary

    # Merge genres with order preservation
    seen_genres = {genre for genre in primary.genres}
    for genre in fallback.genres:
        if genre and genre not in seen_genres:
            primary.genres.append(genre)
            seen_genres.add(genre)

    if not primary.moods and fallback.moods:
        primary.moods = list(fallback.moods)

    if not primary.media_types and fallback.media_types:
        primary.media_types = list(fallback.media_types)

    if primary.min_runtime is None and fallback.min_runtime is not None:
        primary.min_runtime = fallback.min_runtime

    if primary.max_runtime is None and fallback.max_runtime is not None:
        primary.max_runtime = fallback.max_runtime

    _merge_maturity_rating(primary, fallback)

    return primary


def _merge_maturity_rating(primary: IntentFilters, fallback: IntentFilters) -> None:
    """Merge maturity caps, keeping the stricter (lower) rating."""
    fallback_cap = getattr(fallback, "maturity_rating_max", None)
    if not fallback_cap:
        return

    if not primary.maturity_rating_max:
        primary.maturity_rating_max = fallback_cap
        return

    fallback_level = rating_level(fallback_cap)
    primary_level = rating_level(primary.maturity_rating_max)
    if fallback_level is None:
        return
    if primary_level is None or fallback_level < primary_level:
        primary.maturity_rating_max = fallback_cap


def _explicit_query_maturity_cap(query: str | None) -> str | None:
    if not query:
        return None
    for pattern, rating in _EXPLICIT_MATURITY_PATTERNS:
        if pattern.search(query):
            return rating
    return None


def _apply_explicit_query_overrides(
    intent: IntentFilters,
    query: str | None,
    llm_intent: Intent | None = None,
) -> IntentFilters:
    explicit_cap = _explicit_query_maturity_cap(query)
    if explicit_cap:
        intent.maturity_rating_max = explicit_cap
        return intent

    if not query:
        return intent

    query_profile = build_query_profile(query)
    signals = query_profile.signals
    current_level = rating_level(intent.maturity_rating_max)
    if signals.kids_profile:
        if current_level is None or current_level != _PG13_LEVEL:
            intent.maturity_rating_max = "PG-13"
        return intent

    if signals.teen_or_family_safe:
        # Family-safe queries work better with a consistent PG-13 ceiling across movie and TV catalogs.
        if current_level is None or current_level != _PG13_LEVEL:
            intent.maturity_rating_max = "PG-13"
        return intent

    if signals.not_too_dark_fantasy_tv:
        if current_level is None or current_level > _PG13_LEVEL:
            intent.maturity_rating_max = "PG-13"
        return intent

    # Some movie queries inherit an overly restrictive PG/G cap from the LLM even when
    # the user never asked for age safety. Remove that cap before constrained filtering.
    # However, if the LLM intent specifically requested a restrictive cap (provenance check),
    # or if the query contains audience signals, we keep it.
    llm_cap_explicit = bool(llm_intent and llm_intent.maturity_rating_max is not None)
    is_adult_feel_good_comedy_query = signals.adult_feelgood_movieish
    if (
        ("movie" in (intent.media_types or []) or is_adult_feel_good_comedy_query)
        and current_level is not None
        and current_level < _PG13_LEVEL
        and not signals.audience_signal
        and (is_adult_feel_good_comedy_query or not llm_cap_explicit)
    ):
        intent.maturity_rating_max = None
    return intent


def _normalize_merged_intent(
    intent: IntentFilters,
    query: str | None,
    llm_intent: Intent | None = None,
) -> IntentFilters:
    if not query:
        return intent

    query_profile = build_query_profile(query)
    signals = query_profile.signals
    media_types = {media_type for media_type in intent.media_types if media_type}
    if intent.year_min is None and query_profile.hard_constraints.year_min is not None:
        intent.year_min = query_profile.hard_constraints.year_min
    if intent.year_max is None and query_profile.hard_constraints.year_max is not None:
        intent.year_max = query_profile.hard_constraints.year_max
    if (
        intent.min_runtime is None
        and query_profile.hard_constraints.min_runtime is not None
    ):
        intent.min_runtime = query_profile.hard_constraints.min_runtime
    if (
        intent.max_runtime is None
        and query_profile.hard_constraints.max_runtime is not None
    ):
        intent.max_runtime = query_profile.hard_constraints.max_runtime
    if (
        intent.maturity_rating_max is None
        and query_profile.hard_constraints.maturity_rating_max is not None
    ):
        intent.maturity_rating_max = query_profile.hard_constraints.maturity_rating_max

    if (
        media_types != {"movie"}
        and not signals.kids_profile
        and not signals.teen_or_family_safe
        and not signals.adult_feelgood_movieish
        and not signals.not_too_dark_fantasy_tv
        and not signals.teen_friendly_adventure_provider
        and not signals.multilingual_family_adventure
        and not signals.crime_query_signal
        and not signals.scifi_query_signal
        and not signals.fantasy_query_signal
        and not signals.high_concept_thriller
        and not signals.cerebral_temporal_thriller
        and not signals.temporal_thriller
        and not signals.caper_crime_tv
        and not signals.heist_tv
        and not signals.international_crime
        and not signals.anime_scifi_movie
        and not signals.street_level_superhero
        and not signals.rom_com
        and not signals.money_psychology_thriller
        and not signals.optimistic_scifi_tv
        and not signals.fantasy_epic_tv
    ):
        return intent

    # "Animation" is often injected by legacy mood/keyword mappings (like "light").
    # We strip it for movies UNLESS it was explicitly requested by the LLM
    # OR the query clearly contains an animation signal.
    # PROVENANCE FIX: If the LLM specifically picked Animation, we trust it.
    llm_genres = llm_intent.include_genres or [] if llm_intent else []
    animation_requested = "Animation" in llm_genres
    explicit_animation_query = signals.animation_signal
    if "Animation" in intent.genres:
        if (
            not signals.kids_profile
            and not animation_requested
            and not explicit_animation_query
        ):
            intent.genres = [genre for genre in intent.genres if genre != "Animation"]

    if signals.crime_query_signal:
        if "Crime" not in intent.genres:
            intent.genres.append("Crime")

    # scifi_query_signal: don't double-inject when anime_scifi_movie handles it specifically
    if signals.scifi_query_signal and not signals.anime_scifi_movie:
        for g in ("Science Fiction", "Sci-Fi & Fantasy"):
            if g not in intent.genres:
                intent.genres.append(g)

    if signals.fantasy_query_signal:
        for g in ("Fantasy", "Sci-Fi & Fantasy"):
            if g not in intent.genres:
                intent.genres.append(g)

    # fantasy_epic_tv: inject Action & Adventure on top of fantasy genres
    if signals.fantasy_epic_tv:
        for g in ("Sci-Fi & Fantasy", "Action & Adventure"):
            if g not in intent.genres:
                intent.genres.append(g)

    # temporal_thriller: replace genres with the thriller-scifi-mystery set
    if signals.temporal_thriller:
        intent.genres = [
            g for g in intent.genres if g in ("Thriller", "Science Fiction", "Mystery")
        ]
        for g in ("Thriller", "Science Fiction", "Mystery"):
            if g not in intent.genres:
                intent.genres.append(g)

    if (
        signals.high_concept_thriller
        or signals.cerebral_temporal_thriller
        or signals.money_psychology_thriller
    ):
        for g in ("Mystery", "Science Fiction"):
            if g not in intent.genres:
                intent.genres.append(g)

    # caper/heist: inject Crime + Drama only (no Thriller — too broad)
    if signals.caper_crime_tv or signals.heist_tv:
        for g in ("Crime", "Drama"):
            if g not in intent.genres:
                intent.genres.append(g)

    # international_crime: inject Crime + Thriller + Drama
    if signals.international_crime:
        for g in ("Crime", "Thriller", "Drama"):
            if g not in intent.genres:
                intent.genres.append(g)

    # anime_scifi_movie: override to Animation + Science Fiction only
    if signals.anime_scifi_movie:
        for g in ("Animation", "Science Fiction"):
            if g not in intent.genres:
                intent.genres.append(g)

    # street_level_superhero: replace genres with action/crime set
    if signals.street_level_superhero:
        intent.genres = ["Action & Adventure", "Crime", "Drama"]

    if signals.rom_com:
        for g in ("Romance", "Comedy"):
            if g not in intent.genres:
                intent.genres.append(g)

    # not_too_dark_fantasy_tv: hard-reset to Fantasy only (strips dark/thriller spillover and Sci-Fi expansion)
    if signals.not_too_dark_fantasy_tv:
        intent.genres = ["Fantasy"]
        intent.moods = [m for m in intent.moods if m != "dark"]

    # optimistic_scifi_tv: replace unrelated genres with sci-fi set, strip dark mood
    if signals.optimistic_scifi_tv:
        intent.genres = [
            g
            for g in intent.genres
            if g not in ("Crime", "Thriller", "Mystery", "Horror")
        ]
        for g in ("Science Fiction", "Sci-Fi & Fantasy", "Action & Adventure"):
            if g not in intent.genres:
                intent.genres.append(g)
        intent.moods = [m for m in intent.moods if m != "dark"]

    # teen_friendly_adventure_provider: targeted injection — do NOT cascade into full kids block
    if signals.teen_friendly_adventure_provider:
        # Drama is often mis-assigned by the LLM for teen adventure/fantasy mashups
        intent.genres = [
            g for g in intent.genres if g not in ("Drama", "Family", "Animation")
        ]
        for genre in ("Adventure", "Fantasy"):
            if genre not in intent.genres:
                intent.genres.append(genre)

    # kids/family block: only inject if NOT already a specific teen_friendly signal
    elif signals.multilingual_family_adventure:
        # Multilingual/bilingual adventure always gets the full family+fantasy set
        for genre in (
            "Family",
            "Animation",
            "Adventure",
            "Fantasy",
            "Sci-Fi & Fantasy",
        ):
            if genre not in intent.genres:
                intent.genres.append(genre)

    elif signals.kids_profile:
        # Standard kids profile: respect LLM genre choices — only inject what LLM suggested
        desired = (
            set(llm_genres)
            if llm_genres
            else {"Family", "Animation", "Adventure", "Fantasy", "Sci-Fi & Fantasy"}
        )
        for genre in (
            "Family",
            "Animation",
            "Adventure",
            "Fantasy",
            "Sci-Fi & Fantasy",
        ):
            if genre not in intent.genres and genre in desired:
                intent.genres.append(genre)

    elif signals.teen_or_family_safe and signals.family_friendly:
        # "short family movies" and similar — only add Family if not already enriched
        if "Family" not in intent.genres:
            intent.genres.append("Family")

    if signals.feel_good_comedy and not signals.audience_signal:
        intent.genres = [
            genre for genre in intent.genres if genre not in {"Family", "Animation"}
        ]
        if "Comedy" not in intent.genres:
            intent.genres.append("Comedy")

    # "light" injects Animation via legacy mood mapping, which hurts adult feel-good
    # and family adventure movie buckets unless the user explicitly asked for animation.
    if "light" in intent.moods:
        # Strip 'light' tag as it's not a DB field,
        # but ensure secondary genres like Comedy/Family exist.
        intent.moods = [mood for mood in intent.moods if mood != "light"]
        # Only inject Comedy/Family if the query feels like it needs it (audience signal or kid-friendly)
        # to avoid mangling "light drama" or "light sci-fi".
        if signals.audience_signal or signals.family_friendly:
            for genre in ("Comedy", "Family"):
                if genre not in intent.genres:
                    intent.genres.append(genre)
        elif not intent.genres:  # Fallback if no genres at all
            intent.genres.append("Comedy")

    return intent


def _should_soften_provider_preference(
    intent: IntentFilters,
    query: str | None,
) -> bool:
    if not query:
        return False
    media_types = {media_type for media_type in intent.media_types if media_type}
    if media_types != {"movie"}:
        return False
    if build_query_profile(query).signals.audience_signal:
        return False
    if _explicit_query_maturity_cap(query):
        return False
    return True


def _safe_constraint_similarity(
    rank_vector: np.ndarray | None,
    item_vector: Any,
) -> float:
    if rank_vector is None or item_vector is None:
        return -1.0
    candidate = np.asarray(item_vector, dtype="float32")
    if candidate.size == 0:
        return -1.0
    rank_norm = float(np.linalg.norm(rank_vector))
    candidate_norm = float(np.linalg.norm(candidate))
    if rank_norm <= 0 or candidate_norm <= 0:
        return -1.0
    return float(np.dot(rank_vector, candidate) / (rank_norm * candidate_norm))


@dataclass(frozen=True)
class ConstraintBonusContext:
    haystack: str
    genre_names: set[str]
    original_language: str
    media_type: str
    runtime: Any
    release_year: Any
    maturity_rating: str
    vote_count: float


def _build_constraint_bonus_context(item: Item) -> ConstraintBonusContext:
    text_parts = [
        str(getattr(item, "title", "") or "").lower(),
        str(getattr(item, "overview", "") or "").lower(),
    ]
    genres = getattr(item, "genres", None) or []
    genre_names: set[str] = set()
    for genre in genres:
        if isinstance(genre, dict):
            name = str(genre.get("name") or "").lower()
            text_parts.append(name)
            if name:
                genre_names.add(name)
        else:
            name = str(genre).lower()
            text_parts.append(name)
            if name:
                genre_names.add(name)

    return ConstraintBonusContext(
        haystack=" ".join(part for part in text_parts if part),
        genre_names=genre_names,
        original_language=str(getattr(item, "original_language", "") or "").lower(),
        media_type=str(getattr(item, "media_type", "") or "").lower(),
        runtime=getattr(item, "runtime", None),
        release_year=getattr(item, "release_year", None),
        maturity_rating=str(getattr(item, "maturity_rating", "") or "").upper(),
        vote_count=float(getattr(item, "vote_count", None) or 0.0),
    )


def _constraint_bonus_for_contextual_signals(
    signals: Any,
    ctx: ConstraintBonusContext,
) -> float:
    bonus = 0.0
    if (
        signals.modern_setting
        and isinstance(ctx.release_year, int)
        and ctx.release_year >= 1990
    ):
        bonus += 0.18
    if signals.city_setting:
        for keyword in ("city", "urban", "los angeles", "new york", "miami", "chicago"):
            if keyword in ctx.haystack:
                bonus += 0.2
    return bonus


def _constraint_bonus_for_romance_tone(
    intent_signals: IntentSignals,
    signals: Any,
    ctx: ConstraintBonusContext,
) -> float:
    """
    Apply romantic tone bonuses based on LLM-extracted intent signals.

    Conditions:
    1. Romantic mood: Apply romance genre bonuses
    2. Feel-good/uplifting mood: Apply feel-good comedy bonuses
    3. Rom-com signal: Apply romantic comedy bonuses

    All mood detection is LLM-driven and immune to query paraphrasing.
    """
    bonus = 0.0
    genre_names = ctx.genre_names
    haystack = ctx.haystack

    # Condition 1: Romantic mood
    # NEW: Use LLM-extracted mood instead of keyword matching
    is_romantic_date = intent_signals.mood in (
        "romantic",
        "uplifting",
        "heartwarming",
        "light",
    )
    if is_romantic_date:
        if "comedy" in genre_names:
            bonus += 0.22
        if "romance" in genre_names:
            bonus += 0.24
        if "drama" in genre_names:
            bonus += 0.08
        if "music" in genre_names:
            bonus += 0.2
        if "fantasy" in genre_names:
            bonus += 0.08
        if {"comedy", "romance"} <= genre_names:
            bonus += 0.24
        elif {"comedy", "drama"} <= genre_names:
            bonus += 0.1
        for keyword in (
            "romance",
            "romantic",
            "relationship",
            "love",
            "heartwarming",
            "feel-good",
            "warm",
            "uplifting",
            "musician",
            "band",
            "music",
            "song",
            "singer",
            "chef",
            "kitchen",
            "restaurant",
            "cook",
            "food",
            "wedding",
            "summer",
            "date",
            "dating",
            "couple",
            "boyfriend",
            "girlfriend",
            "breakup",
            "heartbreak",
            "falling in love",
            "meet cute",
            "road trip",
            "coming of age",
            "coming-of-age",
        ):
            if keyword in haystack:
                bonus += 0.14
        if isinstance(ctx.runtime, int):
            if ctx.runtime <= 125:
                bonus += 0.12
                if ctx.runtime <= 115:
                    bonus += 0.05
            elif ctx.runtime > 145:
                bonus -= 0.14
        if (
            "action" in genre_names
            or "thriller" in genre_names
            or "crime" in genre_names
        ):
            bonus -= 0.22
        if "horror" in genre_names:
            bonus -= 0.28
        if "science fiction" in genre_names and "comedy" not in genre_names:
            bonus -= 0.18
        if "family" in genre_names and "romance" not in genre_names:
            bonus -= 0.1
        for keyword in (
            "slapstick",
            "gross-out",
            "gross out",
            "spoof",
            "zombie",
            "vampire",
            "killer",
            "gangster",
            "war",
            "vietnam",
            "battle",
            "soldier",
            "assassin",
            "serial killer",
            "forrest",
            "president",
            "history",
            "epic",
            "historical",
            "warfare",
            "biography",
        ):
            if keyword in haystack:
                bonus -= 0.18

    # Condition 2: Feel-good or uplifting mood
    # NEW: Use LLM-extracted mood instead of regex pattern
    is_feel_good = intent_signals.mood in (
        "uplifting",
        "light",
        "heartwarming",
        "cheerful",
    )
    if is_feel_good:
        if "comedy" in genre_names:
            bonus += 0.18
        if "romance" in genre_names:
            bonus += 0.22
        if "drama" in genre_names:
            bonus += 0.14
        if "music" in genre_names:
            bonus += 0.18
        if {"comedy", "romance"} <= genre_names:
            bonus += 0.18
        elif {"comedy", "drama"} <= genre_names:
            bonus += 0.08
        for keyword in (
            "heartwarming",
            "uplifting",
            "feel-good",
            "friendship",
            "family meal",
            "restaurant",
            "chef",
            "music",
            "band",
            "coming of age",
            "summer",
            "love",
            "relationship",
        ):
            if keyword in haystack:
                bonus += 0.14
        if ctx.media_type == "movie":
            bonus += 0.18
        elif ctx.media_type == "tv":
            bonus -= 0.26
        if isinstance(ctx.runtime, int):
            if ctx.runtime <= 125:
                bonus += 0.12
                if ctx.runtime <= 110:
                    bonus += 0.04
            elif ctx.runtime > 145:
                bonus -= 0.16
        if (
            "action" in genre_names
            or "thriller" in genre_names
            or "crime" in genre_names
        ):
            bonus -= 0.2
        if "horror" in genre_names:
            bonus -= 0.26
        for keyword in (
            "slapstick",
            "gross-out",
            "gross out",
            "spoof",
            "killer",
            "zombie",
            "gangster",
        ):
            if keyword in haystack:
                bonus -= 0.18

    # Condition 3: Rom-com signal
    # NEW: Use LLM-extracted mood + semantic facets instead of regex pattern
    is_rom_com = (
        intent_signals.mood == "romantic"
        or "romantic" in intent_signals.semantic_facets
    )
    if is_rom_com:
        if ctx.media_type == "movie":
            bonus += 0.12
        if "romance" in genre_names:
            bonus += 0.22
        if "comedy" in genre_names:
            bonus += 0.22
        if {"romance", "comedy"} <= genre_names:
            bonus += 0.18
        for keyword in (
            "rom com",
            "rom-com",
            "romantic comedy",
            "romantic",
            "fall in love",
            "falls in love",
            "love story",
            "meet cute",
            "dating",
            "date",
            "wedding",
            "breakup",
            "relationship",
            "boyfriend",
            "girlfriend",
            "marriage",
            "proposal",
        ):
            if keyword in haystack:
                bonus += 0.14
        if isinstance(ctx.release_year, int):
            if 2000 <= ctx.release_year <= 2009:
                bonus += 0.18
            elif ctx.release_year > 2010:
                bonus -= 0.08
            elif ctx.release_year < 1995:
                bonus -= 0.12
        if "action" in genre_names and "romance" not in genre_names:
            bonus -= 0.2
        if "thriller" in genre_names:
            bonus -= 0.14
        if "animation" in genre_names or "family" in genre_names:
            bonus -= 0.18
        if "adventure" in genre_names and "romance" not in genre_names:
            bonus -= 0.12
        for keyword in (
            "assassin",
            "spy",
            "killer",
            "warrior",
            "dragon",
            "zoo",
            "animal",
            "monster",
            "superhero",
            "mission",
            "crime fighters",
        ):
            if keyword in haystack:
                bonus -= 0.18
    return bonus


def _constraint_bonus_for_crime_facets(
    intent_signals: IntentSignals,
    signals: Any,
    ctx: ConstraintBonusContext,
) -> float:
    """
    Apply crime/heist/caper bonuses based on LLM-extracted semantic facets.

    This replaces brittle keyword matching in the query with robust semantic detection.
    Bonuses are applied based on the underlying intent (e.g., 'heist') and item
    metadata (genres, media type, language) rather than query wording.
    """
    bonus = 0.0
    genre_names = ctx.genre_names
    haystack = ctx.haystack
    facets = intent_signals.semantic_facets

    # Condition 1: Caper/Crime/Heist (Semantic detection)
    # Triggered by heist, caper, or robbery related facets
    has_caper_crime = any(
        f in facets for f in ("heist", "caper", "con-artist", "con", "robbery")
    )
    if has_caper_crime:
        # Base facet bonus
        bonus += 0.22

        if "crime" in genre_names:
            bonus += 0.25  # Increased from 0.2
        if "drama" in genre_names:
            bonus += 0.08  # Increased from 0.06
        if "action & adventure" in genre_names:
            bonus += 0.1  # Increased from 0.08

        # Increased secondary boost and expanded keyword list
        if any(
            kw in haystack
            for kw in (
                "heist",
                "robbery",
                "con artist",
                "thief",
                "mastermind",
                "score",
                "scheme",
                "inside job",
            )
        ):
            bonus += 0.15  # Increased from 0.05

        if "horror" in genre_names or "mystery" in genre_names:
            bonus -= 0.2

        # Penalty for procedural keywords (remains metadata-based for precision)
        if any(
            kw in haystack
            for kw in ("fbi", "ncis", "csi", "police procedural", "courtroom")
        ):
            bonus -= 0.24

        # Penalty for procedural facets
        if "procedural" in facets:
            bonus -= 0.24

    # Condition 2: Heist TV (Semantic detection)
    has_heist = "heist" in facets
    if has_heist:
        # Base heist facet bonus
        bonus += 0.18

        if ctx.media_type == "tv":
            bonus += 0.12
        if "crime" in genre_names:
            bonus += 0.18
        if "drama" in genre_names:
            bonus += 0.08
        if "action & adventure" in genre_names:
            bonus += 0.1

        if "mystery" in genre_names and "crime" not in genre_names:
            bonus -= 0.12

        # Law enforcement penalty
        if any(kw in haystack for kw in ("fbi", "homicide", "forensic", "cop", "unit")):
            bonus -= 0.22

    # Condition 3: Noir / Mystery / Detective (Semantic detection)
    is_noir_query = any(f in facets for f in ("noir", "mystery", "detective"))
    if is_noir_query:
        # Base noir/detective bonus
        bonus += 0.22

        # Additional boost for investigation facets
        if any(f in facets for f in ("investigation", "homicide", "murder", "missing")):
            bonus += 0.18

        if {"crime", "mystery"} <= genre_names:
            bonus += 0.2
        if "thriller" in genre_names and "crime" in genre_names:
            bonus += 0.12

        # Heavy penalties for non-noir genres
        if any(g in genre_names for g in ("science fiction", "family", "adventure")):
            bonus -= 0.4
        if "action" in genre_names and "mystery" not in genre_names:
            bonus -= 0.22
        if "comedy" in genre_names:
            bonus -= 0.14

        # Penalty for non-noir facets (semantic mismatch)
        if any(
            f in facets
            for f in ("monster", "alien", "ghost", "supernatural", "beach", "wedding")
        ):
            bonus -= 0.22

    # Condition 4: International Crime (Semantic + Context detection)
    has_intl_crime = (
        "crime" in facets and ctx.original_language and ctx.original_language != "en"
    )
    if has_intl_crime:
        # Base international crime bonus
        bonus += 0.16

        if ctx.original_language != "en":
            bonus += 0.26
        if ctx.original_language in {"es", "fr", "it", "pt", "de", "ko"}:
            bonus += 0.08

        # Boost for crime-related facets in international context
        if any(
            f in facets
            for f in ("heist", "mafia", "cartel", "corruption", "underworld")
        ):
            bonus += 0.14

        # Procedural penalty (usually less desired in intl crime searches)
        if any(f in facets for f in ("procedural", "police", "detective")):
            bonus -= 0.22

    return bonus


def _constraint_bonus_for_thriller_facets(
    intent_signals: IntentSignals,
    signals: Any,
    ctx: ConstraintBonusContext,
) -> float:
    bonus = 0.0
    genre_names = ctx.genre_names
    facets = intent_signals.semantic_facets

    is_thriller_query = (
        "thriller" in facets
        or "psychological" in facets
        or "mystery" in facets
        or "temporal" in facets
        or "time-loop" in facets
        or intent_signals.mood == "dark"
    )

    if is_thriller_query:
        if "thriller" in genre_names:
            bonus += 0.2
        if "mystery" in genre_names:
            bonus += 0.12
        if "crime" in genre_names:
            bonus += 0.08

        # Enhance via mood="dark"
        if intent_signals.mood == "dark" or "dark" in facets:
            if {"thriller", "mystery", "crime", "horror"} & genre_names:
                bonus += 0.16
            if {"comedy", "family"} & genre_names:
                bonus -= 0.24

        # Semantic detection instead of signals object
        if "psychological" in facets:
            if "drama" in genre_names:
                bonus += 0.12
            if "action" in genre_names and "crime" not in genre_names:
                bonus -= 0.12
            if "science fiction" in genre_names or "horror" in genre_names:
                bonus -= 0.16
            # Metadata-side keyword matching
            if any(
                kw in ctx.haystack
                for kw in (
                    "psychological",
                    "obsession",
                    "desperate",
                    "parasite",
                    "social climbing",
                    "class divide",
                )
            ):
                bonus += 0.08

        if any(f in facets for f in ("time travel", "temporal", "time-loop")):
            if "science fiction" in genre_names or "sci-fi & fantasy" in genre_names:
                bonus += 0.28
            if "horror" in genre_names:
                bonus -= 0.24
            if "crime" in genre_names and "science fiction" not in genre_names:
                bonus -= 0.14
            # Metadata-side keyword matching
            if any(
                kw in ctx.haystack
                for kw in (
                    "time travel",
                    "time-travel",
                    "temporal",
                    "timeline",
                    "timelines",
                    "time loop",
                    "loop",
                    "paradox",
                )
            ):
                bonus += 0.12

        if "cerebral" in facets:
            if "science fiction" in genre_names or "sci-fi & fantasy" in genre_names:
                bonus += 0.16
            if isinstance(ctx.runtime, int):
                if ctx.runtime <= 125:
                    bonus += 0.16
                elif ctx.runtime > 140:
                    bonus -= 0.16

        # Quality baseline Check
        if ctx.vote_count >= 500:
            bonus += 0.1

    return bonus


def _constraint_bonus_for_prestige_and_superhero_facets(
    intent_signals: IntentSignals,
    signals: Any,
    ctx: ConstraintBonusContext,
) -> float:
    bonus = 0.0
    genre_names = ctx.genre_names
    facets = intent_signals.semantic_facets

    # Prestige replacement
    # Using intent_signals.prestige_indicator which is now the source of truth
    is_prestige = intent_signals.prestige_indicator
    if is_prestige:
        if "war & politics" in genre_names:
            bonus += 0.22
        if "sci-fi & fantasy" in genre_names:
            bonus += 0.24
        if {"drama", "sci-fi & fantasy"} <= genre_names:
            bonus += 0.12
        if {"drama", "action & adventure"} <= genre_names:
            bonus += 0.08
        if "drama" in genre_names or "history" in genre_names:
            bonus += 0.15

        if "soap" in genre_names:
            bonus -= 0.28
        if "family" in genre_names or "comedy" in genre_names:
            bonus -= 0.22
        if "crime" in genre_names and not (
            {"war & politics", "sci-fi & fantasy", "action & adventure"} & genre_names
        ):
            bonus -= 0.08

        # Quality metrics replace generic "oscar"/"acclaimed" keyword searches
        if isinstance(ctx.release_year, int):
            if ctx.release_year >= 2010:
                bonus += 0.16
            elif ctx.release_year < 2005:
                bonus -= 0.18
        if ctx.vote_count >= 5000:
            bonus += 0.25  # boosted
        elif 0 < ctx.vote_count < 200:
            bonus -= 0.1
        # Metadata keywords provide secondary boost only
        if any(
            kw in ctx.haystack
            for kw in (
                "oscar",
                "nominated",
                "nomination",
                "academy award",
                "golden globe",
                "emmy",
                "cannes",
                "sundance",
                "critical acclaim",
                "acclaimed",
                "masterpiece",
                "best picture",
            )
        ):
            bonus += 0.08

    # Superhero / Vigilante
    is_superhero = (
        "superhero" in facets
        or "vigilante" in facets
        or (signals and getattr(signals, "street_level_superhero", False))
    )
    if is_superhero:
        if ctx.media_type == "tv":
            bonus += 0.16
        if "action & adventure" in genre_names or "action" in genre_names:
            bonus += 0.2
        if "crime" in genre_names:
            bonus += 0.16
        if "drama" in genre_names:
            bonus += 0.1

        # Determine level/grittiness
        is_gritty = any(f in facets for f in ("street-level", "gritty", "dark", "noir"))
        if is_gritty:
            if "crime" in genre_names or "thriller" in genre_names:
                bonus += 0.16
            if "comedy" in genre_names or "animation" in genre_names:
                bonus -= 0.22

        if (
            "fantasy" in genre_names
            or "family" in genre_names
            or "animation" in genre_names
        ):
            bonus -= 0.18

    return bonus


def _constraint_bonus_for_audience_facets(
    intent_signals: IntentSignals,
    signals: Any,
    ctx: ConstraintBonusContext,
    explicit_cap: str | None,
) -> float:
    """
    Apply audience-based bonuses (Kids, Family, Teen, Multilingual) using semantic facets.

    Replaces brittle query keyword matching with robust LLM-extracted intent signals.
    """
    bonus = 0.0
    genre_names = ctx.genre_names
    facets = set(intent_signals.semantic_facets)

    # Condition 1: Kids / Animation Adventure (Semantic)
    is_kids_query = any(
        f in facets for f in ("kids", "animation", "child-friendly", "kids-profile")
    )
    if is_kids_query:
        if "adventure" in genre_names or "action & adventure" in genre_names:
            bonus += 0.24
        if "fantasy" in genre_names or "sci-fi & fantasy" in genre_names:
            bonus += 0.24
        if "science fiction" in genre_names:
            bonus += 0.18
        if "family" in genre_names:
            bonus += 0.26
        if "animation" in genre_names:
            bonus += 0.3
        if ctx.media_type == "tv":
            bonus += 0.22
        if ctx.maturity_rating in {"TV-Y7", "TV-PG", "PG", "G", "PG-13"}:
            bonus += 0.12

        # Semantic facet boosts
        for facet in (
            "quest",
            "magic",
            "wizard",
            "galaxy",
            "space",
            "alien",
            "dragon",
            "heroic",
        ):
            if facet in facets:
                bonus += 0.18

        if any(g in genre_names for g in ("horror", "thriller", "crime")):
            bonus -= 0.24

    # Condition 2: Multilingual / International Family (Semantic + Context)
    is_intl_family = "family" in facets and (
        ctx.original_language != "en" or "multilingual" in facets
    )
    if is_intl_family:
        if "family" in genre_names:
            bonus += 0.22
        if "animation" in genre_names:
            bonus += 0.24
        if "kids" in genre_names:
            bonus += 0.28
        if "adventure" in genre_names or "action & adventure" in genre_names:
            bonus += 0.24
        if "fantasy" in genre_names or "sci-fi & fantasy" in genre_names:
            bonus += 0.20
        if "science fiction" in genre_names:
            bonus += 0.12
        if ctx.media_type == "tv":
            bonus += 0.12
        if ctx.maturity_rating in {"TV-Y7", "TV-PG", "PG", "G", "PG-13"}:
            bonus += 0.08
        if ctx.original_language and ctx.original_language != "en":
            bonus += 0.18

        # Semantic facet boosts
        for facet in (
            "magic",
            "dragon",
            "creature",
            "enchanted",
            "legend",
            "quest",
            "warrior",
        ):
            if facet in facets:
                bonus += 0.18

    # Condition 3: Family Adventure (PG-13 Cap)
    is_family_adventure = "family" in facets and "adventure" in facets
    if is_family_adventure and explicit_cap == "PG-13":
        if "adventure" in genre_names or "action & adventure" in genre_names:
            bonus += 0.24
        if "family" in genre_names:
            bonus += 0.22
        if "fantasy" in genre_names or "sci-fi & fantasy" in genre_names:
            bonus += 0.18
        if "science fiction" in genre_names:
            bonus += 0.12
        if ctx.maturity_rating in {"PG", "PG-13", "TV-PG"}:
            bonus += 0.16

        # Semantic facet boosts
        for facet in (
            "quest",
            "heroic",
            "magic",
            "wizard",
            "pirate",
            "dragon",
            "space",
            "princess",
        ):
            if facet in facets:
                bonus += 0.14

        if any(g in genre_names for g in ("crime", "thriller", "horror")):
            bonus -= 0.24

    # Condition 4: Teen Friendly / Young Adult (Semantic)
    is_teen_query = any(
        f in facets for f in ("teen", "young-adult", "ya", "coming-of-age")
    )
    if is_teen_query:
        if "action & adventure" in genre_names or "adventure" in genre_names:
            bonus += 0.22
        if "fantasy" in genre_names or "sci-fi & fantasy" in genre_names:
            bonus += 0.2
        if "family" in genre_names or "animation" in genre_names:
            bonus += 0.16
        if ctx.media_type == "tv":
            bonus += 0.18

        # Semantic facet boosts
        for facet in ("heroic", "quest", "dragon", "legend", "academy", "kingdom"):
            if facet in facets:
                bonus += 0.14

        if any(g in genre_names for g in ("crime", "thriller", "horror")):
            bonus -= 0.24

    return bonus


def _constraint_bonus_for_scifi_fantasy_facets(
    intent_signals: IntentSignals,
    signals: Any,
    ctx: ConstraintBonusContext,
) -> float:
    """
    Applies Sci-Fi & Fantasy bonuses using semantic facets.

    Replaces brittle keyword matching with robust LLM-extracted intent signals.
    """
    bonus = 0.0
    genre_names = ctx.genre_names
    haystack = ctx.haystack
    facets = set(intent_signals.semantic_facets)

    # Condition 1: Bingeable Sci-Fi TV
    if (
        "short-form" in facets
        and "bingeable" in facets
        and any(f in facets for f in ("sci-fi", "science-fiction"))
    ):
        if ctx.media_type == "tv":
            bonus += 0.18
        if "science fiction" in genre_names or "sci-fi & fantasy" in genre_names:
            bonus += 0.3
        if isinstance(ctx.runtime, int):
            if ctx.runtime <= 50:
                bonus += 0.22
            elif ctx.runtime > 60:
                bonus -= 0.18
        if any(
            kw in haystack
            for kw in (
                "space",
                "future",
                "time travel",
                "timeline",
                "android",
                "robot",
                "alien",
                "starship",
            )
        ):
            bonus += 0.16
        if any(g in genre_names for g in ("comedy", "crime", "medical")):
            bonus -= 0.24
        if "war & politics" in genre_names:
            bonus -= 0.18

    # Condition 2: Anime Sci-Fi Movie
    if "anime" in facets and "sci-fi" in facets and ctx.media_type == "movie":
        if "animation" in genre_names:
            bonus += 0.22
        if "science fiction" in genre_names or "sci-fi & fantasy" in genre_names:
            bonus += 0.22
        if ctx.original_language == "ja":
            bonus += 0.2
        if any(
            kw in haystack
            for kw in (
                "cyberpunk",
                "mecha",
                "robot",
                "android",
                "future",
                "futuristic",
                "tokyo",
                "dream",
                "memory",
                "virtual",
                "space",
            )
        ):
            bonus += 0.16
        if any(g in genre_names for g in ("family", "comedy")):
            bonus -= 0.18

    # Condition 3: Optimistic Sci-Fi / Space Opera TV
    if "optimistic" in facets and "sci-fi" in facets and ctx.media_type == "tv":
        if "sci-fi & fantasy" in genre_names or "science fiction" in genre_names:
            bonus += 0.3
        if "action & adventure" in genre_names or "adventure" in genre_names:
            bonus += 0.18
        if any(
            kw in haystack
            for kw in (
                "space",
                "starship",
                "spaceship",
                "crew",
                "captain",
                "galaxy",
                "planet",
                "exploration",
                "explore",
                "alien",
                "future",
                "optimistic",
                "adventure",
            )
        ):
            bonus += 0.16
        if "comedy" in genre_names and "sci-fi & fantasy" in genre_names:
            bonus += 0.08
        if any(g in genre_names for g in ("crime", "medical", "soap")):
            bonus -= 0.26

    # Condition 4: Epic Fantasy TV
    if "epic" in facets and "fantasy" in facets and ctx.media_type == "tv":
        if "fantasy" in genre_names or "sci-fi & fantasy" in genre_names:
            bonus += 0.32
        if "action & adventure" in genre_names or "adventure" in genre_names:
            bonus += 0.18
        if any(
            kw in haystack
            for kw in (
                "kingdom",
                "throne",
                "prophecy",
                "monster",
                "magic",
                "wizard",
                "quest",
                "dragon",
                "sword",
                "saga",
                "realm",
            )
        ):
            bonus += 0.16
        if any(g in genre_names for g in ("crime", "comedy", "family")):
            bonus -= 0.22

    # Condition 5: Fantasy Worlds (Cross-Media)
    if "fantasy-world" in facets:
        if "fantasy" in genre_names or "sci-fi & fantasy" in genre_names:
            bonus += 0.26
        if "action & adventure" in genre_names or "adventure" in genre_names:
            bonus += 0.16
        if ctx.media_type == "tv":
            bonus += 0.14
        if any(
            kw in haystack
            for kw in (
                "world",
                "realm",
                "kingdom",
                "magic",
                "wizard",
                "quest",
                "dragon",
                "prophecy",
                "portal",
                "fantasy",
            )
        ):
            bonus += 0.14
        if any(g in genre_names for g in ("crime", "horror")):
            bonus -= 0.22

    return bonus


def _constraint_query_bonus(intent: IntentFilters | None, item: Item) -> float:
    raw_query = (intent.raw_query or "").lower() if intent else ""
    if not raw_query:
        return 0.0

    intent_signals = extract_intent_signals(raw_query)
    ctx = _build_constraint_bonus_context(item)

    bonus = 0.0
    # All bonus functions are now driven by LLM-extracted intent_signals
    bonus += _constraint_bonus_for_crime_facets(intent_signals, None, ctx)
    bonus += _constraint_bonus_for_romance_tone(intent_signals, None, ctx)
    bonus += _constraint_bonus_for_thriller_facets(intent_signals, None, ctx)
    bonus += _constraint_bonus_for_prestige_and_superhero_facets(
        intent_signals, None, ctx
    )
    bonus += _constraint_bonus_for_audience_facets(
        intent_signals,
        None,
        ctx,
        _explicit_query_maturity_cap(raw_query),
    )
    bonus += _constraint_bonus_for_scifi_fantasy_facets(intent_signals, None, ctx)

    # Contextual signals are lightweight and can remain
    query_profile = build_query_profile(raw_query)
    bonus += _constraint_bonus_for_contextual_signals(query_profile.signals, ctx)

    return bonus


def _constraint_sort_key(
    item: Item,
    *,
    intent: IntentFilters | None,
    item_vector: Any,
    rank_vector: np.ndarray | None,
) -> Tuple[float, int, int, float, float, int]:
    semantic_score = _safe_constraint_similarity(rank_vector, item_vector)
    lexical_bonus = _constraint_query_bonus(intent, item)
    raw_query = (intent.raw_query or "").lower() if intent else ""

    intent_signals = extract_intent_signals(raw_query)
    facets = set(intent_signals.semantic_facets)

    query_is_noir = "noir" in facets
    query_is_time_bending = "temporal" in facets or "time-loop" in facets
    query_is_high_concept = "cerebral" in facets
    query_is_serialized_tv = (
        "serialized" in facets and intent and "tv" in intent.media_types
    )
    query_is_multilingual_family_adventure = (
        "multilingual" in facets and "family" in facets
    )

    catalog_score = (
        (
            (
                0.05
                if query_is_multilingual_family_adventure
                else (
                    0.06
                    if query_is_noir
                    else (
                        0.08
                        if query_is_time_bending
                        else (
                            0.08
                            if query_is_high_concept
                            else 0.1 if query_is_serialized_tv else 0.15
                        )
                    )
                )
            )
            * math.log1p(float(getattr(item, "popularity", None) or 0.0))
        )
        + (
            (
                0.03
                if query_is_multilingual_family_adventure
                else (
                    0.03
                    if query_is_noir
                    else (
                        0.04
                        if query_is_time_bending
                        else (
                            0.04
                            if query_is_high_concept
                            else 0.05 if query_is_serialized_tv else 0.08
                        )
                    )
                )
            )
            * math.log1p(float(getattr(item, "vote_count", None) or 0.0))
        )
        + (
            (
                0.6
                if query_is_multilingual_family_adventure
                else (
                    0.8
                    if query_is_noir
                    else (
                        0.9
                        if query_is_time_bending
                        else (
                            0.9
                            if query_is_high_concept
                            else 1.0 if query_is_serialized_tv else 2.0
                        )
                    )
                )
            )
            / (10.0 + float(getattr(item, "popular_rank", None) or _UNKNOWN_RANK))
        )
        + (
            (
                0.3
                if query_is_multilingual_family_adventure
                else (
                    0.4
                    if query_is_noir
                    else (
                        0.45
                        if query_is_time_bending
                        else (
                            0.45
                            if query_is_high_concept
                            else 0.5 if query_is_serialized_tv else 1.0
                        )
                    )
                )
            )
            / (10.0 + float(getattr(item, "trending_rank", None) or _UNKNOWN_RANK))
        )
    )
    return (
        -(semantic_score + catalog_score + lexical_bonus),
        int(getattr(item, "popular_rank", None) or _UNKNOWN_RANK),
        int(getattr(item, "trending_rank", None) or _UNKNOWN_RANK),
        -float(getattr(item, "popularity", None) or 0.0),
        -float(getattr(item, "vote_count", None) or 0.0),
        int(item.id),
    )


def _copy_intent_filters(intent: IntentFilters) -> IntentFilters:
    return IntentFilters(
        raw_query=intent.raw_query,
        genres=list(intent.genres),
        moods=list(intent.moods),
        media_types=list(intent.media_types),
        year_min=intent.year_min,
        year_max=intent.year_max,
        min_runtime=intent.min_runtime,
        max_runtime=intent.max_runtime,
        maturity_rating_max=intent.maturity_rating_max,
    )


def _serialize_intent_filters(intent: IntentFilters | None) -> Dict[str, Any]:
    if intent is None:
        return {}
    return {
        "raw_query": intent.raw_query,
        "genres": list(intent.genres),
        "effective_genres": intent.effective_genres(),
        "moods": list(intent.moods),
        "media_types": list(intent.media_types),
        "year_min": intent.year_min,
        "year_max": intent.year_max,
        "min_runtime": intent.min_runtime,
        "max_runtime": intent.max_runtime,
        "maturity_rating_max": intent.maturity_rating_max,
    }


def _normalized_preferred_media_types(values: Sequence[str] | None) -> List[str]:
    normalized: List[str] = []
    seen: set[str] = set()
    for value in values or []:
        media_type = str(value).strip().lower()
        if media_type not in {"movie", "tv"} or media_type in seen:
            continue
        seen.add(media_type)
        normalized.append(media_type)
    return normalized


def _preferred_media_rank(
    media_type: str | None,
    preferred_media_types: Sequence[str] | None,
) -> int:
    normalized_media_type = str(media_type).strip().lower() if media_type else ""
    preferred = _normalized_preferred_media_types(preferred_media_types)
    if not preferred or normalized_media_type not in preferred:
        return len(preferred)
    return preferred.index(normalized_media_type)


def _intent_signature(intent: IntentFilters) -> tuple[Any, ...]:
    return (
        tuple(intent.genres),
        tuple(intent.moods),
        tuple(intent.media_types),
        intent.min_runtime,
        intent.max_runtime,
        intent.maturity_rating_max,
    )


def _widen_runtime_bounds(intent: IntentFilters) -> IntentFilters:
    widened = _copy_intent_filters(intent)
    if widened.max_runtime is not None:
        widened.max_runtime = max(
            widened.max_runtime + 10,
            int(round(widened.max_runtime * 1.25)),
        )
    if widened.min_runtime is not None:
        widened.min_runtime = max(0, int(round(widened.min_runtime * 0.8)))
    if (
        widened.min_runtime is not None
        and widened.max_runtime is not None
        and widened.min_runtime > widened.max_runtime
    ):
        widened.min_runtime = widened.max_runtime
    return widened


def _build_constraint_relaxation_ladder(
    intent: IntentFilters | None,
) -> List[IntentFilters]:
    if not intent:
        return []

    has_runtime = intent.min_runtime is not None or intent.max_runtime is not None
    has_maturity_cap = bool(intent.maturity_rating_max)
    if not has_runtime and not has_maturity_cap:
        return []

    ladder: List[IntentFilters] = []
    seen = {_intent_signature(intent)}

    def add(candidate: IntentFilters) -> None:
        signature = _intent_signature(candidate)
        if signature in seen:
            return
        seen.add(signature)
        ladder.append(candidate)

    if has_runtime:
        add(_widen_runtime_bounds(intent))

    if has_maturity_cap:
        no_maturity = _copy_intent_filters(intent)
        no_maturity.maturity_rating_max = None
        add(no_maturity)

    if has_runtime and has_maturity_cap:
        widened_no_maturity = _widen_runtime_bounds(intent)
        widened_no_maturity.maturity_rating_max = None
        add(widened_no_maturity)

    if has_runtime:
        no_runtime = _copy_intent_filters(intent)
        no_runtime.min_runtime = None
        no_runtime.max_runtime = None
        add(no_runtime)

    if has_runtime and has_maturity_cap:
        no_runtime_no_maturity = _copy_intent_filters(intent)
        no_runtime_no_maturity.min_runtime = None
        no_runtime_no_maturity.max_runtime = None
        no_runtime_no_maturity.maturity_rating_max = None
        add(no_runtime_no_maturity)

    return ladder


def _query_semantic_facets(query: str | None) -> set[str]:
    return set(build_query_profile(query).semantic_facets)


def _semantic_specificity_score(
    intent: IntentFilters | None,
    *,
    preferred_services: Sequence[str] | None = None,
) -> int:
    if intent is None:
        return 0

    score = len(_query_semantic_facets(intent.raw_query)) * 2
    if len(intent.effective_genres()) >= 2:
        score += 1
    if intent.year_min is not None or intent.year_max is not None:
        score += 1
    if intent.min_runtime is not None or intent.max_runtime is not None:
        score += 1
    if intent.maturity_rating_max:
        score += 1
    if preferred_services:
        score += 1
    raw_query = _normalize_query_text(intent.raw_query)
    if raw_query and len(raw_query.split()) >= 6:
        score += 1
    return score


def _filter_expressiveness_gap(
    intent: IntentFilters | None,
    *,
    preferred_services: Sequence[str] | None = None,
) -> int:
    if intent is None:
        return 0

    breadth = 0
    media_types = {value for value in intent.media_types if value}
    if not media_types:
        breadth += 2
    elif len(media_types) > 1:
        breadth += 1
    if len(intent.effective_genres()) <= 1:
        breadth += 1
    if intent.year_min is None and intent.year_max is None:
        breadth += 1
    if intent.min_runtime is None and intent.max_runtime is None:
        breadth += 1
    if not intent.maturity_rating_max:
        breadth += 1
    if not preferred_services:
        breadth += 1
    return breadth


def _semantic_recall_boost_level(
    intent: IntentFilters | None,
    *,
    preferred_services: Sequence[str] | None = None,
) -> int:
    if intent is None or not (intent.raw_query or "").strip():
        return 0

    specificity = _semantic_specificity_score(
        intent,
        preferred_services=preferred_services,
    )
    breadth = _filter_expressiveness_gap(
        intent,
        preferred_services=preferred_services,
    )

    pressure = 0
    if specificity >= 4 and breadth >= 2:
        pressure = 1
    if specificity >= 6 and breadth >= 3:
        pressure = 2
    if specificity >= 8 and breadth >= 4:
        pressure = 3
    return pressure


def _needs_semantic_recall_boost(
    intent: IntentFilters | None,
    *,
    preferred_services: Sequence[str] | None = None,
) -> bool:
    return (
        _semantic_recall_boost_level(
            intent,
            preferred_services=preferred_services,
        )
        > 0
    )


def _retrieval_pressure(
    intent: IntentFilters | None,
    *,
    preferred_services: Sequence[str] | None = None,
) -> int:
    # Compatibility wrapper for existing budget/debug consumers.
    return _semantic_recall_boost_level(
        intent,
        preferred_services=preferred_services,
    )


def _prefilter_fetch_limit(
    limit: int,
    intent: IntentFilters | None,
    *,
    preferred_services: Sequence[str] | None = None,
) -> int:
    fetch_limit = max(limit * 10, 500)
    pressure = _semantic_recall_boost_level(
        intent,
        preferred_services=preferred_services,
    )
    if pressure == 1:
        return max(fetch_limit, limit * 20, 1000)
    if pressure == 2:
        return max(fetch_limit, limit * 30, 1500)
    if pressure >= 3:
        return max(fetch_limit, limit * 40, 2000)
    return fetch_limit


def _constraint_prior_fetch_limit(
    limit: int,
    intent: IntentFilters | None,
    *,
    rank_vector: np.ndarray | None = None,
) -> int:
    fetch_limit = max(limit * (10 if rank_vector is not None else 8), 50)
    pressure = _semantic_recall_boost_level(intent)
    if pressure == 1:
        return max(fetch_limit, limit * 12, 80)
    if pressure == 2:
        return max(fetch_limit, limit * 16, 120)
    if pressure >= 3:
        return max(fetch_limit, limit * 20, 160)
    return fetch_limit


def _resolve_candidate_limit(
    params: "RecommendParams",
    intent: IntentFilters,
) -> int:
    candidate_limit = min(
        500,
        max(params.limit, params.limit * _BASE_CANDIDATE_MULTIPLIER),
    )
    if not params.query:
        return min(
            500,
            max(candidate_limit, params.limit * _NO_QUERY_CANDIDATE_MULTIPLIER),
        )
    if intent.has_filters():
        candidate_limit = min(
            500,
            max(candidate_limit, params.limit * _FILTERED_QUERY_CANDIDATE_MULTIPLIER),
        )
    pressure = _semantic_recall_boost_level(intent)
    if pressure == 1:
        candidate_limit = max(candidate_limit, params.limit * 6)
    elif pressure == 2:
        candidate_limit = max(candidate_limit, params.limit * 8)
    elif pressure >= 3:
        candidate_limit = max(candidate_limit, params.limit * 10)
    return candidate_limit


def _first_relaxed_match_tier(
    item: Item,
    relaxed_intents: Sequence[IntentFilters],
    *,
    enforce_genres: bool,
) -> int | None:
    for idx, relaxed_intent in enumerate(relaxed_intents):
        if item_matches_intent(item, relaxed_intent, enforce_genres=enforce_genres):
            return idx
    return None


def _apply_franchise_cap(
    candidates: List[Dict[str, Any]], cap: int = 2
) -> List[Dict[str, Any]]:
    if not candidates or cap <= 0:
        return candidates

    franchise_counts: Dict[int, int] = {}
    filtered_candidates: List[Dict[str, Any]] = []

    for item in candidates:
        collection_id = item.get("collection_id")
        if collection_id is None:
            filtered_candidates.append(item)
            continue

        count = franchise_counts.get(collection_id, 0)
        if count < cap:
            filtered_candidates.append(item)
            franchise_counts[collection_id] = count + 1

    return filtered_candidates


def _is_long_tail(item: Dict[str, Any], limit: int) -> bool:
    if limit <= 0:
        return False
    original_rank = int(item.get("original_rank", 0) or 0)
    return original_rank >= limit


def _serendipity_target(limit: int) -> int:
    """
    Determine how many serendipity slots to allocate.

    We skip serendipity when the limit is very small (<=2) because swapping
    long-tail items into a list that short tends to degrade perceived quality.
    """
    if _SERENDIPITY_RATIO <= 0.0 or limit <= 2:
        return 0
    target = round(limit * _SERENDIPITY_RATIO)
    target = max(1, target)
    return min(limit, target)


def _apply_serendipity_slot(
    current: List[Dict[str, Any]],
    candidate_pool: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    if not current or limit <= 0 or _SERENDIPITY_RATIO <= 0.0:
        return current

    top_count = min(limit, len(current))
    target = _serendipity_target(limit)
    if target == 0:
        return current

    top_section = list(current[:top_count])
    existing_long_tail = [item for item in top_section if _is_long_tail(item, limit)]
    if len(existing_long_tail) >= target:
        return current

    short_tail_candidates = [
        idx for idx, item in enumerate(top_section) if not _is_long_tail(item, limit)
    ]
    if not short_tail_candidates:
        return current

    top_ids = {item.get("id") for item in top_section if item.get("id") is not None}

    replacement_pool: List[Dict[str, Any]] = []
    seen_pool: set[int] = set()
    for item in candidate_pool:
        ident = item.get("id")
        if ident is None or ident in top_ids or ident in seen_pool:
            continue
        if _is_long_tail(item, limit):
            replacement_pool.append(item)
            seen_pool.add(ident)

    if not replacement_pool:
        return current

    needed = min(target - len(existing_long_tail), len(replacement_pool))
    if needed <= 0:
        return current

    short_tail_candidates = short_tail_candidates[-needed:]
    replacements = replacement_pool[:needed]

    new_top = top_section
    for idx, replacement in zip(short_tail_candidates, replacements):
        new_top[idx] = replacement

    deduped: List[Dict[str, Any]] = []
    seen_ids: set[int] = set()
    for item in new_top + list(current):
        ident = item.get("id")
        if ident is None or ident not in seen_ids:
            if ident is not None:
                seen_ids.add(ident)
            deduped.append(item)

    return deduped


@dataclass
class ComputeResult:
    items: List[Dict[str, Any]]
    debug_context: Dict[str, Any]


def _round_debug_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _debug_item_identity(
    item_id: int,
    items_with_data: Dict[int, Tuple[Any, Any, Any]],
) -> Dict[str, Any]:
    item_row = items_with_data.get(item_id, (None, None, None))[0]
    if item_row is None:
        return {"id": item_id}
    return {
        "id": item_row.id,
        "tmdb_id": item_row.tmdb_id,
        "title": item_row.title,
        "media_type": item_row.media_type,
    }


def _debug_id_snapshot(
    ids: Sequence[int],
    items_with_data: Dict[int, Tuple[Any, Any, Any]],
    *,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    return [
        _debug_item_identity(item_id, items_with_data) for item_id in list(ids)[:limit]
    ]


def _debug_scored_id_snapshot(
    pairs: Sequence[Tuple[int, float]],
    items_with_data: Dict[int, Tuple[Any, Any, Any]],
    *,
    score_label: str = "score",
    limit: int = 10,
) -> List[Dict[str, Any]]:
    snapshot: List[Dict[str, Any]] = []
    for item_id, score in list(pairs)[:limit]:
        payload = _debug_item_identity(item_id, items_with_data)
        payload[score_label] = _round_debug_float(score)
        snapshot.append(payload)
    return snapshot


def _debug_candidate_snapshot(
    candidates: Sequence[Dict[str, Any]],
    *,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    snapshot: List[Dict[str, Any]] = []
    for candidate in list(candidates)[:limit]:
        source_scores = {
            key: _round_debug_float(value)
            for key, value in (candidate.get("source_scores") or {}).items()
        }
        snapshot.append(
            {
                "id": candidate.get("id"),
                "tmdb_id": candidate.get("tmdb_id"),
                "title": candidate.get("title"),
                "media_type": candidate.get("media_type"),
                "tagline": candidate.get("tagline"),
                "tmdb_keywords": list(candidate.get("tmdb_keywords") or [])[:5],
                "original_rank": candidate.get("original_rank"),
                "ann_rank": candidate.get("ann_rank"),
                "retrieval_score": _round_debug_float(candidate.get("retrieval_score")),
                "score": _round_debug_float(candidate.get("score")),
                "source_scores": source_scores,
            }
        )
    return snapshot


@dataclass
class RecommendParams:
    user_id: str = Query(..., description="Seen'emAll user_id (e.g., 'u1')")
    limit: int = Query(20, ge=1, le=100)
    query: str | None = Query(
        None, description="Optional natural-language intent (e.g. 'light sci-fi < 2h')"
    )
    diversify: bool = Query(True, description="Whether to diversify recommendations.")
    profile: str | None = Query(None, description="Optional profile identifier")
    use_llm_intent: bool = Query(
        True,
        description="Enable the LLM intent parser (set to false for manual overrides).",
    )
    ann_description_override: str | None = Query(
        None,
        description="Manual ANN description override to blend into the rewrite vector.",
    )
    rewrite_override: str | None = Query(
        None,
        description="Manual rewrite text override (skips rewrite_query when provided).",
    )
    ann_weight_override: float | None = Query(
        None, ge=0.0, description="Override weight for the ANN description component."
    )
    rewrite_weight_override: float | None = Query(
        None, ge=0.0, description="Override weight for the rewrite text component."
    )
    genre_override: str | None = Query(
        None,
        description="Comma-separated manual genres to enforce (e.g., 'Drama, Sci-Fi').",
    )
    mixer_ann_weight: float | None = Query(
        None,
        ge=0.0,
        description="Override hybrid ANN weight (default from HYBRID_ANN_WEIGHT).",
    )
    mixer_collab_weight: float | None = Query(
        None,
        ge=0.0,
        description="Override collaborative weight (default MIXER_COLLAB_WEIGHT).",
    )
    mixer_trending_weight: float | None = Query(
        None,
        ge=0.0,
        description="Override trending weight (default HYBRID_TRENDING_WEIGHT).",
    )
    mixer_popularity_weight: float | None = Query(
        None,
        ge=0.0,
        description="Override popularity weight (default HYBRID_POPULARITY_WEIGHT).",
    )
    mixer_vote_weight: float | None = Query(
        None,
        ge=0.0,
        description="Override vote-count weight (default HYBRID_VOTE_WEIGHT).",
    )
    mixer_novelty_weight: float | None = Query(
        None,
        ge=0.0,
        description="Override novelty weight (default MIXER_NOVELTY_WEIGHT).",
    )


async def _compute_recommendations_async(
    request: Request,
    params: RecommendParams,
    db: Session,
) -> ComputeResult:
    _pipeline_start = time.perf_counter()
    debug_mode = request.url.path.endswith("/debug")
    canonical_id = canonical_profile_id(params.user_id, params.profile)
    long_v, short_v, exclude, profile_meta = load_user_state(db, canonical_id)
    cold_start = short_v is None
    if cold_start:
        METRICS.counter("recommend.cold_start").inc()

    linked_entities = None
    if params.query:
        entity_linker = getattr(request.app.state, "entity_linker", None)
        if entity_linker:
            linked_entities = await entity_linker.link_entities(params.query)

    llm_user_context = {"user_id": canonical_id, "profile_id": params.profile}
    if params.use_llm_intent:
        llm_intent = _parse_llm_intent(params.query, llm_user_context, linked_entities)
    else:
        llm_intent = llm_parser.default_intent()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "LLM intent parser disabled for user %s; using manual/default intent.",
                canonical_id,
            )

    intent_snapshot = {}
    if logger.isEnabledFor(logging.DEBUG):
        intent_snapshot = {
            key: value
            for key, value in llm_intent.model_dump(exclude_none=True).items()
            if key in {"include_genres", "exclude_genres", "maturity_rating_max"}
        }
        logger.debug("LLM intent parsed for user %s: %s", canonical_id, intent_snapshot)
        if linked_entities:
            entity_counts = {
                key: len(value) if isinstance(value, list) else 0
                for key, value in linked_entities.items()
            }
            logger.debug("Linked entity counts: %s", entity_counts)

    intent = _intent_filters_from_llm(params.query, llm_intent)
    if params.genre_override:
        custom_genres = [
            g.strip() for g in params.genre_override.split(",") if g.strip()
        ]
        if custom_genres:
            intent.genres = custom_genres
            llm_intent = llm_parser.default_intent()
            llm_intent.include_genres = custom_genres

    preferred_services = _normalize_streaming_services(llm_intent.streaming_providers)
    llm_media_types = list(intent.media_types or [])
    legacy_filters = legacy_parse_intent(params.query) if params.query else None
    if legacy_filters:
        intent = _merge_with_legacy_filters(intent, legacy_filters)
    intent = _apply_explicit_query_overrides(
        intent, params.query, llm_intent=llm_intent
    )
    intent = _normalize_merged_intent(intent, params.query, llm_intent=llm_intent)
    preferred_media_types = _normalized_preferred_media_types(
        profile_meta.get("preferred_media_types")
    )
    soften_provider_preference = bool(
        preferred_services
    ) and _should_soften_provider_preference(
        intent,
        params.query,
    )
    entity_media_types = linked_media_types(linked_entities)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Media type signals for user %s | llm=%s legacy=%s linked=%s -> merged=%s",
            canonical_id,
            llm_media_types,
            legacy_filters.media_types if legacy_filters else None,
            entity_media_types,
            intent.media_types,
        )
        if preferred_services:
            logger.debug(
                "Requested streaming providers for user %s: %s",
                canonical_id,
                sorted(preferred_services),
            )

    if params.ann_description_override:
        llm_intent.ann_description = params.ann_description_override.strip()
    candidate_limit = _resolve_candidate_limit(params, intent)
    query_profile = build_query_profile(intent.raw_query)
    semantic_facets = list(query_profile.semantic_facets)
    semantic_specificity_score = _semantic_specificity_score(
        intent,
        preferred_services=sorted(preferred_services),
    )
    filter_expressiveness_gap = _filter_expressiveness_gap(
        intent,
        preferred_services=sorted(preferred_services),
    )
    retrieval_pressure = _semantic_recall_boost_level(
        intent,
        preferred_services=sorted(preferred_services),
    )
    needs_semantic_recall_boost = retrieval_pressure > 0

    prefilter = _prefilter_allowed_ids(
        db,
        intent,
        candidate_limit,
        preferred_services=None if soften_provider_preference else preferred_services,
    )
    allowlist = prefilter.allowed_ids
    boost_ids = prefilter.boost_ids or []
    enforce_genres = prefilter.enforce_genres
    constraint_prior_fetch_limit = None
    rewrite_result = None
    rewrite_vec: np.ndarray | None = None
    constraint_rank_vec: np.ndarray | None = None
    primary_retrieval_ids: List[int] = []
    primary_retrieval_stage = "none"
    # Task 2.1: Intent-Injected Vector Query
    # Append semantic facets to the query string to anchor the vector embedding
    effective_query = params.query or ""
    if semantic_facets:
        facet_str = " ".join(f"[{f}]" for f in semantic_facets)
        effective_query = f"{effective_query} {facet_str}"
        logger.debug("Intent-injected query for embedding: %s", effective_query)

    if params.rewrite_override:
        manual_rewrite_text = " ".join(params.rewrite_override.split()[:8])
        rewrite_result = llm_parser.default_rewrite()
        rewrite_result.rewritten_text = manual_rewrite_text
    elif params.query:
        rewrite_result = rewrite_query(effective_query, llm_intent)

    effective_ann_description = _effective_ann_description(
        raw_query=params.query,
        ann_description=getattr(llm_intent, "ann_description", None),
        rewrite_text=getattr(rewrite_result, "rewritten_text", None),
    )
    rewrite_vec = _build_rewrite_vector(
        getattr(rewrite_result, "rewritten_text", None),
        effective_ann_description,
        params.ann_weight_override,
        params.rewrite_weight_override,
    )
    constraint_rank_vec = rewrite_vec

    ids = []
    if cold_start:
        rewrite_used = False
        if rewrite_vec is not None:
            vec_norm = float(np.linalg.norm(rewrite_vec))
            if vec_norm > 0 and np.isfinite(vec_norm):
                with timer("recommend.ann_latency_ms"):
                    ann_ids = _ann_candidates_with_version(
                        db,
                        rewrite_vec,
                        exclude,
                        limit=candidate_limit,
                        allowed_ids=allowlist,
                    )
                if ann_ids:
                    logger.info(
                        "Using rewrite ANN candidates for cold-start user %s",
                        canonical_id,
                    )
                    logger.debug(
                        "Rewrite ANN retrieval | candidate_count=%d allowlist_size=%s",
                        len(ann_ids),
                        len(allowlist) if allowlist is not None else None,
                    )
                    ids = ann_ids
                    primary_retrieval_ids = list(ann_ids)
                    primary_retrieval_stage = "rewrite_ann"
                    rewrite_used = True

        if not rewrite_used:
            logger.info("Using cold-start candidates for user %s", canonical_id)
            ids = _cold_start_candidates(
                db,
                intent,
                candidate_limit,
                allowlist,
                preferred_media_types=(
                    preferred_media_types if not params.query else None
                ),
            )
            logger.debug(
                "Cold-start retrieval | candidate_count=%d allowlist_size=%s enforce_genres=%s",
                len(ids),
                len(allowlist) if allowlist is not None else None,
                enforce_genres,
            )
            primary_retrieval_ids = list(ids)
            primary_retrieval_stage = "cold_start"
    else:
        logger.info("Using ANN candidates for user %s", canonical_id)
        assert short_v is not None
        if rewrite_vec is not None:
            alpha = _REWRITE_BLEND_ALPHA_QUERY if params.query else _REWRITE_BLEND_ALPHA
            alpha = max(0.0, min(1.0, alpha))
            q_vec = (alpha * short_v) + ((1 - alpha) * rewrite_vec)
            q_vec = q_vec / np.linalg.norm(q_vec)
        else:
            q_vec = short_v
        constraint_rank_vec = rewrite_vec if rewrite_vec is not None else q_vec

        with timer("recommend.ann_latency_ms"):
            ann_ids = _ann_candidates_with_version(
                db,
                q_vec,
                exclude,
                limit=candidate_limit,
                allowed_ids=allowlist,
            )

        # New: Lexical candidate retrieval
        lexical_ids = []
        if params.query:
            lexical_ids = lexical_candidates(
                db,
                params.query,
                exclude,
                limit=candidate_limit,
                allowed_ids=allowlist,
            )

        # Merge ANN and Lexical candidates using RRF
        ids = _rrf_merge(ann_ids, lexical_ids)

        bool(rewrite_vec is not None)
        logger.debug(
            "Retrieval results | ann_count=%d lexical_count=%d merged_count=%d",
            len(ann_ids),
            len(lexical_ids),
            len(ids),
        )
        primary_retrieval_ids = list(ids)
        primary_retrieval_stage = "hybrid_rrf" if lexical_ids else "ann"

    collab_results = _collaborative_candidates(
        db,
        profile_meta.get("neighbors"),
        exclude,
        candidate_limit,
        allowed_ids=allowlist,
    )
    logger.debug(
        "Collaborative retrieval | neighbor_count=%d candidate_count=%d allowlist_size=%s",
        len(profile_meta.get("neighbors") or []),
        len(collab_results),
        len(allowlist) if allowlist is not None else None,
    )
    collab_scores = {iid: score for iid, score in collab_results}
    merged_scores: Dict[int, Dict[str, float]] = {}
    if not cold_start:
        ann_scores = {iid: 1.0 / (1.0 + idx) for idx, iid in enumerate(ids)}
        for iid, score in ann_scores.items():
            merged_scores.setdefault(iid, {})["ann"] = score
    else:
        ann_scores = {}

    if collab_scores:
        max_collab = max(collab_scores.values()) or 1.0
        for iid, score in collab_scores.items():
            merged_scores.setdefault(iid, {})["collab"] = score / max_collab

    trending_results: List[Tuple[int, float]] = []
    if not params.query:
        if cold_start or ann_scores or collab_scores or boost_ids:
            trending_results = _trending_prior_candidates(
                db,
                intent,
                exclude,
                candidate_limit,
                allowed_ids=allowlist,
                preferred_media_types=preferred_media_types,
            )
            logger.debug(
                "Trending prior retrieval | candidate_count=%d allowlist_size=%s",
                len(trending_results),
                len(allowlist) if allowlist is not None else None,
            )
            trending_scores = {iid: score for iid, score in trending_results}
            if trending_scores:
                max_trending = max(trending_scores.values()) or 1.0
                for iid, score in trending_scores.items():
                    merged_scores.setdefault(iid, {})["trending"] = score / max_trending

    collab_ids = [iid for iid, _ in collab_results]
    trending_ids = [iid for iid, _ in trending_results]
    merged_ids_before_constraint = list(ids)

    if collab_ids or trending_ids:
        merged: List[int] = []
        seen: set[int] = set()
        sequences = [collab_ids, ids, trending_ids]
        for seq in sequences:
            for iid in seq:
                if iid in seen:
                    continue
                merged.append(iid)
                seen.add(iid)
                if len(merged) >= candidate_limit:
                    break
            if len(merged) >= candidate_limit:
                break
        ids = merged
        merged_ids_before_constraint = list(ids)
    # Limit applied after boost reordering

    constraint_prior_ids: List[int] = []
    if params.query and intent.has_filters():
        constraint_prior_fetch_limit = _constraint_prior_fetch_limit(
            max(params.limit * 2, 20),
            intent,
            rank_vector=constraint_rank_vec,
        )
        constraint_prior_ids = _constraint_prior_candidates(
            db,
            intent,
            limit=max(params.limit * 2, 20),
            allowlist=allowlist,
            enforce_genres=enforce_genres,
            rank_vector=constraint_rank_vec,
        )
        if constraint_prior_ids:
            boost_ids = _ordered_unique(constraint_prior_ids + list(boost_ids))
            logger.debug(
                "Constraint prior retrieval | candidate_count=%d allowlist_size=%s enforce_genres=%s",
                len(constraint_prior_ids),
                len(allowlist) if allowlist is not None else None,
                enforce_genres,
            )
            ids = _ordered_unique(constraint_prior_ids + ids)
            max_constraint_rank = max(1, len(constraint_prior_ids))
            for idx, iid in enumerate(constraint_prior_ids):
                normalized = 1.0 - (idx / max_constraint_rank)
                scores = merged_scores.setdefault(iid, {})
                current = float(scores.get("constraint_prior") or 0.0)
                if normalized > current:
                    scores["constraint_prior"] = normalized

    if boost_ids:
        exclude_set = set(exclude or [])
        priority: List[int] = []
        seen_priority: set[int] = set()
        for candidate in boost_ids:
            if candidate in exclude_set or candidate in seen_priority:
                continue
            priority.append(candidate)
            seen_priority.add(candidate)

        if priority:
            priority = [cand for cand in priority if cand in ids]
            if priority:
                combined: List[int] = list(priority)
                seen_all = set(priority)
                for candidate in ids:
                    if candidate in seen_all:
                        continue
                    combined.append(candidate)
                    seen_all.add(candidate)

                ids = combined
                for idx, candidate in enumerate(priority):
                    score = 1.0 / (1.0 + idx)
                    scores = merged_scores.setdefault(candidate, {})
                    current = scores.get("ann")
                    if current is None or score > current:
                        scores["ann"] = score
                logger.debug(
                    "Boost reordering applied | boost_count=%d merged_length=%d",
                    len(priority),
                    len(ids),
                )

    ids_before_trim = list(ids)
    ids = ids[:candidate_limit]

    negative_items = set(profile_meta.get("negative_items") or [])
    if negative_items:
        ids = [i for i in ids if i not in negative_items]
    if not ids:
        return ComputeResult(items=[], debug_context={})

    # fetch metadata and streaming links, preserve ANN order
    items_with_data = {
        row.id: (row, vec, watch_options)
        for row, vec, watch_options in db.execute(
            select(
                Item,
                ItemEmbedding.vector,
                func.json_agg(
                    func.json_build_object(
                        "service",
                        Availability.service,
                        "url",
                        func.coalesce(Availability.web_url, Availability.deeplink),
                    )
                ).label("watch_options"),
            )
            .join(ItemEmbedding, Item.id == ItemEmbedding.item_id)
            .outerjoin(
                Availability,
                (Item.id == Availability.item_id)
                & (Availability.country == COUNTRY_DEFAULT),
            )
            .where(Item.id.in_(ids))
            .group_by(Item.id, ItemEmbedding.vector)
        ).all()
    }
    ordered: List[Dict[str, Any]] = []
    fallback_candidates: List[Dict[str, Any]] = []
    relaxed_intents = _build_constraint_relaxation_ladder(intent)
    relaxed_candidates_by_tier: List[List[Dict[str, Any]]] = [
        [] for _ in relaxed_intents
    ]
    relaxed_provider_fallback_by_tier: List[List[Dict[str, Any]]] = [
        [] for _ in relaxed_intents
    ]
    intent_relaxation_added = 0
    max_candidates = min(candidate_limit, max(params.limit * 2, 25))
    skipped_intent = 0
    rank_counter = 0
    strict_ordered_count = 0
    for iid in ids:
        it, vec, watch_options = items_with_data.get(iid, (None, None, None))
        if it is None or vec is None or len(vec) == 0:
            continue
        strict_intent_match = item_matches_intent(
            it, intent, enforce_genres=enforce_genres
        )
        relaxed_tier: int | None = None
        if not strict_intent_match:
            relaxed_tier = _first_relaxed_match_tier(
                it, relaxed_intents, enforce_genres=enforce_genres
            )
        if not strict_intent_match and relaxed_tier is None:
            skipped_intent += 1
            continue

        sources = merged_scores.get(iid, {})

        # Clean up watch_options - remove null entries and handle None case
        cleaned_options = []
        if (
            watch_options and watch_options[0] is not None
        ):  # PostgreSQL returns [null] when no matches
            cleaned_options = [
                {"service": opt["service"], "url": opt["url"]}
                for opt in watch_options
                if opt["url"] is not None
            ]
        provider_allowed = True
        filtered_options = cleaned_options
        if preferred_services:
            matching_options = [
                opt
                for opt in cleaned_options
                if isinstance(opt, dict)
                and ((service := str(opt.get("service") or "").strip().lower()))
                and service in preferred_services
            ]
            if soften_provider_preference:
                if matching_options:
                    filtered_options = matching_options
            elif not matching_options:
                provider_allowed = False
            else:
                filtered_options = matching_options

        candidate_payload = {
            "id": it.id,
            "tmdb_id": it.tmdb_id,
            "media_type": it.media_type,
            "title": it.title,
            "tagline": getattr(it, "tagline", None),
            "tmdb_keywords": list(getattr(it, "tmdb_keywords", None) or []),
            "overview": it.overview,
            "poster_url": it.poster_url,
            "runtime": it.runtime,
            "original_language": it.original_language,
            "genres": it.genres,
            "release_year": it.release_year,
            "collection_id": it.collection_id,
            "collection_name": it.collection_name,
            "maturity_rating": getattr(it, "maturity_rating", None),
            "watch_options": filtered_options,
            "watch_url": (filtered_options[0]["url"] if filtered_options else None),
            "original_rank": rank_counter,
            "ann_rank": rank_counter,
            "vector": vec,
            "popularity": getattr(it, "popularity", None),
            "vote_average": getattr(it, "vote_average", None),
            "vote_count": getattr(it, "vote_count", None),
            "popular_rank": getattr(it, "popular_rank", None),
            "trending_rank": getattr(it, "trending_rank", None),
            "top_rated_rank": getattr(it, "top_rated_rank", None),
            "retrieval_score": None,
            "source_scores": sources,
        }

        if strict_intent_match and provider_allowed:
            ordered.append(candidate_payload)
            strict_ordered_count += 1
        elif strict_intent_match:
            candidate_payload["watch_options"] = cleaned_options
            candidate_payload["watch_url"] = (
                cleaned_options[0]["url"] if cleaned_options else None
            )
            fallback_candidates.append(candidate_payload)
        elif relaxed_tier is not None and provider_allowed:
            relaxed_candidates_by_tier[relaxed_tier].append(candidate_payload)
        elif relaxed_tier is not None:
            candidate_payload["watch_options"] = cleaned_options
            candidate_payload["watch_url"] = (
                cleaned_options[0]["url"] if cleaned_options else None
            )
            relaxed_provider_fallback_by_tier[relaxed_tier].append(candidate_payload)

        rank_counter += 1
        if provider_allowed and len(ordered) >= max_candidates:
            break

    if len(ordered) < params.limit and relaxed_candidates_by_tier:
        for tier_index, tier_candidates in enumerate(relaxed_candidates_by_tier):
            if len(ordered) >= params.limit:
                break
            if not tier_candidates:
                continue
            deficit = params.limit - len(ordered)
            selected = tier_candidates[:deficit]
            if not selected:
                continue
            ordered.extend(selected)
            intent_relaxation_added += len(selected)
            logger.debug(
                "Constraint relaxation tier %d backfilled %d candidates.",
                tier_index,
                len(selected),
            )
        if intent_relaxation_added:
            METRICS.counter("recommend.intent_relaxation_applied").inc()

    provider_fallback_pool: List[Dict[str, Any]] = list(fallback_candidates)
    if relaxed_provider_fallback_by_tier:
        for tier_candidates in relaxed_provider_fallback_by_tier:
            provider_fallback_pool.extend(tier_candidates)
    if preferred_services and len(ordered) < params.limit and provider_fallback_pool:
        deficit = params.limit - len(ordered)
        ordered.extend(provider_fallback_pool[:deficit])

    if not ordered:
        return ComputeResult(items=[], debug_context={})

    if skipped_intent and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Post-filter dropped %d candidates due to intent match (enforce_genres=%s)",
            skipped_intent,
            enforce_genres,
        )

    if params.diversify:
        ordered = _apply_franchise_cap(ordered)

    pre_mixer_count = len(ordered)
    pre_mixer_candidates = _debug_candidate_snapshot(ordered) if debug_mode else None
    ann_weight_override = params.mixer_ann_weight
    collab_weight_override = params.mixer_collab_weight
    trending_weight_override = params.mixer_trending_weight
    popularity_weight_override = params.mixer_popularity_weight
    vote_weight_override = params.mixer_vote_weight
    novelty_weight_override = params.mixer_novelty_weight

    if params.query and _QUERY_DISABLE_NON_ANN_SIGNALS:
        if trending_weight_override is None:
            trending_weight_override = 0.0
        if popularity_weight_override is None:
            popularity_weight_override = 0.0
        if vote_weight_override is None:
            vote_weight_override = 0.0
        if novelty_weight_override is None:
            novelty_weight_override = 0.0

    _apply_mixer_scores(
        ordered,
        ann_weight_override=ann_weight_override,
        collab_weight_override=collab_weight_override,
        trending_weight_override=trending_weight_override,
        popularity_weight_override=popularity_weight_override,
        vote_weight_override=vote_weight_override,
        novelty_weight_override=novelty_weight_override,
    )
    post_mixer_count = len(ordered)
    post_mixer_candidates = _debug_candidate_snapshot(ordered) if debug_mode else None
    ordered = apply_business_rules(ordered, intent=intent)
    if not ordered:
        return ComputeResult(items=[], debug_context={})
    post_business_rule_count = len(ordered)
    post_business_rule_candidates = (
        _debug_candidate_snapshot(ordered) if debug_mode else None
    )

    serendipity_context = list(ordered)

    if params.diversify:
        with timer("recommend.mmr_latency_ms"):
            ordered = diversify_with_mmr(ordered, limit=params.limit)
    post_diversify_count = len(ordered)
    post_diversify_candidates = (
        _debug_candidate_snapshot(ordered) if debug_mode else None
    )

    ordered = _apply_serendipity_slot(ordered, serendipity_context, params.limit)

    with timer("recommend.rerank_latency_ms"):
        reranked = rerank_with_explanations(
            ordered,
            intent=intent,
            query=params.query,
            user={
                "user_id": canonical_id,
                "base_user_id": params.user_id,
                "profile": params.profile,
                "genre_prefs": profile_meta.get("genre_prefs"),
                "neighbors": profile_meta.get("neighbors"),
                "negative_items": profile_meta.get("negative_items"),
            },
        )
    METRICS.counter("recommend.reranker_used").inc()

    pipeline_ms = (time.perf_counter() - _pipeline_start) * 1000
    METRICS.histogram("recommend.total_latency_ms").observe(pipeline_ms)

    debug_ctx = {
        "llm_intent": {
            "snapshot": intent_snapshot,
            "raw": {
                key: value
                for key, value in llm_intent.model_dump(exclude_none=True).items()
                if key
                in {
                    "include_genres",
                    "exclude_genres",
                    "media_types",
                    "year_min",
                    "year_max",
                    "runtime_minutes_min",
                    "runtime_minutes_max",
                    "maturity_rating_max",
                    "streaming_providers",
                    "ann_description",
                }
            },
        },
        "legacy_intent": _serialize_intent_filters(legacy_filters),
        "final_intent": _serialize_intent_filters(intent),
        "query_profile": query_profile.to_debug_dict(),
        "prefilter": {
            "allowlist_count": len(allowlist) if allowlist is not None else None,
            "allowlist_preview": list(allowlist[:10]) if allowlist is not None else [],
            "boost_count": len(boost_ids),
            "boost_preview": list(boost_ids[:10]),
            "enforce_genres": enforce_genres,
            "fetch_limit": prefilter.fetch_limit,
            "preferred_services": sorted(preferred_services),
            "soften_provider_preference": soften_provider_preference,
        },
        "rewrite_text": (
            getattr(rewrite_result, "rewritten_text", None) if rewrite_result else None
        ),
        "semantic_facets": semantic_facets,
        "metrics": {
            "initial_candidates": len(ids),
            "post_filter_candidates": len(ordered),
            "neighbors_found": len(profile_meta.get("neighbors") or []),
            "preferred_media_types": preferred_media_types,
            "cold_start": cold_start,
            "constraint_prior_candidates": len(constraint_prior_ids),
            "candidate_limit": candidate_limit,
            "prefilter_fetch_limit": prefilter.fetch_limit,
            "constraint_prior_fetch_limit": constraint_prior_fetch_limit,
            "intent_relaxation_added": intent_relaxation_added,
            "intent_relaxation_steps": len(relaxed_intents),
            "semantic_specificity_score": semantic_specificity_score,
            "filter_expressiveness_gap": filter_expressiveness_gap,
            "needs_semantic_recall_boost": needs_semantic_recall_boost,
            "retrieval_pressure": retrieval_pressure,
            "pipeline_latency_ms": round(pipeline_ms, 2),
        },
    }
    if debug_mode:
        debug_ctx["source_counts"] = {
            "primary_stage": primary_retrieval_stage,
            "primary_candidates": len(primary_retrieval_ids),
            "ann_candidates": (
                len(primary_retrieval_ids)
                if primary_retrieval_stage in {"ann", "rewrite_ann"}
                else 0
            ),
            "collab_candidates": len(collab_results),
            "trending_candidates": len(trending_results),
            "constraint_prior_candidates": len(constraint_prior_ids),
        }
        debug_ctx["stage_counts"] = {
            "primary_retrieval": len(primary_retrieval_ids),
            "collab_retrieval": len(collab_results),
            "trending_retrieval": len(trending_results),
            "merged_before_constraint_prior": len(merged_ids_before_constraint),
            "constraint_prior": len(constraint_prior_ids),
            "boost_candidates": len(boost_ids),
            "merged_before_trim": len(ids_before_trim),
            "after_negative_filter": len(ids),
            "metadata_loaded": len(items_with_data),
            "strict_matches": strict_ordered_count,
            "provider_fallback_candidates": len(provider_fallback_pool),
            "relaxed_provider_fallback_candidates": sum(
                len(tier_candidates)
                for tier_candidates in relaxed_provider_fallback_by_tier
            ),
            "intent_filtered_out": skipped_intent,
            "post_filter_candidates": pre_mixer_count,
            "post_mixer_candidates": post_mixer_count,
            "post_business_rules_candidates": post_business_rule_count,
            "post_diversify_candidates": post_diversify_count,
            "final_candidates": len(reranked),
        }
        debug_ctx["candidates"] = {
            "primary_retrieval": _debug_id_snapshot(
                primary_retrieval_ids, items_with_data
            ),
            "collab_retrieval": _debug_scored_id_snapshot(
                collab_results,
                items_with_data,
                score_label="collab_score",
            ),
            "trending_retrieval": _debug_scored_id_snapshot(
                trending_results,
                items_with_data,
                score_label="trending_score",
            ),
            "constraint_prior": _debug_id_snapshot(
                constraint_prior_ids,
                items_with_data,
            ),
            "merged_before_trim": _debug_id_snapshot(ids_before_trim, items_with_data),
            "pre_mixer": pre_mixer_candidates or [],
            "post_mixer": post_mixer_candidates or [],
            "post_business_rules": post_business_rule_candidates or [],
            "post_diversify": post_diversify_candidates or [],
            "final": _debug_candidate_snapshot(reranked),
        }
    return ComputeResult(items=reranked, debug_context=debug_ctx)


@router.get("")
async def recommend(
    request: Request,
    params: RecommendParams = Depends(),
    cursor: str | None = Query(
        None,
        description="Opaque cursor returned by a previous request for pagination.",
    ),
    db: Session = Depends(get_db),
):
    canonical_id = canonical_profile_id(params.user_id, params.profile)
    cache_key = _get_cache_key(canonical_id, params)

    cached_result = _cache_get(cache_key)

    if cached_result:
        METRICS.counter("recommend.cache_hit").inc()
        reranked = cached_result["items"]
        logger.debug("Served recommendation from cache (key=%s)", cache_key)
    else:
        loop_scoped_key = (id(asyncio.get_running_loop()), cache_key)
        created_task = False
        with _INFLIGHT_TASKS_LOCK:
            task = _INFLIGHT_RECOMMEND_TASKS.get(loop_scoped_key)
            if task is None:
                METRICS.counter("recommend.cache_miss").inc()
                task = asyncio.create_task(
                    _compute_recommendations_async(request, params, db)
                )
                _INFLIGHT_RECOMMEND_TASKS[loop_scoped_key] = task
                created_task = True
            else:
                METRICS.counter("recommend.cache_wait").inc()

        try:
            result = await task
        finally:
            if created_task:
                with _INFLIGHT_TASKS_LOCK:
                    if _INFLIGHT_RECOMMEND_TASKS.get(loop_scoped_key) is task:
                        _INFLIGHT_RECOMMEND_TASKS.pop(loop_scoped_key, None)

        reranked = result.items
        if created_task and reranked:
            _cache_set(canonical_id, cache_key, reranked)

    start_index = _decode_cursor(cursor)
    if start_index < 0:
        raise HTTPException(status_code=400, detail="Invalid cursor")

    page = reranked[start_index : start_index + params.limit]
    response: List[Dict[str, Any]] = []
    for entry in page:
        cleaned = dict(entry)
        cleaned.pop("original_rank", None)
        cleaned.pop("vector", None)
        cleaned.pop("ann_rank", None)
        cleaned.pop("retrieval_score", None)
        cleaned.pop("source_scores", None)
        response.append(cleaned)

    next_cursor = None
    if start_index + params.limit < len(reranked):
        next_cursor = _encode_cursor(start_index + params.limit)

    payload: Dict[str, Any] = {"items": response}
    if next_cursor:
        payload["next_cursor"] = next_cursor

    request_id = request_id_ctx.get() or request.headers.get("x-request-id")
    _record_impressions(
        db,
        canonical_id=canonical_id,
        params=params,
        page=response,
        start_index=start_index,
        request_id=request_id,
    )
    return payload


@router.get("/debug")
async def debug_recommend(
    request: Request,
    params: RecommendParams = Depends(),
    db: Session = Depends(get_db),
):
    result = await _compute_recommendations_async(request, params, db)

    response: List[Dict[str, Any]] = []
    for entry in result.items[: params.limit]:
        cleaned = dict(entry)
        cleaned.pop("vector", None)  # just hide vector representation to save space
        response.append(cleaned)

    payload: Dict[str, Any] = {"items": response, "debug": result.debug_context}
    return payload


def _apply_mixer_scores(
    candidates: List[Dict[str, Any]],
    *,
    ann_weight_override: float | None = None,
    collab_weight_override: float | None = None,
    trending_weight_override: float | None = None,
    popularity_weight_override: float | None = None,
    vote_weight_override: float | None = None,
    novelty_weight_override: float | None = None,
) -> None:
    if not candidates:
        return

    max_popularity = max(
        (float(item.get("popularity") or 0.0) for item in candidates), default=0.0
    )
    max_vote_count = max(
        (float(item.get("vote_count") or 0.0) for item in candidates), default=0.0
    )

    base_ann_weight = (
        _HYBRID_ANN_WEIGHT if ann_weight_override is None else ann_weight_override
    )
    ann_weight = max(_HYBRID_MIN_ANN_WEIGHT, base_ann_weight)
    collab_weight = (
        _MIXER_COLLAB_WEIGHT
        if collab_weight_override is None
        else collab_weight_override
    )
    trending_weight = (
        _MIXER_TRENDING_WEIGHT
        if trending_weight_override is None
        else trending_weight_override
    )
    constraint_prior_weight = _MIXER_CONSTRAINT_PRIOR_WEIGHT
    pop_weight = (
        _HYBRID_POPULARITY_WEIGHT
        if popularity_weight_override is None
        else popularity_weight_override
    )
    vote_weight = (
        _HYBRID_VOTE_WEIGHT if vote_weight_override is None else vote_weight_override
    )
    novelty_weight = (
        _MIXER_NOVELTY_WEIGHT
        if novelty_weight_override is None
        else novelty_weight_override
    )

    for item in candidates:
        ann_rank = float(item.get("ann_rank", item.get("original_rank", 0)))
        ann_score = 1.0 / (1.0 + ann_rank)

        source_scores = item.get("source_scores") or {}
        collab_score = float(source_scores.get("collab") or 0.0)
        trending_source = float(source_scores.get("trending") or 0.0)
        constraint_prior_score = float(source_scores.get("constraint_prior") or 0.0)

        pop_score = (
            float(item.get("popularity") or 0.0) / max_popularity
            if max_popularity > 0
            else 0.0
        )
        vote_bonus = (
            float(item.get("vote_count") or 0.0) / max_vote_count
            if max_vote_count > 0
            else 0.0
        )

        novelty_score = 1.0 - min(1.0, pop_score)
        novelty_score = (novelty_score + (1.0 - vote_bonus)) / 2.0

        retrieval_score = (
            ann_weight * ann_score
            + collab_weight * collab_score
            + trending_weight * trending_source
            + constraint_prior_weight * constraint_prior_score
            + pop_weight * pop_score
            + vote_weight * vote_bonus
            + novelty_weight * novelty_score
        )
        item["retrieval_score"] = retrieval_score

    candidates.sort(
        key=lambda item: (
            -(item.get("retrieval_score") or 0.0),
            item.get("ann_rank", item.get("original_rank", 0)),
        )
    )
    for idx, item in enumerate(candidates):
        item["original_rank"] = idx


def _collaborative_candidates(
    db: Session,
    neighbors: List[Dict[str, Any]] | None,
    exclude_ids: List[int],
    limit: int,
    allowed_ids: List[int] | None,
) -> List[Tuple[int, float]]:
    if not neighbors:
        return []

    if allowed_ids is not None and len(allowed_ids) == 0:
        return []

    neighbor_weights: Dict[str, float] = {}
    for entry in neighbors:
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get("user_id")
        if not raw_id:
            continue
        uid = str(raw_id).strip()
        if not uid:
            continue
        weight = float(entry.get("weight", 0.0) or 0.0)
        if weight <= 0:
            continue
        neighbor_weights[uid] = weight

    if not neighbor_weights:
        return []

    exclude_set = set(exclude_ids or [])
    allowed_set = set(allowed_ids) if allowed_ids is not None else None
    history_limit = max(limit * _COLLAB_HISTORY_LIMIT_MULTIPLIER, 200)

    stmt = (
        select(
            UserHistory.item_id,
            UserHistory.user_id,
            UserHistory.weight,
            UserHistory.event_type,
            UserHistory.ts,
        )
        .where(UserHistory.user_id.in_(list(neighbor_weights)))
        .order_by(UserHistory.ts.desc())
        .limit(history_limit)
    )

    try:
        rows = db.execute(stmt).all()
    except SQLAlchemyError:
        logger.debug("Collaborative recall query failed.", exc_info=True)
        return []

    if not rows:
        return []

    aggregated: Dict[int, tuple[float, float]] = {}
    for item_id, uid, base_weight, event_type, ts in rows:
        if item_id is None or uid is None:
            continue
        uid_str = str(uid)
        neighbor_weight = neighbor_weights.get(uid_str)
        if neighbor_weight is None:
            continue
        if item_id in exclude_set:
            continue
        if allowed_set is not None and item_id not in allowed_set:
            continue
        normalized_event = (event_type or "").lower()
        if normalized_event in NEGATIVE_EVENT_TYPES:
            continue
        event_score = _event_weight(normalized_event, base_weight)
        if event_score <= 0:
            continue
        score = neighbor_weight * event_score
        ts_value = 0.0
        if ts is not None:
            ts_value = float(ts.timestamp()) if hasattr(ts, "timestamp") else 0.0
        if item_id in aggregated:
            current_score, current_ts = aggregated[item_id]
            aggregated[item_id] = (
                current_score + score,
                max(current_ts, ts_value),
            )
        else:
            aggregated[item_id] = (score, ts_value)

    if not aggregated:
        return []

    ranked = sorted(
        aggregated.items(),
        key=lambda kv: (-kv[1][0], -kv[1][1], kv[0]),
    )
    top = ranked[:limit]
    if not top:
        return []
    max_score = max(score for _, (score, _) in top)
    if max_score <= 0:
        max_score = 1.0
    scaled: List[Tuple[int, float]] = []
    for item_id, (score, _) in top:
        normalized = score / max_score if max_score > 0 else 0.0
        scaled.append((item_id, normalized))
    return scaled


def _trending_prior_candidates(
    db: Session,
    intent: IntentFilters | None,
    exclude_ids: List[int],
    limit: int,
    allowed_ids: List[int] | None,
    preferred_media_types: Sequence[str] | None = None,
) -> List[Tuple[int, float]]:
    if limit <= 0:
        return []

    stmt = select(
        Item.id,
        Item.media_type,
        Item.trending_rank,
        Item.popular_rank,
        Item.popularity,
    )

    filters = []

    if allowed_ids is not None:
        if not allowed_ids:
            return []
        filters.append(Item.id.in_(allowed_ids))

    if exclude_ids:
        filters.append(~Item.id.in_(exclude_ids))

    if intent and intent.media_types:
        filters.append(Item.media_type.in_(intent.media_types))

    if intent:
        genres = intent.effective_genres()
        if genres:
            genre_filters = [
                _genre_contains_clause(db, genre) for genre in genres if genre
            ]
            if genre_filters:
                filters.append(or_(*genre_filters))

    if filters:
        stmt = stmt.where(*filters)

    stmt = stmt.order_by(
        Item.trending_rank.asc().nullslast(),
        Item.popular_rank.asc().nullslast(),
        Item.popularity.desc().nullslast(),
        Item.id.asc(),
    ).limit(limit)

    rows = db.execute(stmt).all()
    if not rows:
        return []

    scored: List[Tuple[int, str | None, float, float]] = []
    for item_id, media_type, trending_rank, popular_rank, popularity in rows:
        if item_id is None:
            continue

        if not isinstance(trending_rank, (int, float)):
            trending_rank = None
        if not isinstance(popular_rank, (int, float)):
            popular_rank = None
        if not isinstance(popularity, (int, float)):
            popularity = None

        rank_score = 0.0
        if trending_rank and trending_rank > 0:
            rank_score += 1.0 / (1.0 + float(trending_rank))
        if popular_rank and popular_rank > 0:
            rank_score += 0.5 / (1.0 + float(popular_rank))
        pop_score = float(popularity) if popularity is not None else 0.0
        scored.append((int(item_id), media_type, rank_score, pop_score))

    scored.sort(
        key=lambda entry: (
            _preferred_media_rank(entry[1], preferred_media_types),
            -entry[2],
            -entry[3],
            entry[0],
        )
    )

    max_rank = max((entry[2] for entry in scored), default=0.0)
    max_pop = max((entry[3] for entry in scored), default=0.0)
    if max_rank <= 0:
        max_rank = 1.0
    if max_pop <= 0:
        max_pop = 1.0

    results: List[Tuple[int, float]] = []
    for item_id, _, rank_score, pop_score in scored:
        normalized_rank = rank_score / max_rank if max_rank > 0 else 0.0
        normalized_pop = pop_score / max_pop if max_pop > 0 else 0.0
        combined = 0.7 * normalized_rank + 0.3 * normalized_pop
        results.append((item_id, combined))
    return results


def _prefilter_allowed_ids(
    db: Session,
    intent: IntentFilters | None,
    limit: int,
    preferred_services: Set[str] | None = None,
) -> PrefilterDecision:
    if intent is None or not intent.has_filters():
        # Nothing to prefilter: leave the allowlist unset, but keep genre enforcement
        # enabled so downstream checks still respect catalog intent defaults.
        return PrefilterDecision(None, [], True, None)

    threshold = max(10, limit // 2)
    fetch_limit = _prefilter_fetch_limit(
        limit,
        intent,
        preferred_services=sorted(preferred_services or []),
    )
    strict_ids = _run_prefilter_query(
        db,
        intent,
        fetch_limit=fetch_limit,
        include_genres=True,
        required_services=preferred_services,
    )
    logger.debug(
        "Prefilter strict run | threshold=%d strict_count=%d media_types=%s genres=%s",
        threshold,
        len(strict_ids),
        intent.media_types,
        intent.effective_genres(),
    )
    boost_cap = max(5, min(limit, 15))

    if len(strict_ids) >= threshold:
        logger.debug("Prefilter returning strict allowlist.")
        return PrefilterDecision(strict_ids, strict_ids[:boost_cap], True, fetch_limit)

    relaxed_ids = _run_prefilter_query(
        db,
        intent,
        fetch_limit=fetch_limit,
        include_genres=False,
        required_services=preferred_services,
    )

    if len(relaxed_ids) >= threshold:
        # Genre matches were sparse, treat them as boosts but keep other filters strict.
        logger.debug(
            "Prefilter relaxed pass selected | relaxed_count=%d strict_count=%d",
            len(relaxed_ids),
            len(strict_ids),
        )
        return PrefilterDecision(relaxed_ids, strict_ids, False, fetch_limit)

    # Fall back to ANN-first retrieval. Preserve genre matches as soft boosts.
    logger.debug(
        "Prefilter falling back to ANN-first | relaxed_count=%d strict_count=%d",
        len(relaxed_ids),
        len(strict_ids),
    )
    return PrefilterDecision(None, strict_ids[:boost_cap], False, fetch_limit)


def _constraint_prior_candidates(
    db: Session,
    intent: IntentFilters | None,
    *,
    limit: int,
    allowlist: List[int] | None,
    enforce_genres: bool,
    rank_vector: np.ndarray | None = None,
) -> List[int]:
    """Return query-aware strict matches to backfill constrained query recall."""
    if intent is None or not intent.has_filters() or limit <= 0:
        return []

    fetch_limit = _constraint_prior_fetch_limit(
        limit,
        intent,
        rank_vector=rank_vector,
    )
    stmt = select(Item, ItemEmbedding.vector).join(
        ItemEmbedding, ItemEmbedding.item_id == Item.id
    )

    if allowlist is not None:
        if not allowlist:
            return []
        stmt = stmt.where(Item.id.in_(allowlist))
    else:
        media_types = intent.media_types or []
        if media_types:
            stmt = stmt.where(Item.media_type.in_(media_types))
        if intent.year_min is not None:
            stmt = stmt.where(
                Item.release_year.is_not(None), Item.release_year >= intent.year_min
            )
        if intent.year_max is not None:
            stmt = stmt.where(
                Item.release_year.is_not(None), Item.release_year <= intent.year_max
            )
        if enforce_genres:
            genres = intent.effective_genres()
            if genres:
                genre_filters = [
                    _genre_contains_clause(db, genre) for genre in genres if genre
                ]
                if genre_filters:
                    stmt = stmt.where(or_(*genre_filters))

    if rank_vector is not None:
        stmt = stmt.order_by(
            ItemEmbedding.vector.cosine_distance(list(map(float, rank_vector))).asc(),
            Item.popular_rank.asc().nullslast(),
            Item.trending_rank.asc().nullslast(),
            Item.popularity.desc().nullslast(),
            Item.vote_count.desc().nullslast(),
            Item.id.asc(),
        )
    else:
        stmt = stmt.order_by(
            Item.popular_rank.asc().nullslast(),
            Item.trending_rank.asc().nullslast(),
            Item.popularity.desc().nullslast(),
            Item.vote_count.desc().nullslast(),
            Item.id.asc(),
        )
    stmt = stmt.limit(fetch_limit)

    rows = db.execute(stmt).all()
    matches: List[int] = []
    seen: set[int] = set()
    sorted_rows = sorted(
        rows,
        key=lambda row: _constraint_sort_key(
            row[0],
            intent=intent,
            item_vector=row[1],
            rank_vector=rank_vector,
        ),
    )
    for row in sorted_rows:
        item = row[0]
        if not item_matches_intent(item, intent, enforce_genres=enforce_genres):
            continue
        if item.id in seen:
            continue
        seen.add(item.id)
        matches.append(int(item.id))
        if len(matches) >= limit:
            break
    return matches


def _run_prefilter_query(
    db: Session,
    intent: IntentFilters,
    *,
    fetch_limit: int,
    include_genres: bool,
    required_services: Set[str] | None = None,
) -> List[int]:
    availability_exists = None
    if required_services:
        availability_exists = (
            select(Availability.item_id)
            .where(
                Availability.item_id == Item.id,
                Availability.country == COUNTRY_DEFAULT,
                Availability.service.in_(required_services),
            )
            .exists()
        )

    stmt = select(
        Item.id,
        Item.trending_rank,
        Item.popular_rank,
        Item.popularity,
        Item.vote_count,
        Item.release_year,
    )

    if availability_exists is not None:
        stmt = stmt.where(availability_exists)

    media_types = intent.media_types or []
    if media_types:
        stmt = stmt.where(Item.media_type.in_(media_types))

    if include_genres:
        genres = intent.effective_genres()
        if genres:
            filters = [_genre_contains_clause(db, genre) for genre in genres if genre]
            if filters:
                stmt = stmt.where(or_(*filters))

    if intent.year_min is not None:
        stmt = stmt.where(
            Item.release_year.is_not(None), Item.release_year >= intent.year_min
        )
    if intent.year_max is not None:
        stmt = stmt.where(
            Item.release_year.is_not(None), Item.release_year <= intent.year_max
        )

    stmt = stmt.order_by(
        Item.trending_rank.asc().nullslast(),
        Item.popular_rank.asc().nullslast(),
        Item.popularity.desc().nullslast(),
        Item.vote_count.desc().nullslast(),
        Item.release_year.desc().nullslast(),
        Item.id.asc(),
    )

    rows = db.execute(stmt.limit(fetch_limit)).all()
    return _ordered_unique([row[0] for row in rows if row and row[0] is not None])


def _ordered_unique(values: List[int]) -> List[int]:
    seen: set[int] = set()
    ordered: List[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _genre_contains_clause(db: Session, genre: str):
    bind = getattr(db, "bind", None)
    if bind is None:
        get_bind = getattr(db, "get_bind", None)
        if callable(get_bind):
            try:
                bind = get_bind()
            except Exception:
                bind = None

    dialect_name: Optional[str] = None
    if bind is not None:
        dialect = getattr(bind, "dialect", None)
        if dialect is not None:
            dialect_name = getattr(dialect, "name", None)

    if dialect_name == "postgresql":
        return cast(Item.genres, JSONB).contains([{"name": genre}])

    # Fallback for other dialects used in tests (e.g., SQLite memory stubs)
    return Item.genres.contains([{"name": genre}])


def _cold_start_candidates(
    db: Session,
    intent: IntentFilters,
    limit: int,
    allowlist: List[int] | None,
    preferred_media_types: Sequence[str] | None = None,
) -> List[int]:
    stmt = select(Item.id, Item.media_type).join(
        ItemEmbedding, ItemEmbedding.item_id == Item.id
    )

    if allowlist is not None:
        if not allowlist:
            return []
        stmt = stmt.where(Item.id.in_(allowlist))
    else:
        if intent.media_types:
            stmt = stmt.where(Item.media_type.in_(intent.media_types))
        if intent.year_min is not None:
            stmt = stmt.where(
                Item.release_year.is_not(None), Item.release_year >= intent.year_min
            )
        if intent.year_max is not None:
            stmt = stmt.where(
                Item.release_year.is_not(None), Item.release_year <= intent.year_max
            )
        genres = intent.effective_genres()
        if genres:
            genre_filters = [
                _genre_contains_clause(db, genre) for genre in genres if genre
            ]
            if genre_filters:
                stmt = stmt.where(or_(*genre_filters))

    stmt = stmt.order_by(
        Item.popular_rank.asc().nullslast(),
        Item.trending_rank.asc().nullslast(),
        Item.popularity.desc().nullslast(),
        Item.id.asc(),
    ).limit(limit)

    rows = db.execute(stmt).all()
    rows = sorted(
        rows,
        key=lambda row: (
            _preferred_media_rank(row[1], preferred_media_types),
            row[0],
        ),
    )
    seen: set[int] = set()
    ordered: List[int] = []
    for value, _ in rows:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _encode_cursor(rank: int) -> str:
    payload = json.dumps({"rank": rank}).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("utf-8").rstrip("=")


def _decode_cursor(cursor: str | None) -> int:
    if not cursor:
        return 0
    padding = "=" * (-len(cursor) % 4)
    try:
        decoded = base64.urlsafe_b64decode(cursor + padding)
        payload = json.loads(decoded.decode("utf-8"))
        rank = int(payload["rank"])
        if rank < 0:
            raise ValueError
        return rank
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=400, detail="Invalid cursor") from exc


def _empty_response() -> Dict[str, Any]:
    return {"items": []}
