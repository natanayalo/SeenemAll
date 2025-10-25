from __future__ import annotations

import json
import logging
import os
import re
import atexit
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
import hashlib
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from threading import Lock

import httpx
import numpy as np
from cachetools import TTLCache

from api.core.legacy_intent_parser import IntentFilters
from api.core.prompt_eval import load_prompt_template
from api.core.embeddings import encode_texts

logger = logging.getLogger(__name__)
# Ensure INFO-level reranker messages surface unless overridden globally.
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)
if not logger.handlers:
    _stream: logging.Handler = logging.StreamHandler()
    _stream.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(_stream)
    logger.propagate = False

_DEFAULT_MAX_ITEMS_FOR_LLM = 25
_DEFAULT_TIMEOUT_SECONDS = 12.0

_DEFAULT_EXPLANATION_TEMPLATES = {
    "narrative": {
        "genre_match": "Dialed into your {genres} vibe.",
        "media_type": "It's a {media_type} pick like you asked for.",
        "runtime_match": "Runs about {runtime} minutes, matching your runtime window.",
        "query_fallback": 'Aligns with "{query}".',
        "fallback": "Popular with viewers who share your taste.",
    },
    "heuristic": {
        "trending": "Trending now (#{rank}).",
        "popular": "Top popular pick (#{rank}).",
        "votes": "Rated {vote_average:.1f}/10 by {vote_count} fans.",
        "recent": "Fresh release from {year}.",
        "fallback": "Popular with viewers who share your taste.",
    },
}


def _float_from_env(name: str, default: float) -> float:
    """Read a float override from the environment, falling back to *default*."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_from_env(name: str, default: int) -> int:
    """Read an integer override from the environment, falling back to *default*."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _optional_int_from_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


_MMR_LAMBDA_DEFAULT = _float_from_env("MMR_LAMBDA", 0.7)
_MMR_POOL_MULTIPLIER = max(1, _int_from_env("MMR_POOL_MULTIPLIER", 3))
_MMR_POOL_MIN = max(0, _int_from_env("MMR_POOL_MIN", 25))
_MMR_POOL_MAX = _optional_int_from_env("MMR_POOL_MAX")
if _MMR_POOL_MAX is not None and _MMR_POOL_MAX <= 0:
    _MMR_POOL_MAX = None

_HEURISTIC_POPULARITY_WEIGHT = _float_from_env("HEURISTIC_POPULARITY_WEIGHT", 0.3)
_HEURISTIC_VOTE_QUALITY_WEIGHT = _float_from_env("HEURISTIC_VOTE_QUALITY_WEIGHT", 0.2)
_HEURISTIC_VOTE_VOLUME_WEIGHT = _float_from_env("HEURISTIC_VOTE_VOLUME_WEIGHT", 0.1)
_HEURISTIC_TRENDING_WEIGHT = _float_from_env("HEURISTIC_TRENDING_WEIGHT", 0.25)
_HEURISTIC_POPULAR_WEIGHT = _float_from_env("HEURISTIC_POPULAR_WEIGHT", 0.15)
_HEURISTIC_RECENT_WEIGHT = _float_from_env("HEURISTIC_RECENT_WEIGHT", 0.2)
_HEURISTIC_POSITION_BASE = _float_from_env("HEURISTIC_POSITION_BASE", 0.05)
_HEURISTIC_POSITION_DECAY = _float_from_env("HEURISTIC_POSITION_DECAY", 0.002)
_HEURISTIC_MIN_VOTE_COUNT = max(0, _int_from_env("HEURISTIC_MIN_VOTE_COUNT", 50))
_HEURISTIC_LOW_VOTE_PENALTY = min(
    0.9, max(0.0, _float_from_env("HEURISTIC_LOW_VOTE_PENALTY", 0.35))
)
_HEURISTIC_MIN_POPULARITY = max(0.0, _float_from_env("HEURISTIC_MIN_POPULARITY", 20.0))
_HEURISTIC_LOW_POPULARITY_PENALTY = min(
    0.9, max(0.0, _float_from_env("HEURISTIC_LOW_POPULARITY_PENALTY", 0.25))
)
_HEURISTIC_MIN_SIGNAL_MULTIPLIER = min(
    1.0, max(0.0, _float_from_env("HEURISTIC_MIN_SIGNAL_MULTIPLIER", 0.0))
)

_SMALL_RERANK_INPUT_WINDOW = max(1, _int_from_env("SMALL_RERANK_INPUT_WINDOW", 40))
_SMALL_RERANK_OUTPUT_LIMIT = max(1, _int_from_env("SMALL_RERANK_OUTPUT_LIMIT", 12))
_SMALL_RERANK_TIMEOUT_SECONDS = max(0.1, _float_from_env("SMALL_RERANK_TIMEOUT", 1.5))
_SMALL_RERANK_CACHE_TTL_SECONDS = max(5, _int_from_env("SMALL_RERANK_CACHE_TTL", 120))
_SMALL_RERANK_CACHE_MAXSIZE = max(16, _int_from_env("SMALL_RERANK_CACHE_MAXSIZE", 256))
_SMALL_RERANK_RANK_WEIGHT = min(
    0.3, max(0.0, _float_from_env("SMALL_RERANK_RANK_WEIGHT", 0.05))
)

_SMALL_RERANK_CACHE: TTLCache[str, List[Tuple[int, float]]] = TTLCache(
    maxsize=_SMALL_RERANK_CACHE_MAXSIZE, ttl=_SMALL_RERANK_CACHE_TTL_SECONDS
)
_SMALL_RERANK_CACHE_LOCK = Lock()
_SMALL_RERANK_EXECUTOR = ThreadPoolExecutor(max_workers=1)
# Ensure we clean the worker thread across reloads/tests.
atexit.register(_SMALL_RERANK_EXECUTOR.shutdown, wait=False)


def _get_templates_path() -> str:
    env_path = os.getenv("EXPLANATION_TEMPLATES_PATH")
    if env_path:
        return env_path
    return os.path.join(os.getcwd(), "config", "explanation_templates.json")


@lru_cache(maxsize=32)
def _load_explanation_templates_cached(
    path: str, mtime: float | None
) -> Dict[str, Any]:
    if path == "__defaults__":
        return _DEFAULT_EXPLANATION_TEMPLATES

    try:
        with open(path, "r", encoding="utf-8") as handle:
            templates = json.load(handle)
        if not isinstance(templates, dict):
            raise ValueError("Explanation template file must contain a JSON object.")
        return templates
    except (OSError, json.JSONDecodeError, ValueError):
        logger.warning(
            "Failed to parse explanation templates at %s; falling back to defaults.",
            path,
        )
        return _DEFAULT_EXPLANATION_TEMPLATES


def _load_explanation_templates() -> Dict[str, Any]:
    path = _get_templates_path()
    try:
        mtime = os.path.getmtime(path)
    except FileNotFoundError:
        return _load_explanation_templates_cached("__defaults__", None)
    return _load_explanation_templates_cached(path, mtime)


def _get_prompt_config():
    config = load_prompt_template("reranker")
    return config


@lru_cache(maxsize=1)
def get_system_prompt() -> str:
    """Get the system prompt from the template config."""
    return _get_prompt_config()["system_prompt"]


_SYSTEM_PROMPT = get_system_prompt()


@dataclass(frozen=True)
class RerankerSettings:
    provider: str
    api_key: str | None
    model: str
    endpoint: str
    enabled: bool
    timeout: float


@dataclass(frozen=True)
class LLMDecision:
    item_id: int
    score: float | None
    explanation: str | None


class RerankerError(RuntimeError):
    pass


def rerank_with_explanations(
    items: Sequence[Dict[str, Any]],
    intent: IntentFilters | None,
    query: str | None = None,
    user: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    if not items:
        return []

    base = _with_default_explanations(items, intent, query)
    settings = _get_settings()
    if not settings.enabled:
        logger.warning(
            "Reranker(%s) disabled (missing credentials or RERANK_ENABLED=0); returning ANN order.",
            settings.provider,
        )
        return base

    try:
        if settings.provider == "small":
            window = min(len(base), _SMALL_RERANK_INPUT_WINDOW)
        else:
            window = min(len(base), _DEFAULT_MAX_ITEMS_FOR_LLM)
        llm_items = base[:window]
        if not llm_items:
            return base
        logger.info(
            "Reranker(%s) evaluating %d candidates.",
            settings.provider,
            len(llm_items),
        )
        start_time = time.perf_counter()
        decisions = _call_reranker(settings, llm_items, intent, query, user)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        if not decisions:
            logger.info(
                "Reranker(%s) returned no decisions; using baseline ordering. latency=%.1fms",
                settings.provider,
                latency_ms,
            )
            return base
        logger.info(
            "Reranker(%s) produced %d decisions in %.1fms.",
            settings.provider,
            len(decisions),
            latency_ms,
        )
        return _merge_with_base(base, decisions)
    except Exception:  # pragma: no cover - defensive guardrail
        logger.exception(
            "Reranker(%s) failed; falling back to ANN order.", settings.provider
        )
        return base


def diversify_with_mmr(
    items: List[Dict[str, Any]],
    lambda_param: float | None = None,
    limit: int | None = None,
    pool_size: int | None = None,
) -> List[Dict[str, Any]]:
    if not items:
        return []

    limit = limit or len(items)
    limit = max(1, min(limit, len(items)))

    lambda_value = _resolve_lambda_value(lambda_param)

    pool = _resolve_mmr_pool_size(limit, len(items), pool_size)
    head = items[:pool]

    ranked = _mmr_rank_subset(head, lambda_value, limit)
    seen_keys = {_item_identity(item) for item in ranked}

    if len(ranked) < limit:
        for item in items:
            if len(ranked) >= limit:
                break
            ident = _item_identity(item)
            if ident in seen_keys:
                continue
            ranked.append(item)
            seen_keys.add(ident)

    return ranked[:limit]


def _resolve_mmr_pool_size(
    requested_limit: int, total_items: int, override: int | None
) -> int:
    if total_items <= 0:
        return 0
    if override is not None:
        return max(1, min(total_items, override))

    baseline = max(1, min(requested_limit, total_items))
    pool = baseline
    if _MMR_POOL_MULTIPLIER > 1:
        pool = max(pool, baseline * _MMR_POOL_MULTIPLIER)
    if _MMR_POOL_MIN > 0:
        pool = max(pool, _MMR_POOL_MIN)
    if _MMR_POOL_MAX is not None:
        pool = min(pool, _MMR_POOL_MAX)
    pool = min(pool, total_items)
    return max(1, pool)


def _resolve_lambda_value(lambda_param: float | None) -> float:
    try:
        value = (
            float(lambda_param)
            if lambda_param is not None
            else float(_MMR_LAMBDA_DEFAULT)
        )
    except (TypeError, ValueError):
        value = float(_MMR_LAMBDA_DEFAULT)
    return min(1.0, max(0.0, value))


def _mmr_rank_subset(
    subset: Sequence[Dict[str, Any]],
    lambda_value: float,
    limit: int,
) -> List[Dict[str, Any]]:
    if not subset:
        return []

    ranked: List[Dict[str, Any]] = []
    ranked_vectors: List[np.ndarray] = []

    first_item = subset[0]
    ranked.append(first_item)
    first_vector = _normalized_vector(first_item.get("vector"))
    if first_vector is not None:
        ranked_vectors.append(first_vector)

    if len(ranked) >= limit:
        return ranked[:limit]

    prepared: List[Dict[str, Any]] = []
    for idx, item in enumerate(subset):
        normalized = _normalized_vector(item.get("vector"))
        if normalized is None:
            continue
        original_rank = int(item.get("original_rank", idx))
        relevance = 1.0 / (1.0 + float(original_rank))
        prepared.append(
            {
                "item": item,
                "vector": normalized,
                "original_rank": original_rank,
                "relevance": relevance,
            }
        )

    first_identity = _item_identity(first_item)
    prepared = [
        entry for entry in prepared if _item_identity(entry["item"]) != first_identity
    ]

    if not prepared:
        return ranked

    prepared.sort(key=lambda entry: entry["original_rank"])

    while prepared and len(ranked) < limit:
        best_idx = None
        best_score = -np.inf

        for idx, candidate in enumerate(prepared):
            if ranked_vectors:
                similarity = max(
                    float(np.dot(candidate["vector"], selected))
                    for selected in ranked_vectors
                )
            else:
                similarity = 0.0
            mmr_score = (
                lambda_value * candidate["relevance"]
                - (1.0 - lambda_value) * similarity
            )
            if mmr_score > best_score:
                best_idx = idx
                best_score = mmr_score

        if best_idx is None:
            break

        chosen = prepared.pop(best_idx)
        ranked.append(chosen["item"])
        ranked_vectors.append(chosen["vector"])

    return ranked


def cosine_similarity(v1, v2):
    arr1 = np.asarray(v1, dtype=np.float32)
    arr2 = np.asarray(v2, dtype=np.float32)
    denom = float(np.linalg.norm(arr1) * np.linalg.norm(arr2))
    if denom == 0.0 or not np.isfinite(denom):
        return 0.0
    return float(np.dot(arr1, arr2) / denom)


def _normalized_vector(vector: Any) -> np.ndarray | None:
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim == 0 or arr.size == 0:
        return None
    norm = float(np.linalg.norm(arr))
    if norm == 0.0 or not np.isfinite(norm):
        return None
    return arr / norm


def _item_identity(item: Dict[str, Any]) -> tuple[Any, int]:
    return (item.get("id"), id(item))


@lru_cache(maxsize=1)
def _get_settings() -> RerankerSettings:
    raw_provider = os.getenv("RERANK_PROVIDER", "openai").strip().lower()
    supported = {"openai", "gemini", "small"}
    if raw_provider not in supported:
        logger.warning(
            "Unsupported RERANK_PROVIDER '%s'; falling back to 'openai'.", raw_provider
        )
        provider = "openai"
    else:
        provider = raw_provider

    api_key = None
    if provider in {"openai", "gemini"}:
        api_key = os.getenv("RERANK_API_KEY") or os.getenv("OPENAI_API_KEY")

    if provider == "gemini":
        default_model = "gemini-2.0-flash-lite"
        default_endpoint = "https://generativelanguage.googleapis.com/v1beta/models"
    elif provider == "small":
        default_model = os.getenv("SMALL_RERANK_MODEL", "all-MiniLM-L6-v2")
        default_endpoint = "local://small-rerank"
    else:
        default_model = "gpt-4o-mini"
        default_endpoint = "https://api.openai.com/v1/chat/completions"

    model = os.getenv("RERANK_MODEL", default_model)
    endpoint = os.getenv("RERANK_ENDPOINT", default_endpoint)
    enabled_value = os.getenv("RERANK_ENABLED", "1").strip().lower()
    if provider == "small":
        enabled = enabled_value not in {"0", "false", "no"}
    else:
        enabled = bool(api_key) and enabled_value not in {"0", "false", "no"}
    timeout = (
        _SMALL_RERANK_TIMEOUT_SECONDS
        if provider == "small"
        else _DEFAULT_TIMEOUT_SECONDS
    )
    raw_timeout = os.getenv("RERANK_TIMEOUT")
    if raw_timeout:
        try:
            timeout = max(1.0, float(raw_timeout))
        except ValueError:
            logger.warning(
                "Invalid RERANK_TIMEOUT value '%s'; using default.", raw_timeout
            )
    return RerankerSettings(
        provider=provider,
        api_key=api_key,
        model=model,
        endpoint=endpoint,
        enabled=enabled,
        timeout=timeout,
    )


def _current_year() -> int:
    return datetime.now(UTC).year


def _with_default_explanations(
    items: Sequence[Dict[str, Any]],
    intent: IntentFilters | None,
    query: str | None,
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    intent_genres: List[str] = intent.effective_genres() if intent else []
    current_year = _current_year()
    for idx, item in enumerate(items):
        copy_item = dict(item)
        copy_item.setdefault("original_rank", idx)
        score, heuristics = _heuristic_score(copy_item, idx, current_year)
        copy_item["score"] = score
        if (
            _HEURISTIC_MIN_SIGNAL_MULTIPLIER > 0.0
            and heuristics.get("signal_multiplier", 1.0)
            < _HEURISTIC_MIN_SIGNAL_MULTIPLIER
        ):
            logger.debug(
                "Dropping low-signal item | id=%s title=%s multiplier=%.3f threshold=%.3f",
                copy_item.get("id"),
                copy_item.get("title") or copy_item.get("name"),
                heuristics.get("signal_multiplier"),
                _HEURISTIC_MIN_SIGNAL_MULTIPLIER,
            )
            continue
        narrative = _default_explanation(copy_item, intent, query, intent_genres)
        heuristic_summary = _heuristic_summary(copy_item, heuristics)
        explanation_parts = [narrative, heuristic_summary]
        copy_item["explanation"] = " ".join(
            part for part in explanation_parts if part
        ).strip()
        enriched.append(copy_item)
    return enriched


def _low_signal_multiplier(popularity: float, vote_count: float) -> float:
    multiplier = 1.0

    vote_floor = max(1, _HEURISTIC_MIN_VOTE_COUNT)
    raw_votes = max(0.0, vote_count)
    if vote_floor > 0 and raw_votes < vote_floor:
        deficit = vote_floor - raw_votes
        fraction = min(1.0, deficit / vote_floor)
        multiplier *= max(0.0, 1.0 - _HEURISTIC_LOW_VOTE_PENALTY * fraction)

    popularity_floor = max(1.0, _HEURISTIC_MIN_POPULARITY)
    raw_popularity = max(0.0, popularity)
    if raw_popularity < popularity_floor:
        deficit = popularity_floor - raw_popularity
        fraction = min(1.0, deficit / popularity_floor)
        multiplier *= max(0.0, 1.0 - _HEURISTIC_LOW_POPULARITY_PENALTY * fraction)

    return multiplier


def _heuristic_score(
    item: Dict[str, Any],
    position: int,
    current_year: int,
) -> Tuple[float, Dict[str, Any]]:
    retrieval = float(item.get("retrieval_score") or 0.0)
    popularity = float(item.get("popularity") or 0.0)
    vote_average = float(item.get("vote_average") or 0.0)
    vote_count = float(item.get("vote_count") or 0.0)
    trending_rank = item.get("trending_rank")
    popular_rank = item.get("popular_rank")
    release_year = item.get("release_year")

    popularity_norm = min(popularity / 100.0, 1.0)
    vote_quality = max((vote_average - 6.5) / 3.5, 0.0)
    vote_quality = min(vote_quality, 1.0)
    vote_volume = min(vote_count / 5000.0, 1.0)

    trending_bonus = 0.0
    if isinstance(trending_rank, (int, float)) and trending_rank > 0:
        trending_bonus = max(0.0, 1.0 - min(trending_rank - 1, 50) / 50.0)

    popular_bonus = 0.0
    if isinstance(popular_rank, (int, float)) and popular_rank > 0:
        popular_bonus = max(0.0, 1.0 - min(popular_rank - 1, 100) / 100.0)

    recent_bonus = 0.0
    if isinstance(release_year, int):
        age = max(0, current_year - release_year)
        if age <= 2:
            recent_bonus = 1.0
        elif age <= 5:
            recent_bonus = 0.5

    base_score = retrieval
    score = base_score
    score += _HEURISTIC_POPULARITY_WEIGHT * popularity_norm
    score += _HEURISTIC_VOTE_QUALITY_WEIGHT * vote_quality
    score += _HEURISTIC_VOTE_VOLUME_WEIGHT * vote_volume
    score += _HEURISTIC_TRENDING_WEIGHT * trending_bonus
    score += _HEURISTIC_POPULAR_WEIGHT * popular_bonus
    score += _HEURISTIC_RECENT_WEIGHT * recent_bonus
    score += max(0.0, _HEURISTIC_POSITION_BASE - position * _HEURISTIC_POSITION_DECAY)

    baseline_score = score
    signal_multiplier = _low_signal_multiplier(popularity, vote_count)
    score *= signal_multiplier
    if signal_multiplier < 0.999:
        logger.debug(
            "Low-signal penalty applied | id=%s title=%s votes=%s popularity=%s multiplier=%.3f baseline=%.4f final=%.4f",
            item.get("id"),
            item.get("title") or item.get("name"),
            vote_count,
            popularity,
            signal_multiplier,
            baseline_score,
            score,
        )

    heuristics = {
        "popularity_norm": popularity_norm,
        "vote_quality": vote_quality,
        "vote_volume": vote_volume,
        "trending_rank": trending_rank,
        "popular_rank": popular_rank,
        "recent_bonus": recent_bonus,
        "vote_average": vote_average,
        "vote_count": int(vote_count) if vote_count else 0,
        "release_year": release_year,
        "baseline_score": round(baseline_score, 4),
        "signal_multiplier": round(signal_multiplier, 4),
    }
    return round(score, 4), heuristics


def _safe_template(template: str | None, **kwargs: Any) -> str:
    if not template or not isinstance(template, str):
        return ""
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError, ValueError):
        return template or ""


def _heuristic_summary(item: Dict[str, Any], heuristics: Dict[str, Any]) -> str:
    templates = _load_explanation_templates()
    heuristic_templates = templates.get("heuristic", {})
    reasons: List[str] = []

    trending_rank = heuristics.get("trending_rank")
    if (
        isinstance(trending_rank, (int, float))
        and trending_rank > 0
        and trending_rank <= 20
    ):
        template = heuristic_templates.get("trending") or "Trending now (#{rank})."
        reasons.append(_safe_template(template, rank=int(trending_rank)))

    if heuristics.get("recent_bonus", 0.0) >= 0.5:
        candidate_year = heuristics.get("release_year")
        if isinstance(candidate_year, int) and candidate_year > 1900:
            template = heuristic_templates.get("recent") or "Fresh release from {year}."
            release_reason = _safe_template(template, year=candidate_year)
            if release_reason:
                reasons.append(release_reason)

    popular_rank = heuristics.get("popular_rank")
    if (
        isinstance(popular_rank, (int, float))
        and popular_rank > 0
        and popular_rank <= 20
    ):
        template = heuristic_templates.get("popular") or "Top popular pick (#{rank})."
        reasons.append(_safe_template(template, rank=int(popular_rank)))

    vote_average = heuristics.get("vote_average") or 0.0
    vote_count = heuristics.get("vote_count") or 0
    if vote_average and vote_average >= 7.5 and vote_count >= 500:
        template = (
            heuristic_templates.get("votes")
            or "Rated {vote_average:.1f}/10 by {vote_count} fans."
        )
        formatted_votes = f"{vote_count:,}"
        reasons.append(
            _safe_template(
                template,
                vote_average=vote_average,
                vote_count=formatted_votes,
                votes=formatted_votes,
            )
        )

    if not reasons and heuristics.get("popularity_norm", 0.0) >= 0.7:
        fallback = (
            heuristic_templates.get("fallback")
            or _DEFAULT_EXPLANATION_TEMPLATES["heuristic"]["fallback"]
        )
        reasons.append(fallback)

    return " ".join(reasons[:2])


def _merge_with_base(
    base: Sequence[Dict[str, Any]],
    decisions: Iterable[LLMDecision],
) -> List[Dict[str, Any]]:
    by_id: Dict[int, Dict[str, Any]] = {int(item["id"]): dict(item) for item in base}
    seen: set[int] = set()
    merged: List[Dict[str, Any]] = []

    for decision in decisions:
        if decision.item_id not in by_id or decision.item_id in seen:
            continue
        row = dict(by_id[decision.item_id])
        if decision.score is not None:
            row["score"] = decision.score
        if decision.explanation:
            row["explanation"] = decision.explanation
        merged.append(row)
        seen.add(decision.item_id)

    if not merged:
        return list(by_id.values())

    merged_ids = {m["id"] for m in merged}
    for item in base:
        if item["id"] not in merged_ids:
            merged.append(dict(item))
    return merged


def _small_rerank_cache_key(
    query_text: str,
    items: Sequence[Dict[str, Any]],
    user: Dict[str, Any] | None,
) -> str:
    user_id = None
    if isinstance(user, dict):
        user_id = user.get("user_id") or user.get("id")
    parts: List[str] = [
        "v1",
        str(user_id or ""),
        query_text.strip().lower(),
        str(_SMALL_RERANK_OUTPUT_LIMIT),
    ]
    for item in items:
        ident = item.get("id")
        if ident is None:
            continue
        parts.append(
            "::".join(
                [
                    str(ident),
                    str(item.get("original_rank")),
                    str(item.get("retrieval_score")),
                    str(item.get("score")),
                ]
            )
        )
    payload = "||".join(parts)
    return hashlib.sha256(payload.encode("utf-8", "ignore")).hexdigest()


def _build_small_rerank_query(intent: IntentFilters | None, query: str | None) -> str:
    segments: List[str] = []
    if query:
        cleaned = query.strip()
        if cleaned:
            segments.append(cleaned)

    if intent:
        try:
            genres = intent.effective_genres()
        except AttributeError:
            genres = list(getattr(intent, "genres", []) or [])
        if genres:
            segments.append("Genres: " + ", ".join(genres[:6]))

        media_types = getattr(intent, "media_types", []) or []
        if media_types:
            segments.append("Media: " + ", ".join(str(mt) for mt in media_types[:4]))

        min_runtime = getattr(intent, "min_runtime", None)
        max_runtime = getattr(intent, "max_runtime", None)
        runtime_bits: List[str] = []
        if min_runtime is not None:
            runtime_bits.append(f">={min_runtime}m")
        if max_runtime is not None:
            runtime_bits.append(f"<={max_runtime}m")
        if runtime_bits:
            segments.append("Runtime " + " ".join(runtime_bits))

        maturity = getattr(intent, "maturity_rating_max", None)
        if maturity:
            segments.append(f"Rating ≤ {maturity}")

    return " | ".join(seg for seg in segments if seg).strip()


def _build_small_rerank_documents(items: Sequence[Dict[str, Any]]) -> List[str]:
    documents: List[str] = []
    for item in items:
        title = str(item.get("title") or item.get("name") or "").strip()
        overview = str(item.get("overview") or "").strip()
        genres_field = item.get("genres") or []
        genres: List[str] = []
        for entry in genres_field:
            if isinstance(entry, dict):
                name = entry.get("name")
            else:
                name = getattr(entry, "name", None)
            if name:
                genres.append(str(name))
        media_type = item.get("media_type")
        runtime = item.get("runtime")
        release_year = item.get("release_year")

        parts: List[str] = []
        if title:
            parts.append(title)
        if genres:
            parts.append("Genres: " + ", ".join(genres[:6]))
        if media_type:
            parts.append(f"Media: {media_type}")
        if runtime:
            parts.append(f"Runtime: {runtime} minutes")
        if release_year:
            parts.append(f"Released: {release_year}")
        if overview:
            parts.append(overview)

        doc = ". ".join(segment for segment in parts if segment).strip()
        if not doc:
            doc = f"Item {item.get('id')}"
        documents.append(doc)
    return documents


def _execute_small_rerank(
    query_text: str,
    items: Sequence[Dict[str, Any]],
) -> List[Tuple[int, float]]:
    limit = min(_SMALL_RERANK_OUTPUT_LIMIT, len(items))
    if limit <= 0:
        return []

    documents = _build_small_rerank_documents(items)
    if not documents:
        return []

    text_inputs = [query_text] + documents
    embeddings = encode_texts(text_inputs)
    if (
        not isinstance(embeddings, np.ndarray)
        or embeddings.shape[0] != len(text_inputs)
        or embeddings.ndim != 2
    ):
        return []

    query_vec = embeddings[0]
    if query_vec.size == 0 or float(np.linalg.norm(query_vec)) == 0.0:
        return []

    item_vecs = embeddings[1:]
    similarities = np.matmul(item_vecs, query_vec)
    if similarities.shape[0] != len(items):
        return []

    scored: List[Tuple[int, float, int]] = []
    for idx, (item, similarity) in enumerate(zip(items, similarities)):
        ident = item.get("id")
        if ident is None:
            continue
        base_rank = int(item.get("original_rank", idx))
        # Normalize cosine similarity from [-1, 1] -> [0, 1]
        normalized = float((float(similarity) + 1.0) / 2.0)
        rank_bias = _SMALL_RERANK_RANK_WEIGHT * (1.0 / (1.0 + base_rank))
        final_score = normalized + rank_bias
        scored.append((int(ident), final_score, base_rank))

    if not scored:
        return []

    scored.sort(key=lambda entry: (-entry[1], entry[2]))
    trimmed = scored[:limit]
    return [(item_id, score) for item_id, score, _ in trimmed]


def _call_small_reranker(
    settings: RerankerSettings,
    items: Sequence[Dict[str, Any]],
    intent: IntentFilters | None,
    query: str | None,
    user: Dict[str, Any] | None,
) -> List[LLMDecision]:
    if not items:
        return []

    window = min(len(items), _SMALL_RERANK_INPUT_WINDOW)
    subset = list(items[:window])
    query_text = _build_small_rerank_query(intent, query)
    if not query_text:
        return []

    cache_key = _small_rerank_cache_key(query_text, subset, user)
    with _SMALL_RERANK_CACHE_LOCK:
        cached = _SMALL_RERANK_CACHE.get(cache_key)
    if cached is not None:
        return [
            LLMDecision(item_id=item_id, score=score, explanation=None)
            for item_id, score in cached
        ]

    future = _SMALL_RERANK_EXECUTOR.submit(_execute_small_rerank, query_text, subset)
    try:
        scored = future.result(timeout=settings.timeout)
    except FuturesTimeout:
        future.cancel()
        logger.warning(
            "Small reranker timed out after %.2fs; using baseline ordering.",
            settings.timeout,
        )
        return []
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "Small reranker failed with error: %s; using baseline ordering.", exc
        )
        return []

    if not scored:
        return []

    with _SMALL_RERANK_CACHE_LOCK:
        _SMALL_RERANK_CACHE[cache_key] = list(scored)

    return [
        LLMDecision(item_id=item_id, score=score, explanation=None)
        for item_id, score in scored
    ]


def _reset_small_rerank_cache_for_tests() -> None:
    with _SMALL_RERANK_CACHE_LOCK:
        _SMALL_RERANK_CACHE.clear()


def _call_reranker(
    settings: RerankerSettings,
    items: Sequence[Dict[str, Any]],
    intent: IntentFilters | None,
    query: str | None,
    user: Dict[str, Any] | None,
) -> List[LLMDecision]:
    if settings.provider == "openai":
        decisions = _call_openai_reranker(settings, items, intent, query, user)
    elif settings.provider == "gemini":
        decisions = _call_gemini_reranker(settings, items, intent, query, user)
    elif settings.provider == "small":
        decisions = _call_small_reranker(settings, items, intent, query, user)
    else:
        raise RerankerError(f"Unsupported reranker provider '{settings.provider}'")

    # Validate reranker output in debug/test environments
    if logger.isEnabledFor(logging.DEBUG) and settings.provider != "small":
        config = _get_prompt_config()
        for example in config.get("examples", []):
            if (
                example.get("input", {}).get("query") == query
                and example.get("input", {}).get("intent") == intent
            ):
                from api.core.prompt_eval import evaluate_reranker_output

                output = {
                    "items": [
                        {
                            "id": d.item_id,
                            "score": d.score,
                            "explanation": d.explanation,
                        }
                        for d in decisions
                    ]
                }
                score, errors = evaluate_reranker_output(
                    output,
                    example["expected_output"],
                    example["input"]["intent"],
                    example["input"]["query"],
                )
                if errors:
                    logger.debug("Reranker evaluation found issues: %s", errors)
                logger.debug("Reranker evaluation score: %.2f", score)
                break

    return decisions


def _call_openai_reranker(
    settings: RerankerSettings,
    items: Sequence[Dict[str, Any]],
    intent: IntentFilters | None,
    query: str | None,
    user: Dict[str, Any] | None,
) -> List[LLMDecision]:
    if not settings.api_key:
        raise RerankerError("Missing API key for reranker provider.")

    payload = _build_llm_payload(settings, items, intent, query, user)
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }
    project = os.getenv("OPENAI_PROJECT")
    if project:
        headers["OpenAI-Project"] = project

    endpoint = settings.endpoint
    body = payload
    using_responses_api = bool(project) or endpoint.endswith("/responses")
    if using_responses_api:
        endpoint = endpoint.rstrip("/")
        if not endpoint.endswith("/responses"):
            endpoint = f"{endpoint}/responses"

        messages = payload.get("messages", [])

        def _as_response_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            converted: List[Dict[str, Any]] = []
            for message in messages:
                text = message.get("content")
                if text is None:
                    continue
                converted.append(
                    {
                        "role": message.get("role", "user"),
                        "content": [{"type": "text", "text": str(text)}],
                    }
                )
            return converted

        body = {
            "model": settings.model,
            "input": _as_response_input(messages),
            "max_output_tokens": 2048,
        }

    with httpx.Client(timeout=settings.timeout) as client:
        response = client.post(endpoint, headers=headers, json=body)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - network failure path
            raise RerankerError(
                f"OpenAI API responded with status {exc.response.status_code}"
            ) from exc

    data = response.json()
    if using_responses_api:
        try:
            first_block = data["output"][0]["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RerankerError(
                "Unexpected response structure from Responses API."
            ) from exc
        try:
            parsed = json.loads(first_block)
        except json.JSONDecodeError as exc:
            raise RerankerError("Reranker returned non-JSON content.") from exc
        return _parse_decisions_json(parsed)

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RerankerError("Unexpected response structure from reranker.") from exc
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RerankerError("Reranker returned non-JSON content.") from exc

    return _parse_decisions_json(parsed)


def _call_gemini_reranker(
    settings: RerankerSettings,
    items: Sequence[Dict[str, Any]],
    intent: IntentFilters | None,
    query: str | None,
    user: Dict[str, Any] | None,
) -> List[LLMDecision]:
    if not settings.api_key:
        raise RerankerError("Missing API key for reranker provider.")

    payload = _build_gemini_payload(settings, items, intent, query, user)

    endpoint = settings.endpoint.rstrip("/")
    if endpoint.endswith(":generateContent"):
        url = endpoint
    else:
        url = f"{endpoint}/{settings.model}:generateContent"

    with httpx.Client(timeout=settings.timeout) as client:
        response = client.post(
            url,
            params={"key": settings.api_key},
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        if response.status_code >= 400:
            sanitized_url = str(response.request.url.copy_with(query=None))
            logger.warning(
                "Gemini reranker HTTP %s for %s",
                response.status_code,
                sanitized_url,
            )
            raise RerankerError(
                f"Gemini API responded with status {response.status_code}"
            )
        response.raise_for_status()

    data = response.json()
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return []

    first = candidates[0]
    content = first.get("content")
    parts: List[Dict[str, Any]] = []
    if isinstance(content, dict):
        maybe_parts = content.get("parts")
        if isinstance(maybe_parts, list):
            parts = maybe_parts
    elif isinstance(content, list):
        # Some responses return a list of parts directly.
        parts = [part for part in content if isinstance(part, dict)]

    if not parts and isinstance(first.get("content"), list):
        parts = [
            part
            for part in first.get("content", [])
            if isinstance(part, dict) and "text" in part
        ]

    text_chunks: List[str] = []
    for part in parts:
        text = part.get("text")
        if isinstance(text, str):
            stripped = text.strip()
            if stripped:
                text_chunks.append(stripped)

    if not text_chunks:
        return []

    combined = " ".join(text_chunks)
    try:
        parsed = json.loads(combined)
    except json.JSONDecodeError as exc:
        raise RerankerError("Gemini reranker returned non-JSON content.") from exc

    return _parse_decisions_json(parsed)


def _parse_decisions_json(payload: Dict[str, Any] | List) -> List[LLMDecision]:
    raw_items = _decision_items(payload)
    if not raw_items:
        return []

    decisions: List[LLMDecision] = []
    for entry in raw_items:
        if not isinstance(entry, dict):
            continue
        item_id = entry.get("id")
        if item_id is None:
            continue
        try:
            iid = int(item_id)
        except (TypeError, ValueError):
            continue
        score = entry.get("score")
        try:
            score_val = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_val = None
        explanation = entry.get("explanation")
        if explanation is not None:
            explanation = str(explanation).strip()
        decisions.append(
            LLMDecision(item_id=iid, score=score_val, explanation=explanation)
        )
    return decisions


def _decision_items(payload: Dict[str, Any] | List[Any] | None) -> List[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        items = payload.get("items")
        return items if isinstance(items, list) else []
    return []


def _build_prompt_text(
    items: Sequence[Dict[str, Any]],
    intent: IntentFilters | None,
    query: str | None,
    user: Dict[str, Any] | None,
) -> str:
    filters_lines: List[str] = []
    if intent:
        genres = intent.effective_genres()
        if genres:
            filters_lines.append(f"Preferred genres: {', '.join(genres)}")
        if intent.media_types:
            filters_lines.append(f"Media types: {', '.join(intent.media_types)}")
        if intent.min_runtime is not None:
            filters_lines.append(f"Minimum runtime: {intent.min_runtime} minutes")
        if intent.max_runtime is not None:
            filters_lines.append(f"Maximum runtime: {intent.max_runtime} minutes")
    if query:
        filters_lines.append(f"Original query: {query}")
    if user and user.get("user_id"):
        filters_lines.append(f"User id: {user['user_id']}")
    filters_text = (
        "\n".join(filters_lines) if filters_lines else "No explicit filters provided."
    )

    candidate_lines: List[str] = []
    for item in items:
        genre_text = _format_join(_extract_genre_names(item.get("genres")))
        overview = _short_overview(item.get("overview"))
        candidate_lines.append(
            (
                f"- id={item.get('id')} | title={item.get('title')} | "
                f"type={item.get('media_type')} | year={item.get('release_year')} | "
                f"runtime={item.get('runtime')} | genres={genre_text or 'n/a'} | "
                f"rank={item.get('original_rank')} | overview={overview}"
            )
        )

    prompt = (
        "Review the candidate list and return JSON containing the best ordering.\n"
        "Include all provided items exactly once in your response.\n"
        "For each item supply fields: id (int), score (float between 0 and 1), explanation (<=45 words).\n"
        "Scores should reflect final ranking (higher means better).\n"
        "Keep explanations concise and reference intent filters when relevant.\n\n"
        f"Context:\n{filters_text}\n\n"
        "Candidates:\n" + "\n".join(candidate_lines)
    )

    return prompt


def _build_llm_payload(
    settings: RerankerSettings,
    items: Sequence[Dict[str, Any]],
    intent: IntentFilters | None,
    query: str | None,
    user: Dict[str, Any] | None,
) -> Dict[str, Any]:
    prompt = _build_prompt_text(items, intent, query, user)

    return {
        "model": settings.model,
        "messages": [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }


def _build_gemini_payload(
    settings: RerankerSettings,
    items: Sequence[Dict[str, Any]],
    intent: IntentFilters | None,
    query: str | None,
    user: Dict[str, Any] | None,
) -> Dict[str, Any]:
    prompt = f"{_SYSTEM_PROMPT}\n\n{_build_prompt_text(items, intent, query, user)}"

    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }


def _default_explanation(
    item: Dict[str, Any],
    intent: IntentFilters | None,
    query: str | None,
    intent_genres: Sequence[str],
) -> str:
    templates = _load_explanation_templates()
    narrative_templates = templates.get("narrative", {})
    parts: List[str] = []

    item_genres = _extract_genre_names(item.get("genres"))
    item_genres_lower = {_safe_lower(name) for name in item_genres}
    matched_genres = [
        genre for genre in intent_genres if _safe_lower(genre) in item_genres_lower
    ]
    if matched_genres:
        pretty = ", ".join(matched_genres[:2])
        template = (
            narrative_templates.get("genre_match")
            or _DEFAULT_EXPLANATION_TEMPLATES["narrative"]["genre_match"]
        )
        parts.append(_safe_template(template, genres=pretty))

    if intent and intent.media_types and item.get("media_type") in intent.media_types:
        template = (
            narrative_templates.get("media_type")
            or _DEFAULT_EXPLANATION_TEMPLATES["narrative"]["media_type"]
        )
        parts.append(
            _safe_template(
                template,
                media_type=str(item.get("media_type")),
            )
        )

    runtime = item.get("runtime")
    if (
        runtime
        and intent
        and (
            (intent.min_runtime and runtime >= intent.min_runtime)
            or (intent.max_runtime and runtime <= intent.max_runtime)
        )
    ):
        template = (
            narrative_templates.get("runtime_match")
            or _DEFAULT_EXPLANATION_TEMPLATES["narrative"]["runtime_match"]
        )
        parts.append(_safe_template(template, runtime=runtime))

    summary = _short_overview(item.get("overview"))
    if summary:
        parts.append(summary)

    if query and not parts:
        template = (
            narrative_templates.get("query_fallback")
            or _DEFAULT_EXPLANATION_TEMPLATES["narrative"]["query_fallback"]
        )
        parts.append(_safe_template(template, query=query))

    if not parts:
        fallback = (
            narrative_templates.get("fallback")
            or _DEFAULT_EXPLANATION_TEMPLATES["narrative"]["fallback"]
        )
        parts.append(fallback)

    return " ".join(parts)


def _short_overview(text: Any, limit: int = 180) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    if not text:
        return ""
    first_sentence = re.split(r"(?<=[.!?])\s+", text)[0]
    truncated = (
        first_sentence
        if len(first_sentence) <= limit
        else first_sentence[: limit - 3].rstrip() + "..."
    )
    return truncated


def _extract_genre_names(raw: Any) -> List[str]:
    names: List[str] = []
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                name = entry.get("name")
                if isinstance(name, str):
                    names.append(name)
            elif isinstance(entry, str):
                names.append(entry)
    elif isinstance(raw, dict):
        name = raw.get("name")
        if isinstance(name, str):
            names.append(name)
    elif isinstance(raw, str):
        names.append(raw)
    return names


def _format_join(values: Sequence[str]) -> str:
    return ", ".join(v for v in values if v)


def _safe_lower(value: str) -> str:
    return value.lower() if isinstance(value, str) else ""
