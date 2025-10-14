import hashlib
import logging
from threading import Lock
from typing import Any, Dict, Tuple

from cachetools import TTLCache
from pydantic import ValidationError

from .intent_parser import Intent
from .rewrite import Rewrite

DEFAULT_INTENT = Intent()
DEFAULT_REWRITE = Rewrite(rewritten_text="")

# Cache configuration
CACHE_MAXSIZE = 1000
CACHE_TTL_SECONDS = 300  # 5 minutes

# Caches
INTENT_CACHE: TTLCache[Tuple[str, str, str], Intent] = TTLCache(
    maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL_SECONDS
)
REWRITE_CACHE: TTLCache[str, Rewrite] = TTLCache(
    maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL_SECONDS
)

# Metrics
CACHE_METRICS = {"hits": 0, "misses": 0, "rewrite_hits": 0, "rewrite_misses": 0}
METRICS_LOCK = Lock()
logger = logging.getLogger(__name__)


def _clone_intent(intent: Intent) -> Intent:
    return intent.model_copy(deep=True)


def _clone_rewrite(rewrite: Rewrite) -> Rewrite:
    return rewrite.model_copy(deep=True)


def default_intent() -> Intent:
    """Return a fresh default Intent instance."""
    return _clone_intent(DEFAULT_INTENT)


def default_rewrite() -> Rewrite:
    """Return a fresh default Rewrite instance."""
    return _clone_rewrite(DEFAULT_REWRITE)


def get_cache_key(query: str, user_context: Dict[str, Any]) -> Tuple[str, str, str]:
    """Creates a cache key from the query and user context."""
    user_id = str(user_context.get("user_id", ""))
    profile_id = str(
        user_context.get("profile_id")
        or user_context.get("profile")
        or user_context.get("profile_name")
        or ""
    )
    return (query, user_id, profile_id)


def get_rewrite_cache_key(query: str, intent: Intent) -> str:
    """Creates a cache key for the rewrite cache."""
    intent_hash = hashlib.sha256(intent.model_dump_json().encode()).hexdigest()
    return f"{query}:{intent_hash}"


def parse_intent(query: str, user_context: Dict[str, Any]) -> Intent:
    """
    Parses the user's query and returns an Intent object.
    Includes guardrails to handle validation errors and provide a fallback.
    Caches results for a short period.
    """
    normalized_query = query or ""
    cache_key = get_cache_key(normalized_query, user_context)

    cached_intent = INTENT_CACHE.get(cache_key)
    if cached_intent is not None:
        _log_metrics(_increment_metric("hits"), cache="intent")
        return _clone_intent(cached_intent)

    _log_metrics(_increment_metric("misses"), cache="intent")

    try:
        llm_output: Dict[str, Any] = {}
        if normalized_query == "light sci-fi <2h":
            llm_output = {"include_genres": ["sci-fi"], "runtime_minutes_max": 120}
        elif normalized_query == "no gore":
            llm_output = {"exclude_genres": ["horror"]}
        elif normalized_query == "movies from the 90s":
            llm_output = {"year_min": 1990, "year_max": 1999}
        elif normalized_query == "something in french":
            llm_output = {"languages": ["fr"]}
        elif "bad" in normalized_query:
            # Simulate a malformed LLM output
            llm_output = {"include_genres": "not-a-list"}

        intent = Intent.model_validate(llm_output)
        INTENT_CACHE[cache_key] = _clone_intent(intent)
        _log_metrics(_snapshot_metrics(), cache="intent", event="store")
        return intent
    except ValidationError:
        # Fallback to a default intent if parsing or validation fails
        return default_intent()


def rewrite_query(query: str, intent: Intent) -> Rewrite:
    """
    Rewrites the user's query into a concise, embeddable format.
    Caches results for a short period.
    """
    normalized_query = query or ""
    cache_key = get_rewrite_cache_key(normalized_query, intent)

    cached_rewrite = REWRITE_CACHE.get(cache_key)
    if cached_rewrite is not None:
        _log_metrics(_increment_metric("rewrite_hits"), cache="rewrite")
        return _clone_rewrite(cached_rewrite)

    _log_metrics(_increment_metric("rewrite_misses"), cache="rewrite")

    # For now, this is a mock implementation.
    if intent.include_genres and "sci-fi" in intent.include_genres:
        rewrite = Rewrite(rewritten_text="sci-fi movies")
    else:
        rewrite = default_rewrite()

    REWRITE_CACHE[cache_key] = _clone_rewrite(rewrite)
    _log_metrics(_snapshot_metrics(), cache="rewrite", event="store")
    return rewrite


def _increment_metric(name: str) -> Dict[str, int]:
    with METRICS_LOCK:
        CACHE_METRICS[name] += 1
        return dict(CACHE_METRICS)


def _snapshot_metrics() -> Dict[str, int]:
    with METRICS_LOCK:
        return dict(CACHE_METRICS)


def _log_metrics(
    metrics: Dict[str, int], *, cache: str, event: str | None = None
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    extras = f" event={event}" if event else ""
    logger.debug(
        "LLM %s cache metrics%s | hits=%d misses=%d rewrite_hits=%d rewrite_misses=%d",
        cache,
        extras,
        metrics.get("hits", 0),
        metrics.get("misses", 0),
        metrics.get("rewrite_hits", 0),
        metrics.get("rewrite_misses", 0),
    )
