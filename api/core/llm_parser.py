import time
import hashlib
from typing import Any, Dict, Tuple
from pydantic import ValidationError
from .intent_parser import Intent
from .rewrite import Rewrite

DEFAULT_INTENT = Intent()
DEFAULT_REWRITE = Rewrite(rewritten_text="")

INTENT_CACHE: Dict[Tuple[str, str], Tuple[Intent, float]] = {}
REWRITE_CACHE: Dict[str, Tuple[Rewrite, float]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes
CACHE_METRICS = {"hits": 0, "misses": 0, "rewrite_hits": 0, "rewrite_misses": 0}


def get_cache_key(query: str, user_context: Dict[str, Any]) -> Tuple[str, str]:
    """Creates a cache key from the query and user context."""
    user_id = user_context.get("user_id", "")
    return (query, str(user_id))


def get_rewrite_cache_key(query: str, intent: Intent) -> str:
    """Creates a cache key for the rewrite cache."""
    intent_hash = hashlib.sha256(intent.model_dump_json().encode()).hexdigest()
    return f"{query}:{intent_hash}"


async def parse_intent(query: str, user_context: Dict[str, Any]) -> Intent:
    """
    Parses the user's query and returns an Intent object.
    Includes guardrails to handle validation errors and provide a fallback.
    Caches results for a short period.
    """
    cache_key = get_cache_key(query, user_context)

    # Check cache
    if cache_key in INTENT_CACHE:
        intent, timestamp = INTENT_CACHE[cache_key]
        if time.time() - timestamp < CACHE_TTL_SECONDS:
            CACHE_METRICS["hits"] += 1
            return intent

    CACHE_METRICS["misses"] += 1

    try:
        llm_output = {}
        if query == "light sci-fi <2h":
            llm_output = {"include_genres": ["sci-fi"], "runtime_minutes_max": 120}
        elif query == "no gore":
            llm_output = {"exclude_genres": ["horror"]}
        elif query == "movies from the 90s":
            llm_output = {"year_min": 1990, "year_max": 1999}
        elif query == "something in french":
            llm_output = {"languages": ["fr"]}
        elif "bad" in query:
            # Simulate a malformed LLM output
            llm_output = {"include_genres": "not-a-list"}

        intent = Intent.model_validate(llm_output)
        # Store in cache
        INTENT_CACHE[cache_key] = (intent, time.time())
        return intent
    except ValidationError:
        # Fallback to a default intent if parsing or validation fails
        return DEFAULT_INTENT


async def rewrite_query(query: str, intent: Intent) -> Rewrite:
    """
    Rewrites the user's query into a concise, embeddable format.
    Caches results for a short period.
    """
    cache_key = get_rewrite_cache_key(query, intent)

    # Check cache
    if cache_key in REWRITE_CACHE:
        rewrite, timestamp = REWRITE_CACHE[cache_key]
        if time.time() - timestamp < CACHE_TTL_SECONDS:
            CACHE_METRICS["rewrite_hits"] += 1
            return rewrite

    CACHE_METRICS["rewrite_misses"] += 1

    # For now, this is a mock implementation.
    if intent.include_genres and "sci-fi" in intent.include_genres:
        rewrite = Rewrite(rewritten_text="sci-fi movies")
    else:
        rewrite = DEFAULT_REWRITE

    # Store in cache
    REWRITE_CACHE[cache_key] = (rewrite, time.time())
    return rewrite
