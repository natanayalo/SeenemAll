import hashlib
import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Any, Dict, Tuple

import httpx
from cachetools import TTLCache
from pydantic import ValidationError

from .intent_parser import Intent
from .rewrite import Rewrite
from api.core.prompt_eval import load_prompt_template

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


class IntentParserError(RuntimeError):
    pass


@dataclass(frozen=True)
class IntentParserSettings:
    provider: str
    api_key: str | None
    model: str
    endpoint: str
    enabled: bool
    timeout: float


@lru_cache(maxsize=1)
def _get_settings() -> IntentParserSettings:
    api_key = os.getenv("INTENT_API_KEY") or os.getenv("OPENAI_API_KEY")
    raw_provider = os.getenv("INTENT_PROVIDER", "openai").strip().lower()
    if raw_provider not in {"openai", "gemini"}:
        logger.warning(
            "Unsupported INTENT_PROVIDER '%s'; falling back to 'openai'.", raw_provider
        )
        provider = "openai"
    else:
        provider = raw_provider

    if provider == "gemini":
        default_model = "gemini-2.0-flash-lite"
        default_endpoint = "https://generativelanguage.googleapis.com/v1beta/models"
    else:
        default_model = "gpt-4o-mini"
        default_endpoint = "https://api.openai.com/v1/chat/completions"

    model = os.getenv("INTENT_MODEL", default_model)
    endpoint = os.getenv("INTENT_ENDPOINT", default_endpoint)
    enabled_value = os.getenv("INTENT_ENABLED", "1").strip().lower()
    enabled = bool(api_key) and enabled_value not in {"0", "false", "no"}
    timeout = 12.0
    raw_timeout = os.getenv("INTENT_TIMEOUT")
    if raw_timeout:
        try:
            timeout = max(1.0, float(raw_timeout))
        except ValueError:
            logger.warning(
                "Invalid INTENT_TIMEOUT value '%s'; using default.", raw_timeout
            )
    return IntentParserSettings(
        provider=provider,
        api_key=api_key,
        model=model,
        endpoint=endpoint,
        enabled=enabled,
        timeout=timeout,
    )


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

    settings = _get_settings()

    _log_metrics(_increment_metric("misses"), cache="intent")

    llm_output: Dict[str, Any] | None = None
    if settings.enabled:
        logger.info("Intent parser(%s) parsing query.", settings.provider)
        try:
            if settings.provider == "openai":
                llm_output = _call_openai_parser(
                    settings, normalized_query, user_context
                )
            elif settings.provider == "gemini":
                llm_output = _call_gemini_parser(
                    settings, normalized_query, user_context
                )
            else:  # pragma: no cover - defensive guardrail
                raise IntentParserError(f"Unsupported provider '{settings.provider}'")
        except IntentParserError:
            logger.exception("LLM intent parser failed; falling back to offline stub.")

    if not llm_output:
        llm_output = _offline_intent_stub(normalized_query)

    if not llm_output:
        return default_intent()

    try:
        intent = Intent.model_validate(llm_output)
    except ValidationError:
        logger.debug("Offline intent stub produced invalid payload; using default.")
        return default_intent()

    logger.debug("Returning intent: %s", intent)
    INTENT_CACHE[cache_key] = _clone_intent(intent)
    _log_metrics(_snapshot_metrics(), cache="intent", event="store")
    return intent


def _call_openai_parser(
    settings: IntentParserSettings, query: str, user_context: Dict[str, Any]
) -> Dict[str, Any]:
    if not settings.api_key:
        raise IntentParserError("Missing API key for intent parser provider.")

    payload = _build_llm_payload(settings, query, user_context)
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=settings.timeout) as client:
        response = client.post(settings.endpoint, headers=headers, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise IntentParserError(
                f"OpenAI API responded with status {exc.response.status_code}"
            ) from exc
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise IntentParserError("Unexpected response structure from parser.") from exc
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise IntentParserError("Parser returned non-JSON content.") from exc


def _call_gemini_parser(
    settings: IntentParserSettings, query: str, user_context: Dict[str, Any]
) -> Dict[str, Any]:
    if not settings.api_key:
        raise IntentParserError("Missing API key for intent parser provider.")

    payload = _build_gemini_payload(settings, query, user_context)

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
                "Gemini parser HTTP %s for %s",
                response.status_code,
                sanitized_url,
            )
            raise IntentParserError(
                f"Gemini API responded with status {response.status_code}"
            )
        response.raise_for_status()

    data = response.json()
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return {}

    first = candidates[0]
    content = first.get("content")
    parts: list[Dict[str, Any]] = []
    if isinstance(content, dict):
        maybe_parts = content.get("parts")
        if isinstance(maybe_parts, list):
            parts = maybe_parts
    elif isinstance(content, list):
        parts = [part for part in content if isinstance(part, dict)]

    if not parts and isinstance(first.get("content"), list):
        parts = [
            part
            for part in first.get("content", [])
            if isinstance(part, dict) and "text" in part
        ]

    text_chunks: list[str] = []
    for part in parts:
        text = part.get("text")
        if isinstance(text, str):
            stripped = text.strip()
            if stripped:
                text_chunks.append(stripped)

    if not text_chunks:
        return {}

    combined = " ".join(text_chunks)
    try:
        return json.loads(combined)
    except json.JSONDecodeError as exc:
        raise IntentParserError("Gemini parser returned non-JSON content.") from exc


def _build_prompt_text(query: str, user_context: Dict[str, Any]) -> Tuple[str, str]:
    prompt_config = load_prompt_template("intent_parser")
    system_prompt = prompt_config.get("system_prompt", "")
    examples = prompt_config.get("examples", [])

    example_texts = []
    for example in examples:
        example_texts.append(f"Query: {example['input']['query']}")
        example_texts.append(f"Intent: {json.dumps(example['expected_output'])}")

    example_text = "\n".join(example_texts)

    user_parts = []
    if example_text:
        user_parts.append(example_text)
    user_parts.append(f"Query: {query}")
    user_parts.append("Intent:")
    user_prompt = "\n\n".join(user_parts)
    return system_prompt, user_prompt


def _build_llm_payload(
    settings: IntentParserSettings, query: str, user_context: Dict[str, Any]
) -> Dict[str, Any]:
    system_prompt, user_prompt = _build_prompt_text(query, user_context)

    return {
        "model": settings.model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }


def _build_gemini_payload(
    settings: IntentParserSettings, query: str, user_context: Dict[str, Any]
) -> Dict[str, Any]:
    system_prompt, user_prompt = _build_prompt_text(query, user_context)

    payload: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }
    if system_prompt:
        payload["systemInstruction"] = {
            "role": "system",
            "parts": [{"text": system_prompt}],
        }
    return payload


def _offline_intent_stub(query: str) -> Dict[str, Any]:
    normalized = query.strip().lower()
    if not normalized:
        return {}

    fixtures: Dict[str, Dict[str, Any]] = {
        "light sci-fi <2h": {
            "include_genres": ["Science Fiction"],
            "runtime_minutes_max": 120,
        },
        "no gore": {"exclude_genres": ["Horror"]},
        "movies from the 90s": {"year_min": 1990, "year_max": 1999},
        "something in french": {"languages": ["fr"]},
    }

    if normalized in fixtures:
        return fixtures[normalized]

    if "bad" in normalized:
        return {"include_genres": "not-a-list"}

    return {}


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
    if intent.include_genres:
        lowered = {genre.lower() for genre in intent.include_genres}
        if lowered & {"sci-fi", "science fiction", "science-fiction"}:
            rewrite = Rewrite(rewritten_text="sci-fi movies")
        else:
            rewrite = default_rewrite()
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
