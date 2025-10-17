import hashlib
import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Any, Dict, Tuple, List

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
INTENT_CACHE: TTLCache[Tuple[str, str, str, str], Intent] = TTLCache(
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


def get_cache_key(
    query: str, user_context: Dict[str, Any], linked_entities: Dict[str, Any] | None
) -> Tuple[str, str, str, str]:
    """Creates a cache key from the query and user context."""
    user_id = str(user_context.get("user_id", ""))
    profile_id = str(
        user_context.get("profile_id")
        or user_context.get("profile")
        or user_context.get("profile_name")
        or ""
    )
    linked_entities_hash = ""
    if linked_entities and any(
        linked_entities.get(key) for key in ("movie", "tv", "person")
    ):
        linked_entities_hash = hashlib.sha256(
            json.dumps(linked_entities, sort_keys=True).encode()
        ).hexdigest()

    return (query, user_id, profile_id, linked_entities_hash)


def get_rewrite_cache_key(query: str, intent: Intent) -> str:
    """Creates a cache key for the rewrite cache."""
    intent_hash = hashlib.sha256(intent.model_dump_json().encode()).hexdigest()
    return f"{query}:{intent_hash}"


def _normalize_llm_output(
    payload: Dict[str, Any] | List[Any] | None
) -> Dict[str, Any] | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, list):
        return None

    merged: Dict[str, Any] = {}
    for fragment in payload:
        if not isinstance(fragment, dict):
            continue
        for key, value in fragment.items():
            if value is None:
                continue
            existing = merged.get(key)
            if isinstance(value, list):
                if not value:
                    continue
                if not isinstance(existing, list):
                    existing = []
                combined = list(existing) if existing else []
                for entry in value:
                    if entry not in combined:
                        combined.append(entry)
                merged[key] = combined
            else:
                if existing in (None, []):
                    merged[key] = value
    return merged or None


def parse_intent(
    query: str,
    user_context: Dict[str, Any],
    linked_entities: Dict[str, Any] | None = None,
) -> Intent:
    """
    Parses the user's query and returns an Intent object.
    Includes guardrails to handle validation errors and provide a fallback.
    Caches results for a short period.
    """
    normalized_query = query or ""
    cache_key = get_cache_key(normalized_query, user_context, linked_entities)

    cached_intent = INTENT_CACHE.get(cache_key)
    if cached_intent is not None:
        _log_metrics(_increment_metric("hits"), cache="intent")
        return _clone_intent(cached_intent)

    settings = _get_settings()

    _log_metrics(_increment_metric("misses"), cache="intent")

    llm_output: Dict[str, Any] | None = None
    if settings.enabled:
        logger.info(
            "Intent parser(%s) parsing query '%s'.", settings.provider, normalized_query
        )
        try:
            if settings.provider == "openai":
                llm_output = _call_openai_parser(
                    settings, normalized_query, user_context, linked_entities
                )
            elif settings.provider == "gemini":
                llm_output = _call_gemini_parser(
                    settings, normalized_query, user_context, linked_entities
                )
            else:  # pragma: no cover - defensive guardrail
                raise IntentParserError(f"Unsupported provider '{settings.provider}'")
        except IntentParserError:
            logger.exception("LLM intent parser failed; falling back to offline stub.")
    else:
        logger.info(
            "Intent parser disabled (no API key or INTENT_ENABLED=0); using offline stub for query '%s'.",
            normalized_query,
        )

    if not llm_output:
        llm_output = _offline_intent_stub(normalized_query)
        if llm_output:
            logger.debug("Offline intent stub produced intent: %s", llm_output)
        else:
            logger.debug("Offline intent stub returned empty intent payload.")
    else:
        logger.debug("LLM intent payload: %s", llm_output)

    llm_output = _normalize_llm_output(llm_output)

    if not llm_output:
        logger.info(
            "Intent parser returning default intent for query '%s'.", normalized_query
        )
        return default_intent()

    try:
        intent = Intent.model_validate(llm_output)
    except ValidationError as exc:
        logger.debug(
            "Intent payload failed validation; falling back to default intent. errors=%s payload=%s",
            exc.errors(),
            llm_output,
        )
        return default_intent()

    logger.debug("Returning intent: %s", intent)
    INTENT_CACHE[cache_key] = _clone_intent(intent)
    _log_metrics(_snapshot_metrics(), cache="intent", event="store")
    return intent


def _call_openai_parser(
    settings: IntentParserSettings,
    query: str,
    user_context: Dict[str, Any],
    linked_entities: Dict[str, Any] | None,
) -> Dict[str, Any]:
    if not settings.api_key:
        raise IntentParserError("Missing API key for intent parser provider.")

    payload = _build_llm_payload(settings, query, user_context, linked_entities)
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
    settings: IntentParserSettings,
    query: str,
    user_context: Dict[str, Any],
    linked_entities: Dict[str, Any] | None,
) -> Dict[str, Any]:
    if not settings.api_key:
        raise IntentParserError("Missing API key for intent parser provider.")

    payload = _build_gemini_payload(settings, query, user_context, linked_entities)

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


def _build_prompt_text(
    query: str, user_context: Dict[str, Any], linked_entities: Dict[str, Any] | None
) -> Tuple[str, str]:
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
    if linked_entities and any(
        linked_entities.get(key) for key in ("movie", "tv", "person")
    ):
        user_parts.append(f"Linked Entities: {json.dumps(linked_entities)}")
    user_parts.append(f"Query: {query}")
    user_parts.append("Intent:")
    user_prompt = "\n\n".join(user_parts)
    return system_prompt, user_prompt


def _build_llm_payload(
    settings: IntentParserSettings,
    query: str,
    user_context: Dict[str, Any],
    linked_entities: Dict[str, Any] | None,
) -> Dict[str, Any]:
    system_prompt, user_prompt = _build_prompt_text(
        query, user_context, linked_entities
    )

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
    settings: IntentParserSettings,
    query: str,
    user_context: Dict[str, Any],
    linked_entities: Dict[str, Any] | None,
) -> Dict[str, Any]:
    system_prompt, user_prompt = _build_prompt_text(
        query, user_context, linked_entities
    )

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

    def _append(container: Dict[str, Any], key: str, values: Any) -> None:
        if not values:
            return
        existing = container.setdefault(key, [])
        if not isinstance(existing, list):
            return
        if isinstance(values, (list, tuple, set)):
            for value in values:
                if value not in existing:
                    existing.append(value)
        else:
            if values not in existing:
                existing.append(values)

    tokens = set(normalized.replace("-", " ").split())
    payload: Dict[str, Any] = {}

    kids_keywords = {
        "kid",
        "kids",
        "child",
        "children",
        "toddler",
        "toddlers",
        "family",
        "family-friendly",
        "familyfriendly",
        "cartoon",
    }
    if tokens & kids_keywords or "family friendly" in normalized:
        _append(payload, "include_genres", ["Family", "Animation"])
        payload.setdefault("maturity_rating_max", "PG")
        if "show" in tokens or "series" in tokens:
            _append(payload, "media_types", "tv")
        if "movie" in tokens or "film" in tokens:
            _append(payload, "media_types", "movie")

    if "animated" in tokens or "cartoon" in tokens:
        _append(payload, "include_genres", "Animation")

    if "teen" in tokens and "maturity_rating_max" not in payload:
        payload["maturity_rating_max"] = "PG-13"

    if "no" in tokens and "gore" in tokens:
        _append(payload, "exclude_genres", ["Horror", "Thriller"])

    language_keywords = {
        "french": "fr",
        "spanish": "es",
        "german": "de",
        "japanese": "ja",
        "korean": "ko",
        "hindi": "hi",
    }
    for word, code in language_keywords.items():
        if word in tokens:
            _append(payload, "languages", code)

    if payload:
        return payload

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
