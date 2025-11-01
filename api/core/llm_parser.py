import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Any, Dict, Tuple, List, Optional

import httpx
from cachetools import TTLCache
from pydantic import ValidationError

from .intent_parser import Intent
from .legacy_intent_parser import (
    parse_intent as legacy_parse_intent,
    canonical_genres,
)
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


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no"}


_ENABLE_MEDIA_TYPE_SCOPING = _env_flag("INTENT_SCOPE_GENRES_BY_MEDIA", default=True)
_ENABLE_ANN_DESCRIPTION = _env_flag("INTENT_ENABLE_ANN_DESCRIPTION", default=False)


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


@dataclass(frozen=True)
class IntentFallbackRule:
    keywords: set[str]
    include_genres: List[str]
    exclude_genres: List[str]
    maturity_rating_max: Optional[str] = None


_DEFAULT_FALLBACK_RULES: Tuple[IntentFallbackRule, ...] = tuple()


@lru_cache(maxsize=1)
def _load_fallback_rules() -> Tuple[IntentFallbackRule, ...]:
    path = os.getenv("INTENT_FALLBACKS_PATH")
    if not path:
        path = os.path.join(os.getcwd(), "config", "intent_fallbacks.json")
        if not os.path.exists(path):
            return _DEFAULT_FALLBACK_RULES

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw_rules = json.load(handle)
        if not isinstance(raw_rules, list):
            raise ValueError("Fallback rules file must contain a list of rules")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "Failed to load fallback rules from %s; using defaults. error=%s",
            path,
            exc,
        )
        return _DEFAULT_FALLBACK_RULES

    rules: List[IntentFallbackRule] = []
    for entry in raw_rules:
        if not isinstance(entry, dict):
            continue
        raw_keywords = entry.get("keywords") or []
        keywords = {
            str(keyword).lower()
            for keyword in raw_keywords
            if isinstance(keyword, str) and keyword.strip()
        }
        if not keywords:
            continue
        include_genres = [
            str(genre)
            for genre in entry.get("include_genres", [])
            if isinstance(genre, str)
        ]
        exclude_genres = [
            str(genre)
            for genre in entry.get("exclude_genres", [])
            if isinstance(genre, str)
        ]
        maturity = entry.get("maturity_rating_max")
        if maturity is not None:
            maturity = str(maturity)
        rules.append(
            IntentFallbackRule(
                keywords=keywords,
                include_genres=include_genres,
                exclude_genres=exclude_genres,
                maturity_rating_max=maturity,
            )
        )

    return tuple(rules) if rules else _DEFAULT_FALLBACK_RULES


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

    def _merge_list(existing: Any, values: List[Any]) -> List[Any]:
        combined: List[Any]
        if isinstance(existing, list):
            combined = list(existing)
        elif existing is None:
            combined = []
        else:
            combined = [existing]
        seen = set(combined)
        for entry in values:
            if entry not in seen:
                combined.append(entry)
                seen.add(entry)
        return combined

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
                merged[key] = _merge_list(existing, value)
            else:
                if existing in (None, []):
                    merged[key] = value
    return merged or None


def _augment_intent(intent: Intent, query: str) -> Intent:
    normalized_query = query.lower()
    tokens = set(normalized_query.replace("-", " ").split())

    updated = intent.model_copy(deep=True)
    include_genres = list(updated.include_genres or [])
    lower_includes = {genre.lower() for genre in include_genres}
    exclude_genres = list(updated.exclude_genres or [])
    lower_excludes = {genre.lower() for genre in exclude_genres}

    rules = _load_fallback_rules()
    include_modified = False
    exclude_modified = False

    # Include both individual tokens and the full normalized query so that
    # multi-word keywords like "no gore" from the fallback rules still match.
    phrase_matches = set(tokens)
    phrase_matches.add(normalized_query)

    for rule in rules:
        keyword_hit = any(keyword in phrase_matches for keyword in rule.keywords)
        include_hit = bool(rule.keywords & lower_includes)
        if not (keyword_hit or include_hit):
            continue

        if rule.include_genres:
            for genre in rule.include_genres:
                lowered = genre.lower()
                if lowered not in lower_includes:
                    include_genres.append(genre)
                    lower_includes.add(lowered)
                    include_modified = True

        if rule.exclude_genres:
            for genre in rule.exclude_genres:
                lowered = genre.lower()
                if lowered not in lower_excludes:
                    exclude_genres.append(genre)
                    lower_excludes.add(lowered)
                    exclude_modified = True

        if rule.maturity_rating_max and not updated.maturity_rating_max:
            updated.maturity_rating_max = rule.maturity_rating_max

    if include_modified:
        updated.include_genres = include_genres
    if exclude_modified:
        updated.exclude_genres = exclude_genres

    return updated


def linked_media_types(linked_entities: Dict[str, Any] | None) -> List[str]:
    if not linked_entities:
        return []
    result: List[str] = []
    for media_type in ("tv", "movie"):
        values = linked_entities.get(media_type)
        if isinstance(values, list) and values:
            result.append(media_type)
    return result


def _infer_media_types_for_prompt(
    query: str, linked_entities: Dict[str, Any] | None
) -> List[str]:
    hints: List[str] = []
    for media_type in linked_media_types(linked_entities):
        hints.append(media_type)

    if query:
        legacy_types = legacy_parse_intent(query).media_types
        for media_type in legacy_types:
            lowered = media_type.strip().lower()
            if lowered and lowered not in hints:
                hints.append(lowered)

    return hints


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

    logger.debug("LLM intent raw payload: %s", llm_output)

    if not llm_output:
        llm_output = _offline_intent_stub(normalized_query)
        if llm_output:
            logger.debug("Offline intent stub produced intent: %s", llm_output)
        else:
            logger.debug("Offline intent stub returned empty intent payload.")

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

    intent = _augment_intent(intent, normalized_query)
    if not intent.ann_description:
        truncated = " ".join(normalized_query.split()[:32]).strip()
        if truncated:
            intent.ann_description = truncated
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
    project = os.getenv("OPENAI_PROJECT")
    if project:
        headers["OpenAI-Project"] = project

    # Use Responses API when a project id is provided.
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
            "max_output_tokens": 512,
        }

    with httpx.Client(timeout=settings.timeout) as client:
        response = client.post(endpoint, headers=headers, json=body)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise IntentParserError(
                f"OpenAI API responded with status {exc.response.status_code}"
            ) from exc

    data = response.json()
    if using_responses_api:
        try:
            first_block = data["output"][0]["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise IntentParserError(
                "Unexpected response structure from Responses API."
            ) from exc
        try:
            return json.loads(first_block)
        except json.JSONDecodeError as exc:
            raise IntentParserError("Parser returned non-JSON content.") from exc

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


def _augment_ann_description_prompt(
    system_prompt: str, examples: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]]]:
    if not _ENABLE_ANN_DESCRIPTION or not system_prompt:
        return system_prompt, list(examples)

    updated_prompt = (
        f"{system_prompt}\n\n"
        "When possible, populate `ann_description` with one evocative sentence "
        "(18-35 words, <=200 characters) that conveys the desired mood, themes, and stakes. "
        "Avoid restating obvious filters such as media types, named people, or services unless they add vital context. "
        "Only include the field when there is enough signal; otherwise omit it."
    )
    augmented_examples = list(examples)
    augmented_examples.append(
        {
            "input": {"query": "gritty dystopian series on netflix with pedro pascal"},
            "expected_output": {
                "include_genres": ["Science Fiction"],
                "media_types": ["tv"],
                "include_people": ["Pedro Pascal"],
                "streaming_providers": ["netflix"],
                "ann_description": "A gritty dystopian tale where a controlled future society unravels, confronting authoritarian power and personal cost.",
            },
        }
    )
    return updated_prompt, augmented_examples


def _build_prompt_text(
    query: str, user_context: Dict[str, Any], linked_entities: Dict[str, Any] | None
) -> Tuple[str, str]:
    prompt_config = load_prompt_template("intent_parser")
    system_prompt = prompt_config.get("system_prompt", "")
    examples = list(prompt_config.get("examples", []))

    system_prompt, examples = _augment_ann_description_prompt(system_prompt, examples)

    media_types_hint: List[str] | None = None
    if _ENABLE_MEDIA_TYPE_SCOPING:
        hints = _infer_media_types_for_prompt(query or "", linked_entities)
        if hints:
            media_types_hint = hints
            logger.debug(
                "Scoping catalog genres for LLM prompt to media_types=%s", hints
            )

    genre_list = canonical_genres(media_types_hint)
    if genre_list:
        allowed_block = (
            "Allowed catalog genres (choose only from this list for include/exclude/boost):\n"
            + "\n".join(f"- {genre}" for genre in genre_list)
            + "\nIf the user mentions a synonym or related phrase, map it to the closest genre above."
        )
        if system_prompt:
            system_prompt = f"{system_prompt}\n\n{allowed_block}"
        else:
            system_prompt = allowed_block

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
    stripped = (query or "").strip()
    if not stripped:
        return {}

    intent = Intent()
    intent = _augment_intent(intent, stripped.lower())

    legacy_filters = legacy_parse_intent(query)
    if legacy_filters.genres:
        include_genres = list(intent.include_genres or [])
        for genre in legacy_filters.genres:
            if genre not in include_genres:
                include_genres.append(genre)
        if include_genres:
            intent.include_genres = include_genres

    if legacy_filters.min_runtime is not None and intent.runtime_minutes_min is None:
        intent.runtime_minutes_min = legacy_filters.min_runtime

    if legacy_filters.max_runtime is not None and intent.runtime_minutes_max is None:
        intent.runtime_minutes_max = legacy_filters.max_runtime

    if legacy_filters.maturity_rating_max and not intent.maturity_rating_max:
        intent.maturity_rating_max = legacy_filters.maturity_rating_max

    if intent.year_min is None and intent.year_max is None:
        decade_range = _infer_year_range(stripped.lower())
        if decade_range:
            intent.year_min, intent.year_max = decade_range

    if intent.languages is None:
        detected_languages = _detect_languages(stripped.lower())
        if detected_languages:
            intent.languages = detected_languages

    payload = intent.model_dump(exclude_none=True)
    if legacy_filters.media_types:
        payload.setdefault("media_types", list(legacy_filters.media_types))

    return payload


_DECADE_PATTERN = re.compile(r"\b(?:(?P<century>19|20)?(?P<decade>\d{2}))['’]?s\b")
_GENRE_REWRITE_MAP = {
    "science fiction": "sci-fi",
    "sci fi": "sci-fi",
    "sci-fi": "sci-fi",
}
_LANGUAGE_KEYWORDS = {
    "french": "fr",
    "spanish": "es",
    "german": "de",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "hindi": "hi",
    "mandarin": "zh",
    "chinese": "zh",
    "portuguese": "pt",
}


def _infer_year_range(text: str) -> Optional[Tuple[int, int]]:
    match = _DECADE_PATTERN.search(text)
    if not match:
        return None

    century = match.group("century")
    decade_str = match.group("decade")
    if decade_str is None:
        return None
    decade = int(decade_str)

    if century:
        start_year = int(f"{century}{decade:02d}")
    else:
        base_century = 1900 if decade >= 50 else 2000
        start_year = base_century + decade

    start_year = (start_year // 10) * 10
    end_year = start_year + 9
    return start_year, end_year


def _detect_languages(text: str) -> List[str]:
    detected: List[str] = []
    for keyword, code in _LANGUAGE_KEYWORDS.items():
        if keyword in text and code not in detected:
            detected.append(code)
    return detected


def _rewrite_from_intent(intent: Intent) -> Optional[str]:
    include_genres = intent.include_genres or []
    for genre in include_genres:
        key = genre.lower()
        phrase = _GENRE_REWRITE_MAP.get(key, key)
        return f"{phrase} movies"

    if intent.exclude_genres:
        return ""

    return None


def _heuristic_rewrite(normalized_query: str) -> Optional[str]:
    """Fallback heuristics for common queries when LLM intent is unavailable."""
    lower = normalized_query.lower()
    normalized = " ".join(lower.replace("-", " ").split())
    heuristics = [
        (("post apocalyptic", "tv"), "post-apocalyptic survival resilience tv series"),
        (("feel good",), "feel-good uplifting short comedy movies"),
        (("anime", "sci fi"), "anime science fiction adventure films"),
        (("space opera",), "optimistic space exploration adventure tv series"),
        (("rom com",), "romantic comedy films from the 2000s"),
        (("gritty", "superhero"), "gritty street-level vigilante superhero series"),
        (("fantasy", "witcher"), "high fantasy epic quest tv series"),
    ]

    for keywords, rewrite in heuristics:
        if all(keyword in normalized for keyword in keywords):
            return rewrite
    return None


def _truncate_words(text: str, limit: int = 8) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit])


def rewrite_query(query: str, intent: Intent) -> Rewrite:
    """
    Rewrites the user's query into a concise, embeddable format.
    """
    normalized_query = query or ""
    cache_key = get_rewrite_cache_key(normalized_query, intent)

    cached_rewrite = REWRITE_CACHE.get(cache_key)
    if cached_rewrite is not None:
        _log_metrics(_increment_metric("rewrite_hits"), cache="rewrite")
        return _clone_rewrite(cached_rewrite)

    _log_metrics(_increment_metric("rewrite_misses"), cache="rewrite")

    description = (getattr(intent, "ann_description", "") or "").strip()
    intent_hint = _rewrite_from_intent(intent)
    if intent_hint is not None:
        intent_hint = intent_hint.strip()
    normalized = normalized_query.strip()

    base_text = ""
    if description:
        base_text = description
    elif intent_hint:
        base_text = intent_hint
    else:
        heuristic = _heuristic_rewrite(normalized)
        if heuristic:
            base_text = heuristic
    if not base_text:
        base_text = normalized

    combined = base_text or normalized
    if normalized and normalized.lower() not in (combined or "").lower():
        combined = f"{combined} {normalized}".strip()

    rewritten_text = _truncate_words(combined)
    if not rewritten_text:
        rewritten_text = normalized or ""

    rewrite = Rewrite(rewritten_text=rewritten_text)

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
