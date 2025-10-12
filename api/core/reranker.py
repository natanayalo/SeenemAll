from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Sequence

import httpx
import numpy as np

from api.core.intent_parser import IntentFilters

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
_SYSTEM_PROMPT = (
    "You are Seen'emAll's reranker. Reorder items and explain the top picks succinctly. "
    "Always return valid JSON with an 'items' array."
)


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
            "Reranker disabled (missing API key or RERANK_ENABLED=0); returning ANN order."
        )
        return base

    try:
        llm_items = base[: min(len(base), _DEFAULT_MAX_ITEMS_FOR_LLM)]
        logger.info(
            "Reranker(%s) evaluating %d candidates.",
            settings.provider,
            len(llm_items),
        )
        decisions = _call_reranker(settings, llm_items, intent, query, user)
        if not decisions:
            logger.info(
                "Reranker(%s) returned no decisions; using baseline ordering.",
                settings.provider,
            )
            return base
        logger.debug(
            "Reranker(%s) produced %d decisions.", settings.provider, len(decisions)
        )
        return _merge_with_base(base, decisions)
    except Exception:  # pragma: no cover - defensive guardrail
        logger.exception("LLM reranker failed; falling back to ANN order.")
        return base


def diversify_with_mmr(
    items: List[Dict[str, Any]], lambda_param: float = 0.7, limit: int | None = None
) -> List[Dict[str, Any]]:
    if not items:
        return []

    limit = limit or len(items)
    unranked = list(items)
    ranked: List[Dict[str, Any]] = []

    if unranked:
        ranked.append(unranked.pop(0))

    while unranked and len(ranked) < limit:
        max_mmr = -np.inf
        best_item = None

        for item in unranked:
            relevance = 1 / (item["original_rank"] + 1)  # Use rank as relevance score
            similarity_to_ranked = max(
                [
                    cosine_similarity(item["vector"], r_item["vector"])
                    for r_item in ranked
                ]
            )
            mmr = lambda_param * relevance - (1 - lambda_param) * similarity_to_ranked

            if mmr > max_mmr:
                max_mmr = mmr
                best_item = item

        if best_item:
            ranked.append(best_item)
            unranked.remove(best_item)

    return ranked


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


@lru_cache(maxsize=1)
def _get_settings() -> RerankerSettings:
    api_key = os.getenv("RERANK_API_KEY") or os.getenv("OPENAI_API_KEY")
    raw_provider = os.getenv("RERANK_PROVIDER", "openai").strip().lower()
    if raw_provider not in {"openai", "gemini"}:
        logger.warning(
            "Unsupported RERANK_PROVIDER '%s'; falling back to 'openai'.", raw_provider
        )
        provider = "openai"
    else:
        provider = raw_provider

    if provider == "gemini":
        default_model = "gemini-2.0-flash-exp"
        default_endpoint = "https://generativelanguage.googleapis.com/v1beta/models"
    else:
        default_model = "gpt-4o-mini"
        default_endpoint = "https://api.openai.com/v1/chat/completions"

    model = os.getenv("RERANK_MODEL", default_model)
    endpoint = os.getenv("RERANK_ENDPOINT", default_endpoint)
    enabled_value = os.getenv("RERANK_ENABLED", "1").strip().lower()
    enabled = bool(api_key) and enabled_value not in {"0", "false", "no"}
    timeout = _DEFAULT_TIMEOUT_SECONDS
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


def _with_default_explanations(
    items: Sequence[Dict[str, Any]],
    intent: IntentFilters | None,
    query: str | None,
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    intent_genres: List[str] = intent.effective_genres() if intent else []
    for idx, item in enumerate(items):
        copy_item = dict(item)
        copy_item.setdefault("original_rank", idx)
        copy_item["score"] = max(0.0, 1.0 - idx * 0.05)
        copy_item["explanation"] = _default_explanation(
            copy_item, intent, query, intent_genres
        )
        enriched.append(copy_item)
    return enriched


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


def _call_reranker(
    settings: RerankerSettings,
    items: Sequence[Dict[str, Any]],
    intent: IntentFilters | None,
    query: str | None,
    user: Dict[str, Any] | None,
) -> List[LLMDecision]:
    if settings.provider == "openai":
        return _call_openai_reranker(settings, items, intent, query, user)
    if settings.provider == "gemini":
        return _call_gemini_reranker(settings, items, intent, query, user)
    raise RerankerError(f"Unsupported reranker provider '{settings.provider}'")


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
    with httpx.Client(timeout=settings.timeout) as client:
        response = client.post(settings.endpoint, headers=headers, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - network failure path
            raise RerankerError(
                f"OpenAI API responded with status {exc.response.status_code}"
            ) from exc
    data = response.json()
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


def _parse_decisions_json(payload: Dict[str, Any]) -> List[LLMDecision]:
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
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
    parts: List[str] = []

    item_genres = _extract_genre_names(item.get("genres"))
    item_genres_lower = {_safe_lower(name) for name in item_genres}
    matched_genres = [
        genre for genre in intent_genres if _safe_lower(genre) in item_genres_lower
    ]
    if matched_genres:
        pretty = ", ".join(matched_genres[:2])
        parts.append(f"Fits the {pretty} vibe you asked for.")

    if intent and intent.media_types and item.get("media_type") in intent.media_types:
        parts.append(f"It's a {item['media_type']} pick like you specified.")

    runtime = item.get("runtime")
    if (
        runtime
        and intent
        and (
            (intent.min_runtime and runtime >= intent.min_runtime)
            or (intent.max_runtime and runtime <= intent.max_runtime)
        )
    ):
        parts.append(f"Runs about {runtime} minutes, matching your runtime preference.")

    summary = _short_overview(item.get("overview"))
    if summary:
        parts.append(summary)

    if query and not parts:
        parts.append(f"Aligns with '{query}'.")

    if not parts:
        parts.append("Popular with viewers who share your taste.")

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
