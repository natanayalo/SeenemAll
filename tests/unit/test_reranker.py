from __future__ import annotations

import json
import os
from typing import List
import httpx
import pytest

from api.core import reranker
from api.core.intent_parser import IntentFilters


def _reset_settings() -> None:
    reranker._get_settings.cache_clear()
    for key in [
        "RERANK_API_KEY",
        "OPENAI_API_KEY",
        "RERANK_PROVIDER",
        "RERANK_ENDPOINT",
        "RERANK_MODEL",
        "RERANK_ENABLED",
    ]:
        os.environ.pop(key, None)


def test_rerank_with_explanations_without_api_key(monkeypatch):
    _reset_settings()
    monkeypatch.delenv("RERANK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    items = [
        {
            "id": 1,
            "title": "Comet Trails",
            "overview": "A feel-good comedy about a space crew finding home.",
            "genres": [{"name": "Comedy"}, {"name": "Science Fiction"}],
            "media_type": "movie",
            "runtime": 105,
        }
    ]

    result = reranker.rerank_with_explanations(items, intent=None, query=None)

    assert len(result) == 1
    payload = result[0]
    assert payload["id"] == 1
    assert payload["explanation"]
    assert "score" in payload
    assert payload["original_rank"] == 0
    _reset_settings()


def test_rerank_with_explanations_no_items():
    _reset_settings()
    assert reranker.rerank_with_explanations([], intent=None) == []
    _reset_settings()


def test_rerank_returns_base_when_llm_decisions_empty(monkeypatch):
    _reset_settings()
    monkeypatch.setenv("RERANK_API_KEY", "fake")
    original = [
        {"id": 1, "title": "Alpha"},
        {"id": 2, "title": "Beta"},
    ]
    monkeypatch.setattr(
        reranker, "_call_openai_reranker", lambda s, i, intent, q, u: []
    )

    result = reranker.rerank_with_explanations(original, intent=None)
    assert [item["id"] for item in result] == [1, 2]
    _reset_settings()


def test_get_settings_invalid_timeout_warns(monkeypatch, capfd):
    _reset_settings()
    monkeypatch.setenv("RERANK_API_KEY", "fake")
    monkeypatch.setenv("RERANK_TIMEOUT", "not-a-number")

    settings = reranker._get_settings()

    assert settings.timeout == reranker._DEFAULT_TIMEOUT_SECONDS
    _reset_settings()


def test_call_openai_reranker_parses_payload(monkeypatch):
    _reset_settings()
    monkeypatch.setenv("RERANK_API_KEY", "fake")
    settings = reranker._get_settings()

    captured_request = {}

    class DummyResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            payload = {"items": [{"id": 42, "score": 0.8, "explanation": "Ranked"}]}
            return {
                "choices": [
                    {"message": {"content": json.dumps(payload)}},
                ]
            }

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def post(self, endpoint, headers, json):
            captured_request["endpoint"] = endpoint
            captured_request["headers"] = headers
            captured_request["payload"] = json
            return DummyResponse()

    monkeypatch.setattr(reranker.httpx, "Client", DummyClient)

    decisions = reranker._call_openai_reranker(
        settings,
        [{"id": 1, "title": "Alpha", "media_type": "movie"}],
        intent=None,
        query="hi",
        user={"user_id": "u1"},
    )

    assert decisions and decisions[0].item_id == 42
    assert decisions[0].explanation == "Ranked"
    assert "Authorization" in captured_request["headers"]
    assert captured_request["payload"]["model"] == settings.model
    _reset_settings()


def test_rerank_with_explanations_uses_gemini(monkeypatch):
    _reset_settings()
    monkeypatch.setenv("RERANK_API_KEY", "gem-key")
    monkeypatch.setenv("RERANK_PROVIDER", "gemini")

    items = [
        {"id": 1, "title": "Alpha"},
        {"id": 2, "title": "Beta"},
    ]

    def fake_call(settings, payload, intent, query, user):
        assert settings.provider == "gemini"
        return [
            reranker.LLMDecision(item_id=2, score=0.88, explanation="Gemini pick."),
            reranker.LLMDecision(item_id=1, score=0.75, explanation="Backup."),
        ]

    monkeypatch.setattr(reranker, "_call_gemini_reranker", fake_call)

    result = reranker.rerank_with_explanations(items, intent=None, query=None)
    assert [item["id"] for item in result[:2]] == [2, 1]
    assert result[0]["explanation"] == "Gemini pick."
    _reset_settings()


def test_call_gemini_reranker_parses_payload(monkeypatch):
    _reset_settings()
    monkeypatch.setenv("RERANK_API_KEY", "gem-key")
    monkeypatch.setenv("RERANK_PROVIDER", "gemini")
    monkeypatch.delenv("RERANK_ENDPOINT", raising=False)
    settings = reranker._get_settings()

    captured_request = {}

    class DummyResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            payload = {
                "items": [{"id": 9, "score": 0.91, "explanation": "From Gemini."}]
            }
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": json.dumps(payload)},
                            ]
                        }
                    }
                ]
            }

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def post(self, url, params, headers, json):
            captured_request["url"] = url
            captured_request["params"] = params
            captured_request["headers"] = headers
            captured_request["payload"] = json
            return DummyResponse()

    monkeypatch.setattr(reranker.httpx, "Client", DummyClient)

    decisions = reranker._call_gemini_reranker(
        settings,
        [{"id": 1, "title": "Alpha"}],
        intent=None,
        query="Pick best",
        user={"user_id": "u1"},
    )

    assert decisions and decisions[0].item_id == 9
    assert captured_request["params"]["key"] == "gem-key"
    assert settings.model in captured_request["url"]
    assert captured_request["headers"]["Content-Type"] == "application/json"
    text = captured_request["payload"]["contents"][0]["parts"][0]["text"]
    assert "Candidates:" in text
    _reset_settings()


def test_call_gemini_reranker_sanitizes_error(monkeypatch, capfd):
    _reset_settings()
    monkeypatch.setenv("RERANK_API_KEY", "gem-key")
    monkeypatch.setenv("RERANK_PROVIDER", "gemini")
    monkeypatch.delenv("RERANK_ENDPOINT", raising=False)
    settings = reranker._get_settings()

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def post(self, url, params, headers, json):
            request = httpx.Request(
                "POST",
                f"https://generativelanguage.googleapis.com/v1beta/models/{settings.model}:generateContent",
                params={"key": params["key"]},
            )
            return httpx.Response(status_code=404, request=request)

    monkeypatch.setattr(reranker.httpx, "Client", DummyClient)

    with pytest.raises(reranker.RerankerError) as excinfo:
        reranker._call_gemini_reranker(
            settings,
            [{"id": 1, "title": "Alpha"}],
            intent=None,
            query=None,
            user=None,
        )

    assert "status 404" in str(excinfo.value)
    _reset_settings()


def test_default_explanation_uses_intent_filters():
    _reset_settings()
    filters = IntentFilters(
        raw_query="action movie",
        genres=["Action"],
        moods=[],
        media_types=["movie"],
        min_runtime=90,
        max_runtime=None,
    )
    items = [
        {
            "id": 1,
            "title": "Action Hit",
            "overview": "Explosive action thriller.",
            "genres": [{"name": "Action"}],
            "media_type": "movie",
            "runtime": 95,
        }
    ]

    result = reranker.rerank_with_explanations(
        items, intent=filters, query="action movie"
    )
    explanation = result[0]["explanation"]
    assert "Action" in explanation
    assert "movie" in explanation.lower()
    assert "95" in explanation
    _reset_settings()


def test_default_explanation_falls_back_to_query():
    _reset_settings()
    items = [
        {
            "id": 1,
            "title": "Mystery Void",
            "overview": None,
            "genres": [],
        }
    ]

    result = reranker.rerank_with_explanations(items, intent=None, query="mystery vibe")
    assert "mystery vibe" in result[0]["explanation"]
    _reset_settings()


def test_rerank_with_explanations_uses_llm_decisions(monkeypatch):
    _reset_settings()
    monkeypatch.setenv("RERANK_API_KEY", "fake-key")
    monkeypatch.setenv("RERANK_PROVIDER", "openai")

    items = [
        {"id": 1, "title": "Alpha", "overview": "First film.", "genres": ["Drama"]},
        {"id": 2, "title": "Beta", "overview": "Second film.", "genres": ["Comedy"]},
        {"id": 3, "title": "Gamma", "overview": "Third film.", "genres": ["Sci-Fi"]},
    ]

    decisions: List[reranker.LLMDecision] = [
        reranker.LLMDecision(item_id=2, score=0.95, explanation="Funny and upbeat."),
        reranker.LLMDecision(item_id=1, score=0.74, explanation="Grounded drama."),
    ]

    def fake_call(settings, data, intent, query, user):
        return decisions

    monkeypatch.setattr(reranker, "_call_openai_reranker", fake_call)

    result = reranker.rerank_with_explanations(
        items, intent=None, query="light laughs", user={"user_id": "u1"}
    )

    assert [item["id"] for item in result[:3]] == [2, 1, 3]
    assert result[0]["explanation"] == "Funny and upbeat."
    assert result[1]["explanation"] == "Grounded drama."
    assert result[2]["id"] == 3
    assert result[2]["explanation"]
    _reset_settings()
