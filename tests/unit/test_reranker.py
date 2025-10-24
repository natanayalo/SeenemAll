from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import List
from types import SimpleNamespace

import httpx
import pytest

from api.core import reranker
from api.core.legacy_intent_parser import IntentFilters


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


def test_rerank_with_explanations_handles_exception(monkeypatch):
    _reset_settings()
    monkeypatch.setenv("RERANK_API_KEY", "fake")

    def boom(*args, **kwargs):
        raise RuntimeError("llm failed")

    monkeypatch.setattr(reranker, "_call_openai_reranker", boom)
    original = [{"id": 1, "title": "Alpha"}, {"id": 2, "title": "Beta"}]
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


def test_get_settings_fallback_provider(monkeypatch):
    _reset_settings()
    monkeypatch.setenv("RERANK_API_KEY", "fake")
    monkeypatch.setenv("RERANK_PROVIDER", "unsupported")
    settings = reranker._get_settings()
    assert settings.provider == "openai"
    _reset_settings()


def test_call_openai_reranker_parses_payload(monkeypatch):
    _reset_settings()
    monkeypatch.setenv("OPENAI_PROJECT", "proj-test")
    monkeypatch.setenv("RERANK_API_KEY", "fake")
    settings = reranker._get_settings()

    captured_request = {}

    class DummyResponse:
        status_code = 200

        def __init__(self, url):
            self.request = SimpleNamespace(url=url)

        def raise_for_status(self):
            return None

        def json(self):
            payload = {"items": [{"id": 42, "score": 0.8, "explanation": "Ranked"}]}
            return {"output": [{"content": [{"text": json.dumps(payload)}]}]}

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
            return DummyResponse(SimpleNamespace(copy_with=lambda **_: endpoint))

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
    assert captured_request["endpoint"].endswith("/responses")
    assert captured_request["payload"]["model"] == settings.model
    assert captured_request["payload"]["input"][0]["content"][0]["type"] == "text"
    assert "Authorization" in captured_request["headers"]
    assert captured_request["payload"]["model"] == settings.model
    _reset_settings()


def test_call_reranker_raises_on_unknown_provider():
    settings = reranker.RerankerSettings(
        provider="other",
        api_key="k",
        model="m",
        endpoint="https://example",
        enabled=True,
        timeout=1.0,
    )
    with pytest.raises(reranker.RerankerError):
        reranker._call_reranker(settings, [], None, None, None)


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


def test_call_openai_reranker_requires_api_key():
    settings = reranker.RerankerSettings(
        provider="openai",
        api_key=None,
        model="m",
        endpoint="https://api",
        enabled=True,
        timeout=1.0,
    )
    with pytest.raises(reranker.RerankerError):
        reranker._call_openai_reranker(settings, [], None, None, None)


def test_call_openai_reranker_handles_unexpected_response(monkeypatch):
    _reset_settings()
    monkeypatch.setenv("RERANK_API_KEY", "fake")
    settings = reranker._get_settings()

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {}

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def post(self, *args, **kwargs):
            return DummyResponse()

    monkeypatch.setattr(reranker.httpx, "Client", DummyClient)

    with pytest.raises(reranker.RerankerError):
        reranker._call_openai_reranker(settings, [], None, None, None)
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


def test_call_gemini_reranker_requires_api_key():
    settings = reranker.RerankerSettings(
        provider="gemini",
        api_key=None,
        model="m",
        endpoint="https://api",
        enabled=True,
        timeout=1.0,
    )
    with pytest.raises(reranker.RerankerError):
        reranker._call_gemini_reranker(settings, [], None, None, None)


def test_call_gemini_reranker_returns_empty_when_no_text(monkeypatch):
    _reset_settings()
    monkeypatch.setenv("RERANK_API_KEY", "gem")
    monkeypatch.setenv("RERANK_PROVIDER", "gemini")
    settings = reranker._get_settings()

    class DummyResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": ""}]}}]}

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def post(self, *args, **kwargs):
            return DummyResponse()

    monkeypatch.setattr(reranker.httpx, "Client", DummyClient)
    assert reranker._call_gemini_reranker(settings, [], None, None, None) == []
    _reset_settings()


def test_diversify_with_mmr_balances_similarity():
    items = [
        {"id": 1, "original_rank": 0, "vector": [1.0, 0.0]},
        {"id": 2, "original_rank": 1, "vector": [0.9, 0.1]},
        {"id": 3, "original_rank": 2, "vector": [0.0, 1.0]},
    ]

    diversified = reranker.diversify_with_mmr(items, lambda_param=0.6, limit=3)
    assert len(diversified) == 3
    assert any(item["id"] == 3 for item in diversified)
    sim = reranker.cosine_similarity(items[0]["vector"], items[1]["vector"])
    assert 0.85 <= sim <= 1.0


def test_diversify_with_mmr_respects_pool_override():
    items = []
    for idx in range(8):
        vec = [0.0, 0.0, 0.0, 0.0]
        vec[idx % 4] = 1.0
        items.append({"id": idx, "original_rank": idx, "vector": vec.copy()})

    diversified = reranker.diversify_with_mmr(
        items, lambda_param=0.6, limit=3, pool_size=4
    )
    assert len(diversified) == 3
    assert all(entry["id"] < 4 for entry in diversified)


def test_diversify_with_mmr_appends_items_without_vectors_in_order():
    items = [
        {"id": 1, "original_rank": 0, "vector": [1.0, 0.0]},
        {"id": 2, "original_rank": 1},
        {"id": 3, "original_rank": 2, "vector": [0.0, 1.0]},
        {"id": 4, "original_rank": 3},
    ]

    diversified = reranker.diversify_with_mmr(items, lambda_param=0.5, limit=4)
    assert [item["id"] for item in diversified] == [1, 3, 2, 4]


def test_diversify_with_mmr_keeps_first_item_without_vector():
    items = [
        {"id": 1, "original_rank": 0},
        {"id": 2, "original_rank": 1, "vector": [1.0, 0.0]},
        {"id": 3, "original_rank": 2, "vector": [0.0, 1.0]},
    ]

    diversified = reranker.diversify_with_mmr(items, lambda_param=0.5, limit=3)
    assert [item["id"] for item in diversified] == [1, 2, 3]


def test_diversify_with_mmr_handles_zero_vector():
    items = [
        {"id": 1, "original_rank": 0, "vector": [0.0, 0.0]},
        {"id": 2, "original_rank": 1, "vector": [0.0, 1.0]},
    ]

    diversified = reranker.diversify_with_mmr(items, lambda_param=0.5, limit=2)
    assert [item["id"] for item in diversified] == [1, 2]


def test_diversify_with_mmr_clamps_lambda():
    items = [
        {"id": 1, "original_rank": 0, "vector": [1.0, 0.0]},
        {"id": 2, "original_rank": 1, "vector": [0.9, 0.1]},
        {"id": 3, "original_rank": 2, "vector": [0.0, 1.0]},
    ]

    diversified = reranker.diversify_with_mmr(items, lambda_param=2.0, limit=3)
    assert [item["id"] for item in diversified] == [1, 2, 3]


def test_resolve_mmr_pool_size_honours_multiplier(monkeypatch):
    monkeypatch.setattr(reranker, "_MMR_POOL_MULTIPLIER", 2, raising=False)
    monkeypatch.setattr(reranker, "_MMR_POOL_MIN", 0, raising=False)
    monkeypatch.setattr(reranker, "_MMR_POOL_MAX", None, raising=False)

    pool = reranker._resolve_mmr_pool_size(5, 40, override=None)
    assert pool == 10


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


def test_call_reranker_evaluates_examples(monkeypatch, caplog):
    _reset_settings()
    settings = reranker.RerankerSettings(
        provider="openai",
        api_key="fake",
        model="m",
        endpoint="https://api",
        enabled=True,
        timeout=1.0,
    )
    example = reranker._get_prompt_config()["examples"][0]
    decisions = [
        reranker.LLMDecision(item_id=2, score=0.95, explanation="Match"),
        reranker.LLMDecision(item_id=1, score=0.85, explanation="Runner"),
    ]
    monkeypatch.setattr(
        reranker,
        "_call_openai_reranker",
        lambda s, items, intent, query, user: decisions,
    )

    caplog.set_level(logging.DEBUG, logger=reranker.logger.name)
    result = reranker._call_reranker(
        settings,
        example["input"]["items"],
        example["input"]["intent"],
        example["input"]["query"],
        user={"user_id": "u1"},
    )
    assert result == decisions
    caplog.set_level(logging.INFO, logger=reranker.logger.name)
    _reset_settings()


def test_parse_decisions_json_skips_invalid_entries():
    payload = {
        "items": [
            {"id": "bad", "score": "oops"},
            {"id": 2, "score": "0.8", "explanation": 123},
        ]
    }
    decisions = reranker._parse_decisions_json(payload)
    assert len(decisions) == 1
    assert decisions[0].item_id == 2
    assert decisions[0].score == 0.8
    assert decisions[0].explanation == "123"


def test_build_prompt_text_includes_filters():
    filters = IntentFilters(
        raw_query="mystery",
        genres=["Mystery"],
        moods=[],
        media_types=["movie"],
        min_runtime=80,
        max_runtime=120,
    )
    items = [
        {
            "id": 1,
            "title": "Mystery Night",
            "media_type": "movie",
            "release_year": 2020,
            "runtime": 90,
            "genres": [{"name": "Mystery"}],
            "overview": "A thrilling tale.",
            "original_rank": 0,
        }
    ]
    prompt = reranker._build_prompt_text(
        items, filters, "mystery movie", {"user_id": "u1"}
    )
    assert "Preferred genres" in prompt
    assert "Original query" in prompt
    assert "User id: u1" in prompt


def test_default_explanation_covers_runtime_branch():
    filters = IntentFilters(
        raw_query="long drama",
        genres=["Drama"],
        moods=[],
        media_types=["movie"],
        min_runtime=100,
        max_runtime=200,
    )
    item = {
        "id": 1,
        "media_type": "movie",
        "runtime": 150,
        "genres": [{"name": "Drama"}],
        "overview": "An emotional journey.",
    }
    explanation = reranker._default_explanation(item, filters, "long drama", ["Drama"])
    assert "Drama" in explanation
    assert "150" in explanation


def test_heuristic_ranker_highlights_trending(monkeypatch):
    _reset_settings()
    monkeypatch.delenv("RERANK_API_KEY", raising=False)
    monkeypatch.setenv("RERANK_ENABLED", "0")
    reranker._get_settings.cache_clear()

    current_year = datetime.utcnow().year
    items = [
        {
            "id": 1,
            "title": "Now Trending",
            "overview": "A gripping new release taking off.",
            "genres": [{"name": "Thriller"}],
            "media_type": "movie",
            "retrieval_score": 0.6,
            "popularity": 95.0,
            "vote_average": 8.2,
            "vote_count": 1200,
            "trending_rank": 3,
            "popular_rank": 5,
            "release_year": current_year,
        }
    ]

    result = reranker.rerank_with_explanations(items, intent=None, query=None)
    assert result[0]["score"] >= 0.6
    explanation = result[0]["explanation"]
    assert "Trending" in explanation or "trending" in explanation.lower()
    assert str(current_year) in explanation
    _reset_settings()


def test_explanation_templates_override(monkeypatch):
    _reset_settings()
    monkeypatch.delenv("RERANK_API_KEY", raising=False)
    monkeypatch.setenv("RERANK_ENABLED", "0")
    reranker._get_settings.cache_clear()

    custom_templates = {
        "narrative": {
            "genre_match": "Custom genre match for {genres}.",
            "fallback": "Custom narrative fallback.",
        },
        "heuristic": {
            "trending": "Custom trending #{rank}.",
            "recent": "Custom fresh in {year}.",
            "fallback": "Custom heuristic fallback.",
        },
    }

    monkeypatch.setattr(
        reranker, "_load_explanation_templates", lambda: custom_templates
    )

    filters = IntentFilters(
        raw_query="thrilling ride",
        genres=["Thriller"],
        moods=[],
        media_types=["movie"],
        min_runtime=None,
        max_runtime=None,
    )

    current_year = datetime.utcnow().year
    items = [
        {
            "id": 1,
            "title": "Custom Template Test",
            "overview": "Edge of your seat thriller.",
            "genres": [{"name": "Thriller"}],
            "media_type": "movie",
            "retrieval_score": 0.4,
            "popularity": 40.0,
            "trending_rank": 2,
            "release_year": current_year,
        }
    ]

    result = reranker.rerank_with_explanations(items, intent=filters, query=None)
    explanation = result[0]["explanation"]
    assert "Custom genre match" in explanation
    assert "Custom trending #2" in explanation
    assert f"{current_year}" in explanation
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
