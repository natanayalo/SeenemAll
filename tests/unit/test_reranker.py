from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import List
from types import SimpleNamespace

import httpx
import pytest
import numpy as np

from api.core import reranker
from api.core.legacy_intent_parser import IntentFilters


def _reset_settings() -> None:
    reranker._get_settings.cache_clear()
    reranker._reset_small_rerank_cache_for_tests()
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


def test_small_reranker_orders_items(monkeypatch):
    _reset_settings()
    monkeypatch.setenv("RERANK_PROVIDER", "small")
    filters = IntentFilters(
        raw_query="space opera", genres=["Sci-Fi"], moods=[], media_types=["movie"]
    )

    items = [
        {
            "id": 1,
            "title": "Alpha Adventure",
            "genres": [{"name": "Adventure"}],
            "overview": "Heroic quest.",
            "original_rank": 0,
        },
        {
            "id": 2,
            "title": "Bravo Mystery",
            "genres": [{"name": "Mystery"}],
            "overview": "Slow burn mystery.",
            "original_rank": 1,
        },
        {
            "id": 3,
            "title": "Charlie Cosmos",
            "genres": [{"name": "Science Fiction"}],
            "overview": "Spaceships and stars.",
            "original_rank": 2,
        },
    ]

    def fake_encode(texts):
        vectors = []
        for idx, text in enumerate(texts):
            if idx == 0:
                vectors.append(np.array([1.0, 0.0], dtype="float32"))
            elif "Charlie" in text:
                vectors.append(np.array([0.95, 0.05], dtype="float32"))
            elif "Alpha" in text:
                vectors.append(np.array([0.85, 0.15], dtype="float32"))
            else:
                vectors.append(np.array([0.4, 0.6], dtype="float32"))
        return np.vstack(vectors).astype("float32")

    monkeypatch.setattr(reranker, "encode_texts", fake_encode)

    result = reranker.rerank_with_explanations(
        items,
        intent=filters,
        query="space adventure with ships",
        user={"user_id": "u1"},
    )
    assert [item["id"] for item in result[:3]] == [3, 1, 2]
    _reset_settings()


def test_small_reranker_uses_cache(monkeypatch):
    _reset_settings()
    monkeypatch.setenv("RERANK_PROVIDER", "small")

    items = [
        {
            "id": 1,
            "title": "Alpha",
            "genres": [{"name": "Adventure"}],
            "overview": "Heroic quest.",
        },
        {
            "id": 2,
            "title": "Bravo",
            "genres": [{"name": "Drama"}],
            "overview": "Slow burn mystery.",
        },
    ]

    calls = {"count": 0}

    def fake_encode(texts):
        calls["count"] += 1
        vectors = [np.array([1.0, 0.0], dtype="float32")]
        vectors.append(np.array([0.9, 0.1], dtype="float32"))
        vectors.append(np.array([0.7, 0.3], dtype="float32"))
        return np.vstack(vectors).astype("float32")

    monkeypatch.setattr(reranker, "encode_texts", fake_encode)

    first = reranker.rerank_with_explanations(
        items, intent=None, query="adventure", user={"user_id": "u-cache"}
    )
    second = reranker.rerank_with_explanations(
        items, intent=None, query="adventure", user={"user_id": "u-cache"}
    )

    assert calls["count"] == 1
    assert [item["id"] for item in first] == [1, 2]
    assert [item["id"] for item in second] == [1, 2]
    _reset_settings()


def test_small_reranker_handles_timeout(monkeypatch):
    _reset_settings()
    monkeypatch.setenv("RERANK_PROVIDER", "small")

    items = [
        {"id": 1, "title": "Alpha", "overview": "Heroic quest."},
        {"id": 2, "title": "Bravo", "overview": "Mystery tale."},
    ]

    cancelled = {"flag": False}

    class DummyFuture:
        def result(self, timeout=None):
            raise reranker.FuturesTimeout()

        def cancel(self):
            cancelled["flag"] = True

    class DummyExecutor:
        def submit(self, *args, **kwargs):
            return DummyFuture()

    monkeypatch.setattr(reranker, "_SMALL_RERANK_EXECUTOR", DummyExecutor())
    monkeypatch.setattr(
        reranker,
        "encode_texts",
        lambda texts: np.ones((len(texts), 2), dtype="float32"),
    )

    result = reranker.rerank_with_explanations(
        items, intent=None, query="adventure", user={"user_id": "timeout"}
    )

    assert cancelled["flag"] is True
    assert [item["id"] for item in result[:2]] == [1, 2]
    _reset_settings()


def test_low_signal_penalty_reduces_score(monkeypatch):
    monkeypatch.setattr(reranker, "_HEURISTIC_MIN_VOTE_COUNT", 100, raising=False)
    monkeypatch.setattr(reranker, "_HEURISTIC_LOW_VOTE_PENALTY", 0.5, raising=False)
    monkeypatch.setattr(reranker, "_HEURISTIC_MIN_POPULARITY", 50.0, raising=False)
    monkeypatch.setattr(
        reranker, "_HEURISTIC_LOW_POPULARITY_PENALTY", 0.5, raising=False
    )

    current_year = reranker._current_year()

    low_signal = {
        "retrieval_score": 0.8,
        "popularity": 5.0,
        "vote_average": 8.5,
        "vote_count": 4.0,
        "trending_rank": None,
        "popular_rank": None,
        "release_year": current_year,
    }
    strong_signal = dict(low_signal)
    strong_signal["popularity"] = 120.0
    strong_signal["vote_count"] = 5000.0

    score_low, _ = reranker._heuristic_score(low_signal, 0, current_year)
    score_high, _ = reranker._heuristic_score(strong_signal, 0, current_year)

    assert score_low < score_high


def test_low_signal_items_are_dropped(monkeypatch):
    monkeypatch.setattr(
        reranker, "_HEURISTIC_MIN_SIGNAL_MULTIPLIER", 0.9, raising=False
    )

    low_signal_item = {
        "id": 1,
        "title": "Low Signal",
        "retrieval_score": 0.9,
        "popularity": 10.0,
        "vote_average": 5.5,
        "vote_count": 5.0,
        "release_year": reranker._current_year(),
    }
    strong_item = {
        "id": 2,
        "title": "Trusted",
        "retrieval_score": 0.9,
        "popularity": 150.0,
        "vote_average": 8.5,
        "vote_count": 5000.0,
        "release_year": reranker._current_year(),
    }

    ranked = reranker._with_default_explanations(
        [low_signal_item, strong_item], intent=None, query=None
    )
    assert [item["id"] for item in ranked] == [2]


def test_low_signal_drop_disabled(monkeypatch):
    monkeypatch.setattr(
        reranker, "_HEURISTIC_MIN_SIGNAL_MULTIPLIER", 0.9, raising=False
    )
    low_signal_item = {
        "id": 1,
        "title": "Low Signal",
        "retrieval_score": 0.9,
        "popularity": 0.0,
        "vote_average": 0.0,
        "vote_count": 0.0,
        "release_year": reranker._current_year(),
    }
    strong_item = {
        "id": 2,
        "title": "Trusted",
        "retrieval_score": 0.9,
        "popularity": 150.0,
        "vote_average": 8.5,
        "vote_count": 5000.0,
        "release_year": reranker._current_year(),
    }

    ranked = reranker._with_default_explanations(
        [low_signal_item, strong_item], intent=None, query=None, apply_low_signal=False
    )
    assert [item["id"] for item in ranked] == [1, 2]


def test_build_small_rerank_query_includes_filters():
    intent = IntentFilters(
        raw_query="emotional mystery",
        genres=["Drama"],
        media_types=["tv"],
        min_runtime=90,
        max_runtime=120,
        maturity_rating_max="PG-13",
    )
    query_text = reranker._build_small_rerank_query(intent, "emotional mystery")
    assert "emotional mystery" in query_text
    assert "Genres: Drama" in query_text
    assert "Media: tv" in query_text
    assert "Runtime >=90m" in query_text
    assert "Rating ≤ PG-13" in query_text


def test_build_small_rerank_documents_adds_metadata():
    documents = reranker._build_small_rerank_documents(
        [
            {
                "id": 1,
                "title": "Alpha",
                "overview": "Explorers chart the unknown.",
                "genres": [{"name": "Science Fiction"}],
                "media_type": "movie",
                "runtime": 118,
                "release_year": 2021,
            }
        ]
    )
    assert len(documents) == 1
    doc = documents[0]
    assert "Alpha" in doc
    assert "Genres" in doc
    assert "Runtime" in doc
    assert "Released: 2021" in doc


def test_execute_small_rerank(monkeypatch):
    monkeypatch.setattr(
        reranker,
        "encode_texts",
        lambda texts: np.array(
            [
                [1.0, 0.0],
                [0.9, 0.0],
                [0.1, 0.0],
            ],
            dtype="float32",
        ),
    )
    items = [{"id": 1, "title": "Alpha"}, {"id": 2, "title": "Beta"}]
    result = reranker._execute_small_rerank("space adventure", items)
    assert result[0][0] == 1
    assert result[1][0] == 2


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


def test_env_helpers_coerce_types(monkeypatch):
    monkeypatch.setenv("FLOAT_ENV_TEST", "0.42")
    assert reranker._float_from_env("FLOAT_ENV_TEST", 0.1) == 0.42
    monkeypatch.setenv("FLOAT_ENV_TEST", "not-a-number")
    assert reranker._float_from_env("FLOAT_ENV_TEST", 0.2) == 0.2
    monkeypatch.delenv("FLOAT_ENV_TEST", raising=False)
    assert reranker._float_from_env("FLOAT_ENV_TEST", 0.3) == 0.3

    monkeypatch.setenv("INT_ENV_TEST", "7")
    assert reranker._int_from_env("INT_ENV_TEST", 1) == 7
    monkeypatch.setenv("INT_ENV_TEST", "bad")
    assert reranker._int_from_env("INT_ENV_TEST", 5) == 5
    monkeypatch.delenv("INT_ENV_TEST", raising=False)
    assert reranker._int_from_env("INT_ENV_TEST", 9) == 9

    monkeypatch.setenv("OPT_INT_ENV_TEST", "")
    assert reranker._optional_int_from_env("OPT_INT_ENV_TEST") is None
    monkeypatch.setenv("OPT_INT_ENV_TEST", "11")
    assert reranker._optional_int_from_env("OPT_INT_ENV_TEST") == 11
    monkeypatch.setenv("OPT_INT_ENV_TEST", "oops")
    assert reranker._optional_int_from_env("OPT_INT_ENV_TEST") is None


def test_templates_loader_uses_env(monkeypatch, tmp_path, caplog):
    reranker._load_explanation_templates_cached.cache_clear()
    template_file = tmp_path / "templates.json"
    template_file.write_text(json.dumps({"heuristic": {"fallback": "Custom fallback"}}))
    monkeypatch.setenv("EXPLANATION_TEMPLATES_PATH", str(template_file))

    loaded = reranker._load_explanation_templates()
    assert loaded["heuristic"]["fallback"] == "Custom fallback"

    reranker._load_explanation_templates_cached.cache_clear()
    broken = tmp_path / "broken.json"
    broken.write_text("not-json")
    monkeypatch.setenv("EXPLANATION_TEMPLATES_PATH", str(broken))
    caplog.set_level(logging.WARNING, logger=reranker.logger.name)
    loaded = reranker._load_explanation_templates()
    assert (
        loaded["heuristic"]["fallback"]
        == reranker._DEFAULT_EXPLANATION_TEMPLATES["heuristic"]["fallback"]
    )


def test_diversify_with_mmr_handles_empty_input():
    assert reranker.diversify_with_mmr([]) == []


def test_safe_template_handles_invalid_inputs():
    assert reranker._safe_template(None, rank=1) == ""
    assert reranker._safe_template("Rank {rank}", rank=5) == "Rank 5"
    assert reranker._safe_template("Rank {rank}", position=2) == "Rank {rank}"


def test_heuristic_summary_uses_fallback(monkeypatch):
    monkeypatch.setattr(
        reranker,
        "_load_explanation_templates",
        lambda: reranker._DEFAULT_EXPLANATION_TEMPLATES,
    )
    heuristics = {
        "popularity_norm": 0.8,
        "trending_rank": None,
        "recent_bonus": 0.0,
        "popular_rank": None,
        "vote_average": 0.0,
        "vote_count": 0,
    }
    summary = reranker._heuristic_summary({"id": 1}, heuristics)
    assert "Popular with viewers" in summary


def test_merge_with_base_returns_original_when_no_decisions():
    base = [{"id": 1, "score": 0.9}]
    assert reranker._merge_with_base(base, []) == base


def test_reranker_prompt_evaluation_logs(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger=reranker.logger.name)

    def fake_openai(settings, items, intent, query, user):
        return [reranker.LLMDecision(item_id=1, score=0.8, explanation="Reason")]

    monkeypatch.setattr(reranker, "_call_openai_reranker", fake_openai)
    monkeypatch.setattr(reranker, "_call_gemini_reranker", fake_openai)

    monkeypatch.setattr(
        reranker,
        "_get_prompt_config",
        lambda: {
            "examples": [
                {
                    "input": {"query": "needle", "intent": None},
                    "expected_output": {
                        "items": [{"id": 1, "score": 0.8, "explanation": "Reason"}]
                    },
                }
            ]
        },
    )

    evaluations: list[dict] = []

    def fake_evaluate(output, expected, intent, query):
        evaluations.append(output)
        return 0.9, []

    monkeypatch.setattr(
        "api.core.prompt_eval.evaluate_reranker_output",
        fake_evaluate,
    )

    settings = reranker.RerankerSettings(
        provider="openai",
        api_key="key",
        model="gpt",
        endpoint="https://example.com",
        enabled=True,
        timeout=1.0,
    )

    decisions = reranker._call_reranker(settings, [{"id": 1}], None, "needle", None)
    assert decisions and decisions[0].score == 0.8
    assert evaluations and evaluations[0]["items"][0]["score"] == 0.8


def test_rerank_with_explanations_handles_reranker_error(monkeypatch):
    _reset_settings()
    monkeypatch.setenv("RERANK_API_KEY", "key")
    settings = reranker.RerankerSettings(
        provider="openai",
        api_key="key",
        model="gpt",
        endpoint="https://example.com",
        enabled=True,
        timeout=1.0,
    )
    monkeypatch.setattr(reranker, "_get_settings", lambda: settings)

    def boom(*_args, **_kwargs):
        raise reranker.RerankerError("boom")

    monkeypatch.setattr(reranker, "_call_openai_reranker", boom)

    items = [
        {
            "id": 1,
            "title": "Fallback",
            "overview": "",
            "genres": [{"name": "Drama"}],
        }
    ]

    result = reranker.rerank_with_explanations(items, intent=None, query="needle")
    assert result == reranker._with_default_explanations(items, None, "needle")
