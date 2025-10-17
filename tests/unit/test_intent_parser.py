import json
from types import SimpleNamespace

import pytest

from api.core.intent_parser import Intent
from api.core import llm_parser
from api.core.llm_parser import (
    parse_intent,
    rewrite_query,
    DEFAULT_INTENT,
    IntentParserSettings,
)
from api.core.rewrite import Rewrite


@pytest.fixture(autouse=True)
def _reset_llm_state(monkeypatch):
    llm_parser.INTENT_CACHE.clear()
    llm_parser.REWRITE_CACHE.clear()
    for key in llm_parser.CACHE_METRICS:
        llm_parser.CACHE_METRICS[key] = 0
    monkeypatch.delenv("INTENT_API_KEY", raising=False)
    monkeypatch.delenv("INTENT_ENABLED", raising=False)
    original_get_settings = llm_parser._get_settings
    if hasattr(original_get_settings, "cache_clear"):
        original_get_settings.cache_clear()
    yield
    current_get_settings = getattr(llm_parser, "_get_settings", None)
    if hasattr(current_get_settings, "cache_clear"):
        current_get_settings.cache_clear()
    elif hasattr(original_get_settings, "cache_clear"):
        original_get_settings.cache_clear()


def test_parse_intent_fixtures():
    """
    Tests the parse_intent function with a set of predefined fixtures.
    """
    test_cases = [
        (
            "light sci-fi <2h",
            Intent(include_genres=["Science Fiction"], runtime_minutes_max=120),
        ),
        (
            "no gore",
            Intent(exclude_genres=["Horror", "Thriller"]),
        ),
        (
            "movies from the 90s",
            Intent(year_min=1990, year_max=1999),
        ),
        (
            "something in french",
            Intent(languages=["fr"]),
        ),
        (
            "bad query",
            DEFAULT_INTENT,
        ),
    ]

    for query, expected_intent in test_cases:
        intent = parse_intent(query, {})
        assert (
            intent.model_dump() == expected_intent.model_dump()
        ), f"Query: '{query}' failed"


def test_rewrite_query():
    """
    Tests the rewrite_query function with a set of predefined fixtures.
    """
    test_cases = [
        (
            Intent(include_genres=["sci-fi"], runtime_minutes_max=120),
            Rewrite(rewritten_text="sci-fi movies"),
        ),
        (
            Intent(include_genres=["Science Fiction"]),
            Rewrite(rewritten_text="sci-fi movies"),
        ),
        (
            Intent(exclude_genres=["horror"]),
            Rewrite(rewritten_text=""),
        ),
    ]

    for intent, expected_rewrite in test_cases:
        rewrite = rewrite_query("some query", intent)
        assert rewrite == expected_rewrite


def test_parse_intent_uses_cache(monkeypatch):
    user_context = {"user_id": "u-test"}
    first = parse_intent("no gore", user_context)
    assert first.exclude_genres == ["Horror", "Thriller"]

    hits_before = llm_parser.CACHE_METRICS["hits"]
    second = parse_intent("no gore", user_context)
    assert second == first
    assert second is not first
    assert llm_parser.CACHE_METRICS["hits"] == hits_before + 1


def test_parse_intent_llm_failure_falls_back(monkeypatch):
    settings = IntentParserSettings(
        provider="openai",
        api_key="test-key",
        model="gpt-test",
        endpoint="https://example.com",
        enabled=True,
        timeout=3.0,
    )

    monkeypatch.setattr(llm_parser, "_get_settings", lambda: settings)

    def fake_call(settings, query, user_context, linked_entities):
        raise llm_parser.IntentParserError("boom")

    monkeypatch.setattr(llm_parser, "_call_openai_parser", fake_call)

    intent = parse_intent("no gore", {"user_id": "u2"})
    assert intent.exclude_genres == ["Horror", "Thriller"]


def test_parse_intent_merges_list_payload(monkeypatch):
    llm_parser.INTENT_CACHE.clear()
    settings = IntentParserSettings(
        provider="openai",
        api_key="test-key",
        model="gpt-test",
        endpoint="https://example.com",
        enabled=True,
        timeout=3.0,
    )

    monkeypatch.setattr(llm_parser, "_get_settings", lambda: settings)

    def fake_call(settings, query, user_context, linked_entities):
        return {
            "include_genres": ["Family", "Animation"],
            "exclude_genres": ["Horror"],
            "maturity_rating_max": "PG",
        }

    monkeypatch.setattr(llm_parser, "_call_openai_parser", fake_call)

    intent = parse_intent("kids show", {"user_id": "u3"})
    assert intent.maturity_rating_max == "PG"
    assert set(intent.exclude_genres or []) == {"Horror", "Thriller"}
    assert set(intent.include_genres or []) == {"Family", "Animation"}


def test_parse_intent_llm_success(monkeypatch):
    settings = IntentParserSettings(
        provider="openai",
        api_key="test-key",
        model="gpt-test",
        endpoint="https://example.com",
        enabled=True,
        timeout=3.0,
    )

    monkeypatch.setattr(llm_parser, "_get_settings", lambda: settings)

    def fake_call(settings, query, user_context, linked_entities):
        assert query == "find drama"
        assert linked_entities == {"movie": [101]}
        return {"include_genres": ["Drama"]}

    monkeypatch.setattr(llm_parser, "_call_openai_parser", fake_call)

    intent = parse_intent("find drama", {"user_id": "u3"}, {"movie": [101]})
    assert intent.include_genres == ["Drama"]


def test_get_settings_falls_back_to_openai(monkeypatch):
    monkeypatch.setenv("INTENT_PROVIDER", "invalid")
    monkeypatch.setenv("INTENT_API_KEY", "key")
    llm_parser._get_settings.cache_clear()
    settings = llm_parser._get_settings()
    assert settings.provider == "openai"
    assert settings.enabled is True


def test_call_openai_parser_returns_payload(monkeypatch):
    settings = IntentParserSettings(
        provider="openai",
        api_key="key",
        model="gpt-test",
        endpoint="https://example.com",
        enabled=True,
        timeout=2.0,
    )

    class DummyResponse:
        status_code = 200

        def __init__(self):
            self.request = SimpleNamespace(
                url=SimpleNamespace(copy_with=lambda **_: "https://example.com")
            )

        def json(self):
            return {
                "choices": [
                    {"message": {"content": json.dumps({"include_genres": ["Comedy"]})}}
                ]
            }

        def raise_for_status(self):
            return None

    class DummyClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return DummyResponse()

    monkeypatch.setattr(llm_parser.httpx, "Client", lambda *a, **k: DummyClient())

    payload = llm_parser._call_openai_parser(settings, "query", {}, None)
    assert payload == {"include_genres": ["Comedy"]}


def test_call_gemini_parser_returns_payload(monkeypatch):
    settings = IntentParserSettings(
        provider="gemini",
        api_key="key",
        model="gemini-model",
        endpoint="https://example.com",
        enabled=True,
        timeout=2.0,
    )

    class DummyResponse:
        status_code = 200

        def __init__(self):
            self.request = SimpleNamespace(
                url=SimpleNamespace(copy_with=lambda **_: "https://example.com")
            )

        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": json.dumps({"languages": ["fr"]})}]
                        }
                    }
                ]
            }

        def raise_for_status(self):
            return None

    class DummyClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return DummyResponse()

    monkeypatch.setattr(llm_parser.httpx, "Client", lambda *a, **k: DummyClient())

    payload = llm_parser._call_gemini_parser(settings, "query", {}, None)
    assert payload == {"languages": ["fr"]}


def test_offline_intent_stub_handles_unknown_query():
    assert llm_parser._offline_intent_stub("unknown request") == {}


def test_offline_intent_stub_handles_kids_query():
    payload = llm_parser._offline_intent_stub("kids show")
    assert payload["maturity_rating_max"] == "PG"
    assert set(payload["include_genres"]) >= {"Family", "Animation"}
    assert "tv" in payload.get("media_types", [])
