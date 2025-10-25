import json
import logging
from types import SimpleNamespace
from typing import Any, Dict

import pytest
import httpx

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
    monkeypatch.setattr(llm_parser, "_ENABLE_ANN_DESCRIPTION", False, raising=False)
    yield
    current_get_settings = getattr(llm_parser, "_get_settings", None)
    if hasattr(current_get_settings, "cache_clear"):
        current_get_settings.cache_clear()
    elif hasattr(original_get_settings, "cache_clear"):
        original_get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _stub_canonical_genres(monkeypatch):
    monkeypatch.setattr(
        llm_parser, "canonical_genres", lambda media_types=None: ["Action", "Drama"]
    )
    yield


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
            Intent(year_min=1990, year_max=1999, media_types=["movie"]),
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
        (
            Intent(ann_description="desperate players risk their lives for fortune"),
            Rewrite(rewritten_text="desperate players risk their lives for fortune"),
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
            "media_types": ["tv"],
            "streaming_providers": ["disney_plus"],
        }

    monkeypatch.setattr(llm_parser, "_call_openai_parser", fake_call)

    intent = parse_intent("kids show", {"user_id": "u3"})
    assert intent.maturity_rating_max == "PG"
    assert set(intent.exclude_genres or []) == {"Horror", "Thriller"}
    assert set(intent.include_genres or []) == {"Family", "Animation"}
    assert intent.media_types == ["tv"]
    assert intent.streaming_providers == ["disney_plus"]


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
        return {
            "include_genres": ["Drama"],
            "media_types": ["movie"],
            "include_people": ["Viola Davis"],
            "streaming_providers": ["netflix"],
        }

    monkeypatch.setattr(llm_parser, "_call_openai_parser", fake_call)

    intent = parse_intent("find drama", {"user_id": "u3"}, {"movie": [101]})
    assert intent.include_genres == ["Drama"]
    assert intent.media_types == ["movie"]
    assert intent.include_people == ["Viola Davis"]
    assert intent.streaming_providers == ["netflix"]


def test_get_settings_falls_back_to_openai(monkeypatch):
    monkeypatch.setenv("INTENT_PROVIDER", "invalid")
    monkeypatch.setenv("INTENT_API_KEY", "key")
    llm_parser._get_settings.cache_clear()
    settings = llm_parser._get_settings()
    assert settings.provider == "openai"
    assert settings.enabled is True


def test_call_openai_parser_returns_payload(monkeypatch):
    monkeypatch.setenv("OPENAI_PROJECT", "proj-test")
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

        def __init__(self, url):
            self.request = SimpleNamespace(url=url)

        def json(self):
            return {
                "output": [
                    {"content": [{"text": json.dumps({"include_genres": ["Comedy"]})}]}
                ]
            }

        def raise_for_status(self):
            return None

    def fake_client(*args, **kwargs):
        class DummyClient:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def post(self_inner, endpoint, headers, json):
                return DummyResponse(SimpleNamespace(copy_with=lambda **_: endpoint))

        return DummyClient()

    monkeypatch.setattr(llm_parser.httpx, "Client", fake_client)

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


def test_build_prompt_scopes_genres_when_media_known(monkeypatch):
    monkeypatch.setattr(llm_parser, "_ENABLE_MEDIA_TYPE_SCOPING", True)

    recorded: Dict[str, Any] = {}

    def fake_canonical(media_types=None):
        recorded["media_types"] = media_types
        return ["Drama"]

    monkeypatch.setattr(llm_parser, "canonical_genres", fake_canonical)
    llm_parser._build_prompt_text(
        "light sci-fi tv series", {"user_id": "u"}, {"tv": [5]}
    )
    assert recorded["media_types"] == ["tv"]


def test_parse_intent_returns_ann_description_when_enabled(monkeypatch):
    monkeypatch.setattr(llm_parser, "_ENABLE_ANN_DESCRIPTION", True)
    settings = IntentParserSettings(
        provider="openai",
        api_key="key",
        model="gpt-test",
        endpoint="https://example.com",
        enabled=True,
        timeout=2.0,
    )
    monkeypatch.setattr(llm_parser, "_get_settings", lambda: settings)

    def fake_call(settings, query, user_context, linked_entities):
        return {
            "include_genres": ["Science Fiction"],
            "ann_description": "A dark dystopian TV series set in a controlled future society.",
        }

    monkeypatch.setattr(llm_parser, "_call_openai_parser", fake_call)

    intent = parse_intent("dystopian series", {"user_id": "u42"})
    assert intent.include_genres == ["Science Fiction"]
    assert (
        intent.ann_description
        == "A dark dystopian TV series set in a controlled future society."
    )


def test_load_fallback_rules_handles_missing_and_invalid(monkeypatch, tmp_path, caplog):
    llm_parser._load_fallback_rules.cache_clear()
    missing = tmp_path / "missing.json"
    monkeypatch.setenv("INTENT_FALLBACKS_PATH", str(missing))
    assert llm_parser._load_fallback_rules() == tuple()

    llm_parser._load_fallback_rules.cache_clear()
    bad = tmp_path / "bad.json"
    bad.write_text("not-json", encoding="utf-8")
    monkeypatch.setenv("INTENT_FALLBACKS_PATH", str(bad))
    caplog.set_level(logging.WARNING, logger=llm_parser.__name__)
    assert llm_parser._load_fallback_rules() == tuple()
    assert "Failed to load fallback rules" in caplog.text
    llm_parser._load_fallback_rules.cache_clear()


def test_load_fallback_rules_parses_entries(monkeypatch, tmp_path):
    llm_parser._load_fallback_rules.cache_clear()
    path = tmp_path / "fallbacks.json"
    payload = [
        {
            "keywords": ["No gore", "PG-13", ""],
            "include_genres": ["Drama", 123],
            "exclude_genres": ["Horror", None],
            "maturity_rating_max": 13,
        },
        "ignore-me",
        {"keywords": []},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("INTENT_FALLBACKS_PATH", str(path))

    rules = llm_parser._load_fallback_rules()
    assert len(rules) == 1
    rule = rules[0]
    assert "no gore" in rule.keywords and "pg-13" in rule.keywords
    assert rule.include_genres == ["Drama"]
    assert rule.exclude_genres == ["Horror"]
    assert rule.maturity_rating_max == "13"

    llm_parser._load_fallback_rules.cache_clear()


def test_get_settings_invalid_timeout_warns(monkeypatch, caplog):
    monkeypatch.setenv("INTENT_API_KEY", "token")
    monkeypatch.setenv("INTENT_TIMEOUT", "oops")
    llm_parser._get_settings.cache_clear()

    caplog.set_level(logging.WARNING, logger=llm_parser.__name__)
    settings = llm_parser._get_settings()
    assert settings.timeout == 12.0
    assert "Invalid INTENT_TIMEOUT" in caplog.text

    llm_parser._get_settings.cache_clear()


def test_default_rewrite_returns_new_instance():
    first = llm_parser.default_rewrite()
    second = llm_parser.default_rewrite()
    assert first is not second
    assert first == second


def test_normalize_llm_output_merges_fragments():
    assert llm_parser._normalize_llm_output(None) is None
    assert llm_parser._normalize_llm_output("not a collection") is None

    merged = llm_parser._normalize_llm_output(
        [
            {"include_genres": ["Drama"], "streaming_providers": []},
            {
                "include_genres": ["Comedy"],
                "exclude_genres": ["Horror"],
                "maturity_rating_max": "PG",
            },
            {"include_genres": ["Drama"], "ann_description": None},
            {"languages": "en"},
            {"media_types": None},
            {"include_genres": []},
            "ignored",
        ]
    )

    assert merged["include_genres"] == ["Drama", "Comedy"]
    assert merged["exclude_genres"] == ["Horror"]
    assert merged["languages"] == "en"
    assert "maturity_rating_max" in merged


def test_infer_media_types_deduplicates(monkeypatch):
    monkeypatch.setattr(
        llm_parser,
        "legacy_parse_intent",
        lambda query: SimpleNamespace(media_types=["tv", "movie", "tv"]),
    )
    hints = llm_parser._infer_media_types_for_prompt("something movie", {"tv": [1]})
    assert hints == ["tv", "movie"]


def test_parse_intent_uses_gemini(monkeypatch):
    settings = IntentParserSettings(
        provider="gemini",
        api_key="key",
        model="gemini-model",
        endpoint="https://example.com",
        enabled=True,
        timeout=2.0,
    )
    monkeypatch.setattr(llm_parser, "_get_settings", lambda: settings)
    monkeypatch.setattr(
        llm_parser,
        "_call_gemini_parser",
        lambda *a, **k: {"include_genres": ["Drama"]},
    )
    intent = parse_intent("gemini request", {"user_id": "ug"})
    assert intent.include_genres == ["Drama"]


def test_parse_intent_validation_error_returns_default(monkeypatch, caplog):
    settings = IntentParserSettings(
        provider="openai",
        api_key="key",
        model="gpt-test",
        endpoint="https://example.com",
        enabled=True,
        timeout=2.0,
    )
    monkeypatch.setattr(llm_parser, "_get_settings", lambda: settings)

    def fake_call(*args, **kwargs):
        return {"media_types": "not-a-list"}

    monkeypatch.setattr(llm_parser, "_call_openai_parser", fake_call)

    caplog.set_level(logging.DEBUG, logger=llm_parser.__name__)
    intent = parse_intent("broken payload", {"user_id": "u"})
    assert intent == DEFAULT_INTENT
    assert "failed validation" in caplog.text


def test_call_openai_parser_requires_api_key():
    settings = IntentParserSettings(
        provider="openai",
        api_key="",
        model="gpt-test",
        endpoint="https://example.com",
        enabled=True,
        timeout=1.0,
    )
    with pytest.raises(llm_parser.IntentParserError):
        llm_parser._call_openai_parser(settings, "query", {}, None)


def test_call_openai_parser_http_error(monkeypatch):
    settings = IntentParserSettings(
        provider="openai",
        api_key="key",
        model="gpt-test",
        endpoint="https://example.com",
        enabled=True,
        timeout=1.0,
    )

    request = httpx.Request("POST", "https://example.com")
    response = httpx.Response(502, request=request)

    class DummyResponse:
        def __init__(self):
            self.request = request

        def json(self):
            return {}

        def raise_for_status(self):
            raise httpx.HTTPStatusError("boom", request=request, response=response)

    class DummyClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return DummyResponse()

    monkeypatch.setattr(llm_parser.httpx, "Client", lambda *a, **k: DummyClient())

    with pytest.raises(llm_parser.IntentParserError, match="OpenAI API responded"):
        llm_parser._call_openai_parser(settings, "query", {}, None)


def test_call_openai_parser_filters_empty_messages(monkeypatch):
    settings = IntentParserSettings(
        provider="openai",
        api_key="key",
        model="gpt-test",
        endpoint="https://example.com",
        enabled=True,
        timeout=1.0,
    )
    monkeypatch.setenv("OPENAI_PROJECT", "proj")

    def fake_payload(settings, query, user_context, linked_entities):
        return {
            "messages": [
                {"role": "system", "content": None},
                {"role": "user", "content": "hello"},
            ],
        }

    class DummyResponse:
        status_code = 200

        def __init__(self):
            self.request = SimpleNamespace(
                copy_with=lambda **_: "https://example.com/responses"
            )

        def json(self):
            return {
                "output": [
                    {"content": [{"text": json.dumps({"include_genres": ["Drama"]})}]}
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

    monkeypatch.setattr(llm_parser, "_build_llm_payload", fake_payload)
    monkeypatch.setattr(llm_parser.httpx, "Client", lambda *a, **k: DummyClient())

    payload = llm_parser._call_openai_parser(settings, "query", {}, None)
    assert payload == {"include_genres": ["Drama"]}


def test_call_openai_parser_bad_response_shapes(monkeypatch):
    settings = IntentParserSettings(
        provider="openai",
        api_key="key",
        model="gpt-test",
        endpoint="https://example.com",
        enabled=True,
        timeout=1.0,
    )
    monkeypatch.setenv("OPENAI_PROJECT", "proj")

    def fake_payload(settings, query, user_context, linked_entities):
        return {"messages": [{"role": "user", "content": "hi"}]}

    class EmptyResponse:
        status_code = 200

        def __init__(self):
            self.request = SimpleNamespace(
                copy_with=lambda **_: "https://example.com/responses"
            )

        def json(self):
            return {"output": []}

        def raise_for_status(self):
            return None

    class ResponsesBadJsonResponse:
        status_code = 200

        def __init__(self):
            self.request = SimpleNamespace(
                copy_with=lambda **_: "https://example.com/responses"
            )

        def json(self):
            return {"output": [{"content": [{"text": "not-json"}]}]}

        def raise_for_status(self):
            return None

    class StandardBadJsonResponse:
        status_code = 200

        def __init__(self):
            self.request = SimpleNamespace(url="https://example.com")

        def json(self):
            return {"choices": [{"message": {"content": "not-json"}}]}

        def raise_for_status(self):
            return None

    def make_client(response):
        class DummyClient:
            def __init__(self, resp):
                self._resp = resp

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, *args, **kwargs):
                return self._resp

        return DummyClient(response)

    monkeypatch.setattr(llm_parser, "_build_llm_payload", fake_payload)

    monkeypatch.setattr(
        llm_parser.httpx, "Client", lambda *a, **k: make_client(EmptyResponse())
    )
    with pytest.raises(
        llm_parser.IntentParserError, match="Unexpected response structure"
    ):
        llm_parser._call_openai_parser(settings, "query", {}, None)

    monkeypatch.setattr(
        llm_parser.httpx,
        "Client",
        lambda *a, **k: make_client(ResponsesBadJsonResponse()),
    )
    with pytest.raises(llm_parser.IntentParserError, match="non-JSON content"):
        llm_parser._call_openai_parser(settings, "query", {}, None)

    monkeypatch.delenv("OPENAI_PROJECT", raising=False)
    monkeypatch.setattr(
        llm_parser.httpx,
        "Client",
        lambda *a, **k: make_client(StandardBadJsonResponse()),
    )
    with pytest.raises(llm_parser.IntentParserError, match="non-JSON content"):
        llm_parser._call_openai_parser(settings, "query", {}, None)


def test_call_gemini_parser_missing_api_key():
    settings = IntentParserSettings(
        provider="gemini",
        api_key="",
        model="gemini-model",
        endpoint="https://example.com",
        enabled=True,
        timeout=1.0,
    )
    with pytest.raises(llm_parser.IntentParserError):
        llm_parser._call_gemini_parser(settings, "query", {}, None)


def test_call_gemini_parser_error_paths(monkeypatch):
    settings = IntentParserSettings(
        provider="gemini",
        api_key="key",
        model="gemini-model",
        endpoint="https://example.com",
        enabled=True,
        timeout=1.0,
    )

    class ErrorResponse:
        def __init__(self, status_code, json_payload):
            self.status_code = status_code
            self._json = json_payload
            request_url = "https://example.com"
            self.request = SimpleNamespace(
                url=SimpleNamespace(copy_with=lambda **_: request_url)
            )

        def json(self):
            return self._json

        def raise_for_status(self):
            if self.status_code >= 500:
                raise httpx.HTTPStatusError(
                    "failure",
                    request=httpx.Request("POST", "https://example.com"),
                    response=httpx.Response(self.status_code),
                )

    class DummyClient:
        def __init__(self, responses):
            self._responses = responses
            self._index = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            resp = self._responses[self._index]
            self._index += 1
            return resp

    # HTTP 404 should log and raise.
    bad_responses = [
        ErrorResponse(404, {}),
    ]
    monkeypatch.setattr(
        llm_parser.httpx, "Client", lambda *a, **k: DummyClient(bad_responses)
    )
    with pytest.raises(llm_parser.IntentParserError, match="Gemini API responded"):
        llm_parser._call_gemini_parser(settings, "query", {}, None)

    # Empty candidates returns {}.
    ok_response = ErrorResponse(200, {"candidates": []})
    monkeypatch.setattr(
        llm_parser.httpx, "Client", lambda *a, **k: DummyClient([ok_response])
    )
    assert llm_parser._call_gemini_parser(settings, "query", {}, None) == {}

    # Content list with non-dict entries triggers fallback parts branch and returns {}.
    list_response = ErrorResponse(
        200,
        {
            "candidates": [
                {
                    "content": ["ignored", "still ignored"],
                }
            ]
        },
    )
    monkeypatch.setattr(
        llm_parser.httpx, "Client", lambda *a, **k: DummyClient([list_response])
    )
    assert llm_parser._call_gemini_parser(settings, "query", {}, None) == {}

    # Content list path with JSON decode error.
    bad_json_resp = ErrorResponse(
        200,
        {
            "candidates": [
                {
                    "content": [
                        {"text": "  "},
                        {"text": "not-json"},
                    ]
                }
            ]
        },
    )
    monkeypatch.setattr(
        llm_parser.httpx, "Client", lambda *a, **k: DummyClient([bad_json_resp])
    )
    with pytest.raises(llm_parser.IntentParserError, match="non-JSON content"):
        llm_parser._call_gemini_parser(settings, "query", {}, None)

    direct_endpoint_settings = IntentParserSettings(
        provider="gemini",
        api_key="key",
        model="gemini-model",
        endpoint="https://example.com/models:generateContent",
        enabled=True,
        timeout=1.0,
    )
    success_resp = ErrorResponse(
        200,
        {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": json.dumps({"include_genres": ["Drama"]})}]
                    }
                }
            ]
        },
    )
    monkeypatch.setattr(
        llm_parser.httpx, "Client", lambda *a, **k: DummyClient([success_resp])
    )
    payload = llm_parser._call_gemini_parser(
        direct_endpoint_settings, "query", {}, None
    )
    assert payload == {"include_genres": ["Drama"]}


def test_build_prompt_text_ann_description(monkeypatch):
    monkeypatch.setattr(llm_parser, "_ENABLE_ANN_DESCRIPTION", True, raising=False)

    def fake_prompt(name):
        assert name == "intent_parser"
        return {"system_prompt": "base prompt", "examples": []}

    monkeypatch.setattr(llm_parser, "load_prompt_template", fake_prompt)
    monkeypatch.setattr(
        llm_parser, "canonical_genres", lambda media_types=None: ["Drama"]
    )
    system_prompt, user_prompt = llm_parser._build_prompt_text("query", {}, None)
    assert "When possible, populate `ann_description`" in system_prompt
    assert "Allowed catalog genres" in system_prompt
    assert "Query: query" in user_prompt


def test_build_prompt_text_without_system_prompt(monkeypatch):
    monkeypatch.setattr(llm_parser, "_ENABLE_ANN_DESCRIPTION", False, raising=False)
    monkeypatch.setattr(
        llm_parser,
        "load_prompt_template",
        lambda name: {"system_prompt": "", "examples": []},
    )
    monkeypatch.setattr(
        llm_parser, "canonical_genres", lambda media_types=None: ["Action", "Drama"]
    )
    system_prompt, _ = llm_parser._build_prompt_text("query", {}, None)
    assert system_prompt.startswith("Allowed catalog genres")


def test_offline_intent_stub_handles_blank_and_legacy(monkeypatch):
    assert llm_parser._offline_intent_stub("   ") == {}

    legacy_filters = SimpleNamespace(
        genres=["Drama"],
        media_types=["movie"],
        min_runtime=95,
        max_runtime=120,
        maturity_rating_max="PG-13",
    )
    monkeypatch.setattr(llm_parser, "legacy_parse_intent", lambda _: legacy_filters)

    payload = llm_parser._offline_intent_stub("Drama movies from 90s")
    assert payload["include_genres"] == ["Drama"]
    assert payload["runtime_minutes_min"] == 95
    assert payload["runtime_minutes_max"] == 120
    assert payload["maturity_rating_max"] == "PG-13"
    assert payload["media_types"] == ["movie"]


def test_rewrite_query_truncates_long_queries():
    long_query = "one two three four five six seven eight nine ten"
    rewrite = rewrite_query(long_query, Intent())
    assert rewrite.rewritten_text == "one two three four five six seven eight"


def test_rewrite_query_truncates_ann_description(monkeypatch):
    description = "one two three four five six seven eight nine ten"
    intent = Intent(ann_description=description)
    rewrite = rewrite_query("ignored", intent)
    assert rewrite.rewritten_text == "one two three four five six seven eight"


def test_log_metrics_emits_when_debug(caplog):
    caplog.set_level(logging.DEBUG, logger=llm_parser.__name__)
    llm_parser._log_metrics(
        {"hits": 1, "misses": 2, "rewrite_hits": 0, "rewrite_misses": 3},
        cache="intent",
        event="store",
    )
    assert "LLM intent cache metrics" in caplog.text
