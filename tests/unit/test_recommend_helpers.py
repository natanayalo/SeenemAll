from __future__ import annotations

import importlib.util
import sys

from api.routes import recommend as recommend_routes
from api.core.intent_parser import Intent
from api.core.legacy_intent_parser import IntentFilters


def test_parse_llm_intent_falls_back_on_error(monkeypatch):
    def boom(query, user_context, linked_entities=None):
        raise RuntimeError("fail")

    monkeypatch.setattr(recommend_routes.llm_parser, "parse_intent", boom)
    intent = recommend_routes._parse_llm_intent("hello", {"user_id": "u"})
    assert isinstance(intent, Intent)
    assert intent == recommend_routes.llm_parser.default_intent()


def test_merge_with_legacy_filters_merges_and_preserves(monkeypatch):
    primary = IntentFilters(
        raw_query="",
        genres=["Science Fiction"],
        moods=[],
        media_types=[],
        min_runtime=None,
        max_runtime=None,
        maturity_rating_max="PG-13",
    )
    fallback = IntentFilters(
        raw_query="",
        genres=["Drama", "Science Fiction"],
        moods=["dark"],
        media_types=["movie"],
        min_runtime=90,
        max_runtime=150,
        maturity_rating_max="PG",
    )

    merged = recommend_routes._merge_with_legacy_filters(primary, fallback)
    assert merged.genres == ["Science Fiction", "Drama"]
    assert merged.moods == ["dark"]
    assert merged.media_types == ["movie"]
    assert merged.min_runtime == 90
    assert merged.max_runtime == 150
    assert merged.maturity_rating_max == "PG"


def test_merge_maturity_rating_prefers_stricter(monkeypatch):
    primary = IntentFilters(raw_query="", maturity_rating_max="TV-14")
    fallback = IntentFilters(raw_query="", maturity_rating_max="PG")
    recommend_routes._merge_maturity_rating(primary, fallback)
    assert primary.maturity_rating_max == "PG"


def test_merge_maturity_rating_sets_when_primary_missing():
    primary = IntentFilters(raw_query="", maturity_rating_max=None)
    fallback = IntentFilters(raw_query="", maturity_rating_max="PG-13")
    recommend_routes._merge_maturity_rating(primary, fallback)
    assert primary.maturity_rating_max == "PG-13"


def test_merge_maturity_rating_ignores_unknown_levels():
    primary = IntentFilters(raw_query="", maturity_rating_max="PG-13")
    fallback = IntentFilters(raw_query="", maturity_rating_max="UNKNOWN")
    recommend_routes._merge_maturity_rating(primary, fallback)
    assert primary.maturity_rating_max == "PG-13"


def test_apply_franchise_cap_limits_duplicates():
    items = [
        {"id": 1, "collection_id": 10},
        {"id": 2, "collection_id": 10},
        {"id": 3, "collection_id": 10},
        {"id": 4, "collection_id": None},
    ]
    capped = recommend_routes._apply_franchise_cap(items, cap=2)
    assert [item["id"] for item in capped] == [1, 2, 4]


def test_serendipity_target_respects_disabled_ratio(monkeypatch):
    monkeypatch.setattr(recommend_routes, "_SERENDIPITY_RATIO", 0.0, raising=False)
    assert recommend_routes._serendipity_target(10) == 0


def test_apply_serendipity_slot_inserts_long_tail(monkeypatch):
    monkeypatch.setattr(recommend_routes, "_SERENDIPITY_RATIO", 0.2, raising=False)
    current = [
        {"id": 1, "original_rank": 0},
        {"id": 2, "original_rank": 1},
        {"id": 3, "original_rank": 2},
    ]
    pool = [
        {"id": 99, "original_rank": 5},
        {"id": 100, "original_rank": 6},
    ]
    updated = recommend_routes._apply_serendipity_slot(current, pool, limit=3)
    ids = [item["id"] for item in updated[:3]]
    assert 99 in ids or 100 in ids


def test_apply_serendipity_slot_returns_current_when_no_replacements(monkeypatch):
    monkeypatch.setattr(recommend_routes, "_SERENDIPITY_RATIO", 0.2, raising=False)
    current = [{"id": 1, "original_rank": 5}]
    pool = [{"id": 1, "original_rank": 6}]
    assert recommend_routes._apply_serendipity_slot(current, pool, limit=1) == current


def test_serendipity_ratio_env_branch(monkeypatch):
    monkeypatch.setenv("SERENDIPITY_RATIO", "0")
    path = recommend_routes.__file__
    module_name = "api.routes.recommend_serendipity_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    temp_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = temp_module
    try:
        assert spec.loader is not None
        spec.loader.exec_module(temp_module)  # type: ignore[arg-type]
        assert temp_module._SERENDIPITY_RATIO == 0.0
    finally:
        sys.modules.pop(module_name, None)
        monkeypatch.delenv("SERENDIPITY_RATIO", raising=False)


def test_is_long_tail_respects_limit():
    assert not recommend_routes._is_long_tail({"original_rank": 0}, limit=3)
    assert recommend_routes._is_long_tail({"original_rank": 3}, limit=3)
    assert not recommend_routes._is_long_tail({"original_rank": 5}, limit=0)


def test_merge_with_legacy_filters_returns_primary_when_fallback_missing():
    primary = IntentFilters(raw_query="", genres=["Action"])
    result = recommend_routes._merge_with_legacy_filters(primary, None)
    assert result is primary


def test_apply_franchise_cap_respects_disabled_cap():
    items = [{"id": 1, "collection_id": 5}]
    assert recommend_routes._apply_franchise_cap(items, cap=0) == items


def test_apply_serendipity_slot_returns_current_when_disabled(monkeypatch):
    monkeypatch.setattr(recommend_routes, "_SERENDIPITY_RATIO", 0.2, raising=False)
    current = [{"id": 1, "original_rank": 0}]
    assert recommend_routes._apply_serendipity_slot(current, [], limit=0) == current
