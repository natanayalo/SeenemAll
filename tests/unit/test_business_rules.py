from __future__ import annotations

import json
from typing import Dict, List

import pytest

from api.core import business_rules
from api.core.legacy_intent_parser import IntentFilters


def _make_item(
    item_id: int,
    score: float,
    genres: List[str] | None = None,
    vote_count: int | None = None,
    release_year: int | None = None,
    tmdb_id: int | None = None,
    media_type: str = "movie",
) -> Dict[str, object]:
    return {
        "id": item_id,
        "tmdb_id": tmdb_id or item_id * 10,
        "retrieval_score": score,
        "original_rank": item_id,
        "genres": [{"name": g} for g in (genres or [])],
        "vote_count": vote_count,
        "release_year": release_year,
        "media_type": media_type,
    }


def test_apply_business_rules_filters_items(tmp_path, monkeypatch):
    config = {
        "filters": {
            "exclude_genres": ["Horror"],
            "min_vote_count": 50,
            "exclude_media_types": ["tv"],
        }
    }
    config_path = tmp_path / "rules.json"
    config_path.write_text(json.dumps(config))

    monkeypatch.setenv("BUSINESS_RULES_PATH", str(config_path))
    business_rules.clear_rules_cache()

    items = [
        _make_item(1, 0.9, genres=["Comedy"], vote_count=60),
        _make_item(2, 0.8, genres=["Horror"], vote_count=100),
        _make_item(3, 0.7, genres=["Comedy"], vote_count=10),
        _make_item(4, 0.6, genres=["Comedy"], vote_count=75, media_type="tv"),
    ]

    result = business_rules.apply_business_rules(items, intent=None)
    assert [item["id"] for item in result] == [1]


def test_apply_business_rules_boosts_genres(tmp_path, monkeypatch):
    config = {
        "boosts": {
            "genre_multipliers": {"Comedy": 2.0},
            "recent_release": {"year": 2020, "bonus": 0.1},
        }
    }
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(config))

    monkeypatch.setenv("BUSINESS_RULES_PATH", str(path))
    business_rules.clear_rules_cache()

    items = [
        _make_item(1, 0.5, genres=["Drama"], release_year=2019),
        _make_item(2, 0.4, genres=["Comedy"], release_year=2022),
    ]

    result = business_rules.apply_business_rules(items, intent=None)
    # Comedy item should be boosted above the drama item.
    assert [item["id"] for item in result] == [2, 1]
    boosted_item = result[0]
    assert boosted_item["retrieval_score"] > 0.4  # boosted


def test_apply_business_rules_respects_tmdb_boost(tmp_path, monkeypatch):
    config = {
        "boosts": {"tmdb_id_multipliers": {"100": 3.0}},
    }
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(config))

    monkeypatch.setenv("BUSINESS_RULES_PATH", str(path))
    business_rules.clear_rules_cache()

    items = [
        _make_item(1, 0.3, genres=["Comedy"], tmdb_id=100),
        _make_item(2, 0.4, genres=["Comedy"], tmdb_id=200),
    ]
    result = business_rules.apply_business_rules(items, intent=None)
    assert [item["id"] for item in result] == [1, 2]
    assert result[0]["retrieval_score"] > result[1]["retrieval_score"]


def test_apply_business_rules_with_intent_requirements(monkeypatch):
    intent = IntentFilters(
        raw_query="",
        genres=["Comedy"],
        moods=["light"],
        media_types=["movie"],
    )

    rules = {
        "filters": {
            "require_intent_genres": True,
            "include_genres": ["Comedy"],
            "exclude_ids": ["3"],
            "exclude_tmdb_ids": ["40"],
            "max_release_year": 2025,
        },
        "boosts": {
            "id_multipliers": {"1": 2.0},
            "popularity_multiplier": {"threshold": 50, "factor": 1.5},
        },
    }

    monkeypatch.setattr(business_rules, "load_rules", lambda: rules)

    items = [
        {
            "id": 1,
            "tmdb_id": 10,
            "retrieval_score": 0.5,
            "genres": [{"name": "Comedy"}],
            "media_type": "movie",
            "release_year": 2024,
            "popularity": 100,
        },
        {
            "id": 2,
            "tmdb_id": 20,
            "retrieval_score": 0.6,
            "genres": [{"name": "Drama"}],
            "media_type": "movie",
            "release_year": 2024,
            "popularity": 10,
        },
        {
            "id": 3,
            "tmdb_id": 30,
            "retrieval_score": 0.7,
            "genres": [{"name": "Comedy"}],
            "media_type": "movie",
            "release_year": 2024,
            "popularity": 10,
        },
        {
            "id": 4,
            "tmdb_id": 40,
            "retrieval_score": 0.8,
            "genres": [{"name": "Comedy"}],
            "media_type": "movie",
            "release_year": 2026,
            "popularity": 10,
        },
    ]

    result = business_rules.apply_business_rules(items, intent=intent)
    assert [item["id"] for item in result] == [1]
    assert result[0]["retrieval_score"] > 0.5


def test_load_rules_handles_missing_and_invalid_files(tmp_path, monkeypatch):
    missing_path = tmp_path / "missing.json"
    monkeypatch.setenv("BUSINESS_RULES_PATH", str(missing_path))
    business_rules.clear_rules_cache()

    rules = business_rules.load_rules()
    assert rules == {}

    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{not-json")
    monkeypatch.setenv("BUSINESS_RULES_PATH", str(bad_path))
    business_rules.clear_rules_cache()
    rules = business_rules.load_rules()
    assert rules == {}

    good_path = tmp_path / "good.json"
    good_path.write_text(json.dumps({"filters": {}}))
    monkeypatch.setenv("BUSINESS_RULES_PATH", str(good_path))
    business_rules.clear_rules_cache()
    first = business_rules.load_rules()
    assert first == {"filters": {}}
    # Second call should hit cache and not re-read file
    good_path.unlink()
    second = business_rules.load_rules()
    assert second == {}


def test_helper_sanitizers_handle_edge_cases():
    assert business_rules._safe_int(" 10 ") == 10
    assert business_rules._safe_int("bad") is None
    assert business_rules._safe_int(True) == 1
    assert business_rules._safe_float(" 1.5 ") == 1.5
    assert business_rules._safe_float("not-a-number") is None

    item = {
        "genres": [
            {"name": "Comedy"},
            {"name": "Drama"},
        ]
    }
    assert list(business_rules._genre_names(item)) == ["Comedy", "Drama"]

    multipliers = {"5": "2.0"}
    boosted = business_rules._lookup_multiplier(multipliers, {"id": 5})
    assert boosted == 2.0
    assert business_rules._lookup_multiplier({}, {"id": 10}) == 1.0


@pytest.fixture(autouse=True)
def _reset_rules_cache():
    yield
    business_rules.clear_rules_cache()


def test_get_rules_path_defaults(monkeypatch):
    monkeypatch.delenv("BUSINESS_RULES_PATH", raising=False)
    path = business_rules._get_rules_path()
    assert path.endswith("config/business_rules.json")


def test_load_rules_uses_cache(tmp_path, monkeypatch):
    path = tmp_path / "rules.json"
    payload = {"filters": {"exclude_genres": ["Drama"]}}
    path.write_text(json.dumps(payload))
    monkeypatch.setenv("BUSINESS_RULES_PATH", str(path))
    business_rules.clear_rules_cache()
    first = business_rules.load_rules()
    assert first == payload

    def boom(*args, **kwargs):
        raise RuntimeError("should not open")

    monkeypatch.setattr("builtins.open", boom)
    cached = business_rules.load_rules()
    assert cached == payload


def test_apply_business_rules_returns_empty_when_all_filtered(tmp_path, monkeypatch):
    config = {"filters": {"exclude_genres": ["Comedy"]}}
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("BUSINESS_RULES_PATH", str(path))
    business_rules.clear_rules_cache()

    items = [_make_item(1, 0.5, genres=["Comedy"])]
    assert business_rules.apply_business_rules(items, intent=None) == []
