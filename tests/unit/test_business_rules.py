from __future__ import annotations

import json
from typing import Dict, List

import pytest

from api.core import business_rules


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


@pytest.fixture(autouse=True)
def _reset_rules_cache():
    yield
    business_rules.clear_rules_cache()
