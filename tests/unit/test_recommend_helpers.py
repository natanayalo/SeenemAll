from __future__ import annotations

import importlib.util
import sys

from api.routes import recommend as recommend_routes
from api.core.intent_parser import Intent
from api.core.legacy_intent_parser import IntentFilters
from api.core.filter_matcher import QueryFiltersResult


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
    assert merged.genres[0] == "Science Fiction"
    assert "Drama" in merged.genres
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


def test_filter_excluded_candidate_ids_removes_matches():
    exclude_set = {2}
    assert recommend_routes._filter_excluded_candidate_ids([1, 2, 3], exclude_set) == [
        1,
        3,
    ]
    assert recommend_routes._filter_excluded_candidate_ids([], exclude_set) == []
    assert recommend_routes._filter_excluded_candidate_ids([4], set()) == [4]


def test_prioritize_boosted_items_moves_priority_first():
    items = [{"id": 1}, {"id": 2}, {"id": 3}]
    reordered = recommend_routes._prioritize_boosted_items(items, [3, 1])
    assert [item["id"] for item in reordered] == [3, 1, 2]


def _mock_media_genres(monkeypatch, mapping):
    monkeypatch.setattr(
        recommend_routes, "_load_media_genres", lambda db: mapping, raising=False
    )


def test_strict_required_genres_prefers_custom_and_legacy(monkeypatch):
    _mock_media_genres(
        monkeypatch, {"movie": {"Animation", "Science Fiction"}, "tv": set()}
    )
    intent = IntentFilters(
        raw_query="", genres=["Science Fiction", "Animation"], media_types=["movie"]
    )
    legacy = IntentFilters(
        raw_query="", genres=["Animation", "Science Fiction"], media_types=["movie"]
    )
    strict = recommend_routes._strict_required_genres(
        object(), ["Animation"], legacy, intent
    )
    assert strict == ["Animation", "Science Fiction"]


def test_strict_required_genres_falls_back_to_intent(monkeypatch):
    _mock_media_genres(monkeypatch, {"movie": {"Comedy"}})
    intent = IntentFilters(raw_query="", genres=["Comedy"])
    strict = recommend_routes._strict_required_genres(object(), [], None, intent)
    assert strict == ["Comedy"]


def test_strict_required_genres_keeps_tv_aliases(monkeypatch):
    _mock_media_genres(
        monkeypatch,
        {"tv": {"Sci-Fi & Fantasy", "Animation"}, "movie": {"Science Fiction"}},
    )
    intent = IntentFilters(
        raw_query="", genres=["Science Fiction", "Animation"], media_types=["tv"]
    )
    strict = recommend_routes._strict_required_genres(object(), [], intent, intent)
    assert strict == ["Sci-Fi & Fantasy", "Animation"]


def test_merge_query_filter_hints_fallback_for_soft_keywords(monkeypatch):
    # Test case for "best science fiction movies"
    intent = IntentFilters(
        raw_query="best science fiction movies", genres=["Science Fiction"]
    )
    query_filters = QueryFiltersResult(
        languages=(),
        keywords=("best", "science fiction"),
        genres=("Science Fiction",),
        media_types=("movie",),
        cast=(),
        directors=(),
        producers=(),
        writers=(),
        residual_text="best science fiction movies",
    )

    mock_db = object()
    monkeypatch.setattr(
        recommend_routes,
        "_get_top_query_keywords",
        lambda db: {"best", "top", "epic"},
    )
    monkeypatch.setattr(
        recommend_routes.llm_parser,
        "_normalize_genre_names",
        lambda genres: [g.lower() for g in genres],
    )

    recommend_routes._merge_query_filter_hints(intent, query_filters, mock_db)

    assert "best" in intent.keywords
    assert "science fiction" in intent.keywords
    assert "science fiction" in intent.genre_keywords


def test_merge_query_filter_hints_preserves_hard_keywords(monkeypatch):
    # Test case for "dark cyberpunk sci-fi movies"
    intent = IntentFilters(raw_query="dark cyberpunk sci-fi movies", genres=[])
    query_filters = QueryFiltersResult(
        languages=(),
        keywords=("dark", "cyberpunk", "sci-fi"),
        genres=("Sci-Fi",),
        media_types=("movie",),
        cast=(),
        directors=(),
        producers=(),
        writers=(),
        residual_text="dark cyberpunk sci-fi movies",
    )

    mock_db = object()
    monkeypatch.setattr(
        recommend_routes,
        "_get_top_query_keywords",
        lambda db: {"best", "top", "epic"},
    )

    # Simulate 'sci-fi' normalizing to 'science fiction'
    def mock_normalize(genres):
        normalized = []
        for g in genres:
            if g == "sci-fi":
                normalized.append("science fiction")
            else:
                normalized.append(g.lower())
        return normalized

    monkeypatch.setattr(
        recommend_routes.llm_parser,
        "_normalize_genre_names",
        mock_normalize,
    )

    # The intent's genres should already be normalized before the call
    intent.genres = ["science fiction"]

    recommend_routes._merge_query_filter_hints(intent, query_filters, mock_db)

    assert "dark" in intent.keywords
    assert "cyberpunk" in intent.keywords
    assert "sci-fi" not in intent.keywords
    assert "science fiction" not in intent.keywords
    assert "sci-fi" in intent.genre_keywords
