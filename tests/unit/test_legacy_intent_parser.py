from __future__ import annotations

from types import SimpleNamespace


from api.core import legacy_intent_parser as legacy


def test_parse_intent_extracts_expected_filters():
    filters = legacy.parse_intent("Light sci-fi series under 45 minutes")
    assert "Science Fiction" in filters.genres
    assert "light" in filters.moods
    assert "tv" in filters.media_types
    assert filters.max_runtime == 45
    assert filters.min_runtime is None


def test_parse_intent_between_runtime(monkeypatch):
    filters = legacy.parse_intent("between 1.5 hours and 2 hours")
    assert filters.min_runtime == 90
    assert filters.max_runtime == 120

    filters = legacy.parse_intent("over 120 minutes and under 90 minutes")
    assert filters.min_runtime == 90
    assert filters.max_runtime == 120


def test_effective_genres_include_mood_mapping(monkeypatch):
    monkeypatch.setattr(legacy, "_get_genre_mapping", lambda: {"light": ["FeelGood"]})
    filters = legacy.IntentFilters(raw_query="feel good", moods=["light"], genres=[])
    genres = filters.effective_genres()
    assert "Comedy" in genres  # from legacy mood mapping
    assert "FeelGood" in genres  # from prompt mapping


def test_item_matches_intent_handles_constraints():
    filters = legacy.IntentFilters(
        raw_query="",
        genres=["Science Fiction"],
        media_types=["tv"],
        min_runtime=30,
        max_runtime=60,
        maturity_rating_max="TV-14",
    )

    good_item = SimpleNamespace(
        media_type="tv",
        runtime=45,
        genres=[{"name": "Science Fiction"}],
        maturity_rating="PG-13",
    )
    assert legacy.item_matches_intent(good_item, filters) is True

    wrong_media = SimpleNamespace(
        media_type="movie",
        runtime=45,
        genres=[{"name": "Science Fiction"}],
        maturity_rating="PG-13",
    )
    assert legacy.item_matches_intent(wrong_media, filters) is False

    wrong_runtime = SimpleNamespace(
        media_type="tv",
        runtime=10,
        genres=[{"name": "Science Fiction"}],
        maturity_rating="PG-13",
    )
    assert legacy.item_matches_intent(wrong_runtime, filters) is False

    wrong_genre = SimpleNamespace(
        media_type="tv",
        runtime=45,
        genres=[{"name": "Comedy"}],
        maturity_rating="PG-13",
    )
    assert legacy.item_matches_intent(wrong_genre, filters) is False


def test_item_matches_intent_blocks_maturity_overflow():
    filters = legacy.IntentFilters(
        raw_query="",
        maturity_rating_max="PG",
    )

    allowed_item = SimpleNamespace(maturity_rating="G")
    blocked_item = SimpleNamespace(maturity_rating="TV-MA")
    unknown_item = SimpleNamespace(maturity_rating=None)

    assert legacy.item_matches_intent(allowed_item, filters) is True
    assert legacy.item_matches_intent(blocked_item, filters) is False
    assert legacy.item_matches_intent(unknown_item, filters) is False


def test_parse_intent_empty_query_returns_empty_filters():
    filters = legacy.parse_intent(None)
    assert filters.genres == []
    assert filters.media_types == []
    assert filters.moods == []
    assert filters.min_runtime is None and filters.max_runtime is None
