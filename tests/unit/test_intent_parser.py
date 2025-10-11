from __future__ import annotations

from types import SimpleNamespace

from api.core import intent_parser
from api.core.intent_parser import parse_intent, item_matches_intent


def test_parse_intent_extracts_genre_mood_and_runtime():
    intent = parse_intent("light sci-fi < 2h")
    assert "Science Fiction" in intent.genres
    assert "light" in intent.moods
    assert intent.max_runtime == 120
    assert intent.min_runtime is None
    assert intent.has_filters()

    item = SimpleNamespace(
        media_type="movie",
        runtime=110,
        genres=[{"name": "Science Fiction"}],
    )
    assert item_matches_intent(item, intent) is True

    too_long = SimpleNamespace(
        media_type="movie",
        runtime=140,
        genres=[{"name": "Science Fiction"}],
    )
    assert item_matches_intent(too_long, intent) is False


def test_parse_intent_supports_media_type_and_min_runtime():
    intent = parse_intent("over 3 hours epic drama film")
    assert intent.min_runtime == 180
    assert intent.max_runtime is None
    assert "Drama" in intent.genres
    assert "movie" in intent.media_types
    assert intent.has_filters()

    movie = SimpleNamespace(
        media_type="movie",
        runtime=200,
        genres=[{"name": "Drama"}],
    )
    assert item_matches_intent(movie, intent) is True

    short_movie = SimpleNamespace(
        media_type="movie",
        runtime=150,
        genres=[{"name": "Drama"}],
    )
    assert item_matches_intent(short_movie, intent) is False


def test_mood_only_query_adds_fallback_genres():
    intent = parse_intent("feel-good show")
    # Should infer mood -> effective genres (Comedy/Family)
    assert "light" in intent.moods
    assert "tv" in intent.media_types
    effective = intent.effective_genres()
    assert "Comedy" in effective
    assert intent.has_filters()


def test_item_matches_intent_handles_media_types_and_varied_genre_structures():
    intent = parse_intent("romantic comedy series under 45 minutes")
    assert "tv" in intent.media_types
    matcher = SimpleNamespace(
        media_type="tv",
        runtime=40,
        genres=["Romance", "Comedy"],
    )
    assert item_matches_intent(matcher, intent) is True

    wrong_media = SimpleNamespace(
        media_type="movie",
        runtime=40,
        genres=["Romance"],
    )
    assert item_matches_intent(wrong_media, intent) is False

    dict_genre = SimpleNamespace(
        media_type="tv",
        runtime=40,
        genres={"name": "Drama"},
    )
    assert item_matches_intent(dict_genre, intent) is False

    string_genre = SimpleNamespace(
        media_type="tv",
        runtime=40,
        genres="Comedy",
    )
    assert item_matches_intent(string_genre, intent) is True


def test_runtime_parsing_handles_between_and_inverted_bounds():
    intent = parse_intent("between 1.5 hours and 2 hours")
    assert intent.min_runtime == 90
    assert intent.max_runtime == 120

    inverted = parse_intent("over 120 minutes and under 90 minutes")
    # swap occurs when bounds invert; the max bound should take precedence
    assert inverted.min_runtime == 90
    assert inverted.max_runtime == 120


def test_invalid_runtime_tokens_are_ignored():
    intent = parse_intent("under 0 minutes and minimum ninety minutes")
    # 'under 0 minutes' is discarded because it results in None
    assert intent.max_runtime is None
    # 'minimum ninety minutes' cannot be parsed (non-numeric) so ignored
    assert intent.min_runtime is None


def test_helper_functions_handle_empty_inputs_and_invalid_minutes():
    assert intent_parser._extract_genres("") == []
    assert intent_parser._extract_moods("") == []
    assert intent_parser._extract_media_types("") == []
    assert intent_parser._to_minutes("not_a_number", "min") is None
