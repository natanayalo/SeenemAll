from __future__ import annotations

import logging
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


def test_parse_intent_conflicting_signals_prioritises_explicit_genres():
    filters = legacy.parse_intent(
        "light horror movie under 2 hours but over 3 hours in runtime"
    )
    assert filters.media_types == ["movie"]
    assert filters.genres[0] == "Horror"
    assert "light" in filters.moods
    assert filters.min_runtime == 120
    assert filters.max_runtime == 180


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


def test_canonical_genres_uses_cache(monkeypatch):
    calls = {"count": 0}

    def fake_loader(media_types):
        calls["count"] += 1
        return ["Drama", "Science Fiction"]

    monkeypatch.setattr(legacy, "_load_genres_from_db", fake_loader)
    monkeypatch.setattr(legacy, "_GENRE_CACHE", {}, raising=False)
    result1 = legacy.canonical_genres()
    result2 = legacy.canonical_genres()
    assert result1 == ["Drama", "Science Fiction"]
    assert result2 == ["Drama", "Science Fiction"]
    assert calls["count"] == 1


def test_canonical_genres_cache_respects_ttl(monkeypatch):
    calls = {"count": 0}

    def fake_loader(media_types):
        calls["count"] += 1
        return ["Drama"]

    monkeypatch.setattr(legacy, "_load_genres_from_db", fake_loader)
    monkeypatch.setattr(legacy, "_GENRE_CACHE", {}, raising=False)
    monkeypatch.setattr(legacy, "_GENRE_CACHE_TTL_SECONDS", 0.0, raising=False)
    legacy.canonical_genres()
    legacy.canonical_genres()
    assert calls["count"] == 2


def test_canonical_genres_falls_back_on_failure(monkeypatch):
    def fake_loader(media_types):
        return None

    monkeypatch.setattr(legacy, "_load_genres_from_db", fake_loader)
    monkeypatch.setattr(legacy, "_GENRE_CACHE", {}, raising=False)
    result = legacy.canonical_genres(["tv"])
    assert result == list(legacy._CANONICAL_GENRES)


def test_parse_intent_debug_evaluation(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger=legacy.__name__)

    monkeypatch.setattr(
        legacy,
        "_get_prompt_config",
        lambda: {
            "genre_mapping": {},
            "examples": [
                {
                    "input": "debug movie query",
                    "expected_output": {
                        "media_types": ["movie"],
                        "genres": ["Drama"],
                        "min_runtime": 90,
                        "max_runtime": 120,
                    },
                }
            ],
        },
    )

    calls = {}

    def fake_evaluate(output, expected, query):
        calls["args"] = (output, expected, query)
        return 0.75, ["issue"]

    monkeypatch.setattr(
        "api.core.prompt_eval.evaluate_intent_parser_output", fake_evaluate
    )

    query = "debug movie query"
    filters = legacy.parse_intent(query)
    assert filters.media_types == ["movie"]
    assert calls["args"][2] == query
    assert "Intent parser evaluation score" in caplog.text


def test_load_genres_from_db_filters_and_logs(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger=legacy.__name__)

    rows = [
        ("tv", [{"name": "Drama"}, {"name": "science fiction"}]),
        ("tv", [{"name": "Drama"}]),
    ]

    class DummyResult:
        def __init__(self, data):
            self._data = list(data)
            self.closed = False

        def __iter__(self):
            return iter(self._data)

        def close(self):
            self.closed = True

    class DummySession:
        def __init__(self):
            self.closed = False

        def execute(self, stmt):
            self.stmt = stmt
            return DummyResult(rows)

        def close(self):
            self.closed = True

    monkeypatch.setattr(legacy, "_GENRE_CACHE", {}, raising=False)
    monkeypatch.setattr(legacy, "_GENRE_CACHE_TTL_SECONDS", 0.0, raising=False)
    monkeypatch.setattr(legacy, "get_sessionmaker", lambda: lambda: DummySession())

    genres = legacy._load_genres_from_db(["tv"])
    assert genres == ["Drama", "science fiction"]
    # Subsequent call should repopulate cache due to TTL override.
    genres_second = legacy.canonical_genres(["tv"])
    assert genres_second == ["Drama", "science fiction"]
    assert "Loaded" in caplog.text or "Falling back" in caplog.text


def test_iterate_and_item_genre_helpers():
    payload = [{"name": "Drama"}, {"name": "Comedy"}, "Thriller"]
    names = list(legacy._iterate_genre_names(payload))
    assert names == ["Drama", "Comedy", "Thriller"]

    class ItemWithGenres:
        genres = payload

    extracted = legacy._item_genre_names(ItemWithGenres())
    assert extracted == {"drama", "comedy", "thriller"}

    single = legacy._item_genre_names(SimpleNamespace(genres={"name": "Mystery"}))
    assert single == {"mystery"}
    direct = legacy._item_genre_names(SimpleNamespace(genres="Action"))
    assert direct == {"action"}


def test_runtime_utilities_handle_edge_cases():
    assert legacy._to_minutes("abc", "m") is None
    assert legacy._to_minutes("0", "minutes") is None

    max_only = legacy._extract_runtime("shorter than 0 minutes")
    assert max_only == (None, None)

    min_only = legacy._extract_runtime("longer than 0 minutes")
    assert min_only == (None, None)
