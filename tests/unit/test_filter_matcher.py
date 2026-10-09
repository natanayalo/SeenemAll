from __future__ import annotations

from types import SimpleNamespace

from api.core.filter_matcher import QueryFilterMatcher, get_query_filters


class DummySession:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, stmt):
        return SimpleNamespace(all=lambda: self._rows)


def test_query_filter_matcher_extracts_entities(monkeypatch):
    rows = [
        (
            [{"name": "French", "english_name": "French", "iso_639_1": "fr"}],
            [{"name": "Feel-Good"}],
            [{"name": "Comedy"}],
            [{"name": "Tom Cruise"}],
            [{"name": "Christopher McQuarrie"}],
            [],
            [],
        )
    ]

    monkeypatch.setattr(
        "api.core.filter_matcher.get_sessionmaker",
        lambda: lambda: DummySession(rows),
    )

    matcher = QueryFilterMatcher()
    filters = matcher.match("top french comedy movies with Tom Cruise")

    assert list(filters.languages) == ["French"]
    assert list(filters.keywords) == []
    assert list(filters.genres) == ["Comedy"]
    assert list(filters.media_types) == ["movie"]
    assert list(filters.cast) == ["Tom Cruise"]
    assert "top" in filters.residual_text
    assert "french" not in filters.residual_text.lower()


def test_get_query_filters_handles_empty_query():
    result = get_query_filters(None)
    assert result.residual_text == ""
    assert not result.languages


def test_query_filter_matcher_extracts_reference_titles(monkeypatch):
    rows = [
        ([], [], [], [], [], [], []),
    ]

    monkeypatch.setattr(
        "api.core.filter_matcher.get_sessionmaker",
        lambda: lambda: DummySession(rows),
    )

    matcher = QueryFilterMatcher()
    filters = matcher.match("fantasy TV epics like The Witcher")

    assert list(filters.reference_titles) == ["The Witcher"]
    assert "witcher" not in filters.residual_text.lower()


def test_query_filter_matcher_keeps_keywords_in_residual(monkeypatch):
    rows = [
        (
            [],
            [{"name": "Classic Western"}],
            [],
            [],
            [],
            [],
            [],
        )
    ]

    monkeypatch.setattr(
        "api.core.filter_matcher.get_sessionmaker",
        lambda: lambda: DummySession(rows),
    )

    matcher = QueryFilterMatcher()
    filters = matcher.match("classic western movies")

    assert list(filters.keywords) == ["Classic Western"]
    assert "classic western" in filters.residual_text.lower()


def test_query_filter_matcher_extracts_genre_even_with_keyword_overlap(monkeypatch):
    rows = [
        (
            [],
            [{"name": "Classic Western"}],
            [{"name": "Western"}],
            [],
            [],
            [],
            [],
        )
    ]

    monkeypatch.setattr(
        "api.core.filter_matcher.get_sessionmaker",
        lambda: lambda: DummySession(rows),
    )

    matcher = QueryFilterMatcher()
    filters = matcher.match("classic western movies")

    assert list(filters.genres) == ["Western"]
    assert list(filters.keywords) == ["Classic Western"]


def test_query_filter_matcher_role_cues(monkeypatch):
    rows = [
        (
            [],
            [],
            [],
            [{"name": "Tom Cruise"}],
            [{"name": "Christopher McQuarrie"}],
            [{"name": "Tom Cruise"}],  # Also producer
            [],
        )
    ]

    monkeypatch.setattr(
        "api.core.filter_matcher.get_sessionmaker",
        lambda: lambda: DummySession(rows),
    )

    matcher = QueryFilterMatcher()

    # "starring" cue -> cast only, not producer
    filters_starring = matcher.match("action movie starring Tom Cruise")
    assert list(filters_starring.cast) == ["Tom Cruise"]
    assert not filters_starring.producers
    assert not filters_starring.directors

    # "directed by" cue -> directors only
    filters_dir = matcher.match("thriller directed by Christopher McQuarrie")
    assert list(filters_dir.directors) == ["Christopher McQuarrie"]
    assert not filters_dir.cast

    # "film with" cue -> cast
    filters_with = matcher.match("film with Tom Cruise")
    assert list(filters_with.cast) == ["Tom Cruise"]
    assert not filters_with.producers


def test_query_filter_matcher_comparative_people(monkeypatch):
    rows = [
        (
            [],
            [],
            [],
            [{"name": "Tom Cruise"}],
            [{"name": "Christopher McQuarrie"}],
            [],
            [],
        )
    ]

    monkeypatch.setattr(
        "api.core.filter_matcher.get_sessionmaker",
        lambda: lambda: DummySession(rows),
    )

    matcher = QueryFilterMatcher()

    # "like" cue -> reference_titles, NOT cast
    filters_like = matcher.match("action movies like Tom Cruise")
    assert not filters_like.cast
    assert not filters_like.directors
    assert list(filters_like.reference_titles) == ["Tom Cruise"]

    # "in the style of" cue -> reference_titles, NOT directors
    filters_style = matcher.match("movies in the style of Christopher McQuarrie")
    assert not filters_style.directors
    assert not filters_style.cast
    assert list(filters_style.reference_titles) == ["Christopher McQuarrie"]

    # Multi-person comparative conjunction
    filters_multi = matcher.match("movies like Tom Cruise and Christopher McQuarrie")
    assert not filters_multi.cast
    assert not filters_multi.directors
    assert set(filters_multi.reference_titles) == {
        "Tom Cruise",
        "Christopher McQuarrie",
    }


def test_query_filter_matcher_resolves_collections(monkeypatch):
    rows = [
        ([], [], [], [], [], [], [], 10, "Star Wars Collection"),
        ([], [], [], [], [], [], [], 10194, "Toy Story Collection"),
        ([], [], [], [], [], [], [], 263, "The Dark Knight Collection"),
    ]

    monkeypatch.setattr(
        "api.core.filter_matcher.get_sessionmaker",
        lambda: lambda: DummySession(rows),
    )

    matcher = QueryFilterMatcher()
    filters = matcher.match("Star Wars chronological")
    assert (10, "Star Wars Collection") in filters.matched_collections

    # "series" without explicit tv should not force tv media_type when collection matched
    filters_toy = matcher.match("Toy Story animation series")
    assert (10194, "Toy Story Collection") in filters_toy.matched_collections
    assert "tv" not in filters_toy.media_types

    single = matcher.resolve_collection("Dark Knight")
    assert single == (263, "The Dark Knight Collection")

    all_colls = matcher.resolve_collections("Star Wars")
    assert (10, "Star Wars Collection") in all_colls


def test_query_filter_matcher_multi_collection_grouping(monkeypatch):
    rows = [
        ([], [], [], [], [], [], [], 556, "Spider-Man Collection"),
        ([], [], [], [], [], [], [], 531241, "Spider-Man (MCU) Collection"),
        ([], [], [], [], [], [], [], 225941, "Spider-Man (TV) Collection"),
    ]

    monkeypatch.setattr(
        "api.core.filter_matcher.get_sessionmaker",
        lambda: lambda: DummySession(rows),
    )

    matcher = QueryFilterMatcher()
    matched = matcher.resolve_collections("Spider-Man")
    coll_ids = [cid for cid, _ in matched]
    assert 556 in coll_ids
    assert 531241 in coll_ids
    # TV collection with (TV) should be skipped from generic alias
    assert 225941 not in coll_ids


def test_collection_aliases_fuzzy_names_and_unknowns():
    matcher = QueryFilterMatcher()
    matcher._initialized = True
    matcher._collection_map = {
        "star wars": (1, "Star Wars"),
        "spiderman": (2, "Spider-Man"),
        "alien": (3, "Alien"),
    }
    matcher._multi_collection_map = {
        key: [value] for key, value in matcher._collection_map.items()
    }
    for query, expected in [
        ("", None),
        ("Star Wars movies", 1),
        ("the Star Wars", 1),
        ("star,wars", 1),
        ("spi-derman", 2),
        ("please find star wars", 1),
        ("xx", None),
        ("starr worz", 1),
        ("unknown franchise", None),
        ("star wars (original)", 1),
        ("the star wars (original)", 1),
    ]:
        one = matcher.resolve_collection(query)
        many = matcher.resolve_collections(query, threshold=0.7)
        assert (one[0] if one else None) == expected
        assert [item[0] for item in many] == ([] if expected is None else [expected])


def test_spacy_registry_repair_and_optional_failure(monkeypatch):
    import api.core.filter_matcher as module
    from thinc.backends import registry as thinc_registry
    from unittest.mock import MagicMock

    ops, vectors = MagicMock(), MagicMock()
    ops.__contains__.return_value = vectors.__contains__.return_value = False
    monkeypatch.setattr(thinc_registry, "ops", ops)
    monkeypatch.setattr(module.spacy.util.registry, "vectors", vectors)
    module._ensure_spacy_vectors()
    ops.register.assert_called_once()
    vectors.register.assert_called_once()
    ops.register.side_effect = RuntimeError("optional registration unavailable")
    vectors.register.side_effect = RuntimeError("optional registration unavailable")
    module._ensure_spacy_vectors()  # An optional registry cannot prevent startup.
