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
