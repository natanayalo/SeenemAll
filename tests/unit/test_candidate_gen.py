from __future__ import annotations

import numpy as np
import pytest

from api.core import candidate_gen
from api.core.elasticsearch_search import SearchFilters


def test_ann_candidates_returns_empty_when_vector_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(candidate_gen.config, "ANN_BACKEND", "elasticsearch")
    monkeypatch.setattr(candidate_gen, "knn_search", lambda *a, **k: [])

    assert candidate_gen.ann_candidates(None, None, exclude_ids=[1, 2]) == []


def test_ann_candidates_invokes_knn_and_returns_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(candidate_gen.config, "ANN_BACKEND", "elasticsearch")
    hits = [
        {"item_id": "42", "score": 0.5, "source": {}},
        {"item_id": "7", "score": 0.4, "source": {}},
    ]
    monkeypatch.setattr(candidate_gen, "knn_search", lambda *a, **k: hits)

    vec = np.array([0.1, 0.2, 0.3], dtype="float32")
    result = candidate_gen.ann_candidates(
        None, vec, exclude_ids=[1, 2], limit=2, allowed_ids=[7, 42, 99]
    )

    assert result == [42, 7]


def test_ann_candidates_builds_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(candidate_gen.config, "ANN_BACKEND", "elasticsearch")
    captured = {}

    def fake_knn(query_vector, **kwargs):
        captured["query_vector"] = query_vector
        captured.update(kwargs)
        return []

    monkeypatch.setattr(candidate_gen, "knn_search", fake_knn)
    vec = np.array([0.9, 0.1, 0.0], dtype="float32")

    candidate_gen.ann_candidates(
        None,
        vec,
        exclude_ids=[1, 2],
        limit=10,
        allowed_ids=[10, 11],
    )

    assert captured["filters"].include_item_ids == ("10", "11")
    assert captured["filters"].exclude_item_ids == ("1", "2")
    assert captured["k"] == 10
    assert captured["source_includes"] == ["item_id"]


class DummyResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class DummySession:
    def __init__(self, rows=None):
        self.rows = rows or [(42,), (7,)]
        self.calls: list = []

    def execute(self, statement, params):
        self.calls.append((statement, params))
        return DummyResult(self.rows)


def test_pgvector_ann_candidates_queries_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(candidate_gen.config, "ANN_BACKEND", "pgvector")
    db = DummySession()
    vec = np.array([0.1, 0.9, 0.0], dtype="float32")

    result = candidate_gen.ann_candidates(
        db, vec, exclude_ids=[1], limit=5, allowed_ids=None
    )

    assert result == [42, 7]
    assert len(db.calls) == 1
    _, params = db.calls[0]
    assert params["exclude"] == [1]
    assert params["lim"] == 5
    assert params["uvec"] == [float(x) for x in vec]


def test_pgvector_ann_candidates_respects_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(candidate_gen.config, "ANN_BACKEND", "pgvector")
    db = DummySession(rows=[(42,), (7,)])
    vec = np.array([0.2, 0.8], dtype="float32")

    result = candidate_gen.ann_candidates(
        db, vec, exclude_ids=[], limit=3, allowed_ids=[42, 7, 9]
    )

    assert result == [42, 7]
    _, params = db.calls[0]
    assert params["allowed"] == [42, 7, 9]


def test_pgvector_ann_candidates_empty_allowlist_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(candidate_gen.config, "ANN_BACKEND", "pgvector")
    db = DummySession()
    vec = np.array([0.3, 0.7], dtype="float32")

    assert (
        candidate_gen.ann_candidates(db, vec, exclude_ids=[], limit=10, allowed_ids=[])
        == []
    )
    assert db.calls == []


def test_backend_override_toggles_pgvector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(candidate_gen.config, "ANN_BACKEND", "elasticsearch")
    db = DummySession()
    vec = np.array([0.1, 0.2, 0.3], dtype="float32")

    result = candidate_gen.ann_candidates(
        db, vec, exclude_ids=[], limit=2, allowed_ids=None, backend_override="pgvector"
    )

    assert result == [42, 7]


def test_backend_override_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(candidate_gen.config, "ANN_BACKEND", "elasticsearch")
    vec = np.array([0.1, 0.2, 0.3], dtype="float32")

    with pytest.raises(ValueError):
        candidate_gen.ann_candidates(
            DummySession(),
            vec,
            exclude_ids=[],
            limit=5,
            allowed_ids=None,
            backend_override="invalid",
        )


def test_pgvector_ann_candidates_with_search_filters() -> None:
    db = DummySession(rows=[(101,), (102,)])
    vec = np.array([0.1, 0.2, 0.3], dtype="float32")
    sf = SearchFilters(
        media_types=("movie",),
        release_year_gte=2000,
        release_year_lte=2020,
        runtime_gte=90,
        runtime_lte=150,
        directors=("Christopher Nolan",),
        cast=("Christian Bale",),
        producers=("Emma Thomas",),
        writers=("Jonathan Nolan",),
        genres=("Action",),
        languages=("en",),
        include_item_ids=("101", "102"),
        exclude_item_ids=("999",),
    )

    result = candidate_gen.ann_candidates(
        db,
        vec,
        exclude_ids=[55],
        limit=10,
        allowed_ids=[101],
        backend_override="pgvector",
        search_filters=sf,
    )

    assert result == [101, 102]
    assert len(db.calls) == 1
    stmt, params = db.calls[0]
    sql_text = str(stmt)
    assert "JOIN items i ON i.id = e.item_id" in sql_text
    assert "<=>" in sql_text
    assert params["media_types"] == ["movie"]
    assert params["year_gte"] == 2000
    assert params["year_lte"] == 2020
    assert params["runtime_gte"] == 90
    assert params["runtime_lte"] == 150
    assert params["dir_1"] == "%christopher nolan%"
    assert params["cast_1"] == "%christian bale%"
    assert params["prod_1"] == "%emma thomas%"
    assert params["writ_1"] == "%jonathan nolan%"
    assert params["genre_1"] == "%action%"
    assert params["lang_1"] == "en"
    assert set(params["exclude"]) == {55, 999}
    assert set(params["allowed"]) == {101, 102}


def test_pgvector_ann_candidates_with_providers() -> None:
    db = DummySession(rows=[(10,)])
    vec = np.array([0.5, 0.5], dtype="float32")
    sf = SearchFilters(providers=("netflix", "prime"))

    result = candidate_gen.ann_candidates(
        db,
        vec,
        exclude_ids=[],
        limit=5,
        backend_override="pgvector",
        search_filters=sf,
    )

    assert result == [10]
    assert len(db.calls) == 1
    stmt, params = db.calls[0]
    sql_text = str(stmt)
    assert "JOIN availability a ON a.item_id = e.item_id" in sql_text
    assert params["providers"] == ["netflix", "prime"]
    assert params["country"] == "US"


def test_pgvector_ann_candidates_empty_include_short_circuits() -> None:
    db = DummySession()
    vec = np.array([0.1, 0.2], dtype="float32")
    sf = SearchFilters(include_item_ids=("invalid",))

    result = candidate_gen.ann_candidates(
        db,
        vec,
        exclude_ids=[],
        limit=5,
        backend_override="pgvector",
        search_filters=sf,
    )

    assert result == []
    assert len(db.calls) == 0
