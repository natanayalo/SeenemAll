from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient
from types import SimpleNamespace

from tests.helpers import FakeResult

from api.main import app
from api.db.session import get_db
from api.routes import recommend as recommend_routes


class DummySession:
    def execute(self, *_, **__):
        raise AssertionError("execute should not be called when no IDs are returned")


def test_recommend_returns_empty_when_no_candidates(monkeypatch):
    def override_get_db():
        yield DummySession()

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(
        recommend_routes,
        "load_user_state",
        lambda db, user_id: (
            np.array([1.0], dtype="float32"),
            np.array([1.0], dtype="float32"),
            [],
        ),
    )
    monkeypatch.setattr(
        recommend_routes, "ann_candidates", lambda db, vec, exclude, limit: []
    )

    with TestClient(app) as client:
        resp = client.get("/recommend", params={"user_id": "u1"})

    app.dependency_overrides.clear()
    assert resp.status_code == 200
    assert resp.json() == []


class CandidateSession:
    def __init__(self, rows):
        self._rows = list(rows)

    def execute(self, *_args, **_kwargs):
        return FakeResult(self._rows)


def test_recommend_includes_reranker_output(monkeypatch):
    items = [
        SimpleNamespace(
            id=1,
            tmdb_id=101,
            media_type="movie",
            title="Alpha",
            overview="Alpha saves the world.",
            poster_url="alpha.jpg",
            runtime=100,
            original_language="en",
            genres=[{"name": "Action"}],
            release_year=2020,
        ),
        SimpleNamespace(
            id=2,
            tmdb_id=202,
            media_type="movie",
            title="Beta",
            overview="Beta uncovers a mystery.",
            poster_url="beta.jpg",
            runtime=90,
            original_language="en",
            genres=[{"name": "Mystery"}],
            release_year=2022,
        ),
    ]

    session = CandidateSession(items)

    def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    monkeypatch.setattr(
        recommend_routes,
        "load_user_state",
        lambda db, user_id: (
            np.zeros(384, dtype="float32"),
            np.ones(384, dtype="float32"),
            [],
        ),
    )
    monkeypatch.setattr(
        recommend_routes, "ann_candidates", lambda db, vec, exclude, limit: [1, 2]
    )

    def fake_rerank(items_payload, intent, query, user):
        reordered = [
            {**items_payload[1], "explanation": "Because mysteries are trending."},
            {**items_payload[0], "explanation": "Classic action follow-up."},
        ]
        return reordered

    monkeypatch.setattr(recommend_routes, "rerank_with_explanations", fake_rerank)

    try:
        with TestClient(app) as client:
            resp = client.get("/recommend", params={"user_id": "u1", "limit": 2})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    payload = resp.json()
    assert [item["id"] for item in payload] == [2, 1]
    assert payload[0]["explanation"] == "Because mysteries are trending."
    assert "original_rank" not in payload[0]
