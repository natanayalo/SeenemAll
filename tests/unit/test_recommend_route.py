from __future__ import annotations

import numpy as np
from types import SimpleNamespace
from typing import Any, Sequence

from fastapi.testclient import TestClient
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
    captured = {}

    def fake_load_user_state(db, user_id):
        captured["user_id"] = user_id
        return (
            np.array([1.0], dtype="float32"),
            np.array([1.0], dtype="float32"),
            [],
            {"genre_prefs": {}, "neighbors": [], "negative_items": []},
        )

    monkeypatch.setattr(recommend_routes, "load_user_state", fake_load_user_state)
    monkeypatch.setattr(
        recommend_routes, "ann_candidates", lambda db, vec, exclude, limit: []
    )

    with TestClient(app) as client:
        resp = client.get("/recommend", params={"user_id": "u1"})

    app.dependency_overrides.clear()
    assert resp.status_code == 200
    assert resp.json() == []
    assert captured["user_id"] == "u1"


class MockRow:
    """Mock SQL result row with Item, vector, and watch_options."""

    def __init__(self, ns):
        self.id = ns.id
        self.ns = ns
        vector = np.ones(384, dtype="float32")
        self._data = (ns, vector, [])  # Using tuple for immutability

    def __iter__(self):
        """Make this row behave like a SQLAlchemy Row when unpacked."""
        return iter(self._data)

    def __getattr__(self, name):
        """Delegate unknown attributes to the namespace object."""
        return getattr(self.ns, name)

    def __getitem__(self, key):
        """Support both index and attribute access."""
        if isinstance(key, int):
            return self._data[key]
        return getattr(self, key)


class CandidateSession:
    """Mock session that returns predefined rows for recommend route testing."""

    def __init__(self, rows: Sequence[Any]):
        self._rows = [MockRow(row) for row in rows]
        self._closed = False

    def execute(self, statement, *args, **kwargs):
        if self._closed:
            raise RuntimeError("Session is closed")
        # Handle different statement types
        if hasattr(statement, "where"):
            # This is a select statement, return all rows
            return FakeResult(self._rows)
        return FakeResult([])

    def close(self):
        self._closed = True

    def commit(self):
        if self._closed:
            raise RuntimeError("Session is closed")

    def rollback(self):
        if self._closed:
            raise RuntimeError("Session is closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def test_recommend_includes_reranker_output(monkeypatch):
    # Test data setup
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

    # Create session and ensure proper cleanup
    session = CandidateSession(items)

    def override_get_db():
        yield session  # Use yield to ensure FastAPI handles session lifecycle

    # Clear any existing overrides
    app.dependency_overrides.clear()
    app.dependency_overrides[get_db] = override_get_db

    captured = {}

    def fake_load_state(db, user_id):
        captured["user_id"] = user_id
        return (
            np.zeros(384, dtype="float32"),
            np.ones(384, dtype="float32"),
            [],
            {"genre_prefs": {}, "neighbors": [], "negative_items": []},
        )

    monkeypatch.setattr(recommend_routes, "load_user_state", fake_load_state)
    monkeypatch.setattr(
        recommend_routes, "ann_candidates", lambda db, vec, exclude, limit: [1, 2]
    )

    def fake_rerank(items_payload, intent, query, user):
        reordered = [
            {**items_payload[1], "explanation": "Because mysteries are trending."},
            {**items_payload[0], "explanation": "Classic action follow-up."},
        ]
        assert user["user_id"] == "u1"
        assert user["profile"] is None
        assert user["base_user_id"] == "u1"
        return reordered

    monkeypatch.setattr(recommend_routes, "rerank_with_explanations", fake_rerank)

    # Create client and make request
    client = TestClient(app)
    resp = client.get("/recommend", params={"user_id": "u1", "limit": 2})

    # Clean up resources immediately after use
    client.close()
    session.close()
    app.dependency_overrides.clear()

    # Now check the response
    assert resp.status_code == 200
    payload = resp.json()
    assert [item["id"] for item in payload] == [2, 1]
    assert payload[0]["explanation"] == "Because mysteries are trending."
    assert "original_rank" not in payload[0]
    assert captured["user_id"] == "u1"


def test_recommend_hybrid_boosts_trending_items(monkeypatch):
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
            popularity=10.0,
            trending_rank=5,
            popular_rank=3,
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
            popularity=30.0,
            trending_rank=None,
            popular_rank=2,
        ),
        SimpleNamespace(
            id=3,
            tmdb_id=303,
            media_type="movie",
            title="Gamma",
            overview="Gamma is the sleeper hit.",
            poster_url="gamma.jpg",
            runtime=110,
            original_language="en",
            genres=[{"name": "Drama"}],
            release_year=2023,
            popularity=5.0,
            trending_rank=1,
            popular_rank=4,
        ),
    ]

    session = CandidateSession(items)

    def override_get_db():
        yield session

    app.dependency_overrides.clear()
    app.dependency_overrides[get_db] = override_get_db

    def fake_load_state(db, user_id):
        return (
            np.zeros(384, dtype="float32"),
            np.ones(384, dtype="float32"),
            [],
            {"genre_prefs": {}, "neighbors": [], "negative_items": []},
        )

    monkeypatch.setattr(recommend_routes, "load_user_state", fake_load_state)
    monkeypatch.setattr(
        recommend_routes, "ann_candidates", lambda db, vec, exclude, limit: [1, 2, 3]
    )

    def fake_rerank(items_payload, intent, query, user):
        reranked = []
        for item in items_payload:
            copy = dict(item)
            copy.setdefault("explanation", f"Hybrid base item {item['id']}.")
            reranked.append(copy)
        return reranked

    monkeypatch.setattr(recommend_routes, "rerank_with_explanations", fake_rerank)

    client = TestClient(app)
    resp = client.get(
        "/recommend", params={"user_id": "u1", "limit": 3, "diversify": "false"}
    )
    client.close()
    session.close()
    app.dependency_overrides.clear()

    assert resp.status_code == 200
    payload = resp.json()
    assert [item["id"] for item in payload] == [1, 3, 2]
    assert all("ann_rank" not in item for item in payload)
    assert payload[1]["trending_rank"] == 1


def test_recommend_supports_profile_parameter(monkeypatch):
    class DummySession:
        def execute(self, *_, **__):
            raise AssertionError

    def override_get_db():
        yield DummySession()

    app.dependency_overrides[get_db] = override_get_db

    captured = {}

    def fake_load_state(db, user_id):
        captured["user_id"] = user_id
        return (
            np.zeros(384, dtype="float32"),
            np.ones(384, dtype="float32"),
            [],
            {"genre_prefs": {}, "neighbors": [], "negative_items": []},
        )

    monkeypatch.setattr(recommend_routes, "load_user_state", fake_load_state)
    monkeypatch.setattr(
        recommend_routes, "ann_candidates", lambda db, vec, exclude, limit: []
    )

    with TestClient(app) as client:
        resp = client.get("/recommend", params={"user_id": "u1", "profile": "kids"})

    app.dependency_overrides.clear()
    assert resp.status_code == 200
    assert resp.json() == []
    assert captured["user_id"] == "u1::kids"


def test_recommend_filters_negative_items(monkeypatch):
    session = CandidateSession(
        [
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
        ]
    )

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
            {"genre_prefs": {}, "neighbors": [], "negative_items": [1]},
        ),
    )
    monkeypatch.setattr(
        recommend_routes,
        "ann_candidates",
        lambda db, vec, exclude, limit: [1],
    )
    monkeypatch.setattr(
        recommend_routes,
        "rerank_with_explanations",
        lambda items, **_: items,
    )

    with TestClient(app) as client:
        resp = client.get("/recommend", params={"user_id": "u1"})

    app.dependency_overrides.clear()
    assert resp.status_code == 200
    assert resp.json() == []
