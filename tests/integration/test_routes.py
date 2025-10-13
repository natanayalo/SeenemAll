from __future__ import annotations

from typing import Iterator, List

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.db.models import Item, UserHistory
from api.db.session import get_db
from api.main import app
from api.routes import recommend as recommend_routes
from api.routes import user as user_routes
from tests.helpers import FakeResult
from api.core import reranker


@pytest.fixture(autouse=True)
def _disable_db_startup(monkeypatch):
    monkeypatch.setattr("api.main.init_engine", lambda: None)
    monkeypatch.setattr("api.main.get_sessionmaker", lambda: None)


class _HistorySession:
    def __init__(self):
        self.added: List[UserHistory] = []
        self.commits = 0

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commits += 1

    def close(self):
        pass


class _RecommendSession(_HistorySession):
    def __init__(self, items: List[Item]):
        super().__init__()
        self._items = items
        self.last_statement = None
        self.last_params = None

    def execute(self, statement, params=None):
        self.last_statement = statement
        self.last_params = params or {}
        # Simulate the complex join query result
        rows = [
            (
                item,
                np.array([0.1] * 384),
                [{"service": "netflix", "url": f"http://test/{item.id}"}],
            )
            for item in self._items
        ]
        return FakeResult(rows)


def _make_items() -> List[Item]:
    first = Item(
        tmdb_id=1,
        media_type="movie",
        title="First",
        overview="First overview",
        poster_url="url1",
        runtime=90,
        original_language="en",
        genres=[{"id": 1, "name": "Drama"}],
        release_year=2020,
    )
    first.id = 100
    second = Item(
        tmdb_id=2,
        media_type="tv",
        title="Second",
        overview="Second overview",
        poster_url="url2",
        runtime=45,
        original_language="es",
        genres=[{"id": 2, "name": "Comedy"}],
        release_year=2022,
    )
    second.id = 200
    return [first, second]


def test_post_history_adds_events_and_upserts_profile(monkeypatch):
    session = _HistorySession()

    def override_get_db() -> Iterator[_HistorySession]:
        yield session

    called = {}

    def fake_upsert(db, user_id):
        called["user_id"] = user_id

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(user_routes, "upsert_user_vectors", fake_upsert)

    with TestClient(app) as client:
        resp = client.post(
            "/user/history",
            json={
                "user_id": "u1",
                "items": [1, 2],
                "event_type": "watched",
                "weight": 3,
            },
        )

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert len(session.added) == 2
    assert {h.item_id for h in session.added} == {1, 2}
    assert session.commits == 2
    assert called["user_id"] == "u1"


def test_post_history_with_profile_scopes_user(monkeypatch):
    session = _HistorySession()

    def override_get_db() -> Iterator[_HistorySession]:
        yield session

    called = {}

    def fake_upsert(db, user_id):
        called.setdefault("user_ids", []).append(user_id)

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(user_routes, "upsert_user_vectors", fake_upsert)

    with TestClient(app) as client:
        resp = client.post(
            "/user/history",
            json={
                "user_id": "u1",
                "profile": "kids",
                "items": [3],
                "event_type": "not_interested",
            },
        )

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["profile"] == "kids"
    assert session.added[0].user_id == "u1::kids"
    assert called["user_ids"] == ["u1::kids"]


def test_recommend_endpoint_returns_ranked_items(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "0")
    reranker._get_settings.cache_clear()
    items = _make_items()
    session = _RecommendSession(items)

    def override_get_db() -> Iterator[_RecommendSession]:
        yield session

    app.dependency_overrides[get_db] = override_get_db

    def fake_load_user_state(db, user_id):
        assert user_id == "u1"
        return (
            np.array([0.1, 0.2]),
            np.array([0.3, 0.7], dtype="float32"),
            [999],
            {"genre_prefs": {}, "neighbors": [], "negative_items": []},
        )

    def fake_ann_candidates(db, vec, exclude, limit):
        assert np.allclose(vec, np.array([0.3, 0.7], dtype="float32"))
        return [items[1].id, items[0].id]

    monkeypatch.setattr(recommend_routes, "load_user_state", fake_load_user_state)
    monkeypatch.setattr(recommend_routes, "ann_candidates", fake_ann_candidates)

    with TestClient(app) as client:
        resp = client.get("/recommend", params={"user_id": "u1", "limit": 5})

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    data = resp.json()
    assert [item["id"] for item in data] == [200, 100]
    assert data[0]["title"] == "Second"
    assert data[0]["runtime"] == 45
    assert data[0]["original_language"] == "es"
    assert data[0]["genres"] == [{"id": 2, "name": "Comedy"}]
    assert data[0]["release_year"] == 2022


def test_recommend_endpoint_requires_user_vector(monkeypatch):
    session = _RecommendSession([])

    def override_get_db() -> Iterator[_RecommendSession]:
        yield session

    app.dependency_overrides[get_db] = override_get_db

    def fake_load_user_state(db, user_id):
        assert user_id == "missing"
        return (
            None,
            None,
            [],
            {"genre_prefs": {}, "neighbors": [], "negative_items": []},
        )

    monkeypatch.setattr(recommend_routes, "load_user_state", fake_load_user_state)

    with TestClient(app) as client:
        resp = client.get("/recommend", params={"user_id": "missing"})

    app.dependency_overrides.clear()

    assert resp.status_code == 400
    assert resp.json()["detail"] == "No user vector. POST /user/history first."


def test_recommend_endpoint_diversifies_items(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "0")
    reranker._get_settings.cache_clear()
    items = _make_items()
    session = _RecommendSession(items)

    def override_get_db() -> Iterator[_RecommendSession]:
        yield session

    app.dependency_overrides[get_db] = override_get_db

    def fake_load_user_state(db, user_id):
        assert user_id == "u1"
        return (
            np.array([0.1, 0.2]),
            np.array([0.3, 0.7], dtype="float32"),
            [999],
            {"genre_prefs": {}, "neighbors": [], "negative_items": []},
        )

    def fake_ann_candidates(db, vec, exclude, limit):
        return [items[1].id, items[0].id]

    called = {}

    def fake_diversify(items, limit):
        called["diversify"] = True
        return items

    monkeypatch.setattr(recommend_routes, "load_user_state", fake_load_user_state)
    monkeypatch.setattr(recommend_routes, "ann_candidates", fake_ann_candidates)
    monkeypatch.setattr(recommend_routes, "diversify_with_mmr", fake_diversify)

    with TestClient(app) as client:
        client.get(
            "/recommend", params={"user_id": "u1", "limit": 5, "diversify": "true"}
        )

    assert called.get("diversify") is True

    # Test with diversify=False
    called.clear()
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        client.get(
            "/recommend", params={"user_id": "u1", "limit": 5, "diversify": "false"}
        )

    app.dependency_overrides.clear()
    assert called.get("diversify") is None


def test_recommend_endpoint_honors_profile(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "0")
    reranker._get_settings.cache_clear()
    session = _RecommendSession(_make_items())

    def override_get_db() -> Iterator[_RecommendSession]:
        yield session

    app.dependency_overrides[get_db] = override_get_db

    captured = {}

    def fake_load_user_state(db, user_id):
        captured["user_id"] = user_id
        return (
            np.array([0.1, 0.2]),
            np.array([0.3, 0.7], dtype="float32"),
            [999],
            {"genre_prefs": {}, "neighbors": [], "negative_items": []},
        )

    monkeypatch.setattr(recommend_routes, "load_user_state", fake_load_user_state)
    monkeypatch.setattr(
        recommend_routes, "ann_candidates", lambda db, vec, exclude, limit: []
    )

    with TestClient(app) as client:
        resp = client.get(
            "/recommend", params={"user_id": "u1", "profile": "kids", "limit": 5}
        )

    app.dependency_overrides.clear()
    assert resp.status_code == 200
    assert resp.json() == []
    assert captured["user_id"] == "u1::kids"
