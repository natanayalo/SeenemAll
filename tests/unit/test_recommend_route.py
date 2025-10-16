from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
import numpy as np
from types import SimpleNamespace
from typing import Any, Dict, Sequence

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from tests.helpers import FakeResult

from api.main import app
from api.db.session import get_db
from api.routes import recommend as recommend_routes
from api.core import business_rules
from api.core.legacy_intent_parser import IntentFilters
from api.core.entity_linker import ENTITY_LINKER_CACHE

ORIGINAL_PREFILTER = recommend_routes._prefilter_allowed_ids


@pytest.fixture(autouse=True)
def _tmdb_client_stub():
    previous = getattr(app.state, "tmdb_client", None)
    previous_linker = getattr(app.state, "entity_linker", None)

    class _StubClient:
        async def search(self, query, media_type=None):
            return {"results": []}

        async def aclose(self):
            return None

    ENTITY_LINKER_CACHE.clear()
    app.state.tmdb_client = _StubClient()
    app.state.entity_linker = None
    try:
        yield
    finally:
        app.state.tmdb_client = previous
        app.state.entity_linker = previous_linker


@pytest.fixture(autouse=True)
def _reset_prefilter(monkeypatch):
    monkeypatch.setattr(
        recommend_routes, "_prefilter_allowed_ids", lambda db, intent, limit: None
    )
    yield


@pytest.fixture(autouse=True)
def _clear_business_rules(monkeypatch):
    monkeypatch.setattr(business_rules, "load_rules", lambda: {})
    yield


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
        recommend_routes,
        "ann_candidates",
        lambda db, vec, exclude, limit, allowed_ids=None: [],
    )

    with TestClient(app) as client:
        resp = client.get("/recommend", params={"user_id": "u1"})

    app.dependency_overrides.clear()
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert "next_cursor" not in body
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


class CollabSession:
    """Lightweight session for collaborative recall tests."""

    def __init__(self, rows: Sequence[Any]):
        self._rows = list(rows)

    def execute(self, statement, *args, **kwargs):
        return FakeResult(self._rows)


def test_collaborative_candidates_scores_items():
    now = datetime.now(UTC)
    rows = [
        (10, "ally", 1, "watched", now),
        (10, "bob", 1, "liked", now - timedelta(minutes=5)),
        (20, "ally", 1, "liked", now - timedelta(hours=1)),
        (30, "bob", 1, "disliked", now),
        (40, "ally", 1, "not_interested", now),
        (50, "stranger", 1, "watched", now),
    ]
    session = CollabSession(rows)
    neighbors = [
        {"user_id": "ally", "weight": 0.5},
        {"user_id": "bob", "weight": 1.5},
        {"user_id": "zero", "weight": 0.0},
    ]

    result = recommend_routes._collaborative_candidates(
        session, neighbors, exclude_ids=[20], limit=5, allowed_ids=None
    )

    assert result == [10]

    only_allowed = recommend_routes._collaborative_candidates(
        session, neighbors, exclude_ids=[], limit=5, allowed_ids=[20]
    )
    assert only_allowed == [20]

    blocked = recommend_routes._collaborative_candidates(
        session, neighbors, exclude_ids=[], limit=5, allowed_ids=[]
    )
    assert blocked == []


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
        recommend_routes,
        "ann_candidates",
        lambda db, vec, exclude, limit, allowed_ids=None: [1, 2],
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
    body = resp.json()
    payload = body["items"]
    assert [item["id"] for item in payload] == [2, 1]
    assert payload[0]["explanation"] == "Because mysteries are trending."
    assert "original_rank" not in payload[0]
    assert captured["user_id"] == "u1"


def test_recommend_paginates_with_cursor(monkeypatch):
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
            genres=[{"name": "Drama"}],
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
            genres=[{"name": "Drama"}],
            release_year=2021,
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
            {"genre_prefs": {}, "neighbors": [], "negative_items": []},
        ),
    )

    monkeypatch.setattr(
        recommend_routes,
        "ann_candidates",
        lambda db, vec, exclude, limit, allowed_ids=None: [1, 2],
    )

    monkeypatch.setattr(
        recommend_routes,
        "rerank_with_explanations",
        lambda items, **_: items,
    )

    with TestClient(app) as client:
        first = client.get(
            "/recommend",
            params={"user_id": "u1", "limit": 1, "diversify": "false"},
        )
        body = first.json()
        assert [item["id"] for item in body["items"]] == [1]
        cursor = body.get("next_cursor")
        assert cursor

        second = client.get(
            "/recommend",
            params={
                "user_id": "u1",
                "limit": 1,
                "diversify": "false",
                "cursor": cursor,
            },
        )
        body2 = second.json()
        assert [item["id"] for item in body2["items"]] == [2]
        assert "next_cursor" not in body2

    app.dependency_overrides.clear()


def test_recommend_prefilter_passes_allowed_ids(monkeypatch):
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
            genres=[{"name": "Comedy"}],
            release_year=2020,
        )
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
            {"genre_prefs": {}, "neighbors": [], "negative_items": []},
        ),
    )

    monkeypatch.setattr(
        recommend_routes,
        "_prefilter_allowed_ids",
        lambda db, intent, limit: [101, 202],
    )

    recorded = {}

    def fake_ann(db, vec, exclude, limit, allowed_ids=None):
        recorded["allowed"] = allowed_ids
        return [1]

    monkeypatch.setattr(recommend_routes, "ann_candidates", fake_ann)
    monkeypatch.setattr(
        recommend_routes,
        "rerank_with_explanations",
        lambda items, **_: items,
    )

    with TestClient(app) as client:
        client.get(
            "/recommend",
            params={"user_id": "u1", "limit": 1, "query": "comedy movie"},
        )

    app.dependency_overrides.clear()
    session.close()
    assert recorded["allowed"] == [101, 202]


def test_recommend_merges_collaborative_candidates(monkeypatch):
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
        SimpleNamespace(
            id=3,
            tmdb_id=303,
            media_type="movie",
            title="Gamma",
            overview="Gamma surprises.",
            poster_url="gamma.jpg",
            runtime=95,
            original_language="en",
            genres=[{"name": "Drama"}],
            release_year=2023,
        ),
    ]

    session = CandidateSession(items)

    def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    short_v = np.ones(384, dtype="float32")

    monkeypatch.setattr(
        recommend_routes,
        "load_user_state",
        lambda db, user_id: (
            np.zeros(384, dtype="float32"),
            short_v,
            [],
            {
                "genre_prefs": {},
                "neighbors": [{"user_id": "ally", "weight": 0.7}],
                "negative_items": [],
            },
        ),
    )

    recorded: Dict[str, Any] = {}

    def fake_collab(db, neighbors, exclude, limit, allowed_ids=None):
        recorded["neighbors"] = neighbors
        recorded["exclude"] = list(exclude)
        recorded["limit"] = limit
        recorded["allowed"] = allowed_ids
        return [2, 3]

    monkeypatch.setattr(recommend_routes, "_collaborative_candidates", fake_collab)

    monkeypatch.setattr(
        recommend_routes,
        "ann_candidates",
        lambda db, vec, exclude, limit, allowed_ids=None: [1, 2],
    )

    monkeypatch.setattr(
        recommend_routes,
        "rewrite_query",
        lambda query, intent: SimpleNamespace(rewritten_text=""),
    )

    monkeypatch.setattr(
        recommend_routes,
        "rerank_with_explanations",
        lambda items, **_: items,
    )

    with TestClient(app) as client:
        resp = client.get(
            "/recommend", params={"user_id": "u1", "limit": 3, "diversify": "false"}
        )

    app.dependency_overrides.clear()
    session.close()

    assert resp.status_code == 200
    body = resp.json()
    payload_ids = [item["id"] for item in body["items"]]
    assert payload_ids == [2, 3, 1]
    assert recorded["limit"] == 9  # candidate_limit with limit=3
    assert recorded["neighbors"][0]["user_id"] == "ally"


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
        recommend_routes,
        "ann_candidates",
        lambda db, vec, exclude, limit, allowed_ids=None: [1, 2, 3],
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
    body = resp.json()
    payload = body["items"]
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
        recommend_routes,
        "ann_candidates",
        lambda db, vec, exclude, limit, allowed_ids=None: [],
    )

    with TestClient(app) as client:
        resp = client.get("/recommend", params={"user_id": "u1", "profile": "kids"})

    app.dependency_overrides.clear()
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
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
        lambda db, vec, exclude, limit, allowed_ids=None: [1],
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
    body = resp.json()
    assert body["items"] == []


def test_cold_start_candidates_respects_allowlist(monkeypatch):
    intent = IntentFilters(
        raw_query="",
        genres=["Comedy"],
        moods=[],
        media_types=["tv"],
    )

    class ColdSession:
        def __init__(self, rows):
            self._rows = rows
            self.bind = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))

        def execute(self, statement, params=None):
            return FakeResult([(row,) for row in self._rows])

        def get_bind(self):  # pragma: no cover - compatibility shim
            return self.bind

    session = ColdSession([5, 6, 5])
    result = recommend_routes._cold_start_candidates(
        session, intent, limit=5, allowlist=[5, 6, 7]
    )
    assert result == [5, 6]

    assert (
        recommend_routes._cold_start_candidates(
            ColdSession([1, 2, 3]), intent, limit=5, allowlist=[]
        )
        == []
    )

    session_no_allowlist = ColdSession([3, 4, 3])
    result_no_allowlist = recommend_routes._cold_start_candidates(
        session_no_allowlist, intent, limit=5, allowlist=None
    )
    assert result_no_allowlist == [3, 4]


def test_genre_contains_clause_uses_jsonb_when_available():
    class Dialect:
        name = "postgresql"

    class Bind:
        dialect = Dialect()

    session = SimpleNamespace(bind=Bind())
    clause = recommend_routes._genre_contains_clause(session, "Comedy")
    assert "genres" in str(clause)
    assert "jsonb" in str(clause).lower()


def test_prefilter_allowed_ids_short_circuits_without_filters():
    intent = IntentFilters(raw_query="", genres=[], moods=[], media_types=[])
    result = ORIGINAL_PREFILTER(object(), intent, limit=5)
    assert result is None


def test_prefilter_allowed_ids_returns_ordered_unique(monkeypatch):
    class PrefilterSession:
        def __init__(self, rows):
            self.rows = rows
            self.bind = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))
            self.last_statement = None

        def execute(self, statement):
            self.last_statement = statement
            return FakeResult([(row,) for row in self.rows])

    session = PrefilterSession([2, 1, 2])
    intent = IntentFilters(
        raw_query="",
        genres=["Comedy"],
        moods=[],
        media_types=["movie"],
    )
    result = ORIGINAL_PREFILTER(session, intent, limit=10)
    assert result == [2, 1]
    assert session.last_statement is not None


def test_float_from_env_parses_values(monkeypatch):
    monkeypatch.setenv("FLOAT_ENV", "1.75")
    assert recommend_routes._float_from_env("FLOAT_ENV", 0.0) == 1.75
    monkeypatch.setenv("FLOAT_ENV", "not-a-number")
    assert recommend_routes._float_from_env("FLOAT_ENV", 0.0) == 0.0
    monkeypatch.delenv("FLOAT_ENV", raising=False)
    assert recommend_routes._float_from_env("FLOAT_ENV", 2.5) == 2.5


@pytest.mark.asyncio
async def test_recommend_uses_entity_linker_and_blends_query_vector(monkeypatch):
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

    short_v = np.ones(384, dtype="float32")
    monkeypatch.setattr(
        recommend_routes,
        "load_user_state",
        lambda db, user_id: (
            np.zeros(384, dtype="float32"),
            short_v,
            [],
            {"genre_prefs": {}, "neighbors": [], "negative_items": []},
        ),
    )

    recorded = {}

    async def fake_link(self, query):
        recorded["searched_query"] = query
        return {"movie": [101], "tv": [], "person": []}

    monkeypatch.setattr(recommend_routes.EntityLinker, "link_entities", fake_link)

    previous_tmdb = app.state.tmdb_client
    app.state.tmdb_client = object()

    original_parse = recommend_routes._parse_llm_intent

    def fake_parse(query, user_context, linked_entities=None):
        recorded["linked_entities"] = linked_entities
        return original_parse(query, user_context, linked_entities)

    monkeypatch.setattr(recommend_routes, "_parse_llm_intent", fake_parse)

    def fake_ann(db, vec, exclude, limit, allowed_ids=None):
        recorded["allowed_ids"] = allowed_ids
        recorded["q_vec"] = vec
        return [1, 2]

    monkeypatch.setattr(recommend_routes, "ann_candidates", fake_ann)

    monkeypatch.setattr(
        recommend_routes,
        "rerank_with_explanations",
        lambda items, **_: items,
    )
    monkeypatch.setattr(
        recommend_routes, "_prefilter_allowed_ids", lambda db, intent, limit: [1]
    )

    rewrite_vec = np.full(384, 0.5, dtype="float32")
    monkeypatch.setattr(
        recommend_routes,
        "rewrite_query",
        lambda query, intent: SimpleNamespace(rewritten_text="rewritten query"),
    )
    monkeypatch.setattr(recommend_routes, "encode_texts", lambda texts: [rewrite_vec])

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/recommend", params={"user_id": "u1", "query": "alpha movie"}
        )

    assert response.status_code == 200

    app.dependency_overrides.clear()
    app.state.tmdb_client = previous_tmdb
    session.close()

    assert recorded.get("linked_entities") == {"movie": [101], "tv": [], "person": []}
    assert recorded.get("searched_query") == "alpha movie"
    assert recorded["allowed_ids"] == [1]

    alpha = 0.5
    expected_q_vec = (alpha * short_v) + ((1 - alpha) * rewrite_vec)
    expected_q_vec = expected_q_vec / np.linalg.norm(expected_q_vec)
    assert np.allclose(recorded["q_vec"], expected_q_vec)


def test_recommend_logs_cold_start_path(monkeypatch, caplog):
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
    ]

    session = CandidateSession(items)

    def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    monkeypatch.setattr(
        recommend_routes,
        "load_user_state",
        lambda db, user_id: (
            None,
            None,
            [],
            {"genre_prefs": {}, "neighbors": [], "negative_items": []},
        ),
    )

    monkeypatch.setattr(
        recommend_routes,
        "_cold_start_candidates",
        lambda db, intent, limit, allowlist: [1],
    )

    monkeypatch.setattr(
        recommend_routes,
        "rerank_with_explanations",
        lambda items, **_: items,
    )

    with caplog.at_level(logging.INFO, logger=recommend_routes.logger.name):
        with TestClient(app) as client:
            client.get("/recommend", params={"user_id": "u1"})

    app.dependency_overrides.clear()
    session.close()

    assert any(
        "Using cold-start candidates for user u1" in message
        for message in caplog.messages
    )
