from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
import numpy as np
from types import SimpleNamespace
from typing import Any, Dict, Sequence
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request
from tests.helpers import FakeResult

from api.main import app
from api.db.session import get_db
from api.routes import recommend as recommend_routes
from api.routes.recommend import PrefilterDecision
from api.core import business_rules
from api.core.legacy_intent_parser import IntentFilters
from api.core.entity_linker import ENTITY_LINKER_CACHE
from api.core.intent_parser import Intent
from api.core.rewrite import Rewrite

ORIGINAL_PREFILTER = recommend_routes._prefilter_allowed_ids


@pytest.fixture(autouse=True)
def _clear_recommend_cache():
    recommend_routes._clear_recommend_cache_for_tests()
    yield
    recommend_routes._clear_recommend_cache_for_tests()


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
def _stub_llm_intent(monkeypatch):
    captured: Dict[str, Any] = {}

    def fake_parse(query, user_context, linked_entities=None):
        captured["query"] = query
        captured["user_context"] = user_context
        captured["linked_entities"] = linked_entities
        return Intent(include_genres=[])

    monkeypatch.setattr(recommend_routes, "_parse_llm_intent", fake_parse)
    yield captured


@pytest.fixture(autouse=True)
def _reset_prefilter(monkeypatch):
    monkeypatch.setattr(
        recommend_routes,
        "_prefilter_allowed_ids",
        lambda db, intent, limit, preferred_services=None: PrefilterDecision(
            None, [], True
        ),
    )
    yield


@pytest.fixture(autouse=True)
def _clear_business_rules(monkeypatch):
    monkeypatch.setattr(business_rules, "load_rules", lambda: {})
    yield


@pytest.fixture(autouse=True)
def _disable_trending_prior(monkeypatch, request):
    if request.node.get_closest_marker("enable_trending_helper"):
        yield
        return
    monkeypatch.setattr(
        recommend_routes,
        "_trending_prior_candidates",
        lambda *args, **kwargs: [],
    )
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
        watch_options = getattr(ns, "watch_options", [])
        self._data = (ns, vector, watch_options)  # Using tuple for immutability

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

    assert result[0][0] == 10
    assert pytest.approx(result[0][1], rel=1e-6) == 1.0

    only_allowed = recommend_routes._collaborative_candidates(
        session, neighbors, exclude_ids=[], limit=5, allowed_ids=[20]
    )
    assert only_allowed == [(20, 1.0)]

    blocked = recommend_routes._collaborative_candidates(
        session, neighbors, exclude_ids=[], limit=5, allowed_ids=[]
    )
    assert blocked == []


@pytest.mark.enable_trending_helper
def test_trending_prior_candidates_normalises_scores():
    session = Mock()
    session.execute = Mock(
        return_value=FakeResult(
            [
                (1, "movie", 1, 3, 100.0),
                (2, "movie", 2, None, 50.0),
                (3, "tv", None, None, 10.0),
            ]
        )
    )

    results = recommend_routes._trending_prior_candidates(
        session, None, exclude_ids=[], limit=3, allowed_ids=None
    )

    assert len(results) == 3
    assert results[0][0] == 1
    assert pytest.approx(results[0][1], rel=1e-6) == 1.0
    assert results[1][0] == 2
    assert results[2][0] == 3
    assert results[2][1] >= 0.0


@pytest.mark.enable_trending_helper
def test_trending_prior_candidates_prefers_requested_media_types():
    session = Mock()
    session.execute = Mock(
        return_value=FakeResult(
            [
                (1, "tv", 1, 1, 100.0),
                (2, "movie", 2, 2, 90.0),
                (3, "movie", 3, 3, 80.0),
            ]
        )
    )

    results = recommend_routes._trending_prior_candidates(
        session,
        None,
        exclude_ids=[],
        limit=3,
        allowed_ids=None,
        preferred_media_types=["movie"],
    )

    assert [item_id for item_id, _ in results] == [2, 3, 1]


def test_cold_start_candidates_prefers_requested_media_types():
    session = Mock()
    session.execute = Mock(
        return_value=FakeResult(
            [
                (1, "tv"),
                (2, "movie"),
                (3, "movie"),
            ]
        )
    )

    intent = IntentFilters(raw_query="", genres=[], moods=[], media_types=[])
    results = recommend_routes._cold_start_candidates(
        session,
        intent,
        limit=3,
        allowlist=None,
        preferred_media_types=["movie"],
    )

    assert results == [2, 3, 1]


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
            collection_id=None,
            collection_name=None,
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
            collection_id=None,
            collection_name=None,
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
    assert "collection_id" in payload[0]
    assert "collection_name" in payload[0]
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
            collection_id=None,
            collection_name=None,
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
            collection_id=None,
            collection_name=None,
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
            collection_id=None,
            collection_name=None,
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
        lambda db, intent, limit, preferred_services=None: PrefilterDecision(
            [101, 202], [101, 202], True
        ),
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


def test_recommend_relaxed_prefilter_allows_mismatched_genres(monkeypatch):
    sci_fi_item = SimpleNamespace(
        id=1,
        tmdb_id=101,
        media_type="tv",
        title="Nebula Frontiers",
        overview="Explorers chart the unknown.",
        poster_url="nebula.jpg",
        runtime=45,
        original_language="en",
        genres=[{"name": "Sci-Fi & Fantasy"}],
        release_year=2023,
        collection_id=None,
        collection_name=None,
    )

    session = CandidateSession([sci_fi_item])

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
        lambda db, intent, limit, preferred_services=None: PrefilterDecision(
            None, [], False
        ),
    )
    monkeypatch.setattr(
        recommend_routes,
        "ann_candidates",
        lambda db, vec, exclude, limit, allowed_ids=None: [sci_fi_item.id],
    )
    monkeypatch.setattr(
        recommend_routes,
        "rerank_with_explanations",
        lambda items, **_: items,
    )
    monkeypatch.setattr(
        recommend_routes,
        "rewrite_query",
        lambda query, intent: SimpleNamespace(rewritten_text=""),
    )

    with TestClient(app) as client:
        response = client.get(
            "/recommend",
            params={"user_id": "u1", "limit": 1, "query": "science fiction series"},
        )

    app.dependency_overrides.clear()
    session.close()

    assert response.status_code == 200
    body = response.json()
    assert [item["id"] for item in body["items"]] == [sci_fi_item.id]
    assert body["items"][0]["media_type"] == "tv"


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
            collection_id=None,
            collection_name=None,
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
            collection_id=None,
            collection_name=None,
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
            collection_id=None,
            collection_name=None,
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
        return [(2, 1.0), (3, 0.8)]

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
        "_trending_prior_candidates",
        lambda db, intent, exclude, limit, allowed_ids=None, preferred_media_types=None: [(3, 0.6)],
    )

    monkeypatch.setattr(
        recommend_routes,
        "rerank_with_explanations",
        lambda items, **_: items,
    )

    with TestClient(app) as client:
        resp = client.get(
            "/recommend",
            params={
                "user_id": "u1",
                "limit": 3,
                "diversify": "false",
                "mixer_popularity_weight": "0",
                "mixer_trending_weight": "0",
            },
        )

    app.dependency_overrides.clear()
    session.close()

    assert resp.status_code == 200
    body = resp.json()
    payload_ids = [item["id"] for item in body["items"]]
    assert payload_ids == [2, 3, 1]
    assert recorded["limit"] == 30  # no-query candidate_limit with limit=3
    assert recorded["neighbors"][0]["user_id"] == "ally"


def test_recommend_mixer_scores_items(monkeypatch):
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
            collection_id=None,
            collection_name=None,
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
            collection_id=None,
            collection_name=None,
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
            collection_id=None,
            collection_name=None,
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

    with TestClient(app) as client:
        resp = client.get(
            "/recommend", params={"user_id": "u1", "limit": 3, "diversify": "false"}
        )
        resp_override = client.get(
            "/recommend",
            params={
                "user_id": "u1",
                "limit": 3,
                "diversify": "false",
                "mixer_ann_weight": "0",
                "mixer_collab_weight": "0",
                "mixer_trending_weight": "0",
                "mixer_popularity_weight": "5",
                "mixer_novelty_weight": "0",
            },
        )
    session.close()
    app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    payload = body["items"]
    assert [item["id"] for item in payload] == [1, 2, 3]
    assert all("ann_rank" not in item for item in payload)

    assert resp_override.status_code == 200
    override_ids = [item["id"] for item in resp_override.json()["items"]]
    assert override_ids[0] == 2  # popularity dominates when overrides applied


def test_recommend_injects_serendipity_items(monkeypatch):
    limit = 6
    items = []
    for idx in range(1, 11):
        items.append(
            SimpleNamespace(
                id=idx,
                tmdb_id=1000 + idx,
                media_type="movie",
                title=f"Item {idx}",
                overview=f"Overview {idx}",
                poster_url=f"poster{idx}.jpg",
                runtime=90 + idx,
                original_language="en",
                genres=[{"name": "Action"}],
                release_year=2010 + idx,
                popularity=float(100 - idx * 3),
                vote_average=7.0,
                vote_count=100 - idx,
                trending_rank=idx,
                collection_id=None,
                collection_name=None,
            )
        )

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
        lambda db, vec, exclude, limit, allowed_ids=None: [item.id for item in items],
    )

    monkeypatch.setattr(
        recommend_routes,
        "apply_business_rules",
        lambda ordered, intent=None: ordered,
    )

    monkeypatch.setattr(
        recommend_routes,
        "rerank_with_explanations",
        lambda ordered, **_: ordered,
    )

    with TestClient(app) as client:
        resp = client.get(
            "/recommend",
            params={"user_id": "u1", "limit": limit, "diversify": "false"},
        )

    app.dependency_overrides.clear()
    session.close()

    assert resp.status_code == 200
    body = resp.json()
    payload = body["items"]
    assert payload

    target = recommend_routes._serendipity_target(limit)
    long_tail_ids = [ns.id for ns in items[limit:]]
    top_ids = [entry["id"] for entry in payload[:limit]]
    serendipity_count = sum(1 for iid in top_ids if iid in long_tail_ids)
    assert serendipity_count >= target


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
                collection_id=None,
                collection_name=None,
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


def test_recommend_filters_streaming_providers(monkeypatch):
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
            collection_id=None,
            collection_name=None,
            watch_options=[{"service": "nfx", "url": "http://alpha"}],
        ),
        SimpleNamespace(
            id=2,
            tmdb_id=202,
            media_type="movie",
            title="Beta",
            overview="Beta saves the world.",
            poster_url="beta.jpg",
            runtime=95,
            original_language="en",
            genres=[{"name": "Action"}],
            release_year=2020,
            collection_id=None,
            collection_name=None,
            watch_options=[{"service": "hlu", "url": "http://beta"}],
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
        lambda db, vec, exclude, limit, allowed_ids=None: [item.id for item in items],
    )
    monkeypatch.setattr(
        recommend_routes,
        "rerank_with_explanations",
        lambda candidates, **_: candidates,
    )

    def fake_intent(query, user_context, linked_entities=None):
        return Intent(streaming_providers=["netflix"])

    monkeypatch.setattr(recommend_routes, "_parse_llm_intent", fake_intent)

    with TestClient(app) as client:
        resp = client.get(
            "/recommend",
            params={"user_id": "u1", "query": "netflix", "limit": 1},
        )

    app.dependency_overrides.clear()
    assert resp.status_code == 200
    body = resp.json()
    assert [item["id"] for item in body["items"]] == [1]
    assert body["items"][0]["watch_options"] == [
        {"service": "nfx", "url": "http://alpha"}
    ]


def test_recommend_provider_fallback_when_insufficient(monkeypatch):
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
            collection_id=None,
            collection_name=None,
            watch_options=[{"service": "nfx", "url": "http://alpha"}],
        ),
        SimpleNamespace(
            id=2,
            tmdb_id=202,
            media_type="movie",
            title="Beta",
            overview="Beta saves the world.",
            poster_url="beta.jpg",
            runtime=95,
            original_language="en",
            genres=[{"name": "Action"}],
            release_year=2020,
            collection_id=None,
            collection_name=None,
            watch_options=[{"service": "hlu", "url": "http://beta"}],
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
        lambda db, vec, exclude, limit, allowed_ids=None: [item.id for item in items],
    )
    monkeypatch.setattr(
        recommend_routes,
        "rerank_with_explanations",
        lambda candidates, **_: candidates,
    )

    def fake_intent(query, user_context, linked_entities=None):
        return Intent(streaming_providers=["netflix"])

    monkeypatch.setattr(recommend_routes, "_parse_llm_intent", fake_intent)

    with TestClient(app) as client:
        resp = client.get(
            "/recommend",
            params={"user_id": "u1", "query": "netflix", "limit": 2},
        )

    app.dependency_overrides.clear()
    assert resp.status_code == 200
    body = resp.json()
    ids = [item["id"] for item in body["items"]]
    assert ids == [1, 2]
    assert (
        body["items"][0]["watch_options"]
        and body["items"][0]["watch_options"][0]["service"] == "nfx"
    )
    assert body["items"][1]["watch_options"][0]["service"] == "hlu"


def _build_franchise_item(
    idx: int, franchise_id: int | None, franchise_name: str | None
):
    return SimpleNamespace(
        id=idx,
        tmdb_id=100 + idx,
        media_type="movie",
        title=f"Item {idx}",
        overview=f"Overview {idx}",
        poster_url=f"poster{idx}.jpg",
        runtime=100 + idx,
        original_language="en",
        genres=[{"name": "Action"}],
        release_year=2020 + idx,
        collection_id=franchise_id,
        collection_name=franchise_name,
    )


def test_recommend_applies_franchise_cap_when_diversify_enabled(monkeypatch):
    session = CandidateSession(
        [
            _build_franchise_item(1, 42, "Franchise"),
            _build_franchise_item(2, 42, "Franchise"),
            _build_franchise_item(3, 42, "Franchise"),
            _build_franchise_item(4, 7, "Standalone"),
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
            {"genre_prefs": {}, "neighbors": [], "negative_items": []},
        ),
    )
    monkeypatch.setattr(
        recommend_routes,
        "ann_candidates",
        lambda db, vec, exclude, limit, allowed_ids=None: [1, 2, 3, 4],
    )
    monkeypatch.setattr(
        recommend_routes,
        "_apply_mixer_scores",
        lambda items, **kwargs: None,
    )
    monkeypatch.setattr(
        recommend_routes,
        "apply_business_rules",
        lambda items, intent: items,
    )
    monkeypatch.setattr(
        recommend_routes,
        "diversify_with_mmr",
        lambda items, limit: items,
    )
    monkeypatch.setattr(
        recommend_routes,
        "rerank_with_explanations",
        lambda items, **_: items,
    )

    with TestClient(app) as client:
        resp = client.get("/recommend", params={"user_id": "u1", "limit": 4})

    app.dependency_overrides.clear()
    session.close()

    assert resp.status_code == 200
    body = resp.json()
    payload = body["items"]
    assert [item["id"] for item in payload] == [1, 2, 4]
    assert all(
        item.get("collection_id") != 42 or idx < 2 for idx, item in enumerate(payload)
    )


def test_recommend_skips_franchise_cap_when_diversify_disabled(monkeypatch):
    session = CandidateSession(
        [
            _build_franchise_item(1, 42, "Franchise"),
            _build_franchise_item(2, 42, "Franchise"),
            _build_franchise_item(3, 42, "Franchise"),
            _build_franchise_item(4, 7, "Standalone"),
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
            {"genre_prefs": {}, "neighbors": [], "negative_items": []},
        ),
    )
    monkeypatch.setattr(
        recommend_routes,
        "ann_candidates",
        lambda db, vec, exclude, limit, allowed_ids=None: [1, 2, 3, 4],
    )
    monkeypatch.setattr(
        recommend_routes,
        "_apply_mixer_scores",
        lambda items, **kwargs: None,
    )
    monkeypatch.setattr(
        recommend_routes,
        "apply_business_rules",
        lambda items, intent: items,
    )
    monkeypatch.setattr(
        recommend_routes,
        "diversify_with_mmr",
        lambda items, limit: items,
    )
    monkeypatch.setattr(
        recommend_routes,
        "rerank_with_explanations",
        lambda items, **_: items,
    )

    with TestClient(app) as client:
        resp = client.get(
            "/recommend",
            params={"user_id": "u1", "limit": 4, "diversify": "false"},
        )

    app.dependency_overrides.clear()
    session.close()

    assert resp.status_code == 200
    body = resp.json()
    payload = body["items"]
    assert [item["id"] for item in payload] == [1, 2, 3, 4]
    assert sum(1 for item in payload if item.get("collection_id") == 42) == 3


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
            return FakeResult(self._rows)

        def get_bind(self):  # pragma: no cover - compatibility shim
            return self.bind

    session = ColdSession([(5, "tv"), (6, "tv"), (5, "tv")])
    result = recommend_routes._cold_start_candidates(
        session, intent, limit=5, allowlist=[5, 6, 7]
    )
    assert result == [5, 6]

    assert (
        recommend_routes._cold_start_candidates(
            ColdSession([(1, "tv"), (2, "tv"), (3, "tv")]), intent, limit=5, allowlist=[]
        )
        == []
    )

    session_no_allowlist = ColdSession([(3, "tv"), (4, "tv"), (3, "tv")])
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
    assert isinstance(result, PrefilterDecision)
    assert result.allowed_ids is None
    assert result.boost_ids == []
    assert result.enforce_genres is True


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
    assert isinstance(result, PrefilterDecision)
    assert result.allowed_ids is None
    assert result.boost_ids == [2, 1]
    assert result.enforce_genres is False
    assert session.last_statement is not None


def test_constraint_prior_candidates_returns_strict_matches():
    class ConstraintSession:
        def __init__(self, rows):
            self.rows = rows
            self.bind = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))

        def execute(self, statement):
            return FakeResult(self.rows)

    matching = SimpleNamespace(
        id=1,
        media_type="movie",
        runtime=105,
        maturity_rating="PG",
        genres=[{"name": "Family"}, {"name": "Adventure"}],
        popular_rank=10,
        trending_rank=20,
        popularity=50.0,
        vote_count=1000,
    )
    too_long = SimpleNamespace(
        id=2,
        media_type="movie",
        runtime=145,
        maturity_rating="PG",
        genres=[{"name": "Family"}, {"name": "Adventure"}],
        popular_rank=5,
        trending_rank=10,
        popularity=60.0,
        vote_count=2000,
    )
    too_mature = SimpleNamespace(
        id=3,
        media_type="movie",
        runtime=100,
        maturity_rating="TV-MA",
        genres=[{"name": "Family"}, {"name": "Adventure"}],
        popular_rank=1,
        trending_rank=1,
        popularity=90.0,
        vote_count=5000,
    )

    session = ConstraintSession(
        [
            (matching, np.array([1.0, 0.0], dtype="float32")),
            (too_long, np.array([0.7, 0.0], dtype="float32")),
            (too_mature, np.array([0.6, 0.0], dtype="float32")),
        ]
    )
    intent = IntentFilters(
        raw_query="family adventure movies for teens",
        genres=["Family", "Adventure"],
        moods=[],
        media_types=["movie"],
        max_runtime=120,
        maturity_rating_max="PG-13",
    )

    result = recommend_routes._constraint_prior_candidates(
        session,
        intent,
        limit=5,
        allowlist=None,
        enforce_genres=True,
        rank_vector=np.array([1.0, 0.0], dtype="float32"),
    )

    assert result == [1]


def test_constraint_prior_candidates_prefers_query_similar_items():
    class ConstraintSession:
        def __init__(self, rows):
            self.rows = rows
            self.bind = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))

        def execute(self, statement):
            return FakeResult(self.rows)

    semantically_close = SimpleNamespace(
        id=1,
        media_type="movie",
        runtime=105,
        maturity_rating="PG-13",
        genres=[{"name": "Comedy"}, {"name": "Romance"}],
        popular_rank=100,
        trending_rank=100,
        popularity=10.0,
        vote_count=100,
    )
    semantically_far_but_popular = SimpleNamespace(
        id=2,
        media_type="movie",
        runtime=105,
        maturity_rating="PG-13",
        genres=[{"name": "Comedy"}, {"name": "Romance"}],
        popular_rank=1,
        trending_rank=1,
        popularity=90.0,
        vote_count=5000,
    )

    session = ConstraintSession(
        [
            (semantically_far_but_popular, np.array([0.0, 1.0], dtype="float32")),
            (semantically_close, np.array([1.0, 0.0], dtype="float32")),
        ]
    )
    intent = IntentFilters(
        raw_query="rom-coms from the 2000s",
        genres=["Comedy", "Romance"],
        moods=[],
        media_types=["movie"],
    )

    result = recommend_routes._constraint_prior_candidates(
        session,
        intent,
        limit=2,
        allowlist=None,
        enforce_genres=True,
        rank_vector=np.array([1.0, 0.0], dtype="float32"),
    )

    assert result == [1, 2]


def test_constraint_prior_candidates_applies_noir_lexical_bonus():
    class ConstraintSession:
        def __init__(self, rows):
            self.rows = rows
            self.bind = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))

        def execute(self, statement):
            return FakeResult(self.rows)

    noir_match = SimpleNamespace(
        id=1,
        title="Neon City",
        overview="A modern noir mystery in the city.",
        media_type="movie",
        runtime=110,
        maturity_rating="R",
        genres=[{"name": "Mystery"}, {"name": "Crime"}],
        popular_rank=100,
        trending_rank=100,
        popularity=5.0,
        vote_count=50,
        release_year=2014,
    )
    generic_popular = SimpleNamespace(
        id=2,
        title="Big Popular Thriller",
        overview="A blockbuster thriller.",
        media_type="movie",
        runtime=110,
        maturity_rating="R",
        genres=[{"name": "Thriller"}],
        popular_rank=1,
        trending_rank=1,
        popularity=95.0,
        vote_count=5000,
        release_year=2014,
    )

    session = ConstraintSession(
        [
            (generic_popular, np.array([0.9, 0.1], dtype="float32")),
            (noir_match, np.array([0.85, 0.15], dtype="float32")),
        ]
    )
    intent = IntentFilters(
        raw_query="neo-noir mysteries in modern cities",
        genres=["Mystery", "Crime", "Thriller"],
        moods=[],
        media_types=["movie"],
        year_min=1990,
    )

    result = recommend_routes._constraint_prior_candidates(
        session,
        intent,
        limit=2,
        allowlist=None,
        enforce_genres=True,
        rank_vector=np.array([1.0, 0.0], dtype="float32"),
    )

    assert result == [1, 2]


def test_constraint_query_bonus_penalizes_scifi_action_spillover_for_noir():
    noir_intent = IntentFilters(
        raw_query="neo-noir mysteries in modern cities",
        genres=["Mystery", "Crime", "Thriller"],
        moods=[],
        media_types=["movie"],
        year_min=1990,
    )
    noir_item = SimpleNamespace(
        title="Neon City",
        overview="A homicide detective investigates a murder in the city.",
        genres=[{"name": "Crime"}, {"name": "Mystery"}, {"name": "Thriller"}],
        release_year=2010,
    )
    spillover_item = SimpleNamespace(
        title="Monster Attack",
        overview="A monster descends on a beach town.",
        genres=[{"name": "Action"}, {"name": "Science Fiction"}, {"name": "Thriller"}],
        release_year=2010,
    )

    assert recommend_routes._constraint_query_bonus(
        noir_intent, noir_item
    ) > recommend_routes._constraint_query_bonus(noir_intent, spillover_item)


def test_constraint_query_bonus_prefers_noir_detective_over_buddy_cop_for_urban_noir_movies():
    noir_intent = IntentFilters(
        raw_query="urban noir detective movies",
        genres=["Crime", "Mystery", "Thriller"],
        moods=[],
        media_types=["movie"],
    )
    noir_item = SimpleNamespace(
        title="Neon City",
        overview="A detective investigates a murder in a corrupt urban underworld.",
        genres=[{"name": "Crime"}, {"name": "Mystery"}, {"name": "Thriller"}],
        media_type="movie",
    )
    buddy_item = SimpleNamespace(
        title="Rush Hour Two",
        overview="A buddy cop duo heads on a vacation chase after criminals.",
        genres=[{"name": "Action"}, {"name": "Comedy"}],
        media_type="movie",
    )

    assert recommend_routes._constraint_query_bonus(
        noir_intent, noir_item
    ) > recommend_routes._constraint_query_bonus(noir_intent, buddy_item)


def test_constraint_query_bonus_prefers_wealth_thriller_over_generic_crime_for_money_obsession_queries():
    greed_intent = IntentFilters(
        raw_query="psychological thrillers about wealth and obsession",
        genres=["Thriller"],
        moods=[],
        media_types=["movie"],
    )
    wealth_thriller = SimpleNamespace(
        title="Empire of Desire",
        overview="A wealthy investment banker spirals into greed, status obsession, and murder.",
        genres=[{"name": "Thriller"}, {"name": "Drama"}, {"name": "Crime"}],
        media_type="movie",
    )
    generic_crime = SimpleNamespace(
        title="City Pursuit",
        overview="A detective chases a killer across the city.",
        genres=[{"name": "Thriller"}, {"name": "Crime"}],
        media_type="movie",
    )

    assert recommend_routes._constraint_query_bonus(
        greed_intent, wealth_thriller
    ) > recommend_routes._constraint_query_bonus(greed_intent, generic_crime)


def test_constraint_prior_candidates_applies_money_psychology_bonus():
    class ConstraintSession:
        def __init__(self, rows):
            self.rows = rows
            self.bind = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))

        def execute(self, statement):
            return FakeResult(self.rows)

    wealth_match = SimpleNamespace(
        id=1,
        title="Empire of Desire",
        overview="A wealthy banker and his elite social circle spiral into greed and obsession.",
        media_type="movie",
        runtime=110,
        maturity_rating="R",
        genres=[{"name": "Thriller"}, {"name": "Drama"}, {"name": "Crime"}],
        popular_rank=200,
        trending_rank=200,
        popularity=8.0,
        vote_count=120,
        release_year=2005,
    )
    generic_popular = SimpleNamespace(
        id=2,
        title="Big Popular Thriller",
        overview="A blockbuster thriller with a murder investigation.",
        media_type="movie",
        runtime=110,
        maturity_rating="R",
        genres=[{"name": "Thriller"}],
        popular_rank=1,
        trending_rank=1,
        popularity=95.0,
        vote_count=5000,
        release_year=2005,
    )

    session = ConstraintSession(
        [
            (generic_popular, np.array([0.9, 0.1], dtype="float32")),
            (wealth_match, np.array([0.85, 0.15], dtype="float32")),
        ]
    )
    intent = IntentFilters(
        raw_query="mind games and thrillers about wealth and greed",
        genres=["Thriller"],
        moods=[],
        media_types=["movie"],
    )

    result = recommend_routes._constraint_prior_candidates(
        session,
        intent,
        limit=2,
        allowlist=None,
        enforce_genres=True,
        rank_vector=np.array([1.0, 0.0], dtype="float32"),
    )

    assert result == [1, 2]


def test_constraint_query_bonus_prefers_rom_com_over_action_romance_for_rom_com_queries():
    rom_com_intent = IntentFilters(
        raw_query="romantic comedies from the 2000s",
        genres=["Romance", "Comedy"],
        moods=[],
        media_types=["movie"],
        year_min=2000,
        year_max=2009,
    )
    rom_com_item = SimpleNamespace(
        title="Meet Cute Again",
        overview="Two strangers fall in love after a disastrous first date.",
        genres=[{"name": "Comedy"}, {"name": "Romance"}],
        media_type="movie",
        release_year=2004,
    )
    action_romance_item = SimpleNamespace(
        title="Target Hearts",
        overview="Two assassins on a mission discover unexpected chemistry.",
        genres=[{"name": "Action"}, {"name": "Romance"}],
        media_type="movie",
        release_year=2005,
    )

    assert recommend_routes._constraint_query_bonus(
        rom_com_intent, rom_com_item
    ) > recommend_routes._constraint_query_bonus(rom_com_intent, action_romance_item)


def test_constraint_prior_candidates_applies_rom_com_bonus():
    class ConstraintSession:
        def __init__(self, rows):
            self.rows = rows
            self.bind = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))

        def execute(self, statement):
            return FakeResult(self.rows)

    rom_com_match = SimpleNamespace(
        id=1,
        title="Meet Cute Again",
        overview="Two strangers fall in love after a disastrous first date.",
        media_type="movie",
        runtime=105,
        maturity_rating="PG-13",
        genres=[{"name": "Comedy"}, {"name": "Romance"}],
        popular_rank=120,
        trending_rank=120,
        popularity=8.0,
        vote_count=400,
        release_year=2004,
    )
    broad_popular = SimpleNamespace(
        id=2,
        title="Action Hearts",
        overview="Two spies flirt while stopping an international plot.",
        media_type="movie",
        runtime=118,
        maturity_rating="PG-13",
        genres=[{"name": "Action"}, {"name": "Romance"}],
        popular_rank=1,
        trending_rank=1,
        popularity=95.0,
        vote_count=5000,
        release_year=2005,
    )

    session = ConstraintSession(
        [
            (broad_popular, np.array([0.9, 0.1], dtype="float32")),
            (rom_com_match, np.array([0.85, 0.15], dtype="float32")),
        ]
    )
    intent = IntentFilters(
        raw_query="romantic comedies from the 2000s",
        genres=["Romance", "Comedy"],
        moods=[],
        media_types=["movie"],
        year_min=2000,
        year_max=2009,
    )

    result = recommend_routes._constraint_prior_candidates(
        session,
        intent,
        limit=2,
        allowlist=None,
        enforce_genres=True,
        rank_vector=np.array([1.0, 0.0], dtype="float32"),
    )

    assert result == [1, 2]


def test_constraint_prior_candidates_applies_serialized_tv_bonus():
    class ConstraintSession:
        def __init__(self, rows):
            self.rows = rows
            self.bind = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))

        def execute(self, statement):
            return FakeResult(self.rows)

    prestige_match = SimpleNamespace(
        id=1,
        title="Kingdom of Power",
        overview="A political power struggle erupts during wartime.",
        media_type="tv",
        runtime=55,
        maturity_rating="TV-MA",
        genres=[{"name": "Drama"}, {"name": "War & Politics"}],
        popular_rank=100,
        trending_rank=100,
        popularity=12.0,
        vote_count=200,
        release_year=2018,
    )
    procedural_popular = SimpleNamespace(
        id=2,
        title="Case Files",
        overview="FBI agents solve a new case every week.",
        media_type="tv",
        runtime=45,
        maturity_rating="TV-14",
        genres=[{"name": "Drama"}, {"name": "Crime"}],
        popular_rank=1,
        trending_rank=1,
        popularity=95.0,
        vote_count=5000,
        release_year=2018,
    )

    session = ConstraintSession(
        [
            (procedural_popular, np.array([0.9, 0.1], dtype="float32")),
            (prestige_match, np.array([0.85, 0.15], dtype="float32")),
        ]
    )
    intent = IntentFilters(
        raw_query="prestige survival drama series",
        genres=["Drama"],
        moods=[],
        media_types=["tv"],
    )

    result = recommend_routes._constraint_prior_candidates(
        session,
        intent,
        limit=2,
        allowlist=None,
        enforce_genres=True,
        rank_vector=np.array([1.0, 0.0], dtype="float32"),
    )

    assert result == [1, 2]


def test_constraint_query_bonus_penalizes_procedural_spillover_for_serialized_tv():
    serialized_intent = IntentFilters(
        raw_query="prestige survival drama series",
        genres=["Drama"],
        moods=[],
        media_types=["tv"],
    )
    serialized_item = SimpleNamespace(
        title="Kingdom of Power",
        overview="A political power struggle erupts during wartime.",
        genres=[{"name": "Drama"}, {"name": "War & Politics"}],
        release_year=2018,
    )
    procedural_item = SimpleNamespace(
        title="Case Files",
        overview="FBI agents solve a new case every week.",
        genres=[{"name": "Drama"}, {"name": "Crime"}],
        release_year=2018,
    )

    assert recommend_routes._constraint_query_bonus(
        serialized_intent, serialized_item
    ) > recommend_routes._constraint_query_bonus(serialized_intent, procedural_item)


def test_constraint_query_bonus_prefers_dark_prestige_over_finance_drama_for_serialized_tv():
    serialized_intent = IntentFilters(
        raw_query="prestige survival drama series",
        genres=["Drama"],
        moods=[],
        media_types=["tv"],
    )
    dark_prestige_item = SimpleNamespace(
        title="Empire of Ashes",
        overview="A corrupt empire descends into rebellion and survival war after an apocalypse.",
        genres=[
            {"name": "Drama"},
            {"name": "War & Politics"},
            {"name": "Sci-Fi & Fantasy"},
        ],
        release_year=2020,
    )
    finance_drama_item = SimpleNamespace(
        title="Capital Billions",
        overview="A billionaire hedge fund founder battles rivals in the courtroom and boardroom.",
        genres=[{"name": "Drama"}],
        release_year=2020,
    )

    assert recommend_routes._constraint_query_bonus(
        serialized_intent, dark_prestige_item
    ) > recommend_routes._constraint_query_bonus(serialized_intent, finance_drama_item)


def test_constraint_query_bonus_prefers_fantasy_scifi_adventure_for_kids_profile_queries():
    kids_intent = IntentFilters(
        raw_query="kids profile: bilingual adventure picks across movies and TV",
        genres=["Adventure"],
        moods=[],
        media_types=["movie", "tv"],
    )
    fantasy_scifi_item = SimpleNamespace(
        title="Quest Across the Galaxy",
        overview="A heroic space quest with magic, aliens, and a starship crew.",
        genres=[{"name": "Action & Adventure"}, {"name": "Sci-Fi & Fantasy"}],
        release_year=2017,
    )
    generic_family_item = SimpleNamespace(
        title="Holiday Escape",
        overview="A family takes a trip across the city during the holidays.",
        genres=[{"name": "Family"}, {"name": "Comedy"}],
        release_year=2017,
    )

    assert recommend_routes._constraint_query_bonus(
        kids_intent, fantasy_scifi_item
    ) > recommend_routes._constraint_query_bonus(kids_intent, generic_family_item)


def test_constraint_query_bonus_prefers_multilingual_kids_adventure_titles_for_family_language_queries():
    multilingual_intent = IntentFilters(
        raw_query="family-friendly bilingual adventure movies and shows for kids",
        genres=["Family", "Adventure"],
        moods=[],
        media_types=["movie", "tv"],
    )
    multilingual_kids_item = SimpleNamespace(
        title="Paris Hero Club",
        overview="Two friends begin a magical quest across Paris to protect their city.",
        genres=[
            {"name": "Animation"},
            {"name": "Action & Adventure"},
            {"name": "Kids"},
        ],
        original_language="fr",
        maturity_rating="TV-Y7",
        media_type="tv",
        release_year=2021,
    )
    generic_blockbuster_item = SimpleNamespace(
        title="Big Franchise Adventure",
        overview="A famous team battles villains across the universe.",
        genres=[{"name": "Action"}, {"name": "Adventure"}, {"name": "Fantasy"}],
        original_language="en",
        maturity_rating="PG-13",
        media_type="movie",
        release_year=2025,
    )

    assert recommend_routes._constraint_query_bonus(
        multilingual_intent, multilingual_kids_item
    ) > recommend_routes._constraint_query_bonus(
        multilingual_intent, generic_blockbuster_item
    )


def test_constraint_query_bonus_prefers_romantic_comedy_for_date_night_queries():
    date_night_intent = IntentFilters(
        raw_query="light comedy date-night movies",
        genres=["Comedy", "Romance"],
        moods=["romantic"],
        media_types=["movie"],
    )
    romantic_comedy_item = SimpleNamespace(
        title="Summer of Love",
        overview="A heartwarming romantic relationship blossoms at a wedding.",
        genres=[{"name": "Comedy"}, {"name": "Romance"}, {"name": "Drama"}],
    )
    slapstick_item = SimpleNamespace(
        title="Bro Pranks",
        overview="A gross-out spoof full of slapstick chaos and zombies.",
        genres=[{"name": "Comedy"}],
    )

    assert recommend_routes._constraint_query_bonus(
        date_night_intent, romantic_comedy_item
    ) > recommend_routes._constraint_query_bonus(date_night_intent, slapstick_item)


def test_constraint_query_bonus_penalizes_epic_drama_spillover_for_date_night_queries():
    date_night_intent = IntentFilters(
        raw_query="light comedy date-night movies",
        genres=["Comedy", "Romance"],
        moods=["romantic"],
        media_types=["movie"],
    )
    romantic_comedy_item = SimpleNamespace(
        title="Summer of Love",
        overview="A heartwarming romantic relationship blossoms at a wedding.",
        genres=[{"name": "Comedy"}, {"name": "Romance"}, {"name": "Drama"}],
        runtime=104,
    )
    epic_drama_item = SimpleNamespace(
        title="History of a Life",
        overview="A man moves through war, history, and presidents in an epic life story.",
        genres=[{"name": "Drama"}, {"name": "Romance"}],
        runtime=148,
    )

    assert recommend_routes._constraint_query_bonus(
        date_night_intent, romantic_comedy_item
    ) > recommend_routes._constraint_query_bonus(date_night_intent, epic_drama_item)


def test_constraint_query_bonus_prefers_warm_food_romance_over_scifi_romance_for_date_night_queries():
    date_night_intent = IntentFilters(
        raw_query="light comedy date-night movies",
        genres=["Comedy", "Romance"],
        moods=["romantic"],
        media_types=["movie"],
    )
    warm_food_item = SimpleNamespace(
        title="Dinner for Two",
        overview="A warm chef comedy about food, romance, and a summer road trip.",
        genres=[{"name": "Comedy"}, {"name": "Drama"}, {"name": "Romance"}],
        runtime=110,
    )
    scifi_romance_item = SimpleNamespace(
        title="Future Hearts",
        overview="A lonely man falls in love with an operating system in a melancholy future.",
        genres=[{"name": "Romance"}, {"name": "Drama"}, {"name": "Science Fiction"}],
        runtime=126,
    )

    assert recommend_routes._constraint_query_bonus(
        date_night_intent, warm_food_item
    ) > recommend_routes._constraint_query_bonus(date_night_intent, scifi_romance_item)


def test_constraint_query_bonus_prefers_music_romcom_over_family_fantasy_for_date_night_queries():
    date_night_intent = IntentFilters(
        raw_query="light comedy date-night movies",
        genres=["Comedy", "Romance"],
        moods=["romantic"],
        media_types=["movie"],
    )
    music_romcom_item = SimpleNamespace(
        title="Summer Songs",
        overview="A feel-good band romance about music, heartbreak, and falling in love.",
        genres=[{"name": "Comedy"}, {"name": "Drama"}, {"name": "Romance"}, {"name": "Music"}],
        runtime=102,
    )
    family_fantasy_item = SimpleNamespace(
        title="Wish Clock",
        overview="A boy makes a magical wish and becomes an adult overnight.",
        genres=[{"name": "Comedy"}, {"name": "Family"}, {"name": "Fantasy"}],
        runtime=104,
    )

    assert recommend_routes._constraint_query_bonus(
        date_night_intent, music_romcom_item
    ) > recommend_routes._constraint_query_bonus(date_night_intent, family_fantasy_item)


def test_constraint_query_bonus_prefers_heartwarming_over_slapstick_for_feel_good_comedy_queries():
    feel_good_intent = IntentFilters(
        raw_query="feel-good comedies under 125 minutes",
        genres=["Comedy"],
        moods=[],
        media_types=["movie"],
    )
    heartwarming_item = SimpleNamespace(
        title="Kitchen Dreams",
        overview="An uplifting chef story about friendship, music, and summer.",
        genres=[{"name": "Comedy"}, {"name": "Drama"}],
    )
    slapstick_item = SimpleNamespace(
        title="Chaos Brothers",
        overview="A gross-out slapstick spoof with gangsters and zombies.",
        genres=[{"name": "Comedy"}],
    )

    assert recommend_routes._constraint_query_bonus(
        feel_good_intent, heartwarming_item
    ) > recommend_routes._constraint_query_bonus(feel_good_intent, slapstick_item)


def test_constraint_query_bonus_prefers_family_fantasy_over_spy_action_for_family_adventure_queries():
    family_intent = IntentFilters(
        raw_query="family adventure movies rated PG-13 or below",
        genres=["Family", "Adventure"],
        moods=[],
        media_types=["movie"],
        maturity_rating_max="PG-13",
    )
    family_fantasy_item = SimpleNamespace(
        title="Quest for the Kingdom",
        overview="A heroic wizard leads a family quest through a magical kingdom.",
        genres=[{"name": "Family"}, {"name": "Adventure"}, {"name": "Fantasy"}],
        maturity_rating="PG-13",
    )
    spy_action_item = SimpleNamespace(
        title="Agent Storm",
        overview="A spy agent races across the world to stop assassins.",
        genres=[{"name": "Action"}],
        maturity_rating="PG-13",
    )

    assert recommend_routes._constraint_query_bonus(
        family_intent, family_fantasy_item
    ) > recommend_routes._constraint_query_bonus(family_intent, spy_action_item)


def test_constraint_query_bonus_prefers_caper_over_procedural_for_short_crime_tv_queries():
    short_crime_intent = IntentFilters(
        raw_query="quick caper crime series",
        genres=["Crime"],
        moods=[],
        media_types=["tv"],
        max_runtime=50,
    )
    caper_item = SimpleNamespace(
        title="The Gentleman Thief",
        overview="A con artist leads a heist crew through a string of robberies.",
        genres=[{"name": "Crime"}, {"name": "Drama"}],
    )
    procedural_item = SimpleNamespace(
        title="CSI Squad",
        overview="A forensic unit solves homicide cases with the FBI each week.",
        genres=[{"name": "Crime"}, {"name": "Drama"}],
    )

    assert recommend_routes._constraint_query_bonus(
        short_crime_intent, caper_item
    ) > recommend_routes._constraint_query_bonus(short_crime_intent, procedural_item)


def test_constraint_query_bonus_prefers_caper_for_short_heist_show_paraphrase():
    short_caper_intent = IntentFilters(
        raw_query="short grifter crime shows",
        genres=["Crime"],
        moods=[],
        media_types=["tv"],
        max_runtime=50,
    )
    caper_item = SimpleNamespace(
        title="The Gentleman Thief",
        overview="A con artist leads a heist crew through a string of robberies.",
        genres=[{"name": "Crime"}, {"name": "Drama"}],
    )
    procedural_item = SimpleNamespace(
        title="CSI Squad",
        overview="A forensic unit solves homicide cases with the FBI each week.",
        genres=[{"name": "Crime"}, {"name": "Drama"}],
    )

    assert recommend_routes._constraint_query_bonus(
        short_caper_intent, caper_item
    ) > recommend_routes._constraint_query_bonus(short_caper_intent, procedural_item)


def test_constraint_query_bonus_prefers_heist_shows_over_procedurals_for_heist_tv_queries():
    heist_intent = IntentFilters(
        raw_query="robbery caper show",
        genres=["Crime", "Drama"],
        moods=[],
        media_types=["tv"],
    )
    heist_item = SimpleNamespace(
        title="The Crew Job",
        overview="A mastermind recruits a crew for a daring robbery and inside job.",
        genres=[{"name": "Crime"}, {"name": "Drama"}],
        media_type="tv",
    )
    procedural_item = SimpleNamespace(
        title="CSI Squad",
        overview="A forensic unit solves homicide cases with the FBI each week.",
        genres=[{"name": "Crime"}, {"name": "Drama"}],
        media_type="tv",
    )

    assert recommend_routes._constraint_query_bonus(
        heist_intent, heist_item
    ) > recommend_routes._constraint_query_bonus(heist_intent, procedural_item)


def test_constraint_query_bonus_penalizes_horror_spillover_for_caper_crime_tv_queries():
    short_caper_intent = IntentFilters(
        raw_query="short grifter crime shows",
        genres=["Crime"],
        moods=[],
        media_types=["tv"],
        max_runtime=50,
    )
    caper_item = SimpleNamespace(
        title="The Gentleman Thief",
        overview="A con artist leads a heist crew through a string of robberies.",
        genres=[{"name": "Crime"}, {"name": "Drama"}],
    )
    horror_item = SimpleNamespace(
        title="Night Terrors",
        overview="A supernatural monster haunts a crime-ridden town.",
        genres=[{"name": "Crime"}, {"name": "Horror"}],
    )

    assert recommend_routes._constraint_query_bonus(
        short_caper_intent, caper_item
    ) > recommend_routes._constraint_query_bonus(short_caper_intent, horror_item)


def test_constraint_query_bonus_prefers_non_english_crime_for_international_queries():
    international_crime_intent = IntentFilters(
        raw_query="international crime picks across movies and TV",
        genres=["Crime"],
        moods=[],
        media_types=["movie", "tv"],
    )
    international_item = SimpleNamespace(
        title="Le Gang",
        overview="A French heist crew dives into the criminal underworld.",
        genres=[{"name": "Crime"}, {"name": "Drama"}],
        original_language="fr",
    )
    procedural_item = SimpleNamespace(
        title="NCIS Metro",
        overview="An FBI forensic team solves crime in New York.",
        genres=[{"name": "Crime"}, {"name": "Drama"}],
        original_language="en",
    )

    assert recommend_routes._constraint_query_bonus(
        international_crime_intent, international_item
    ) > recommend_routes._constraint_query_bonus(
        international_crime_intent, procedural_item
    )


def test_constraint_query_bonus_prefers_non_english_crime_for_european_crime_paraphrase():
    european_crime_intent = IntentFilters(
        raw_query="european crime films and series",
        genres=["Crime"],
        moods=[],
        media_types=["movie", "tv"],
    )
    international_item = SimpleNamespace(
        title="Le Gang",
        overview="A French heist crew dives into the criminal underworld.",
        genres=[{"name": "Crime"}, {"name": "Drama"}],
        original_language="fr",
    )
    procedural_item = SimpleNamespace(
        title="NCIS Metro",
        overview="An FBI forensic team solves crime in New York.",
        genres=[{"name": "Crime"}, {"name": "Drama"}],
        original_language="en",
    )

    assert recommend_routes._constraint_query_bonus(
        european_crime_intent, international_item
    ) > recommend_routes._constraint_query_bonus(
        european_crime_intent, procedural_item
    )


def test_constraint_query_bonus_prefers_grounded_vigilante_for_street_level_superhero_queries():
    vigilante_intent = IntentFilters(
        raw_query="grounded vigilante series",
        genres=["Action & Adventure", "Crime", "Drama"],
        moods=[],
        media_types=["tv"],
    )
    vigilante_item = SimpleNamespace(
        title="Night Watch",
        overview="A grounded masked vigilante takes on urban crime and corruption.",
        genres=[{"name": "Action & Adventure"}, {"name": "Crime"}, {"name": "Drama"}],
        media_type="tv",
    )
    cosmic_item = SimpleNamespace(
        title="Galaxy Squad",
        overview="A cosmic hero battles threats across the multiverse.",
        genres=[{"name": "Action & Adventure"}, {"name": "Fantasy"}],
        media_type="tv",
    )

    assert recommend_routes._constraint_query_bonus(
        vigilante_intent, vigilante_item
    ) > recommend_routes._constraint_query_bonus(vigilante_intent, cosmic_item)


def test_constraint_query_bonus_penalizes_procedural_and_animation_for_grounded_vigilante_queries():
    vigilante_intent = IntentFilters(
        raw_query="grounded vigilante series",
        genres=["Action & Adventure", "Crime", "Drama"],
        moods=[],
        media_types=["tv"],
    )
    vigilante_item = SimpleNamespace(
        title="Night Watch",
        overview="A grounded masked vigilante takes on urban crime and corruption.",
        genres=[{"name": "Action & Adventure"}, {"name": "Crime"}, {"name": "Drama"}],
        media_type="tv",
    )
    procedural_item = SimpleNamespace(
        title="Metro Unit",
        overview="An FBI consultant works homicide cases with a forensic police unit.",
        genres=[{"name": "Crime"}, {"name": "Drama"}],
        media_type="tv",
    )
    animated_item = SimpleNamespace(
        title="Hero Toons",
        overview="A cartoon superhero protects the city from colorful villains.",
        genres=[{"name": "Animation"}, {"name": "Action & Adventure"}],
        media_type="tv",
    )

    assert recommend_routes._constraint_query_bonus(
        vigilante_intent, vigilante_item
    ) > recommend_routes._constraint_query_bonus(vigilante_intent, procedural_item)
    assert recommend_routes._constraint_query_bonus(
        vigilante_intent, vigilante_item
    ) > recommend_routes._constraint_query_bonus(vigilante_intent, animated_item)


def test_constraint_query_bonus_prefers_japanese_scifi_anime_for_anime_scifi_queries():
    anime_intent = IntentFilters(
        raw_query="anime sci-fi films",
        genres=["Animation", "Science Fiction"],
        moods=[],
        media_types=["movie"],
    )
    anime_item = SimpleNamespace(
        title="Neo Tokyo",
        overview="A cyberpunk anime about android memories in futuristic Tokyo.",
        genres=[{"name": "Animation"}, {"name": "Science Fiction"}],
        media_type="movie",
        original_language="ja",
    )
    family_item = SimpleNamespace(
        title="Holiday Dragons",
        overview="A family adventure with dragons and musical fun.",
        genres=[{"name": "Animation"}, {"name": "Family"}],
        media_type="movie",
        original_language="en",
    )

    assert recommend_routes._constraint_query_bonus(
        anime_intent, anime_item
    ) > recommend_routes._constraint_query_bonus(anime_intent, family_item)


def test_constraint_query_bonus_prefers_time_loop_scifi_for_time_bending_queries():
    time_bending_intent = IntentFilters(
        raw_query="time-travel thrillers",
        genres=["Thriller", "Science Fiction", "Mystery"],
        moods=[],
        media_types=["movie"],
        year_min=2000,
    )
    time_loop_item = SimpleNamespace(
        title="Loop Protocol",
        overview="A detective is trapped in a time loop and uncovers a paradox.",
        genres=[{"name": "Science Fiction"}, {"name": "Thriller"}, {"name": "Mystery"}],
        media_type="movie",
        release_year=2011,
    )
    creature_item = SimpleNamespace(
        title="Sea Monster",
        overview="A creature attacks a coastal town.",
        genres=[{"name": "Thriller"}, {"name": "Horror"}],
        media_type="movie",
        release_year=2011,
    )

    assert recommend_routes._constraint_query_bonus(
        time_bending_intent, time_loop_item
    ) > recommend_routes._constraint_query_bonus(time_bending_intent, creature_item)


def test_constraint_query_bonus_prefers_short_twisty_thriller_for_high_concept_queries():
    high_concept_intent = IntentFilters(
        raw_query="short high-concept thriller movies",
        genres=["Thriller", "Mystery"],
        moods=[],
        media_types=["movie"],
        max_runtime=125,
    )
    twisty_item = SimpleNamespace(
        title="Night Shift",
        overview="A journalist becomes obsessed with a mysterious double life.",
        genres=[{"name": "Thriller"}, {"name": "Mystery"}],
        media_type="movie",
        runtime=110,
    )
    creature_item = SimpleNamespace(
        title="Predator Bay",
        overview="A creature stalks the coast.",
        genres=[{"name": "Thriller"}, {"name": "Horror"}],
        media_type="movie",
        runtime=110,
    )

    assert recommend_routes._constraint_query_bonus(
        high_concept_intent, twisty_item
    ) > recommend_routes._constraint_query_bonus(high_concept_intent, creature_item)


def test_constraint_query_bonus_prefers_cerebral_time_thrillers_over_family_scifi_for_cerebral_queries():
    brainy_intent = IntentFilters(
        raw_query="cerebral paradox thrillers",
        genres=["Thriller", "Science Fiction", "Mystery"],
        moods=[],
        media_types=["movie"],
        year_min=2000,
    )
    cerebral_item = SimpleNamespace(
        title="Dream Architects",
        overview="A dream inversion puzzle bends memory and identity through time.",
        genres=[{"name": "Science Fiction"}, {"name": "Thriller"}, {"name": "Mystery"}],
        media_type="movie",
        release_year=2010,
    )
    family_scifi_item = SimpleNamespace(
        title="Tomorrow Park",
        overview="A family visits a futuristic theme park full of gadgets.",
        genres=[{"name": "Science Fiction"}, {"name": "Family"}, {"name": "Action & Adventure"}],
        media_type="movie",
        release_year=2015,
    )

    assert recommend_routes._constraint_query_bonus(
        brainy_intent, cerebral_item
    ) > recommend_routes._constraint_query_bonus(brainy_intent, family_scifi_item)


def test_constraint_query_bonus_penalizes_spy_action_spillover_for_cerebral_temporal_queries():
    cerebral_intent = IntentFilters(
        raw_query="cerebral time-bending thrillers",
        genres=["Thriller", "Science Fiction", "Mystery"],
        moods=[],
        media_types=["movie"],
        year_min=2000,
    )
    puzzle_item = SimpleNamespace(
        title="The Thought Experiment",
        overview="A scientist's experiment triggers a paradox that fractures memory and time.",
        genres=[{"name": "Science Fiction"}, {"name": "Thriller"}, {"name": "Mystery"}],
        media_type="movie",
        release_year=2011,
    )
    spy_item = SimpleNamespace(
        title="The Operative Files",
        overview="A spy agent races through a dystopian mission after a rogue operative.",
        genres=[{"name": "Action & Adventure"}, {"name": "Thriller"}],
        media_type="movie",
        release_year=2012,
    )

    assert recommend_routes._constraint_query_bonus(
        cerebral_intent, puzzle_item
    ) > recommend_routes._constraint_query_bonus(cerebral_intent, spy_item)


def test_apply_mixer_scores_respects_constraint_prior_signal():
    candidates = [
        {
            "id": 1,
            "title": "Weak ANN",
            "ann_rank": 8,
            "original_rank": 8,
            "source_scores": {"ann": 1.0},
            "popularity": 1.0,
            "vote_count": 10,
        },
        {
            "id": 2,
            "title": "Constraint Prior Match",
            "ann_rank": 20,
            "original_rank": 20,
            "source_scores": {"ann": 0.2, "constraint_prior": 1.0},
            "popularity": 1.0,
            "vote_count": 10,
        },
    ]

    recommend_routes._apply_mixer_scores(candidates)

    assert candidates[0]["id"] == 2
    assert candidates[0]["retrieval_score"] > candidates[1]["retrieval_score"]


def test_constraint_query_bonus_prefers_space_crew_scifi_for_optimistic_tv_queries():
    scifi_tv_intent = IntentFilters(
        raw_query="optimistic sci-fi TV",
        genres=["Sci-Fi & Fantasy"],
        moods=[],
        media_types=["tv"],
    )
    optimistic_item = SimpleNamespace(
        title="Starbound Crew",
        overview="An optimistic starship crew explores the galaxy and new planets.",
        genres=[{"name": "Sci-Fi & Fantasy"}, {"name": "Action & Adventure"}],
        media_type="tv",
    )
    procedural_item = SimpleNamespace(
        title="Metro Cases",
        overview="A detective squad investigates murders in New York.",
        genres=[{"name": "Crime"}, {"name": "Drama"}],
        media_type="tv",
    )

    assert recommend_routes._constraint_query_bonus(
        scifi_tv_intent, optimistic_item
    ) > recommend_routes._constraint_query_bonus(scifi_tv_intent, procedural_item)


def test_constraint_query_bonus_prefers_short_scifi_over_non_scifi_tv_for_bingeable_queries():
    short_scifi_intent = IntentFilters(
        raw_query="short bingeable sci-fi TV",
        genres=["Sci-Fi & Fantasy"],
        moods=[],
        media_types=["tv"],
        max_runtime=50,
    )
    short_scifi_item = SimpleNamespace(
        title="Continuum Drift",
        overview="A time travel detective races through a future timeline.",
        genres=[{"name": "Sci-Fi & Fantasy"}],
        media_type="tv",
        runtime=44,
    )
    sitcom_item = SimpleNamespace(
        title="Family Laughs",
        overview="A family sitcom follows daily life at the hospital.",
        genres=[{"name": "Comedy"}],
        media_type="tv",
        runtime=44,
    )

    assert recommend_routes._constraint_query_bonus(
        short_scifi_intent, short_scifi_item
    ) > recommend_routes._constraint_query_bonus(short_scifi_intent, sitcom_item)


def test_constraint_query_bonus_prefers_ensemble_scifi_over_procedural_for_bingeable_queries():
    short_scifi_intent = IntentFilters(
        raw_query="short bingeable sci-fi TV",
        genres=["Sci-Fi & Fantasy"],
        moods=[],
        media_types=["tv"],
        max_runtime=50,
    )
    ensemble_item = SimpleNamespace(
        title="Star Crew",
        overview="A ragtag crew explores space on a mission across the frontier.",
        genres=[{"name": "Sci-Fi & Fantasy"}, {"name": "Action & Adventure"}],
        media_type="tv",
        runtime=44,
        release_year=2018,
    )
    procedural_item = SimpleNamespace(
        title="Future Detective",
        overview="A detective investigates strange crimes in a futuristic city.",
        genres=[{"name": "Sci-Fi & Fantasy"}, {"name": "Crime"}],
        media_type="tv",
        runtime=44,
        release_year=2018,
    )

    assert recommend_routes._constraint_query_bonus(
        short_scifi_intent, ensemble_item
    ) > recommend_routes._constraint_query_bonus(short_scifi_intent, procedural_item)


def test_constraint_query_bonus_penalizes_station_politics_for_bingeable_scifi_queries():
    short_scifi_intent = IntentFilters(
        raw_query="short bingeable sci-fi TV",
        genres=["Sci-Fi & Fantasy"],
        moods=[],
        media_types=["tv"],
        max_runtime=50,
    )
    adventure_item = SimpleNamespace(
        title="Star Crew",
        overview="A ragtag crew explores space on an exploratory ship.",
        genres=[{"name": "Sci-Fi & Fantasy"}, {"name": "Action & Adventure"}],
        media_type="tv",
        runtime=44,
        release_year=2018,
    )
    station_item = SimpleNamespace(
        title="Orbital Station",
        overview="A commander navigates politics on a remote space station.",
        genres=[
            {"name": "Sci-Fi & Fantasy"},
            {"name": "Drama"},
            {"name": "War & Politics"},
        ],
        media_type="tv",
        runtime=45,
        release_year=1995,
    )

    assert recommend_routes._constraint_query_bonus(
        short_scifi_intent, adventure_item
    ) > recommend_routes._constraint_query_bonus(short_scifi_intent, station_item)


def test_constraint_query_bonus_prefers_fantasy_tv_for_epic_comparison_queries():
    fantasy_tv_intent = IntentFilters(
        raw_query="fantasy epics like classic monster sagas",
        genres=["Fantasy", "Sci-Fi & Fantasy"],
        moods=[],
        media_types=["tv"],
    )
    fantasy_tv_item = SimpleNamespace(
        title="Kingdom of Ash",
        overview="A prophecy drives a monster hunter through a magical kingdom.",
        genres=[{"name": "Sci-Fi & Fantasy"}, {"name": "Action & Adventure"}],
        media_type="tv",
    )
    comedy_movie_item = SimpleNamespace(
        title="Road Trip",
        overview="Friends take a funny road trip across the country.",
        genres=[{"name": "Comedy"}],
        media_type="movie",
    )

    assert recommend_routes._constraint_query_bonus(
        fantasy_tv_intent, fantasy_tv_item
    ) > recommend_routes._constraint_query_bonus(fantasy_tv_intent, comedy_movie_item)


def test_constraint_query_bonus_prefers_modern_high_signal_prestige_over_old_cult_for_serialized_tv():
    serialized_intent = IntentFilters(
        raw_query="prestige survival drama series",
        genres=["Drama"],
        moods=[],
        media_types=["tv"],
    )
    modern_prestige_item = SimpleNamespace(
        title="Survivors of the Throne",
        overview="A brutal power struggle erupts after an apocalyptic civil war.",
        genres=[
            {"name": "Drama"},
            {"name": "War & Politics"},
            {"name": "Sci-Fi & Fantasy"},
        ],
        release_year=2023,
        vote_count=12000,
    )
    old_cult_item = SimpleNamespace(
        title="Cult Star Empire",
        overview="An offbeat fantasy crew drifts through the galaxy.",
        genres=[{"name": "Drama"}, {"name": "Sci-Fi & Fantasy"}],
        release_year=1999,
        vote_count=80,
    )

    assert recommend_routes._constraint_query_bonus(
        serialized_intent, modern_prestige_item
    ) > recommend_routes._constraint_query_bonus(serialized_intent, old_cult_item)


def test_effective_ann_description_drops_query_echo_when_rewrite_is_more_specific():
    assert (
        recommend_routes._effective_ann_description(
            raw_query="prestige survival drama series",
            ann_description="prestige survival drama series",
            rewrite_text="dark prestige fantasy apocalypse antihero political tv series",
        )
        is None
    )


def test_effective_ann_description_keeps_non_echo_description():
    assert (
        recommend_routes._effective_ann_description(
            raw_query="prestige survival drama series",
            ann_description="dark prestige antihero political survival drama series",
            rewrite_text="dark prestige fantasy apocalypse antihero political tv series",
        )
        == "dark prestige antihero political survival drama series"
    )


def test_apply_explicit_query_overrides_prefers_explicit_pg13_cap():
    intent = IntentFilters(
        raw_query="family adventure movies rated PG-13 or below",
        genres=["Family", "Adventure"],
        moods=[],
        media_types=["movie"],
        maturity_rating_max="PG",
    )

    updated = recommend_routes._apply_explicit_query_overrides(
        intent,
        "family adventure movies rated PG-13 or below",
    )

    assert updated.maturity_rating_max == "PG-13"


def test_apply_explicit_query_overrides_widens_teen_queries_to_pg13():
    intent = IntentFilters(
        raw_query="family adventure movies for teens",
        genres=["Family", "Adventure"],
        moods=[],
        media_types=["movie"],
        maturity_rating_max="PG",
    )

    updated = recommend_routes._apply_explicit_query_overrides(
        intent,
        "family adventure movies for teens",
    )

    assert updated.maturity_rating_max == "PG-13"


def test_apply_explicit_query_overrides_widens_family_friendly_queries_to_pg13():
    intent = IntentFilters(
        raw_query="family-friendly fantasy adventure movies",
        genres=["Family", "Adventure", "Fantasy"],
        moods=[],
        media_types=["movie"],
        maturity_rating_max="PG",
    )

    updated = recommend_routes._apply_explicit_query_overrides(
        intent,
        "family-friendly fantasy adventure movies",
    )

    assert updated.maturity_rating_max == "PG-13"


def test_apply_explicit_query_overrides_clears_unsolicited_low_cap_for_adult_movie_query():
    intent = IntentFilters(
        raw_query="feel-good movies on netflix with short runtime",
        genres=["Comedy", "Family"],
        moods=[],
        media_types=["movie"],
        maturity_rating_max="PG",
    )

    updated = recommend_routes._apply_explicit_query_overrides(
        intent,
        "feel-good movies on netflix with short runtime",
    )

    assert updated.maturity_rating_max is None


def test_apply_explicit_query_overrides_clears_unsolicited_low_cap_for_adult_comedy_runtime_query():
    intent = IntentFilters(
        raw_query="feel-good comedies on netflix under 125 minutes",
        genres=["Comedy", "Family"],
        moods=[],
        media_types=[],
        maturity_rating_max="PG",
    )

    updated = recommend_routes._apply_explicit_query_overrides(
        intent,
        "feel-good comedies on netflix under 125 minutes",
    )

    assert updated.maturity_rating_max is None


def test_apply_explicit_query_overrides_caps_not_too_dark_fantasy_tv_to_pg13():
    intent = IntentFilters(
        raw_query="not-too-dark fantasy tv from the last decade",
        genres=["Fantasy"],
        moods=["dark"],
        media_types=["tv"],
        maturity_rating_max=None,
    )

    updated = recommend_routes._apply_explicit_query_overrides(
        intent,
        "not-too-dark fantasy tv from the last decade",
    )

    assert updated.maturity_rating_max == "PG-13"


def test_apply_explicit_query_overrides_widens_kids_profile_queries_to_pg13():
    intent = IntentFilters(
        raw_query="kids profile: bilingual adventure picks across movies and TV",
        genres=["Family", "Animation", "Adventure"],
        moods=[],
        media_types=["movie", "tv"],
        maturity_rating_max="PG",
    )

    updated = recommend_routes._apply_explicit_query_overrides(
        intent,
        "kids profile: bilingual adventure picks across movies and TV",
    )

    assert updated.maturity_rating_max == "PG-13"


def test_normalize_merged_intent_strips_animation_from_movie_queries():
    intent = IntentFilters(
        raw_query="feel-good movies on netflix with short runtime",
        genres=["Comedy", "Family", "Animation"],
        moods=["light"],
        media_types=["movie"],
        max_runtime=125,
        maturity_rating_max="PG",
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "feel-good movies on netflix with short runtime",
    )

    assert updated.genres == ["Comedy", "Family"]
    assert updated.moods == []


def test_normalize_merged_intent_strips_family_from_adult_feel_good_comedy_queries_when_movie_detected():
    intent = IntentFilters(
        raw_query="feel-good comedies on netflix under 125 minutes",
        genres=["Comedy", "Family", "Animation"],
        moods=["light"],
        media_types=["movie"],
        max_runtime=125,
        maturity_rating_max="PG",
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "feel-good comedies on netflix under 125 minutes",
    )

    assert updated.media_types == ["movie"]
    assert updated.genres == ["Comedy"]
    assert updated.moods == []


def test_normalize_merged_intent_keeps_family_and_animation_for_kids_profile_queries():
    intent = IntentFilters(
        raw_query="kids profile: bilingual adventure picks across movies and TV",
        genres=["Family", "Animation", "Adventure"],
        moods=[],
        media_types=["movie", "tv"],
    )
    llm_intent = Intent(include_genres=["Family", "Animation", "Adventure"])

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "kids profile: bilingual adventure picks across movies and TV",
        llm_intent=llm_intent,
    )

    assert updated.genres == [
        "Family",
        "Animation",
        "Adventure",
        "Fantasy",
        "Sci-Fi & Fantasy",
    ]


def test_normalize_merged_intent_keeps_family_for_explicit_family_kids_queries():
    intent = IntentFilters(
        raw_query="kids profile: family adventure picks across movies and TV",
        genres=["Family", "Animation", "Adventure"],
        moods=[],
        media_types=["movie", "tv"],
    )
    llm_intent = Intent(include_genres=["Family", "Animation", "Adventure"])

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "kids profile: family adventure picks across movies and TV",
        llm_intent=llm_intent,
    )

    assert updated.genres == ["Family", "Animation", "Adventure"]


def test_normalize_merged_intent_keeps_animation_for_explicit_animation_queries():
    intent = IntentFilters(
        raw_query="animated family movies",
        genres=["Family", "Animation"],
        moods=["light"],
        media_types=["movie"],
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "animated family movies for kids",
    )

    assert "Animation" in updated.genres
    assert updated.moods == []


def test_normalize_merged_intent_keeps_media_types_empty_for_neo_noir_queries_without_upstream_detection():
    intent = IntentFilters(
        raw_query="neo-noir mysteries in modern cities",
        genres=["Mystery", "Crime", "Thriller"],
        moods=[],
        media_types=[],
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "neo-noir mysteries in modern cities",
    )

    assert updated.media_types == []
    assert updated.year_min == 1990


def test_normalize_merged_intent_keeps_media_types_empty_for_urban_noir_queries_without_upstream_detection():
    intent = IntentFilters(
        raw_query="modern city noir detective films",
        genres=["Crime", "Thriller"],
        moods=[],
        media_types=[],
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "modern city noir detective films",
    )

    assert updated.media_types == []
    assert updated.genres == ["Crime", "Thriller"]


def test_normalize_merged_intent_keeps_media_types_empty_for_time_bending_thrillers_without_upstream_detection():
    intent = IntentFilters(
        raw_query="paradox mystery thrillers",
        genres=["Thriller", "Crime"],
        moods=[],
        media_types=[],
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "paradox mystery thrillers",
    )

    assert updated.media_types == []
    assert updated.year_min == 2000
    assert updated.genres == ["Thriller", "Science Fiction", "Mystery"]


def test_normalize_merged_intent_keeps_media_types_empty_for_cerebral_temporal_thrillers_without_upstream_detection():
    intent = IntentFilters(
        raw_query="mind-bending temporal thrillers",
        genres=["Thriller"],
        moods=[],
        media_types=[],
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "mind-bending temporal thrillers",
    )

    assert updated.media_types == []
    assert updated.year_min == 2000
    assert updated.genres == ["Thriller", "Science Fiction", "Mystery"]


def test_normalize_merged_intent_keeps_media_types_empty_for_cerebral_paradox_thrillers_without_upstream_detection():
    intent = IntentFilters(
        raw_query="cerebral paradox thrillers",
        genres=["Thriller"],
        moods=[],
        media_types=[],
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "cerebral paradox thrillers",
    )

    assert updated.media_types == []
    assert updated.year_min == 2000
    assert updated.genres == ["Thriller", "Science Fiction", "Mystery"]


def test_normalize_merged_intent_keeps_short_cap_for_high_concept_thrillers_without_forcing_media_type():
    intent = IntentFilters(
        raw_query="short high-concept thriller movies",
        genres=["Thriller"],
        moods=[],
        media_types=[],
        max_runtime=None,
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "short high-concept thriller movies",
    )

    assert updated.media_types == []
    assert updated.max_runtime == 125
    assert updated.genres == ["Thriller", "Mystery", "Science Fiction"]


def test_normalize_merged_intent_strips_dark_spillover_from_not_too_dark_fantasy_tv_without_forcing_media_type():
    intent = IntentFilters(
        raw_query="not-too-dark fantasy tv from the last decade",
        genres=["Fantasy", "Mystery", "Thriller", "Crime"],
        moods=["dark"],
        media_types=[],
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "not-too-dark fantasy tv from the last decade",
    )

    assert updated.media_types == []
    assert updated.moods == []
    assert updated.genres == ["Fantasy"]


def test_normalize_merged_intent_adds_last_decade_lower_bound():
    intent = IntentFilters(
        raw_query="not-too-dark fantasy tv from the last decade",
        genres=["Fantasy"],
        moods=[],
        media_types=["tv"],
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "not-too-dark fantasy tv from the last decade",
    )

    assert updated.year_min == 2016


def test_normalize_merged_intent_shapes_teen_friendly_netflix_adventure_queries_without_forcing_media_type():
    intent = IntentFilters(
        raw_query="teen-friendly adventure picks on netflix",
        genres=["Drama", "Adventure"],
        moods=[],
        media_types=[],
        maturity_rating_max="PG-13",
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "teen-friendly adventure picks on netflix",
    )

    assert updated.media_types == []
    assert updated.genres == ["Adventure", "Fantasy"]


def test_normalize_merged_intent_shapes_grounded_vigilante_series_without_forcing_media_type():
    intent = IntentFilters(
        raw_query="grounded vigilante series",
        genres=["Fantasy"],
        moods=[],
        media_types=[],
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "grounded vigilante series",
    )

    assert updated.media_types == []
    assert updated.genres == ["Action & Adventure", "Crime", "Drama"]


def test_normalize_merged_intent_shapes_short_heist_crime_shows_without_forcing_media_type():
    intent = IntentFilters(
        raw_query="short grifter crime shows",
        genres=["Crime"],
        moods=[],
        media_types=[],
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "short grifter crime shows",
    )

    assert updated.media_types == []
    assert updated.genres == ["Crime", "Drama"]


def test_normalize_merged_intent_shapes_heist_tv_series_without_forcing_media_type():
    intent = IntentFilters(
        raw_query="robbery caper show",
        genres=["Thriller"],
        moods=[],
        media_types=[],
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "robbery caper show",
    )

    assert updated.media_types == []
    assert updated.genres == ["Thriller", "Crime", "Drama"]


def test_normalize_merged_intent_shapes_european_crime_films_and_series_without_forcing_media_type():
    intent = IntentFilters(
        raw_query="european crime films and series",
        genres=["Crime"],
        moods=[],
        media_types=[],
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "european crime films and series",
    )

    assert updated.media_types == []
    assert updated.genres == ["Crime", "Thriller", "Drama"]


def test_normalize_merged_intent_shapes_anime_scifi_films_without_forcing_media_type():
    intent = IntentFilters(
        raw_query="anime sci-fi films",
        genres=["Animation"],
        moods=[],
        media_types=[],
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "anime sci-fi films",
    )

    assert updated.media_types == []
    assert updated.genres == ["Animation", "Science Fiction"]


def test_normalize_merged_intent_shapes_space_opera_tv_queries_without_forcing_media_type():
    intent = IntentFilters(
        raw_query="hopeful space exploration series",
        genres=["Crime", "Thriller"],
        moods=["dark"],
        media_types=[],
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "hopeful space exploration series",
    )

    assert updated.media_types == []
    assert updated.moods == []
    assert updated.genres == ["Science Fiction", "Sci-Fi & Fantasy", "Action & Adventure"]


def test_normalize_merged_intent_shapes_hopeful_space_series_without_forcing_media_type():
    intent = IntentFilters(
        raw_query="hopeful space adventure series",
        genres=["Drama"],
        moods=["dark"],
        media_types=[],
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "hopeful space adventure series",
    )

    assert updated.media_types == []
    assert updated.moods == []
    assert updated.genres == ["Drama", "Science Fiction", "Sci-Fi & Fantasy", "Action & Adventure"]


def test_normalize_merged_intent_shapes_epic_fantasy_comparison_queries_without_forcing_media_type():
    intent = IntentFilters(
        raw_query="fantasy epics like classic monster sagas",
        genres=["Fantasy"],
        moods=[],
        media_types=[],
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "fantasy epics like classic monster sagas",
    )

    assert updated.media_types == []
    assert updated.genres == ["Fantasy", "Sci-Fi & Fantasy", "Action & Adventure"]


def test_normalize_merged_intent_shapes_epic_fantasy_series_without_forcing_media_type():
    intent = IntentFilters(
        raw_query="epic fantasy quest series",
        genres=["Fantasy"],
        moods=[],
        media_types=[],
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "epic fantasy quest series",
    )

    assert updated.media_types == []
    assert updated.genres == ["Fantasy", "Sci-Fi & Fantasy", "Action & Adventure"]


def test_normalize_merged_intent_adds_short_runtime_cap_for_bingeable_scifi_tv_without_forcing_media_type():
    intent = IntentFilters(
        raw_query="short bingeable sci-fi TV",
        genres=["Drama"],
        moods=[],
        media_types=[],
        max_runtime=None,
    )

    updated = recommend_routes._normalize_merged_intent(
        intent,
        "short bingeable sci-fi TV",
    )

    assert updated.media_types == []
    assert updated.max_runtime == 50
    assert updated.genres == ["Drama", "Science Fiction", "Sci-Fi & Fantasy"]


def test_should_soften_provider_preference_for_adult_movie_queries():
    intent = IntentFilters(
        raw_query="feel-good movies on netflix with short runtime",
        genres=["Comedy", "Family"],
        moods=[],
        media_types=["movie"],
    )

    assert (
        recommend_routes._should_soften_provider_preference(
            intent,
            "feel-good movies on netflix with short runtime",
        )
        is True
    )


def test_should_not_soften_provider_preference_for_family_queries():
    intent = IntentFilters(
        raw_query="family-friendly fantasy adventure movies on netflix",
        genres=["Family", "Adventure", "Fantasy"],
        moods=[],
        media_types=["movie"],
    )

    assert (
        recommend_routes._should_soften_provider_preference(
            intent,
            "family-friendly fantasy adventure movies on netflix",
        )
        is False
    )


def test_intent_filters_from_llm_copies_year_bounds():
    llm_intent = Intent(
        include_genres=["Comedy"],
        year_min=2000,
        year_max=2009,
    )

    filters = recommend_routes._intent_filters_from_llm(
        "rom-coms from the 2000s",
        llm_intent,
    )

    assert filters.year_min == 2000
    assert filters.year_max == 2009


def test_build_constraint_relaxation_ladder_runtime_and_maturity():
    intent = IntentFilters(
        raw_query="short family movies",
        genres=["Family"],
        moods=[],
        media_types=["movie"],
        max_runtime=90,
        maturity_rating_max="PG",
    )

    ladder = recommend_routes._build_constraint_relaxation_ladder(intent)

    assert len(ladder) >= 4
    assert ladder[0].max_runtime is not None
    assert ladder[0].max_runtime > 90
    assert any(step.maturity_rating_max is None for step in ladder)
    assert any(step.max_runtime is None for step in ladder)


def test_run_prefilter_query_orders_by_catalog_strength():
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
        raw_query="family adventure movies",
        genres=["Adventure"],
        moods=[],
        media_types=["movie"],
    )

    result = recommend_routes._run_prefilter_query(
        session,
        intent,
        fetch_limit=10,
        include_genres=True,
    )

    assert result == [2, 1]
    assert session.last_statement is not None
    compiled = str(session.last_statement)
    assert "ORDER BY" in compiled
    assert "trending_rank" in compiled
    assert "popular_rank" in compiled


def test_run_prefilter_query_provider_filter_uses_exists_clause():
    class PrefilterSession:
        def __init__(self, rows):
            self.rows = rows
            self.bind = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))
            self.last_statement = None

        def execute(self, statement):
            self.last_statement = statement
            return FakeResult(self.rows)

    session = PrefilterSession([(2,), (1,), (2,)])
    intent = IntentFilters(
        raw_query="heist TV series",
        genres=["Crime", "Thriller", "Drama"],
        moods=[],
        media_types=["tv"],
    )

    result = recommend_routes._run_prefilter_query(
        session,
        intent,
        fetch_limit=10,
        include_genres=True,
        required_services={"nfx"},
    )

    assert result == [2, 1]
    assert session.last_statement is not None
    compiled = str(session.last_statement)
    assert "EXISTS" in compiled
    assert "DISTINCT" not in compiled


def test_float_from_env_parses_values(monkeypatch):
    monkeypatch.setenv("FLOAT_ENV", "1.75")
    assert recommend_routes._float_from_env("FLOAT_ENV", 0.0) == 1.75
    monkeypatch.setenv("FLOAT_ENV", "not-a-number")
    assert recommend_routes._float_from_env("FLOAT_ENV", 0.0) == 0.0
    monkeypatch.delenv("FLOAT_ENV", raising=False)
    assert recommend_routes._float_from_env("FLOAT_ENV", 2.5) == 2.5


def _make_recommend_params(**overrides):
    payload = {
        "user_id": "u1",
        "limit": 20,
        "query": "space opera",
        "diversify": True,
        "profile": None,
        "use_llm_intent": True,
        "ann_description_override": "high stakes",
        "rewrite_override": "space survival",
        "ann_weight_override": 0.6,
        "rewrite_weight_override": 0.4,
        "genre_override": "Drama, Sci-Fi",
        "mixer_ann_weight": 0.7,
        "mixer_collab_weight": 0.2,
        "mixer_trending_weight": 0.1,
        "mixer_popularity_weight": 0.05,
        "mixer_vote_weight": 0.03,
        "mixer_novelty_weight": 0.02,
    }
    payload.update(overrides)
    return recommend_routes.RecommendParams(**payload)


def test_get_cache_key_tracks_all_recommendation_overrides():
    canonical_id = "u1"
    baseline = _make_recommend_params()
    baseline_key = recommend_routes._get_cache_key(canonical_id, baseline)

    overrides = [
        {"use_llm_intent": False},
        {"ann_description_override": "grim dystopia"},
        {"rewrite_override": "dark competition"},
        {"ann_weight_override": 0.9},
        {"rewrite_weight_override": 0.1},
        {"genre_override": "Thriller"},
        {"mixer_ann_weight": 0.4},
        {"mixer_collab_weight": 0.35},
        {"mixer_trending_weight": 0.25},
        {"mixer_popularity_weight": 0.2},
        {"mixer_vote_weight": 0.15},
        {"mixer_novelty_weight": 0.05},
        {"query": "mystery"},
        {"limit": 10},
        {"diversify": False},
    ]

    for override in overrides:
        key = recommend_routes._get_cache_key(
            canonical_id, _make_recommend_params(**override)
        )
        assert key != baseline_key

    assert recommend_routes._get_cache_key("u1:kids", baseline) != baseline_key


def test_resolve_candidate_limit_expands_no_query_pool():
    params = _make_recommend_params(query=None, limit=3)
    intent = IntentFilters(raw_query="", genres=[], moods=[], media_types=[])

    assert recommend_routes._resolve_candidate_limit(params, intent) == 30


def test_resolve_candidate_limit_keeps_filtered_query_expansion():
    params = _make_recommend_params(limit=3, query="family movies")
    intent = IntentFilters(
        raw_query="family movies",
        genres=["Family"],
        moods=[],
        media_types=["movie"],
    )

    assert recommend_routes._resolve_candidate_limit(params, intent) == 15


def test_semantic_recall_boost_detects_specific_broad_query():
    specific_intent = IntentFilters(
        raw_query="short grifter crime series on netflix",
        genres=["Crime", "Thriller", "Drama"],
        moods=[],
        media_types=[],
    )
    generic_intent = IntentFilters(
        raw_query="comedy movies",
        genres=["Comedy"],
        moods=[],
        media_types=["movie"],
    )

    assert recommend_routes._semantic_specificity_score(
        specific_intent,
        preferred_services=["nfx"],
    ) >= 6
    assert recommend_routes._filter_expressiveness_gap(
        specific_intent,
        preferred_services=["nfx"],
    ) >= 3
    assert recommend_routes._semantic_recall_boost_level(
        specific_intent,
        preferred_services=["nfx"],
    ) >= 2
    assert (
        recommend_routes._needs_semantic_recall_boost(
            specific_intent,
            preferred_services=["nfx"],
        )
        is True
    )
    assert recommend_routes._semantic_recall_boost_level(generic_intent) == 0
    assert recommend_routes._needs_semantic_recall_boost(generic_intent) is False


def test_resolve_candidate_limit_expands_for_high_pressure_query():
    params = _make_recommend_params(limit=3, query="short grifter crime series on netflix")
    intent = IntentFilters(
        raw_query="short grifter crime series on netflix",
        genres=["Crime", "Thriller", "Drama"],
        moods=[],
        media_types=[],
    )

    assert recommend_routes._resolve_candidate_limit(params, intent) == 24


def test_prefilter_fetch_limit_expands_for_high_pressure_query():
    intent = IntentFilters(
        raw_query="short grifter crime series on netflix",
        genres=["Crime", "Thriller", "Drama"],
        moods=[],
        media_types=[],
    )

    assert (
        recommend_routes._prefilter_fetch_limit(
            20,
            intent,
            preferred_services=["nfx"],
        )
        == 1500
    )


def test_constraint_prior_fetch_limit_expands_for_high_pressure_query():
    intent = IntentFilters(
        raw_query="brainy paradox mystery thrillers",
        genres=["Mystery", "Thriller"],
        moods=[],
        media_types=[],
    )

    assert (
        recommend_routes._constraint_prior_fetch_limit(
            20,
            intent,
            rank_vector=np.array([1.0, 0.0], dtype="float32"),
        )
        == 400
    )


def test_retrieval_pressure_treats_noir_movie_query_as_specific():
    intent = IntentFilters(
        raw_query="neo-noir crime movies",
        genres=["Crime", "Mystery", "Thriller"],
        moods=[],
        media_types=["movie"],
    )

    assert "noir" in recommend_routes._query_semantic_facets(intent.raw_query)
    assert recommend_routes._semantic_recall_boost_level(intent) >= 1
    assert recommend_routes._prefilter_fetch_limit(10, intent) == 1000
    assert recommend_routes._resolve_candidate_limit(
        recommend_routes.RecommendParams(user_id="u1", limit=10, query="neo-noir crime movies"),
        intent,
    ) == 60


def test_clear_user_cache_removes_only_target_user_entries():
    params = _make_recommend_params()
    user_one = "u1"
    user_two = "u2"
    key_one = recommend_routes._get_cache_key(user_one, params)
    key_two = recommend_routes._get_cache_key(user_two, params)

    recommend_routes._cache_set(user_one, key_one, [{"id": 1}])
    recommend_routes._cache_set(user_two, key_two, [{"id": 2}])

    recommend_routes.clear_user_cache(user_one)

    assert recommend_routes._cache_get(key_one) is None
    assert recommend_routes._cache_get(key_two) is not None


@pytest.mark.anyio
async def test_recommend_deduplicates_inflight_cache_miss(monkeypatch):
    params = _make_recommend_params(user_id="u1")
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/recommend",
            "headers": [],
            "query_string": b"",
        }
    )

    call_count = {"value": 0}

    async def fake_compute(*args, **kwargs):
        call_count["value"] += 1
        await asyncio.sleep(0.01)
        return recommend_routes.ComputeResult(
            items=[{"id": 1, "title": "Alpha"}], debug_context={}
        )

    monkeypatch.setattr(
        recommend_routes, "_compute_recommendations_async", fake_compute
    )

    results = await asyncio.gather(
        recommend_routes.recommend(request, params=params, cursor=None, db=object()),
        recommend_routes.recommend(request, params=params, cursor=None, db=object()),
    )

    assert call_count["value"] == 1
    assert results[0]["items"] == results[1]["items"]


def test_recommend_uses_entity_linker_and_blends_query_vector(
    monkeypatch, _stub_llm_intent
):
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
            collection_id=None,
            collection_name=None,
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
            collection_id=None,
            collection_name=None,
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
    monkeypatch.setattr("api.main.TMDB_API_KEY", "")

    class _Linker:
        async def link_entities(self, query):
            recorded["searched_query"] = query
            return {"movie": [101], "tv": [], "person": []}

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
        recommend_routes,
        "_prefilter_allowed_ids",
        lambda db, intent, limit, preferred_services=None: PrefilterDecision(
            [1], [1], True
        ),
    )
    monkeypatch.setattr(
        recommend_routes,
        "_trending_prior_candidates",
        lambda *args, **kwargs: [],
    )

    rewrite_vec = np.full(384, 0.5, dtype="float32")
    monkeypatch.setattr(
        recommend_routes,
        "rewrite_query",
        lambda query, intent: SimpleNamespace(rewritten_text="rewritten query"),
    )

    def fake_encode(texts):
        stacked = np.stack([rewrite_vec for _ in texts], axis=0)
        return stacked

    monkeypatch.setattr(recommend_routes, "encode_texts", fake_encode)

    with TestClient(app) as client:
        app.state.entity_linker = _Linker()
        response = client.get(
            "/recommend", params={"user_id": "u1", "query": "alpha movie"}
        )

    assert response.status_code == 200

    app.dependency_overrides.clear()
    session.close()
    app.state.entity_linker = None

    assert _stub_llm_intent.get("linked_entities") == {
        "movie": [101],
        "tv": [],
        "person": [],
    }
    assert recorded.get("searched_query") == "alpha movie"
    assert recorded["allowed_ids"] == [1]


def test_recommend_backfills_with_runtime_relaxation(monkeypatch):
    items = [
        SimpleNamespace(
            id=1,
            tmdb_id=101,
            media_type="movie",
            title="Fast Family Fun",
            overview="A short, upbeat family story.",
            poster_url="one.jpg",
            runtime=28,
            original_language="en",
            genres=[{"name": "Family"}],
            release_year=2020,
            collection_id=None,
            collection_name=None,
        ),
        SimpleNamespace(
            id=2,
            tmdb_id=202,
            media_type="movie",
            title="Long Family Quest",
            overview="A longer adventure for all ages.",
            poster_url="two.jpg",
            runtime=45,
            original_language="en",
            genres=[{"name": "Family"}],
            release_year=2021,
            collection_id=None,
            collection_name=None,
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
        "_parse_llm_intent",
        lambda query, user_context, linked_entities=None: Intent(
            include_genres=["Family"],
            media_types=["movie"],
            runtime_minutes_max=30,
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
    monkeypatch.setattr(
        recommend_routes,
        "rewrite_query",
        lambda query, intent: Rewrite(rewritten_text="short family movie"),
    )

    with TestClient(app) as client:
        response = client.get(
            "/recommend/debug",
            params={
                "user_id": "u1",
                "query": "short family movies",
                "limit": 2,
                "diversify": "false",
            },
        )

    app.dependency_overrides.clear()
    session.close()

    assert response.status_code == 200
    body = response.json()
    assert [item["id"] for item in body["items"]] == [1, 2]
    assert body["debug"]["final_intent"]["effective_genres"] == ["Family"]
    assert body["debug"]["final_intent"]["max_runtime"] == 30
    assert body["debug"]["prefilter"]["enforce_genres"] is True
    assert body["debug"]["prefilter"]["fetch_limit"] is None
    assert body["debug"]["metrics"]["candidate_limit"] == 10
    assert body["debug"]["metrics"]["prefilter_fetch_limit"] is None
    assert body["debug"]["metrics"]["constraint_prior_fetch_limit"] == 200
    assert body["debug"]["metrics"]["intent_relaxation_added"] == 2
    assert body["debug"]["metrics"]["intent_relaxation_steps"] >= 1
    assert body["debug"]["semantic_facets"] == []
    assert body["debug"]["metrics"]["semantic_specificity_score"] >= 1
    assert body["debug"]["metrics"]["filter_expressiveness_gap"] >= 3
    assert body["debug"]["metrics"]["needs_semantic_recall_boost"] is False
    assert body["debug"]["metrics"]["retrieval_pressure"] == 0


def test_recommend_debug_exposes_normalized_final_intent(monkeypatch):
    item = SimpleNamespace(
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
        collection_id=None,
        collection_name=None,
        watch_options=[{"service": "nfx", "url": "http://alpha"}],
    )

    session = CandidateSession([item])

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
        "_parse_llm_intent",
        lambda query, user_context, linked_entities=None: Intent(
            include_genres=["Comedy", "Family"],
            media_types=["movie"],
            runtime_minutes_max=125,
            maturity_rating_max="PG",
            streaming_providers=["netflix"],
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
        response = client.get(
            "/recommend/debug",
            params={
                "user_id": "u1",
                "query": "feel-good movies on netflix with short runtime",
                "limit": 1,
                "diversify": "false",
            },
        )

    app.dependency_overrides.clear()
    session.close()

    assert response.status_code == 200
    body = response.json()
    assert body["debug"]["final_intent"]["genres"] == ["Comedy", "Family"]
    assert body["debug"]["final_intent"]["effective_genres"] == ["Comedy", "Family"]
    assert body["debug"]["llm_intent"]["raw"]["streaming_providers"] == ["netflix"]
    assert body["debug"]["prefilter"]["fetch_limit"] is None
    assert body["debug"]["metrics"]["candidate_limit"] == 6
    assert body["debug"]["metrics"]["prefilter_fetch_limit"] is None
    assert body["debug"]["metrics"]["constraint_prior_fetch_limit"] == 240
    assert body["debug"]["semantic_facets"] == []
    assert body["debug"]["metrics"]["semantic_specificity_score"] >= 1
    assert body["debug"]["metrics"]["filter_expressiveness_gap"] >= 1
    assert body["debug"]["metrics"]["needs_semantic_recall_boost"] is False
    assert body["debug"]["metrics"]["retrieval_pressure"] == 0
    assert body["debug"]["source_counts"]["primary_stage"] == "ann"
    assert body["debug"]["stage_counts"]["primary_retrieval"] == 1
    assert body["debug"]["stage_counts"]["post_filter_candidates"] == 1
    assert body["debug"]["candidates"]["primary_retrieval"][0]["id"] == 1
    assert body["debug"]["candidates"]["pre_mixer"][0]["source_scores"]["ann"] == 1.0
    assert body["debug"]["candidates"]["post_mixer"][0]["retrieval_score"] is not None
    assert body["debug"]["candidates"]["final"][0]["id"] == 1


def test_recommend_merges_constraint_prior_candidates_when_ann_misses(monkeypatch):
    items = [
        SimpleNamespace(
            id=1,
            tmdb_id=101,
            media_type="movie",
            title="Loose Match",
            overview="A generic match from ANN.",
            poster_url="one.jpg",
            runtime=100,
            original_language="en",
            genres=[{"name": "Family"}],
            release_year=2020,
            collection_id=None,
            collection_name=None,
            watch_options=[],
        ),
        SimpleNamespace(
            id=2,
            tmdb_id=202,
            media_type="movie",
            title="Strict Match",
            overview="A better constrained match.",
            poster_url="two.jpg",
            runtime=105,
            original_language="en",
            genres=[{"name": "Family"}, {"name": "Adventure"}],
            release_year=2021,
            collection_id=None,
            collection_name=None,
            watch_options=[],
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
        "_parse_llm_intent",
        lambda query, user_context, linked_entities=None: Intent(
            include_genres=["Family", "Adventure"],
            media_types=["movie"],
            runtime_minutes_max=120,
            maturity_rating_max="PG-13",
        ),
    )
    monkeypatch.setattr(
        recommend_routes,
        "ann_candidates",
        lambda db, vec, exclude, limit, allowed_ids=None: [1],
    )
    monkeypatch.setattr(
        recommend_routes,
        "_constraint_prior_candidates",
        lambda db, intent, limit, allowlist, enforce_genres, rank_vector=None: [2],
    )
    monkeypatch.setattr(
        recommend_routes,
        "rewrite_query",
        lambda query, intent: Rewrite(rewritten_text="family adventure movie"),
    )
    monkeypatch.setattr(
        recommend_routes,
        "rerank_with_explanations",
        lambda items, **_: items,
    )

    with TestClient(app) as client:
        response = client.get(
            "/recommend/debug",
            params={
                "user_id": "u1",
                "query": "family adventure movies for teens",
                "limit": 2,
                "diversify": "false",
            },
        )

    app.dependency_overrides.clear()
    session.close()

    assert response.status_code == 200
    body = response.json()
    assert [item["id"] for item in body["items"]][:2] == [2, 1]
    assert body["debug"]["metrics"]["constraint_prior_candidates"] == 1
    assert body["debug"]["source_counts"]["constraint_prior_candidates"] == 1
    assert body["debug"]["stage_counts"]["constraint_prior"] == 1
    assert body["debug"]["candidates"]["constraint_prior"][0]["id"] == 2


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
            collection_id=None,
            collection_name=None,
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
        lambda db, intent, limit, allowlist, preferred_media_types=None: [1],
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


def test_recommend_cold_start_uses_rewrite_ann(monkeypatch):
    item = SimpleNamespace(
        id=1,
        tmdb_id=101,
        media_type="tv",
        title="Game Arena",
        overview="Contestants face deadly games.",
        poster_url="arena.jpg",
        runtime=50,
        original_language="en",
        genres=[{"name": "Drama"}],
        release_year=2023,
        collection_id=None,
        collection_name=None,
    )

    session = CandidateSession([item])

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
        "_prefilter_allowed_ids",
        lambda db, intent, limit, preferred_services=None: PrefilterDecision(
            [1], [], True
        ),
    )

    rewrite_vec = np.ones(384, dtype="float32") / np.sqrt(384)

    monkeypatch.setattr(
        recommend_routes,
        "rewrite_query",
        lambda query, intent: SimpleNamespace(rewritten_text="desperate deadly games"),
    )

    monkeypatch.setattr(
        recommend_routes,
        "_build_rewrite_vector",
        lambda rewrite_text, ann_desc, ann_w, rewrite_w: rewrite_vec,
    )

    ann_called = {}

    def fake_ann(db, vec, exclude, limit, allowed_ids=None):
        ann_called["vec"] = vec
        return [1]

    monkeypatch.setattr(recommend_routes, "ann_candidates", fake_ann)

    def _fail_cold_start(*args, **kwargs):
        raise AssertionError("cold-start fallback should not run")

    monkeypatch.setattr(
        recommend_routes,
        "_cold_start_candidates",
        _fail_cold_start,
    )
    monkeypatch.setattr(
        recommend_routes,
        "rerank_with_explanations",
        lambda items, **_: items,
    )

    with TestClient(app) as client:
        response = client.get(
            "/recommend",
            params={"user_id": "u1", "query": "tv shows like Squid Game"},
        )

    app.dependency_overrides.clear()
    session.close()

    assert response.status_code == 200
    assert ann_called


def test_recommend_skips_llm_when_disabled(monkeypatch):
    items = [
        SimpleNamespace(
            id=1,
            tmdb_id=1,
            media_type="movie",
            title="Alpha",
            overview="alpha",
            poster_url="alpha.jpg",
            runtime=100,
            original_language="en",
            genres=[{"name": "Drama"}],
            release_year=2020,
            collection_id=None,
            collection_name=None,
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

    def fail_parse(*args, **kwargs):
        raise AssertionError("LLM parser should be disabled")

    monkeypatch.setattr(recommend_routes, "_parse_llm_intent", fail_parse)

    def fake_ann(db, vec, exclude, limit, allowed_ids=None):
        return [1]

    monkeypatch.setattr(recommend_routes, "ann_candidates", fake_ann)
    monkeypatch.setattr(
        recommend_routes,
        "rewrite_query",
        lambda query, intent: Rewrite(rewritten_text="rewritten"),
    )
    monkeypatch.setattr(
        recommend_routes,
        "encode_texts",
        lambda texts: np.array(
            [[1.0 for _ in range(384)] for _ in texts], dtype="float32"
        ),
    )
    monkeypatch.setattr(
        recommend_routes,
        "rerank_with_explanations",
        lambda items, **_: items,
    )
    monkeypatch.setattr(
        recommend_routes,
        "_prefilter_allowed_ids",
        lambda db, intent, limit, preferred_services=None: PrefilterDecision(
            [1], [], True
        ),
    )

    with TestClient(app) as client:
        resp = client.get(
            "/recommend",
            params={
                "user_id": "u1",
                "query": "alpha",
                "use_llm_intent": "false",
            },
        )

    app.dependency_overrides.clear()
    session.close()

    assert resp.status_code == 200


def test_recommend_manual_rewrite_override(monkeypatch):
    item = SimpleNamespace(
        id=1,
        tmdb_id=1,
        media_type="movie",
        title="Alpha",
        overview="alpha",
        poster_url="alpha.jpg",
        runtime=100,
        original_language="en",
        genres=[{"name": "Drama"}],
        release_year=2020,
        collection_id=None,
        collection_name=None,
    )

    session = CandidateSession([item])

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

    def fail_rewrite(*args, **kwargs):
        raise AssertionError("rewrite_query should be bypassed")

    monkeypatch.setattr(recommend_routes, "rewrite_query", fail_rewrite)

    def fake_encode(texts):
        assert texts == ["manual rewrite"]
        return np.ones((1, 384), dtype="float32")

    monkeypatch.setattr(recommend_routes, "encode_texts", fake_encode)

    ann_calls = {"count": 0}

    def fake_ann(db, vec, exclude, limit, allowed_ids=None):
        ann_calls["count"] += 1
        assert vec.shape[0] == 384
        return [1]

    monkeypatch.setattr(recommend_routes, "ann_candidates", fake_ann)
    monkeypatch.setattr(
        recommend_routes,
        "rerank_with_explanations",
        lambda items, **_: items,
    )
    monkeypatch.setattr(
        recommend_routes,
        "_prefilter_allowed_ids",
        lambda db, intent, limit, preferred_services=None: PrefilterDecision(
            [1], [], True
        ),
    )

    with TestClient(app) as client:
        resp = client.get(
            "/recommend",
            params={
                "user_id": "u1",
                "rewrite_override": "manual rewrite",
            },
        )

    app.dependency_overrides.clear()
    session.close()

    assert resp.status_code == 200
    assert ann_calls["count"] == 1


def test_build_rewrite_vector_blends_description(monkeypatch):
    desc = "grim survival stakes"
    rewrite_text = "sci-fi survival"

    def fake_encode(texts):
        assert texts == [desc, rewrite_text]
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")

    monkeypatch.setattr(recommend_routes, "encode_texts", fake_encode)

    vec = recommend_routes._build_rewrite_vector(
        rewrite_text, desc, ann_weight_override=0.5, rewrite_weight_override=1.0
    )
    expected = np.array([0.4472136, 0.8944272], dtype="float32")
    assert vec is not None
    assert np.allclose(vec[:2], expected, atol=1e-6)


def test_recommend_query_resets_mixer_weights_when_env_flag_enabled(monkeypatch):
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
            collection_id=None,
            collection_name=None,
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
        "ann_candidates",
        lambda db, vec, exclude, limit, allowed_ids=None: [1],
    )
    monkeypatch.setattr(
        recommend_routes,
        "rerank_with_explanations",
        lambda items, **_: items,
    )

    captured = {}
    original_apply = recommend_routes._apply_mixer_scores

    def spy_apply(candidates, **overrides):
        captured.update(overrides)
        return original_apply(candidates, **overrides)

    monkeypatch.setattr(recommend_routes, "_apply_mixer_scores", spy_apply)

    monkeypatch.setattr(
        recommend_routes, "_QUERY_DISABLE_NON_ANN_SIGNALS", True
    )

    with TestClient(app) as client:
        client.get(
            "/recommend",
            params={"user_id": "u1", "limit": 1, "query": "rom-coms from the 2000s"},
        )

    app.dependency_overrides.clear()
    session.close()
    assert captured["trending_weight_override"] == 0.0
    assert captured["popularity_weight_override"] == 0.0
    assert captured["vote_weight_override"] == 0.0
    assert captured["novelty_weight_override"] == 0.0


def test_recommend_query_keeps_mixer_defaults_when_env_flag_disabled(monkeypatch):
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
            collection_id=None,
            collection_name=None,
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
        "ann_candidates",
        lambda db, vec, exclude, limit, allowed_ids=None: [1],
    )
    monkeypatch.setattr(
        recommend_routes,
        "rerank_with_explanations",
        lambda items, **_: items,
    )

    captured = {}
    original_apply = recommend_routes._apply_mixer_scores

    def spy_apply(candidates, **overrides):
        captured.update(overrides)
        return original_apply(candidates, **overrides)

    monkeypatch.setattr(recommend_routes, "_apply_mixer_scores", spy_apply)
    monkeypatch.setattr(
        recommend_routes, "_QUERY_DISABLE_NON_ANN_SIGNALS", False
    )

    with TestClient(app) as client:
        client.get(
            "/recommend",
            params={"user_id": "u1", "limit": 1, "query": "rom-coms from the 2000s"},
        )

    app.dependency_overrides.clear()
    session.close()
    assert captured["trending_weight_override"] is None
    assert captured["popularity_weight_override"] is None
    assert captured["vote_weight_override"] is None
    assert captured["novelty_weight_override"] is None
