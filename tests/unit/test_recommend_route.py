from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
import numpy as np
from types import SimpleNamespace
from typing import Any, Dict, Sequence
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
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
from api.core.filter_matcher import QueryFiltersResult

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
        lambda db, intent, limit, preferred_services=None, prefer_top_rated=False: PrefilterDecision(
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


@pytest.fixture(autouse=True)
def _stub_query_filters(monkeypatch):
    monkeypatch.setattr(
        recommend_routes,
        "get_query_filters",
        lambda query: QueryFiltersResult((), (), (), (), (), (), (), (), query or ""),
    )
    yield


class DummySession:
    def execute(self, *_, **__):
        raise AssertionError("execute should not be called when no IDs are returned")


def test_recommend_returns_empty_when_no_candidates(monkeypatch):
    def override_get_db():
        yield DummySession()

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(
        recommend_routes,
        "_load_media_genres",
        lambda db: {
            "movie": {"Animation", "Science Fiction"},
            "tv": {"Sci-Fi & Fantasy", "Animation"},
        },
        raising=False,
    )
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
        lambda db, vec, exclude, limit, allowed_ids=None, backend_override=None, **kwargs: [],
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
                (1, 1, 3, 100.0),
                (2, 2, None, 50.0),
                (3, None, None, 10.0),
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
        lambda db, vec, exclude, limit, allowed_ids=None, backend_override=None, **kwargs: [
            1,
            2,
        ],
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
        lambda db, vec, exclude, limit, allowed_ids=None, backend_override=None, **kwargs: [
            1,
            2,
        ],
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
        lambda db, intent, limit, preferred_services=None, prefer_top_rated=False: PrefilterDecision(
            [101, 202], [101, 202], True
        ),
    )

    recorded = {}

    def fake_ann(
        db,
        vec,
        exclude,
        limit,
        allowed_ids=None,
        backend_override=None,
        search_filters=None,
        text_query=None,
    ):
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
        lambda db, intent, limit, preferred_services=None, prefer_top_rated=False: PrefilterDecision(
            None, [], False
        ),
    )
    monkeypatch.setattr(
        recommend_routes,
        "ann_candidates",
        lambda db, vec, exclude, limit, allowed_ids=None, backend_override=None, **kwargs: [
            sci_fi_item.id
        ],
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
        lambda db, vec, exclude, limit, allowed_ids=None, backend_override=None, **kwargs: [
            1,
            2,
        ],
    )

    monkeypatch.setattr(
        recommend_routes,
        "rewrite_query",
        lambda query, intent: SimpleNamespace(rewritten_text=""),
    )

    monkeypatch.setattr(
        recommend_routes,
        "_trending_prior_candidates",
        lambda db, intent, exclude, limit, allowed_ids=None: [(3, 0.6)],
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
    assert recorded["limit"] == 9  # candidate_limit with limit=3
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
        lambda db, vec, exclude, limit, allowed_ids=None, backend_override=None, **kwargs: [
            1,
            2,
            3,
        ],
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
        lambda db, vec, exclude, limit, allowed_ids=None, backend_override=None, **kwargs: [
            item.id for item in items
        ],
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
        lambda db, vec, exclude, limit, allowed_ids=None, backend_override=None, **kwargs: [],
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
        lambda db, vec, exclude, limit, allowed_ids=None, backend_override=None, **kwargs: [
            1
        ],
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
        lambda db, vec, exclude, limit, allowed_ids=None, backend_override=None, **kwargs: [
            item.id for item in items
        ],
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
        lambda db, vec, exclude, limit, allowed_ids=None, backend_override=None, **kwargs: [
            item.id for item in items
        ],
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
        lambda db, vec, exclude, limit, allowed_ids=None, backend_override=None, **kwargs: [
            1,
            2,
            3,
            4,
        ],
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
        lambda db, vec, exclude, limit, allowed_ids=None, backend_override=None, **kwargs: [
            1,
            2,
            3,
            4,
        ],
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
    assert result.enforce_genres is True
    assert session.last_statement is not None


def test_prefilter_prefers_keyword_boosts(monkeypatch):
    class PrefilterSession:
        def __init__(self):
            self.bind = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))

    session = PrefilterSession()
    intent = IntentFilters(
        raw_query="",
        genres=["Crime"],
        keywords=["heist"],
        media_types=["movie"],
    )

    def fake_run_prefilter(
        db,
        intent,
        *,
        fetch_limit,
        include_genres,
        required_services=None,
        prefer_top_rated=False,
        require_all_genres=False,
        genres_override=None,
        include_keywords=False,
        require_all_keywords=False,
        keywords_override=None,
    ):
        if include_keywords and include_genres:
            return [10, 11]
        if include_keywords and not include_genres:
            return [12]
        if include_genres:
            return list(range(30, 43))
        return list(range(50, 55))

    monkeypatch.setattr(
        recommend_routes, "_run_prefilter_query", fake_run_prefilter, raising=False
    )

    result = ORIGINAL_PREFILTER(session, intent, limit=10)
    assert isinstance(result, PrefilterDecision)
    assert result.allowed_ids == list(range(30, 43))
    assert result.boost_ids[:3] == [10, 11, 12]


def test_float_from_env_parses_values(monkeypatch):
    monkeypatch.setenv("FLOAT_ENV", "1.75")
    assert recommend_routes._float_from_env("FLOAT_ENV", 0.0) == 1.75
    monkeypatch.setenv("FLOAT_ENV", "not-a-number")
    assert recommend_routes._float_from_env("FLOAT_ENV", 0.0) == 0.0
    monkeypatch.delenv("FLOAT_ENV", raising=False)
    assert recommend_routes._float_from_env("FLOAT_ENV", 2.5) == 2.5


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

    def fake_ann(
        db,
        vec,
        exclude,
        limit,
        allowed_ids=None,
        backend_override=None,
        search_filters=None,
        text_query=None,
    ):
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
        lambda db, intent, limit, preferred_services=None, prefer_top_rated=False: PrefilterDecision(
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
        lambda db, intent, limit, allowlist, prefer_top_rated=False: [1],
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
        lambda db, intent, limit, preferred_services=None, prefer_top_rated=False: PrefilterDecision(
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
        lambda rewrite_text, ann_desc, ann_w, rewrite_w, reference_titles=None: rewrite_vec,
    )

    ann_called = {}

    def fake_ann(
        db,
        vec,
        exclude,
        limit,
        allowed_ids=None,
        backend_override=None,
        search_filters=None,
        text_query=None,
    ):
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

    def fake_ann(
        db,
        vec,
        exclude,
        limit,
        allowed_ids=None,
        backend_override=None,
        search_filters=None,
        text_query=None,
    ):
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
        lambda db, intent, limit, preferred_services=None, prefer_top_rated=False: PrefilterDecision(
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

    def fake_ann(
        db,
        vec,
        exclude,
        limit,
        allowed_ids=None,
        backend_override=None,
        search_filters=None,
        text_query=None,
    ):
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
        lambda db, intent, limit, preferred_services=None, prefer_top_rated=False: PrefilterDecision(
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


def test_recommend_query_resets_mixer_weights(monkeypatch):
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
        lambda db, vec, exclude, limit, allowed_ids=None, backend_override=None, **kwargs: [
            1
        ],
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


def test_recommend_normalizes_query_filters(monkeypatch):
    items = [
        SimpleNamespace(
            id=1,
            tmdb_id=101,
            media_type="movie",
            title="Sample Anime Sci-Fi",
            overview="Animated sci-fi adventure.",
            poster_url=None,
            runtime=100,
            original_language="ja",
            genres=[
                {"id": 16, "name": "Animation"},
                {"id": 878, "name": "Science Fiction"},
            ],
            release_year=2010,
            collection_id=None,
            collection_name=None,
            watch_options=[],
        )
    ]
    session = CandidateSession(items)

    def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    monkeypatch.setattr(
        recommend_routes,
        "_load_media_genres",
        lambda db: {
            "movie": {"Animation", "Science Fiction"},
            "tv": {"Sci-Fi & Fantasy", "Animation"},
        },
        raising=False,
    )

    monkeypatch.setattr(
        recommend_routes,
        "legacy_parse_intent",
        lambda query: IntentFilters(
            raw_query=query or "",
            genres=["Science Fiction", "Animation"],
            media_types=["movie"],
        ),
    )

    def fake_get_query_filters(query):
        return QueryFiltersResult(
            languages=("ja",),
            keywords=("anime", "sci-fi"),
            genres=("Sci-Fi & Fantasy",),
            media_types=("tv",),
            cast=(),
            directors=(),
            producers=(),
            writers=(),
            residual_text=query or "",
            reference_titles=("Akira",),
        )

    monkeypatch.setattr(recommend_routes, "get_query_filters", fake_get_query_filters)

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

    captured: Dict[str, Any] = {}

    def fake_ann_candidates(
        db,
        vec,
        exclude,
        limit,
        allowed_ids=None,
        backend_override=None,
        search_filters=None,
        **kwargs,
    ):
        captured["search_filters"] = search_filters
        return [items[0].id]

    monkeypatch.setattr(recommend_routes, "ann_candidates", fake_ann_candidates)

    def fake_prefilter(
        db, intent, limit, preferred_services=None, prefer_top_rated=False, **kwargs
    ):
        return PrefilterDecision([items[0].id], [], True)

    monkeypatch.setattr(recommend_routes, "_prefilter_allowed_ids", fake_prefilter)

    with TestClient(app) as client:
        resp = client.get(
            "/recommend",
            params={
                "user_id": "u1",
                "query": "anime sci-fi movies",
                "limit": 1,
                "strict_filters": "true",
            },
        )

    assert resp.status_code == 200
    search_filters = captured.get("search_filters")
    assert search_filters is not None
    assert search_filters.media_types == ("movie",)
    assert search_filters.genres == ("Science Fiction", "Animation")
    assert search_filters.keywords == ("anime",)


def test_query_filter_keywords_skip_genre_synonyms(monkeypatch):
    items = [
        SimpleNamespace(
            id=1,
            tmdb_id=101,
            media_type="movie",
            title="Classic Horror",
            overview="Found footage horror.",
            poster_url=None,
            runtime=100,
            original_language="en",
            genres=[{"name": "Horror"}],
            release_year=1980,
            collection_id=None,
            collection_name=None,
            watch_options=[],
        )
    ]
    session = CandidateSession(items)

    def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    monkeypatch.setattr(
        recommend_routes,
        "_load_media_genres",
        lambda db: {
            "movie": {"Horror"},
        },
        raising=False,
    )

    monkeypatch.setattr(
        recommend_routes,
        "legacy_parse_intent",
        lambda query: IntentFilters(
            raw_query=query or "",
            genres=["Horror"],
            media_types=["movie"],
        ),
    )

    def fake_get_query_filters(query):
        return QueryFiltersResult(
            languages=(),
            keywords=("horror", "possession"),
            genres=("Horror",),
            media_types=("movie",),
            cast=(),
            directors=(),
            producers=(),
            writers=(),
            residual_text=query or "",
        )

    monkeypatch.setattr(recommend_routes, "get_query_filters", fake_get_query_filters)

    def fake_prefilter(
        db, intent, limit, preferred_services=None, prefer_top_rated=False, **kwargs
    ):
        return PrefilterDecision([items[0].id], [], True)

    monkeypatch.setattr(recommend_routes, "_prefilter_allowed_ids", fake_prefilter)

    search_capture: Dict[str, Any] = {}

    def fake_ann_candidates(
        db,
        vec,
        exclude,
        limit,
        allowed_ids=None,
        backend_override=None,
        search_filters=None,
        **kwargs,
    ):
        search_capture["filters"] = search_filters
        return [items[0].id]

    monkeypatch.setattr(recommend_routes, "ann_candidates", fake_ann_candidates)

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

    with TestClient(app) as client:
        resp = client.get(
            "/recommend",
            params={
                "user_id": "u1",
                "query": "must-watch horror movies",
                "limit": 1,
                "strict_filters": "true",
            },
        )

    assert resp.status_code == 200
    filters = search_capture.get("filters")
    assert filters is not None
    assert filters.genres == ("Horror",)
    assert filters.keywords == ("possession",)


def test_recommend_preserves_boost_order_after_rerank(monkeypatch, _stub_llm_intent):
    items = [
        SimpleNamespace(
            id=1,
            tmdb_id=101,
            media_type="movie",
            title="Prime Western",
            overview="Classic western hit.",
            poster_url=None,
            runtime=120,
            original_language="en",
            genres=[{"name": "Western"}],
            release_year=1960,
            collection_id=None,
            collection_name=None,
            watch_options=[],
        ),
        SimpleNamespace(
            id=2,
            tmdb_id=202,
            media_type="movie",
            title="Second Western",
            overview="Another staple.",
            poster_url=None,
            runtime=110,
            original_language="en",
            genres=[{"name": "Western"}],
            release_year=1970,
            collection_id=None,
            collection_name=None,
            watch_options=[],
        ),
        SimpleNamespace(
            id=3,
            tmdb_id=303,
            media_type="movie",
            title="LLM Favorite",
            overview="Recent stylized western.",
            poster_url=None,
            runtime=105,
            original_language="en",
            genres=[{"name": "Western"}],
            release_year=2024,
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
        "_prefilter_allowed_ids",
        lambda *args, **kwargs: PrefilterDecision(
            [items[0].id, items[1].id, items[2].id],
            [items[0].id, items[1].id, items[2].id],
            True,
            True,
        ),
    )
    monkeypatch.setattr(recommend_routes, "_SERENDIPITY_RATIO", 0.0, raising=False)
    monkeypatch.setattr(
        recommend_routes,
        "legacy_parse_intent",
        lambda query: IntentFilters(
            raw_query=query or "", genres=["Western"], media_types=["movie"]
        ),
    )

    def fake_get_query_filters(query):
        return QueryFiltersResult(
            languages=(),
            keywords=("classic western",),
            genres=("Western",),
            media_types=("movie",),
            cast=(),
            directors=(),
            producers=(),
            writers=(),
            residual_text=query or "",
            reference_titles=(),
        )

    monkeypatch.setattr(recommend_routes, "get_query_filters", fake_get_query_filters)

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

    def fake_ann_candidates(*_, **__):
        return [items[2].id, items[0].id, items[1].id]

    monkeypatch.setattr(recommend_routes, "ann_candidates", fake_ann_candidates)

    def fake_rerank(payload, *_, **__):
        # Force reranker to prefer the non-boosted item first.
        sorted_payload = sorted(payload, key=lambda entry: entry["id"], reverse=True)
        return sorted_payload

    monkeypatch.setattr(recommend_routes, "rerank_with_explanations", fake_rerank)

    with TestClient(app) as client:
        resp = client.get(
            "/recommend",
            params={"user_id": "u1", "query": "classic western movies", "limit": 2},
        )
        assert resp.status_code == 200
        payload = resp.json()["items"]
        assert [item["tmdb_id"] for item in payload] == [
            items[0].tmdb_id,
            items[1].tmdb_id,
        ]

        resp_full = client.get(
            "/recommend",
            params={"user_id": "u1", "query": "classic western movies", "limit": 3},
        )

    assert resp_full.status_code == 200
    payload_full = resp_full.json()["items"]
    assert [item["tmdb_id"] for item in payload_full] == [
        items[0].tmdb_id,
        items[1].tmdb_id,
        items[2].tmdb_id,
    ]


def test_recommend_falls_back_to_keyword_text(monkeypatch):
    items = [
        SimpleNamespace(
            id=1,
            tmdb_id=201,
            media_type="movie",
            title="Classic Western Sample",
            overview="A classic western adventure.",
            poster_url=None,
            runtime=120,
            original_language="en",
            genres=[{"name": "Western"}],
            release_year=1965,
            collection_id=None,
            collection_name=None,
            watch_options=[],
        )
    ]
    session = CandidateSession(items)

    def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    monkeypatch.setattr(
        recommend_routes,
        "_load_media_genres",
        lambda db: {"movie": {"Western"}},
        raising=False,
    )

    monkeypatch.setattr(
        recommend_routes,
        "legacy_parse_intent",
        lambda query: IntentFilters(raw_query=query or "", media_types=["movie"]),
    )

    def fake_get_query_filters(query):
        return QueryFiltersResult(
            languages=(),
            keywords=("classic western",),
            genres=(),
            media_types=("movie",),
            cast=(),
            directors=(),
            producers=(),
            writers=(),
            residual_text="",
            reference_titles=(),
        )

    monkeypatch.setattr(recommend_routes, "get_query_filters", fake_get_query_filters)

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

    def fake_prefilter(
        db,
        intent,
        limit,
        preferred_services=None,
        prefer_top_rated=False,
        **kwargs,
    ):
        return PrefilterDecision([items[0].id], [], True)

    monkeypatch.setattr(recommend_routes, "_prefilter_allowed_ids", fake_prefilter)

    captured: Dict[str, Any] = {}

    def fake_ann_candidates(
        db,
        vec,
        exclude,
        limit,
        allowed_ids=None,
        backend_override=None,
        *,
        search_filters=None,
        text_query=None,
    ):
        captured["text_query"] = text_query
        return [items[0].id]

    monkeypatch.setattr(recommend_routes, "ann_candidates", fake_ann_candidates)

    with TestClient(app) as client:
        resp = client.get(
            "/recommend",
            params={
                "user_id": "u1",
                "query": "classic western movies",
                "limit": 1,
                "strict_filters": "true",
            },
        )

    assert resp.status_code == 200
    assert captured.get("text_query") == "classic western"
