from __future__ import annotations

import numpy as np
from types import SimpleNamespace

import api.core.user_profile as user_profile
from api.core.user_profile import (
    DIM,
    _time_decay_weights,
    compute_user_vector,
    upsert_user_vectors,
)
from api.db.models import User
from tests.helpers import FakeSession
from sqlalchemy.exc import SQLAlchemyError


def test_time_decay_weights_normalises_and_prefers_recent():
    weights = _time_decay_weights(5)
    assert weights.shape == (5,)
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    assert weights[0] > weights[-1]


def test_time_decay_weights_respects_config(monkeypatch):
    monkeypatch.setattr(user_profile, "USER_PROFILE_DECAY_HALF_LIFE", 2.0)
    fast_decay = _time_decay_weights(5)
    monkeypatch.setattr(user_profile, "USER_PROFILE_DECAY_HALF_LIFE", 20.0)
    slow_decay = _time_decay_weights(5)
    fast_ratio = fast_decay[0] / fast_decay[-1]
    slow_ratio = slow_decay[0] / slow_decay[-1]
    assert fast_ratio > slow_ratio


def test_compute_user_vector_returns_normalised_vectors():
    history_rows = [(101, 1.0), (202, 1.0)]
    embeddings = [
        (101, [1.0, 0.0, 0.0]),
        (202, [0.0, 1.0, 0.0]),
    ]
    session = FakeSession(history_rows=history_rows, embedding_vectors=embeddings)

    long_vec, short_vec, genre_prefs, neighbors, negatives = compute_user_vector(
        session, "user-1"
    )

    assert long_vec is not None and short_vec is not None
    assert np.isclose(np.linalg.norm(long_vec), 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(short_vec), 1.0, atol=1e-6)
    # short vector should lean towards the most recent embedding (index 0)
    assert short_vec[0] > short_vec[1]
    assert genre_prefs is None
    assert neighbors == []
    assert negatives == []


def test_compute_user_vector_with_no_history_returns_none():
    session = FakeSession(history_rows=[], embedding_vectors=[])
    long_vec, short_vec, genre_prefs, neighbors, negatives = compute_user_vector(
        session, "empty-user"
    )
    assert long_vec is None and short_vec is None and genre_prefs is None
    assert neighbors == []
    assert negatives == []


def test_upsert_user_vectors_creates_new_user():
    session = FakeSession(
        history_rows=[(10, 1.0)],
        embedding_vectors=[(10, [1.0] + [0.0] * (DIM - 1))],
        item_rows=[(10, [{"name": "Action"}])],
        user=None,
    )

    upsert_user_vectors(session, "new-user")

    assert len(session.added) == 1
    created = session.added[0]
    assert isinstance(created, User)
    assert created.user_id == "new-user"
    assert created.short_vec is not None
    assert created.genre_prefs is not None
    assert isinstance(created.neighbors, list)


def test_upsert_user_vectors_updates_existing_user():
    existing = User(user_id="existing")
    session = FakeSession(
        history_rows=[(99, 1.0)],
        embedding_vectors=[(99, [0.0, 1.0] + [0.0] * (DIM - 2))],
        item_rows=[(99, [{"name": "Drama"}])],
        user=existing,
    )

    upsert_user_vectors(session, "existing")

    assert session.added == []
    assert existing.short_vec is not None
    assert existing.long_vec is not None
    assert existing.genre_prefs is not None
    assert isinstance(existing.neighbors, list)


def test_explicit_feedback_has_higher_weight():
    history_rows = [
        (101, 1.0, "watched"),
        (202, 5.0, "rated"),
    ]
    embeddings = [
        (101, [1.0, 0.0, 0.0]),
        (202, [0.0, 1.0, 0.0]),
    ]
    session = FakeSession(
        history_rows=history_rows,
        embedding_vectors=embeddings,
        item_rows=[
            (101, [{"name": "Action"}]),
            (202, [{"name": "Drama"}]),
        ],
    )

    long_vec, short_vec, genre_prefs, neighbors, negatives = compute_user_vector(
        session, "user-1"
    )

    assert long_vec is not None and short_vec is not None
    assert long_vec[1] > long_vec[0]
    assert genre_prefs is not None
    assert neighbors == []
    assert negatives == []
    assert "Drama" in genre_prefs and "Action" in genre_prefs
    assert genre_prefs["Drama"] > genre_prefs["Action"]


def test_negative_feedback_excluded_from_profiles():
    history_rows = [
        (301, 1.0, "not_interested"),
        (302, 1.0, "watched"),
    ]
    embeddings = [
        (301, [1.0, 0.0, 0.0]),
        (302, [0.0, 1.0, 0.0]),
    ]
    session = FakeSession(
        history_rows=history_rows,
        embedding_vectors=embeddings,
        item_rows=[
            (301, [{"name": "Horror"}]),
            (302, [{"name": "Comedy"}]),
        ],
    )

    long_vec, short_vec, genre_prefs, neighbors, negatives = compute_user_vector(
        session, "user-1"
    )

    assert negatives == [301]
    assert np.isclose(short_vec[1], 1.0, atol=1e-6)
    assert np.isclose(short_vec[0], 0.0, atol=1e-6)
    assert np.isclose(long_vec[1], 1.0, atol=1e-6)
    assert np.isclose(long_vec[0], 0.0, atol=1e-6)
    assert genre_prefs == {"Comedy": 1.0}
    assert neighbors == []


def test_collaborative_vector_blends_into_long(monkeypatch):
    history_rows = [(1, 1.0)]
    embeddings = [
        (1, [1.0, 0.0, 0.0]),
    ]
    session = FakeSession(
        history_rows=history_rows,
        embedding_vectors=embeddings,
        item_rows=[(1, [{"name": "Action"}])],
    )

    def fake_collab(db, user_id, item_ids):
        return np.array([0.0, 1.0, 0.0], dtype="float32"), []

    monkeypatch.setattr(
        "api.core.user_profile._collect_collaborative_vector", fake_collab
    )

    long_vec, short_vec, genre_prefs, neighbors, negatives = compute_user_vector(
        session, "user-1"
    )

    assert long_vec is not None and short_vec is not None
    assert long_vec[1] > 0.2  # collaborative signal pulls toward second dimension
    assert np.isclose(short_vec[1], 0.0, atol=1e-6)  # short vector unaffected
    assert genre_prefs is not None
    assert neighbors == []
    assert negatives == []


def test_collect_collaborative_vector_handles_sqlalchemy_error():
    class BrokenSession:
        def query(self, *args, **kwargs):
            raise SQLAlchemyError("boom")

    vec, neighbors = user_profile._collect_collaborative_vector(
        BrokenSession(), "user", [1, 2]
    )
    assert vec is None and neighbors == []


def test_collect_collaborative_vector_returns_neighbors():
    class WeightQuery:
        def __init__(self, rows):
            self._rows = rows

        def filter(self, *args, **kwargs):
            return self

        def all(self):
            return self._rows

    class UserQuery:
        def __init__(self, users):
            self._users = users

        def filter(self, *args, **kwargs):
            return self

        def all(self):
            return self._users

    neighbor_user = User(user_id="neighbor", long_vec=[0.0, 1.0, 0.0])
    neighbor_user.long_vec = [0.0, 1.0, 0.0]

    class StubSession:
        def __init__(self):
            self.weights = [("neighbor", 2.0)]

        def query(self, *entities):
            if set(entities) == {
                user_profile.UserHistory.user_id,
                user_profile.UserHistory.weight,
            }:
                return WeightQuery(self.weights)
            if len(entities) == 1 and entities[0] is User:
                return UserQuery([neighbor_user])
            raise AssertionError("Unexpected query")

    vec, neighbors = user_profile._collect_collaborative_vector(
        StubSession(), "user", [5]
    )
    assert vec is not None
    assert neighbors and neighbors[0].user_id == "neighbor"


def test_event_weight_clamps_negative_base():
    assert user_profile._event_weight("liked", -2.0) == 0.0


def test_collect_collaborative_vector_short_circuits_on_empty_items():
    class Stub:
        def query(self, *args, **kwargs):
            raise AssertionError("query should not be invoked")

    vec, neighbors = user_profile._collect_collaborative_vector(Stub(), "u", [])
    assert vec is None and neighbors == []


def test_collect_collaborative_vector_skips_null_users(monkeypatch):
    class Session:
        def query(self, *entities):
            class Query:
                def filter(self, *args, **kwargs):
                    return self

                def all(self):
                    return [(None, 2.0)]

            return Query()

    vec, neighbors = user_profile._collect_collaborative_vector(Session(), "user", [1])
    assert vec is None and neighbors == []


def test_collect_collaborative_vector_handles_missing_vectors(monkeypatch):
    class Session:
        def __init__(self):
            self._stage = 0

        def query(self, *entities):
            class UserHistoryQuery:
                def __init__(self, parent):
                    self.parent = parent

                def filter(self, *args, **kwargs):
                    return self

                def all(self):
                    return [("friend", 1.0)]

            class UserQuery:
                def __init__(self, parent):
                    self.parent = parent

                def filter(self, *args, **kwargs):
                    return self

                def all(self):
                    return [SimpleNamespace(user_id="friend", long_vec=None)]

            if entities and entities[0] is user_profile.UserHistory.user_id:
                return UserHistoryQuery(self)
            return UserQuery(self)

    vec, neighbors = user_profile._collect_collaborative_vector(Session(), "user", [1])
    assert vec is None and neighbors == []


def test_time_decay_weights_zero_items():
    weights = user_profile._time_decay_weights(0)
    assert weights.shape == (0,)


def test_compute_user_vector_handles_zero_weights(monkeypatch):
    session = FakeSession(
        history_rows=[(1, 0.0)],
        embedding_vectors=[(1, [1.0, 0.0, 0.0])],
        item_rows=[(1, [{"name": "Drama"}])],
    )
    monkeypatch.setattr(
        user_profile,
        "_collect_collaborative_vector",
        lambda *args, **kwargs: (None, []),
    )
    long_vec, short_vec, genre_prefs, neighbors, negatives = (
        user_profile.compute_user_vector(session, "user")
    )
    assert np.isclose(np.linalg.norm(long_vec), 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(short_vec), 1.0, atol=1e-6)
    assert genre_prefs is not None
    assert neighbors == []
    assert negatives == []


def test_compute_user_vector_collab_fills_missing_long_vec(monkeypatch):
    session = FakeSession(
        history_rows=[(1, 1.0)],
        embedding_vectors=[(1, [1.0, 0.0, 0.0])],
        item_rows=[(1, [{"name": "Action"}])],
    )

    monkeypatch.setattr(user_profile.np, "average", lambda *args, **kwargs: None)

    def fake_collect(db, user_id, item_ids):
        return np.array([0.0, 1.0, 0.0], dtype="float32"), [
            user_profile.NeighborInfo(user_id="friend", weight=2.0)
        ]

    monkeypatch.setattr(user_profile, "_collect_collaborative_vector", fake_collect)

    long_vec, short_vec, genre_prefs, neighbors, negatives = (
        user_profile.compute_user_vector(session, "user")
    )
    assert long_vec[1] > 0.5
    assert neighbors and neighbors[0].user_id == "friend"


def test_compute_user_vector_skips_nonpositive_genre_weights(monkeypatch):
    session = FakeSession(
        history_rows=[(1, 1.0), (2, 1.0)],
        embedding_vectors=[
            (1, [1.0, 0.0, 0.0]),
            (2, [0.0, 1.0, 0.0]),
        ],
        item_rows=[
            (1, [{"name": "Drama"}]),
            (2, [{"name": None}]),
        ],
    )

    monkeypatch.setattr(
        user_profile,
        "_collect_collaborative_vector",
        lambda *args, **kwargs: (None, []),
    )
    monkeypatch.setattr(
        user_profile, "_time_decay_weights", lambda n: np.zeros(n, dtype="float32")
    )

    long_vec, short_vec, genre_prefs, neighbors, negatives = (
        user_profile.compute_user_vector(session, "user")
    )
    assert genre_prefs is None
    assert np.linalg.norm(short_vec) == 0.0
