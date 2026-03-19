from __future__ import annotations

import numpy as np

from api.core.user_utils import load_user_state, canonical_profile_id
from api.db.models import User
from tests.helpers import FakeSession


def test_load_user_state_missing_user_returns_defaults():
    session = FakeSession(user=None, history_rows=[], embedding_vectors=[])
    long_vec, short_vec, exclude, meta = load_user_state(session, "ghost")

    assert long_vec is None
    assert short_vec is None
    assert exclude == []
    assert meta == {
        "genre_prefs": {},
        "neighbors": [],
        "negative_items": [],
        "preferred_media_types": [],
    }


def test_load_user_state_with_existing_user_returns_numpy_vectors():
    user = User(
        user_id="u1",
        long_vec=[0.1, 0.2],
        short_vec=[0.3, 0.4],
        neighbors=[{"user_id": "ally", "weight": 0.5}],
    )
    session = FakeSession(
        user=user,
        history_rows=[(11, 1.0), (22, 2.0)],
        embedding_vectors=[],
        history_ids=[11, 22, 33],
        item_media_types={11: "movie", 22: "tv"},
    )

    long_vec, short_vec, exclude, meta = load_user_state(session, "u1")

    assert np.allclose(long_vec, np.array([0.1, 0.2], dtype="float32"))
    assert np.allclose(short_vec, np.array([0.3, 0.4], dtype="float32"))
    assert exclude == [11, 22]
    assert meta["genre_prefs"] == {}
    assert meta["neighbors"] == user.neighbors
    assert meta["negative_items"] == []
    assert meta["preferred_media_types"] == ["movie", "tv"]


def test_load_user_state_includes_profile_meta(monkeypatch):
    user = User(
        user_id="u42",
        long_vec=[0.5, 0.5],
        short_vec=[0.0, 1.0],
        genre_prefs={"Action": 0.7, "Drama": 0.3},
        neighbors=[{"user_id": "canon", "weight": 1.0}],
    )
    session = FakeSession(
        user=user,
        history_rows=[(101, 1.0), (202, 1.0, "not_interested")],
        embedding_vectors=[(101, [1.0, 0.0])],
        history_ids=[101, 202],
        item_media_types={101: "movie", 202: "tv"},
    )

    _, short_vec, exclude, meta = load_user_state(session, "u42")

    assert np.allclose(short_vec, np.array([0.0, 1.0], dtype="float32"))
    assert exclude == [101, 202]
    assert meta["genre_prefs"] == {"Action": 0.7, "Drama": 0.3}
    assert meta["neighbors"] == user.neighbors
    assert meta["negative_items"] == [202]
    assert meta["preferred_media_types"] == ["movie"]


def test_load_user_state_falls_back_to_base_profile():
    base_user = User(
        user_id="u1",
        long_vec=[0.1, 0.2],
        short_vec=[0.3, 0.4],
        genre_prefs={"Sci-Fi": 1.0},
        neighbors=[{"user_id": "ally", "weight": 0.5}],
    )
    scoped_user = User(
        user_id="u1::main",
        long_vec=None,
        short_vec=None,
    )
    session = FakeSession(
        user=scoped_user,
        other_users=[base_user],
        history_rows=[(11, 1.0), (22, 1.0)],
        history_ids=[11, 22],
        item_media_types={11: "movie", 22: "movie"},
    )

    long_vec, short_vec, exclude, meta = load_user_state(session, "u1::main")

    assert np.allclose(long_vec, np.array([0.1, 0.2], dtype="float32"))
    assert np.allclose(short_vec, np.array([0.3, 0.4], dtype="float32"))
    assert exclude == [11, 22]
    assert meta["genre_prefs"] == {"Sci-Fi": 1.0}
    assert meta["neighbors"] == base_user.neighbors
    assert meta["preferred_media_types"] == ["movie"]


def test_load_user_state_scoped_profile_without_base_returns_defaults():
    scoped_user = User(
        user_id="u1::main",
        long_vec=None,
        short_vec=None,
    )
    session = FakeSession(user=scoped_user, history_rows=[], embedding_vectors=[])

    long_vec, short_vec, exclude, meta = load_user_state(session, "u1::main")

    assert long_vec is None
    assert short_vec is None
    assert exclude == []
    assert meta["preferred_media_types"] == []


def test_canonical_profile_id():
    assert canonical_profile_id("u1", None) == "u1"
    assert canonical_profile_id("u1", " ") == "u1"
    assert canonical_profile_id("u1", "kids") == "u1::kids"
