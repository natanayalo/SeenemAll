from __future__ import annotations

import numpy as np

from api.core.user_profile import (
    DIM,
    _time_decay_weights,
    compute_user_vector,
    upsert_user_vectors,
)
from api.db.models import User
from tests.helpers import FakeSession


def test_time_decay_weights_normalises_and_prefers_recent():
    weights = _time_decay_weights(5)
    assert weights.shape == (5,)
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    assert weights[0] > weights[-1]


def test_compute_user_vector_returns_normalised_vectors():
    history_rows = [(101, 1.0), (202, 1.0)]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    session = FakeSession(history_rows=history_rows, embedding_vectors=embeddings)

    long_vec, short_vec = compute_user_vector(session, "user-1")

    assert long_vec is not None and short_vec is not None
    assert np.isclose(np.linalg.norm(long_vec), 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(short_vec), 1.0, atol=1e-6)
    # short vector should lean towards the most recent embedding (index 0)
    assert short_vec[0] > short_vec[1]


def test_compute_user_vector_with_no_history_returns_none():
    session = FakeSession(history_rows=[], embedding_vectors=[])
    long_vec, short_vec = compute_user_vector(session, "empty-user")
    assert long_vec is None and short_vec is None


def test_upsert_user_vectors_creates_new_user():
    session = FakeSession(
        history_rows=[(10, 1.0)],
        embedding_vectors=[[1.0] + [0.0] * (DIM - 1)],
        user=None,
    )

    upsert_user_vectors(session, "new-user")

    assert len(session.added) == 1
    created = session.added[0]
    assert isinstance(created, User)
    assert created.user_id == "new-user"
    assert created.short_vec is not None


def test_upsert_user_vectors_updates_existing_user():
    existing = User(user_id="existing")
    session = FakeSession(
        history_rows=[(99, 1.0)],
        embedding_vectors=[[0.0, 1.0] + [0.0] * (DIM - 2)],
        user=existing,
    )

    upsert_user_vectors(session, "existing")

    assert session.added == []
    assert existing.short_vec is not None
    assert existing.long_vec is not None
