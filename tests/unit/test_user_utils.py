from __future__ import annotations

import numpy as np

from api.core.user_utils import load_user_state
from api.db.models import User
from tests.helpers import FakeSession


def test_load_user_state_missing_user_returns_defaults():
    session = FakeSession(user=None, history_rows=[], embedding_vectors=[])
    long_vec, short_vec, exclude = load_user_state(session, "ghost")

    assert long_vec is None
    assert short_vec is None
    assert exclude == []


def test_load_user_state_with_existing_user_returns_numpy_vectors():
    user = User(user_id="u1", long_vec=[0.1, 0.2], short_vec=[0.3, 0.4])
    session = FakeSession(
        user=user,
        history_rows=[(11, 1.0), (22, 2.0)],
        embedding_vectors=[],
        history_ids=[11, 22, 33],
    )

    long_vec, short_vec, exclude = load_user_state(session, "u1")

    assert np.allclose(long_vec, np.array([0.1, 0.2], dtype="float32"))
    assert np.allclose(short_vec, np.array([0.3, 0.4], dtype="float32"))
    assert exclude == [11, 22, 33]
