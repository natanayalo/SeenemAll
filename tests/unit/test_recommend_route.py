from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

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
