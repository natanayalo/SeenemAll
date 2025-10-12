from __future__ import annotations

from typing import Iterator, List

import pytest
from fastapi.testclient import TestClient

from api.db.models import Feedback
from api.db.session import get_db
from api.main import app


@pytest.fixture(autouse=True)
def _disable_db_startup(monkeypatch):
    monkeypatch.setattr("api.main.init_engine", lambda: None)
    monkeypatch.setattr("api.main.get_sessionmaker", lambda: None)


class _FeedbackSession:
    def __init__(self):
        self.added: List[Feedback] = []
        self.commits = 0

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commits += 1

    def close(self):
        pass


def test_post_feedback_adds_event(monkeypatch):
    session = _FeedbackSession()

    def override_get_db() -> Iterator[_FeedbackSession]:
        yield session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        resp = client.post(
            "/feedback",
            json={
                "user_id": "u1",
                "item_id": 123,
                "event_type": "click",
                "meta": {"rank": 5},
            },
        )

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert len(session.added) == 1
    feedback = session.added[0]
    assert feedback.user_id == "u1"
    assert feedback.item_id == 123
    assert feedback.type == "click"
    assert feedback.meta == {"rank": 5}
    assert session.commits == 1
