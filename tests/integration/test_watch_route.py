from __future__ import annotations

from typing import Iterator, List, Tuple

import pytest
from fastapi.testclient import TestClient

from api.db.session import get_db
from api.main import app
from tests.helpers import FakeResult


@pytest.fixture(autouse=True)
def _disable_db_startup(monkeypatch):
    monkeypatch.setattr("api.main.init_engine", lambda: None)
    monkeypatch.setattr("api.main.get_sessionmaker", lambda: None)


class _WatchSession:
    def __init__(self, items: List[Tuple[str | None, str | None]]):
        self._items = items
        self.last_statement = None
        self.last_params = None

    def execute(self, statement, params=None):
        self.last_statement = statement
        self.last_params = params or {}
        return FakeResult(self._items)


def test_get_watch_link_redirects_to_deeplink(monkeypatch):
    session = _WatchSession([("app://deeplink", "http://web.url")])

    def override_get_db() -> Iterator[_WatchSession]:
        yield session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        resp = client.get(
            "/watch-link/123?service=nfx&country=US", follow_redirects=False
        )

    app.dependency_overrides.clear()

    assert resp.status_code == 307
    assert resp.headers["location"] == "app://deeplink"


def test_get_watch_link_redirects_to_web_url_if_no_deeplink(monkeypatch):
    session = _WatchSession([(None, "http://web.url")])

    def override_get_db() -> Iterator[_WatchSession]:
        yield session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        resp = client.get(
            "/watch-link/123?service=nfx&country=US", follow_redirects=False
        )

    app.dependency_overrides.clear()

    assert resp.status_code == 307
    assert resp.headers["location"] == "http://web.url"


def test_get_watch_link_404s_if_not_found(monkeypatch):
    session = _WatchSession([])

    def override_get_db() -> Iterator[_WatchSession]:
        yield session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        resp = client.get("/watch-link/123?service=nfx&country=US")

    app.dependency_overrides.clear()

    assert resp.status_code == 404
    assert resp.json()["detail"] == "Link not found"


def test_get_watch_link_404s_if_link_is_empty(monkeypatch):
    session = _WatchSession([(None, None)])

    def override_get_db() -> Iterator[_WatchSession]:
        yield session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        resp = client.get("/watch-link/123?service=nfx&country=US")

    app.dependency_overrides.clear()

    assert resp.status_code == 404
    assert resp.json()["detail"] == "Link not found"
