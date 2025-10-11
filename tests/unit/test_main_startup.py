from __future__ import annotations

from api.main import app, on_startup
from api.db.session import get_db


def test_on_startup_initialises_engine(monkeypatch):
    called = {"init": 0, "sessionmaker": 0}

    def fake_init():
        called["init"] += 1

    def fake_get_sessionmaker():
        called["sessionmaker"] += 1

    monkeypatch.setattr("api.main.init_engine", fake_init)
    monkeypatch.setattr("api.main.get_sessionmaker", fake_get_sessionmaker)

    # ensure no overrides so startup should call both functions
    app.dependency_overrides.clear()
    on_startup()

    assert called["init"] == 1
    assert called["sessionmaker"] == 1


def test_on_startup_skips_when_override_present(monkeypatch):
    called = {"init": 0}

    def fake_init():
        called["init"] += 1

    monkeypatch.setattr("api.main.init_engine", fake_init)

    def override_get_db():
        yield None

    app.dependency_overrides[get_db] = override_get_db
    on_startup()
    app.dependency_overrides.clear()

    assert called["init"] == 0
