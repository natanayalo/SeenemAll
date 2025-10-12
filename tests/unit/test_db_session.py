from __future__ import annotations

import pytest

from api.db import session as session_mod


def test_init_engine_sets_engine_and_sessionmaker(monkeypatch):
    calls = {}

    def fake_create_engine(url, pool_pre_ping, future):
        calls["create_engine"] = {
            "url": url,
            "pool_pre_ping": pool_pre_ping,
            "future": future,
        }
        return "engine"

    def fake_sessionmaker(*args, **kwargs):
        calls["sessionmaker"] = kwargs
        return "SessionFactory"

    monkeypatch.setattr(session_mod, "create_engine", fake_create_engine)
    monkeypatch.setattr(session_mod, "sessionmaker", fake_sessionmaker)
    monkeypatch.setattr(session_mod, "_engine", None, raising=False)
    monkeypatch.setattr(session_mod, "_SessionLocal", None, raising=False)

    session_mod.init_engine()

    assert session_mod._engine == "engine"
    assert session_mod._SessionLocal == "SessionFactory"
    assert calls["create_engine"]["url"].startswith("postgresql+psycopg2://")
    assert calls["sessionmaker"]["bind"] == "engine"
    assert calls["sessionmaker"]["autoflush"] is False
    assert calls["sessionmaker"]["autocommit"] is False
    assert calls["sessionmaker"]["future"] is True


def test_get_engine_lazy_initialises(monkeypatch):
    monkeypatch.setattr(session_mod, "_engine", None, raising=False)

    def fake_init():
        session_mod._engine = "lazy-engine"

    monkeypatch.setattr(session_mod, "init_engine", fake_init)

    engine = session_mod.get_engine()

    assert engine == "lazy-engine"


def test_get_sessionmaker_lazy_initialises(monkeypatch):
    monkeypatch.setattr(session_mod, "_SessionLocal", None, raising=False)

    def fake_init():
        session_mod._SessionLocal = "lazy-session"

    monkeypatch.setattr(session_mod, "init_engine", fake_init)

    factory = session_mod.get_sessionmaker()

    assert factory == "lazy-session"


def test_get_db_yields_and_closes(monkeypatch):
    class DummySession:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    dummy = DummySession()

    def fake_get_sessionmaker():
        return lambda: dummy

    monkeypatch.setattr(session_mod, "get_sessionmaker", fake_get_sessionmaker)

    gen = session_mod.get_db()
    db = next(gen)
    assert db is dummy
    with pytest.raises(StopIteration):
        next(gen)
    assert dummy.closed is True
