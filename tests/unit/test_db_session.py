from __future__ import annotations

import logging
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from api.db import session as session_mod


@pytest.fixture(autouse=True)
def isolated_session_state(monkeypatch):
    monkeypatch.setattr(session_mod, "_engine", None)
    monkeypatch.setattr(session_mod, "_SessionLocal", None)
    monkeypatch.setattr(session_mod, "_query_logging_attached", False)
    monkeypatch.delenv("EVAL_DB_DSN", raising=False)
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg2://test:test@database.invalid:5432/test"
    )


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


def test_sql_logging_attaches_once_and_reports_queries_and_errors(caplog):
    # A test substitute cannot prevent a later real engine from getting listeners.
    session_mod._attach_sql_logging(object())
    assert not session_mod._query_logging_attached
    engine = create_engine("sqlite://", future=True)
    try:
        session_mod._attach_sql_logging(engine)
        session_mod._attach_sql_logging(engine)
        with caplog.at_level(logging.INFO, logger="api.db.sql"):
            with engine.connect() as conn:
                assert (
                    conn.execute(
                        text("SELECT :value"), {"value": "opaque-test-value"}
                    ).scalar_one()
                    == "opaque-test-value"
                )
                with pytest.raises(OperationalError, match="no such column"):
                    conn.execute(text("SELECT missing_column"))
        query_logs = [
            record.getMessage()
            for record in caplog.records
            if record.name == "api.db.sql" and record.levelno == logging.INFO
        ]
        assert len(query_logs) == 1  # Registering twice must not duplicate logs.
        assert "SELECT query took" in query_logs[0]
        assert "ms | rows=" in query_logs[0]
        assert "opaque-test-value" not in query_logs[0]
        assert any(
            "SQL error during 'SELECT missing_column'" in record.getMessage()
            and "no such column" in record.getMessage()
            for record in caplog.records
        )
    finally:
        engine.dispose()


@pytest.mark.parametrize("statement", ["", "  SELECT\n" + "long_identifier " * 20])
def test_query_logging_handles_missing_timing_and_bounds_statement(caplog, statement):
    with caplog.at_level(logging.INFO, logger="api.db.sql"):
        session_mod._after_cursor_execute(
            None,
            SimpleNamespace(rowcount=None),
            statement,
            (),
            SimpleNamespace(),
            False,
        )
    message = caplog.records[-1].getMessage()
    assert "query | rows=? |" in message
    summary = message.split(" | ")[-1]
    if statement:
        assert len(summary) == 120 and summary.endswith("…")
        assert "\n" not in summary
    else:
        assert message.startswith("SQL query")


def test_disabled_logging_does_not_access_query_or_error_details(caplog):
    with caplog.at_level(logging.ERROR, logger="api.db.sql"):
        session_mod._after_cursor_execute(
            None, object(), "SELECT 1", (), object(), False
        )
        session_mod._handle_error(object())
    assert not caplog.records


@pytest.mark.parametrize(
    "evaluation_url,database_url,dns_result,expected_url",
    [
        (
            "postgresql+psycopg2://eval:eval@db:5432/eval",
            "postgresql+psycopg2://app:app@production.invalid:5432/reco",
            "10.0.0.1",
            "postgresql+psycopg2://eval:eval@db:5432/eval",
        ),
        (
            None,
            "postgresql+psycopg2://app:app@db:5432/reco",
            socket.gaierror("Docker host unavailable"),
            "postgresql+psycopg2://app:app@localhost:5432/reco",
        ),
        (
            None,
            "postgresql+psycopg2://app:app@db:5432/reco",
            OSError("DNS unavailable"),
            "postgresql+psycopg2://app:app@localhost:5432/reco",
        ),
        (
            "",
            "postgresql+psycopg2://app:app@explicit.invalid:5432/reco",
            None,
            "postgresql+psycopg2://app:app@explicit.invalid:5432/reco",
        ),
        (
            None,
            None,
            "10.0.0.1",
            "postgresql+psycopg2://app:app@db:5432/reco",
        ),
    ],
)
def test_engine_url_precedence_and_container_resolution(
    monkeypatch, evaluation_url, database_url, dns_result, expected_url
):
    for name, value in (
        ("EVAL_DB_DSN", evaluation_url),
        ("DATABASE_URL", database_url),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    resolve = MagicMock()
    if isinstance(dns_result, Exception):
        resolve.side_effect = dns_result
    else:
        resolve.return_value = dns_result
    factory = MagicMock(return_value=object())
    make_session = MagicMock()
    monkeypatch.setattr(socket, "gethostbyname", resolve)
    monkeypatch.setattr(session_mod, "create_engine", factory)
    monkeypatch.setattr(session_mod, "sessionmaker", make_session)

    session_mod.init_engine()

    factory.assert_called_once_with(expected_url, pool_pre_ping=True, future=True)
    assert session_mod._engine is factory.return_value
    assert session_mod._SessionLocal is make_session.return_value
    if dns_result is None:
        resolve.assert_not_called()
    else:
        resolve.assert_called_once_with("db")
