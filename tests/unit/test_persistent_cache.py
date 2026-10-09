import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from api.core.persistent_cache import PersistentCache, get_persistent_cache


def test_persistence_namespace_isolation_corruption_and_clear(tmp_path):
    path = str(tmp_path / "nested" / "cache.sqlite")
    first = PersistentCache(path, "first")
    second = PersistentCache(path, "second")
    try:
        first.set({"id": 1}, {"value": "first"})
        second.set({"id": 1}, {"value": "second"})
        assert first.get({"id": 1}) == {"value": "first"}
        assert second.get({"id": 1}) == {"value": "second"}
        assert first.path == path
        first.clear()
        assert first.get({"id": 1}) is None
        assert second.get({"id": 1}) == {"value": "second"}
        second.delete({"id": 1})
        assert second.get({"id": 1}) is None
        first._conn.execute(
            "INSERT INTO cache_entries VALUES (?, ?, ?, ?)",
            ("first", "bad", "broken{", 0),
        )
        first._conn.commit()
        assert first.get("bad") is None
        assert (
            first._conn.execute(
                "SELECT COUNT(*) FROM cache_entries WHERE key='bad'"
            ).fetchone()[0]
            == 0
        )
    finally:
        first._conn.close()
        second._conn.close()


@pytest.mark.parametrize("operation", ["get", "set", "delete", "clear"])
def test_sqlite_failure_disables_cache_without_crashing(operation, tmp_path):
    cache = PersistentCache(str(tmp_path / "cache.sqlite"), "test")
    cache._conn.close()
    connection = MagicMock()
    connection.execute.side_effect = sqlite3.OperationalError("unavailable")
    connection.close.side_effect = sqlite3.OperationalError("already closed")
    cache._conn = connection
    args = {"get": ("key",), "set": ("key", {}), "delete": ("key",), "clear": ()}[
        operation
    ]
    assert getattr(cache, operation)(*args) is None
    assert cache._disabled and cache._conn is None
    # Subsequent operations must safely no-op.
    assert cache.get("key") is None
    cache.set("key", {})
    cache.delete("key")
    cache.clear()
    cache._disable(RuntimeError("repeat"))


def test_connection_failure_serialization_and_factory(tmp_path):
    with patch("sqlite3.connect", side_effect=sqlite3.OperationalError("offline")):
        cache = PersistentCache(str(tmp_path / "missing.sqlite"), "test")
    assert cache.get("key") is None and cache._disabled
    assert PersistentCache._serialise_key(b"key") == "key"
    assert PersistentCache._serialise_key({"b": 2, "a": 1}) == '{"a": 1, "b": 2}'
    assert PersistentCache._serialise_key({1, 2}) == str({1, 2})
    shared = get_persistent_cache(str(tmp_path / "shared.sqlite"), "first")
    distinct = get_persistent_cache(str(tmp_path / "shared.sqlite"), "second")
    try:
        assert get_persistent_cache(shared.path, "first") is shared
        assert distinct is not shared
    finally:
        shared._conn.close()
        distinct._conn.close()
