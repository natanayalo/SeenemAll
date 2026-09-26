import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Optional, Tuple


logger = logging.getLogger(__name__)


class PersistentCache:
    """
    Tiny SQLite-backed cache for storing structured payloads.

    Values are stored as JSON strings keyed by a (namespace, key) pair.
    Callers are expected to provide JSON-serialisable payloads.
    """

    def __init__(self, path: str, namespace: str) -> None:
        self._path = os.path.abspath(path)
        self._namespace = namespace
        directory = os.path.dirname(self._path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        self._lock = threading.Lock()
        self._disabled = False
        self._conn: Optional[sqlite3.Connection]
        try:
            self._conn = sqlite3.connect(self._path, check_same_thread=False)
            with self._conn:
                self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA synchronous=NORMAL;")
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        namespace TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        updated_at REAL NOT NULL,
                        PRIMARY KEY(namespace, key)
                    )
                    """
                )
        except sqlite3.Error as exc:
            logger.warning(
                "Failed to initialize persistent cache at %s (%s). Disabling storage.",
                self._path,
                exc,
            )
            self._conn = None
            self._disabled = True

    @property
    def path(self) -> str:
        return self._path

    def get(self, key: Any) -> Optional[Dict[str, Any]]:
        if self._disabled or self._conn is None:
            return None
        key_str = self._serialise_key(key)
        with self._lock:
            try:
                cursor = self._conn.execute(
                    "SELECT value FROM cache_entries WHERE namespace = ? AND key = ?",
                    (self._namespace, key_str),
                )
                row = cursor.fetchone()
            except sqlite3.Error as exc:
                self._disable(exc)
                return None
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            self.delete(key)
            return None

    def set(self, key: Any, value: Any) -> None:
        if self._disabled or self._conn is None:
            return
        key_str = self._serialise_key(key)
        value_str = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            try:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO cache_entries (namespace, key, value, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (self._namespace, key_str, value_str, time.time()),
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                self._disable(exc)

    def delete(self, key: Any) -> None:
        if self._disabled or self._conn is None:
            return
        key_str = self._serialise_key(key)
        with self._lock:
            try:
                self._conn.execute(
                    "DELETE FROM cache_entries WHERE namespace = ? AND key = ?",
                    (self._namespace, key_str),
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                self._disable(exc)

    @staticmethod
    def _serialise_key(key: Any) -> str:
        if isinstance(key, str):
            return key
        if isinstance(key, bytes):
            return key.decode("utf-8", errors="ignore")
        try:
            return json.dumps(key, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(key)

    def _disable(self, exc: Exception) -> None:
        if not self._disabled:
            logger.warning(
                "Persistent cache at %s became unavailable (%s); disabling storage.",
                self._path,
                exc,
            )
        self._disabled = True
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


_CACHE_SINGLETONS: Dict[Tuple[str, str], PersistentCache] = {}
_FACTORY_LOCK = threading.Lock()


def get_persistent_cache(path: str, namespace: str) -> PersistentCache:
    key = (os.path.abspath(path), namespace)
    with _FACTORY_LOCK:
        cache = _CACHE_SINGLETONS.get(key)
        if cache is None:
            cache = PersistentCache(path, namespace)
            _CACHE_SINGLETONS[key] = cache
        return cache
