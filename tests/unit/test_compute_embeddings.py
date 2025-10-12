from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from etl import compute_embeddings as mod
from api.db.models import ItemEmbedding


def test_build_text_truncates_and_concatenates():
    long_overview = "x" * (mod.MAX_LEN + 10)
    result = mod._build_text("Title", long_overview)
    assert result.startswith("Title :: ")
    assert result.endswith("...")
    assert len(result) == len("Title :: ") + mod.MAX_LEN + 3


def test_select_missing_ids_returns_ids():
    captured = {}

    class DummyResult:
        def __init__(self, rows):
            self.rows = rows

        def fetchall(self):
            return self.rows

    class DummySession:
        def execute(self, query, params):
            captured["query"] = query
            captured["params"] = params
            return DummyResult([(1,), (2,), (3,)])

    ids = mod._select_missing_ids(
        DummySession(), limit=5, version=mod.DEFAULT_EMBED_VERSION
    )
    assert ids == [1, 2, 3]
    assert captured["params"] == {"lim": 5, "version": mod.DEFAULT_EMBED_VERSION}


def test_fetch_items_returns_text_pairs():
    class ScalarResult:
        def __init__(self, items):
            self._items = items
            self._idx = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._idx >= len(self._items):
                raise StopIteration
            result = self._items[self._idx]
            self._idx += 1
            return result

    class DummySession(Session):
        def __init__(self):
            # Initialize parent without engine connection
            super().__init__(bind=None)
            self.seen = []

        def execute(self, stmt, *args, **kwargs):
            self.seen.append(stmt)
            return self

        def scalars(self):
            items = [
                type(
                    "DummyItem",
                    (),
                    {
                        "id": 1,
                        "title": "Title",
                        "overview": "Overview",
                        "genres": [],
                        "release_year": None,
                        "media_type": "movie",
                        "runtime": None,
                    },
                ),
                type(
                    "DummyItem",
                    (),
                    {
                        "id": 2,
                        "title": "Another",
                        "overview": None,
                        "genres": [],
                        "release_year": None,
                        "media_type": "movie",
                        "runtime": None,
                    },
                ),
            ]
            return ScalarResult(items)

    results = mod._fetch_items(DummySession(), [1, 2])
    assert len(results) == 2
    assert results[0][0] == 1
    assert "Title :: Overview" in results[0][1]
    assert results[1][0] == 2
    # Verify the exact content
    assert results == [
        (1, "Title :: Overview"),
        (2, "Another :: None"),  # None overview is included in the output
    ]

    empty = mod._fetch_items(DummySession(), [])
    assert empty == []


def test_upsert_vectors_handles_existing_and_new(monkeypatch):
    class ExistingEmbedding:
        def __init__(self, item_id, version):
            self.item_id = item_id
            self.version = version
            self.vector = [0.0, 0.0]

    class DummyQuery:
        def __init__(self, session):
            self.session = session
            self._item_id = None
            self._version = None

        def filter(self, *exprs):
            for expr in exprs:
                left_name = getattr(getattr(expr, "left", None), "name", None)
                right_value = getattr(getattr(expr, "right", None), "value", None)
                if left_name == "item_id":
                    self._item_id = right_value
                if left_name == "version":
                    self._version = right_value
            return self

        def one_or_none(self):
            key = (self._item_id, self._version)
            return self.session.existing.get(key)

    class DummySession(Session):
        def __init__(self):
            # Initialize parent without engine connection
            super().__init__(bind=None)
            self.existing = {(10, "v1"): ExistingEmbedding(10, "v1")}
            self.added = []
            self.queries = 0

        def query(self, model):
            assert model is ItemEmbedding
            self.queries += 1
            return DummyQuery(self)

        def add(self, obj):
            self.added.append(obj)

    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    session = DummySession()

    mod._upsert_vectors(session, [10, 20], vectors, version="v1")

    assert session.existing[(10, "v1")].vector == [1.0, 0.0]
    assert len(session.added) == 1
    assert isinstance(session.added[0], ItemEmbedding)
    assert session.added[0].item_id == 20
    assert session.added[0].version == "v1"


def test_run_processes_batches(monkeypatch, capsys):
    class FakeSession(Session):
        def __init__(self):
            # Initialize parent without engine connection
            super().__init__(bind=None)
            self.commits = 0
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.closed = True

        def commit(self):
            self.commits += 1

    session = FakeSession()
    monkeypatch.setattr(mod, "get_engine", lambda: "engine")
    monkeypatch.setattr(mod, "get_sessionmaker", lambda: (lambda: session))

    batches = [[101, 102], []]
    select_calls = []

    def fake_select(db, limit, version):
        select_calls.append(limit)
        assert version == mod.DEFAULT_EMBED_VERSION
        return batches.pop(0)

    fetched = []

    def fake_fetch(db, ids):
        fetched.append(list(ids))
        return [(iid, f"text-{iid}") for iid in ids]

    encoded = []

    def fake_encode(texts):
        encoded.append(list(texts))
        return np.array(
            [[idx, idx] for idx, _ in enumerate(texts, start=1)], dtype="float32"
        )

    upserts = []

    def fake_upsert(db, ids, vectors, version):
        upserts.append((db, list(ids), vectors.tolist(), version))

    monkeypatch.setattr(mod, "_select_missing_ids", fake_select)
    monkeypatch.setattr(mod, "_fetch_items", fake_fetch)
    monkeypatch.setattr(mod, "encode_texts", fake_encode)
    monkeypatch.setattr(mod, "_upsert_vectors", fake_upsert)

    mod.run(batch=50)

    captured = capsys.readouterr()
    assert f"Embedded 2 items for version '{mod.DEFAULT_EMBED_VERSION}'" in captured.out
    assert (
        f"Embedding run complete for version '{mod.DEFAULT_EMBED_VERSION}'"
        in captured.out
    )
    assert fetched == [[101, 102]]
    assert encoded == [["text-101", "text-102"]]
    assert len(upserts) == 1
    assert upserts[0][0] is session
    assert upserts[0][1] == [101, 102]
    assert upserts[0][3] == mod.DEFAULT_EMBED_VERSION
    assert session.commits == 1
    assert session.closed is True
    assert select_calls == [50, 50]


def test_run_respects_max_items(monkeypatch):
    class FakeSession:
        def __init__(self):
            self.commits = 0
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.closed = True

        def commit(self):
            self.commits += 1

    session = FakeSession()
    monkeypatch.setattr(mod, "get_engine", lambda: "engine")
    monkeypatch.setattr(mod, "get_sessionmaker", lambda: (lambda: session))

    batches = [[1, 2, 3, 4], [5]]

    def fake_select(db, limit, version):
        assert version == mod.DEFAULT_EMBED_VERSION
        return batches.pop(0)

    fetch_calls = []

    def fake_fetch(db, ids):
        fetch_calls.append(list(ids))
        return [(iid, f"text-{iid}") for iid in ids]

    def fake_encode(texts):
        return np.ones((len(texts), 2), dtype="float32")

    upserts = []

    def fake_upsert(db, ids, vectors, version):
        upserts.append((list(ids), version))

    monkeypatch.setattr(mod, "_select_missing_ids", fake_select)
    monkeypatch.setattr(mod, "_fetch_items", fake_fetch)
    monkeypatch.setattr(mod, "encode_texts", fake_encode)
    monkeypatch.setattr(mod, "_upsert_vectors", fake_upsert)

    mod.run(batch=5, max_items=3)

    assert fetch_calls == [[1, 2, 3]]
    assert upserts == [([1, 2, 3], mod.DEFAULT_EMBED_VERSION)]
    assert session.commits == 1
    assert session.closed is True


def test_run_accepts_explicit_version(monkeypatch):
    class FakeSession:
        def __init__(self):
            self.commits = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def commit(self):
            self.commits += 1

    session = FakeSession()
    monkeypatch.setattr(mod, "get_engine", lambda: "engine")
    monkeypatch.setattr(mod, "get_sessionmaker", lambda: (lambda: session))

    batches = [[11], []]
    versions_seen = []
    upsert_versions = []

    def fake_select(db, limit, version):
        versions_seen.append(version)
        return batches.pop(0)

    def fake_fetch(db, ids):
        return [(iid, f"text-{iid}") for iid in ids]

    def fake_encode(texts):
        return np.ones((len(texts), 2), dtype="float32")

    def fake_upsert(db, ids, vectors, version):
        upsert_versions.append(version)

    monkeypatch.setattr(mod, "_select_missing_ids", fake_select)
    monkeypatch.setattr(mod, "_fetch_items", fake_fetch)
    monkeypatch.setattr(mod, "encode_texts", fake_encode)
    monkeypatch.setattr(mod, "_upsert_vectors", fake_upsert)

    mod.run(batch=5, version="exp-v2")

    assert versions_seen == ["exp-v2", "exp-v2"]
    assert upsert_versions == ["exp-v2"]
    assert session.commits == 1
