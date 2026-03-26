import types

import pytest

from etl import elasticsearch_index
from etl.elasticsearch_index import build_items_index_body, ensure_items_index


def test_items_index_has_expected_embedding_config() -> None:
    body = build_items_index_body()
    embedding = body["mappings"]["properties"]["embedding"]

    assert embedding["type"] == "dense_vector"
    assert embedding["dims"] == 384
    assert embedding["index"] is True
    assert embedding["similarity"] == "cosine"
    opts = embedding["index_options"]
    assert opts["type"] == "hnsw"
    assert opts["ef_construction"] == 128


def test_items_index_includes_structured_fields() -> None:
    body = build_items_index_body()
    props = body["mappings"]["properties"]

    assert props["item_id"]["type"] == "keyword"
    assert props["genres"]["type"] == "keyword"
    assert props["media_type"]["type"] == "keyword"
    assert props["runtime"]["type"] == "short"
    assert props["release_year"]["type"] == "short"
    assert props["streaming_providers"]["type"] == "keyword"


def test_items_index_uses_strict_dynamic_mapping() -> None:
    body = build_items_index_body()

    assert body["mappings"]["dynamic"] == "strict"


def test_ensure_items_index_skips_when_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    created = False

    class FakeIndices:
        def exists(self, index: str) -> bool:
            return True

        def create(self, *args, **kwargs):
            nonlocal created
            created = True

    client = types.SimpleNamespace(indices=FakeIndices())
    ensure_items_index(client)
    assert created is False


def test_ensure_items_index_force_recreates(monkeypatch: pytest.MonkeyPatch) -> None:
    created = False
    deleted = False

    class DummyNotFound(Exception):
        pass

    class FakeIndices:
        def exists(self, index: str) -> bool:
            return False

        def delete(self, index: str) -> None:
            nonlocal deleted
            deleted = True
            raise DummyNotFound()

        def create(self, *args, **kwargs):
            nonlocal created
            created = True

    client = types.SimpleNamespace(indices=FakeIndices())
    monkeypatch.setattr(elasticsearch_index, "NotFoundError", DummyNotFound)
    ensure_items_index(client, force=True)
    assert deleted is True
    assert created is True


def test_ensure_items_index_handles_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    created_attempts = 0

    class DummyConflict(Exception):
        pass

    class FakeIndices:
        def exists(self, index: str) -> bool:
            return False

        def create(self, *args, **kwargs):
            nonlocal created_attempts
            created_attempts += 1
            raise DummyConflict()

    client = types.SimpleNamespace(indices=FakeIndices())
    monkeypatch.setattr(elasticsearch_index, "ConflictError", DummyConflict)
    ensure_items_index(client)
    assert created_attempts == 1
