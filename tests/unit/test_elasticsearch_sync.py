import argparse
import datetime as dt
from types import SimpleNamespace

import pytest

from etl import elasticsearch_sync
from etl.elasticsearch_sync import (
    ItemSnapshot,
    _build_document,
    _chunker,
    _generate_actions,
    _normalise_genres,
    _normalise_providers,
    _parse_datetime,
    _vector_to_list,
)


def test_normalise_genres_handles_list_of_dicts():
    genres = [{"id": 1, "name": "Drama"}, {"id": 2, "name": "Sci-Fi"}]
    assert _normalise_genres(genres) == ["Drama", "Sci-Fi"]


def test_normalise_genres_handles_nested_mapping():
    payload = {"genres": ["Action", "Comedy"]}
    assert _normalise_genres(payload) == ["Action", "Comedy"]


def test_normalise_providers_deduplicates_and_sorts():
    rows = [
        (1, "netflix"),
        (1, "netflix"),
        (1, "hulu"),
        (2, "prime"),
    ]
    assert _normalise_providers(rows) == {1: ["hulu", "netflix"], 2: ["prime"]}


def test_vector_to_list_handles_pgvector_objects():
    class FakeVector:
        def __init__(self, data):
            self._data = data

        def tolist(self):
            return self._data

    vec = FakeVector([0.1, 0.2, 0.3])
    assert _vector_to_list(vec) == [0.1, 0.2, 0.3]


def test_build_document_formats_snapshot():
    snapshot = ItemSnapshot(
        id=42,
        media_type="movie",
        title="Example",
        overview="Test overview",
        runtime=123,
        release_year=2024,
        maturity_rating="PG-13",
        genres=[{"id": 1, "name": "Drama"}],
        popularity=9.5,
        cast=[{"name": "Actor One"}, "Actor Two"],
        directors=[{"name": "Director One"}],
        producers=["Producer One"],
        writers=[{"name": "Writer One"}],
        keywords=[{"name": "Heist"}, "Thriller"],
        spoken_languages=[{"english_name": "English"}, "Spanish"],
        updated_at=dt.datetime(2024, 1, 1, 12, 0, 0),
    )

    doc = _build_document(
        snapshot,
        embedding=[0.1, 0.2],
        providers=["netflix"],
    )

    assert doc["item_id"] == "42"
    assert doc["media_type"] == "movie"
    assert doc["genres"] == ["Drama"]
    assert doc["streaming_providers"] == ["netflix"]
    assert doc["embedding"] == [0.1, 0.2]
    assert doc["updated_at"].startswith("2024-01-01T12:00:00")
    assert doc["updated_at"].endswith("Z")
    assert doc["cast"] == ["Actor One", "Actor Two"]
    assert doc["directors"] == ["Director One"]
    assert doc["producers"] == ["Producer One"]
    assert doc["writers"] == ["Writer One"]
    assert doc["keywords"] == ["Heist", "Thriller"]
    assert doc["spoken_languages"] == ["English", "Spanish"]


def test_generate_actions_skips_missing_embedding():
    now = dt.datetime(2024, 1, 1, 0, 0)
    items = [
        ItemSnapshot(
            id=1,
            media_type="movie",
            title="One",
            overview=None,
            runtime=None,
            release_year=None,
            maturity_rating=None,
            genres=None,
            popularity=None,
            cast=None,
            directors=None,
            producers=None,
            writers=None,
            keywords=None,
            spoken_languages=None,
            updated_at=now,
        ),
        ItemSnapshot(
            id=2,
            media_type="movie",
            title="Two",
            overview=None,
            runtime=None,
            release_year=None,
            maturity_rating=None,
            genres=None,
            popularity=None,
            cast=None,
            directors=None,
            producers=None,
            writers=None,
            keywords=None,
            spoken_languages=None,
            updated_at=now,
        ),
    ]
    embeddings = {1: [0.1, 0.2]}
    providers = {1: ["netflix"], 2: ["hulu"]}

    actions = list(_generate_actions(items, embeddings, providers, index_name="items"))

    assert len(actions) == 1
    assert actions[0]["_id"] == "1"
    assert actions[0]["_source"]["streaming_providers"] == ["netflix"]


def test_chunker_batches_records():
    now = dt.datetime(2024, 1, 1, 0, 0)

    class FakeScalarResult:
        def __init__(self, rows):
            self._rows = rows

        def scalars(self):
            return self

        def all(self):
            return self._rows

    batches = [
        [
            SimpleNamespace(
                id=1,
                media_type="movie",
                title="One",
                overview=None,
                runtime=None,
                release_year=None,
                maturity_rating=None,
                genres=None,
                popularity=None,
                cast=None,
                directors=None,
                producers=None,
                writers=None,
                keywords=None,
                spoken_languages=None,
                updated_at=now,
            )
        ],
        [
            SimpleNamespace(
                id=2,
                media_type="tv",
                title="Two",
                overview=None,
                runtime=None,
                release_year=None,
                maturity_rating=None,
                genres=None,
                popularity=None,
                cast=None,
                directors=None,
                producers=None,
                writers=None,
                keywords=None,
                spoken_languages=None,
                updated_at=now,
            )
        ],
        [],
    ]

    class FakeSession:
        def __init__(self):
            self.calls = 0

        def execute(self, stmt):
            result = batches[self.calls]
            self.calls += 1
            return FakeScalarResult(result)

    chunks = list(_chunker(FakeSession(), batch_size=1, updated_after=None))
    assert len(chunks) == 2
    assert chunks[0][0].id == 1
    assert chunks[1][0].media_type == "tv"


def test_parse_datetime_valid_and_invalid():
    value = _parse_datetime("2024-01-02T03:04:05Z")
    assert value.year == 2024
    assert value.tzinfo is not None

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_datetime("invalid")


def _make_sessionmaker() -> object:
    class FakeSession:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeSessionmaker:
        def __call__(self):
            return FakeSession()

    return FakeSessionmaker()


def test_sync_catalog_respects_max_items(monkeypatch: pytest.MonkeyPatch) -> None:
    now = dt.datetime(2024, 1, 1, 0, 0)
    snapshots = [
        ItemSnapshot(
            id=1,
            media_type="movie",
            title="One",
            overview=None,
            runtime=None,
            release_year=None,
            maturity_rating=None,
            genres=None,
            popularity=None,
            cast=None,
            directors=None,
            producers=None,
            writers=None,
            keywords=None,
            spoken_languages=None,
            updated_at=now,
        ),
        ItemSnapshot(
            id=2,
            media_type="movie",
            title="Two",
            overview=None,
            runtime=None,
            release_year=None,
            maturity_rating=None,
            genres=None,
            popularity=None,
            cast=None,
            directors=None,
            producers=None,
            writers=None,
            keywords=None,
            spoken_languages=None,
            updated_at=now,
        ),
    ]

    def fake_chunker(session, batch_size, updated_after):
        yield snapshots

    def fake_load_embeddings(session, item_ids, version):
        assert item_ids == [1, 2]
        assert version == "v-test"
        return {1: [0.1, 0.2], 2: [0.3, 0.4]}

    def fake_load_providers(session, item_ids):
        assert item_ids == [1, 2]
        return {1: ["netflix"], 2: ["hulu"]}

    bulk_calls = []

    def fake_bulk(client, actions, refresh=False):
        bulk_calls.append((list(actions), refresh))

    class FakeClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    client_instance = FakeClient()

    monkeypatch.setattr(elasticsearch_sync, "_chunker", fake_chunker)
    monkeypatch.setattr(elasticsearch_sync, "_load_embeddings", fake_load_embeddings)
    monkeypatch.setattr(elasticsearch_sync, "_load_providers", fake_load_providers)
    monkeypatch.setattr(elasticsearch_sync.helpers, "bulk", fake_bulk)
    monkeypatch.setattr(
        elasticsearch_sync, "create_elasticsearch_client", lambda: client_instance
    )
    monkeypatch.setattr(elasticsearch_sync, "get_sessionmaker", _make_sessionmaker)

    stats = elasticsearch_sync.sync_catalog(
        batch_size=100, max_items=1, embed_version="v-test", refresh=True
    )

    assert stats == {"indexed": 1, "skipped_missing_embedding": 0}
    assert len(bulk_calls) == 1
    assert len(bulk_calls[0][0]) == 1
    assert bulk_calls[0][1] is True
    assert client_instance.closed is True


def test_sync_catalog_handles_max_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = ItemSnapshot(
        id=1,
        media_type="movie",
        title="Only",
        overview=None,
        runtime=None,
        release_year=None,
        maturity_rating=None,
        genres=None,
        popularity=None,
        cast=None,
        directors=None,
        producers=None,
        writers=None,
        keywords=None,
        spoken_languages=None,
        updated_at=dt.datetime(2024, 1, 1, 0, 0),
    )

    def fake_chunker(session, batch_size, updated_after):
        yield [snapshot]

    monkeypatch.setattr(elasticsearch_sync, "_chunker", fake_chunker)
    monkeypatch.setattr(
        elasticsearch_sync, "_load_embeddings", lambda *args, **kwargs: {1: [0.1, 0.2]}
    )
    monkeypatch.setattr(
        elasticsearch_sync, "_load_providers", lambda *args, **kwargs: {1: ["netflix"]}
    )

    def fail_bulk(*args, **kwargs):
        raise AssertionError("bulk should not run")

    monkeypatch.setattr(elasticsearch_sync.helpers, "bulk", fail_bulk)
    monkeypatch.setattr(
        elasticsearch_sync,
        "create_elasticsearch_client",
        lambda: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(elasticsearch_sync, "get_sessionmaker", _make_sessionmaker)

    stats = elasticsearch_sync.sync_catalog(max_items=0)
    assert stats == {"indexed": 0, "skipped_missing_embedding": 0}


def test_sync_catalog_skips_when_missing_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = ItemSnapshot(
        id=1,
        media_type="movie",
        title="Only",
        overview=None,
        runtime=None,
        release_year=None,
        maturity_rating=None,
        genres=None,
        popularity=None,
        cast=None,
        directors=None,
        producers=None,
        writers=None,
        keywords=None,
        spoken_languages=None,
        updated_at=dt.datetime(2024, 1, 1, 0, 0),
    )

    def fake_chunker(session, batch_size, updated_after):
        yield [snapshot]

    monkeypatch.setattr(elasticsearch_sync, "_chunker", fake_chunker)
    monkeypatch.setattr(
        elasticsearch_sync, "_load_embeddings", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        elasticsearch_sync, "_load_providers", lambda *args, **kwargs: {1: ["netflix"]}
    )

    def fail_bulk(*args, **kwargs):
        raise AssertionError("bulk should not run")

    monkeypatch.setattr(elasticsearch_sync.helpers, "bulk", fail_bulk)
    monkeypatch.setattr(
        elasticsearch_sync,
        "create_elasticsearch_client",
        lambda: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(elasticsearch_sync, "get_sessionmaker", _make_sessionmaker)

    stats = elasticsearch_sync.sync_catalog()
    assert stats == {"indexed": 0, "skipped_missing_embedding": 1}


def test_main_success(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        elasticsearch_sync,
        "sync_catalog",
        lambda **kwargs: {"indexed": 3, "skipped_missing_embedding": 2},
    )
    exit_code = elasticsearch_sync.main(["--batch-size", "10"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Indexed 3 items" in out


def test_main_handles_errors(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    class DummyError(Exception):
        pass

    def fail(**kwargs):
        raise DummyError("boom")

    monkeypatch.setattr(elasticsearch_sync, "sync_catalog", fail)
    monkeypatch.setattr(elasticsearch_sync, "TransportError", DummyError)

    code = elasticsearch_sync.main([])
    assert code == 1
    out = capsys.readouterr().out
    assert "Elasticsearch sync failed" in out
