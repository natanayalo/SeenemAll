from __future__ import annotations

import asyncio

from api.db.models import Item
from etl import tmdb_sync as mod
from etl.tmdb_sync import map_item_payload


def test_map_item_payload_handles_movie_fields():
    payload = map_item_payload(
        {
            "id": 123,
            "media_type": "movie",
            "title": "Example",
            "overview": "Something happens.",
            "runtime": 95,
            "original_language": "en",
            "genres": [{"id": 1, "name": "Action"}],
            "poster_path": "/poster.jpg",
            "release_date": "2024-05-01",
            "popularity": 123.4,
            "vote_average": 8.7,
            "vote_count": 4567,
            "_list_ranks": {
                "popular_rank": 2,
                "trending_rank": 5,
                "top_rated_rank": 4,
            },
        }
    )

    assert payload["tmdb_id"] == 123
    assert payload["media_type"] == "movie"
    assert payload["runtime"] == 95
    assert payload["poster_url"].endswith("/poster.jpg")
    assert payload["release_year"] == 2024
    assert payload["popularity"] == 123.4
    assert payload["vote_average"] == 8.7
    assert payload["vote_count"] == 4567
    assert payload["popular_rank"] == 2
    assert payload["trending_rank"] == 5
    assert payload["top_rated_rank"] == 4


def test_map_item_payload_falls_back_for_tv_runtime_and_name():
    payload = map_item_payload(
        {
            "id": 999,
            "name": "Series Name",
            "episode_run_time": [42],
            "genres": [],
            "first_air_date": "2018-01-01",
        }
    )

    assert payload["media_type"] == "tv"
    assert payload["title"] == "Series Name"
    assert payload["runtime"] == 42
    assert payload["release_year"] == 2018


def test_upsert_items_inserts_and_updates(monkeypatch):
    class DummyResult:
        def __init__(self, rows):
            self.rows = rows

        def all(self):
            return self.rows

    class DummyQuery:
        def __init__(self, session):
            self.session = session
            self.tmdb_id = None

        def filter(self, expr):
            self.tmdb_id = getattr(expr.right, "value", None)
            return self

        def update(self, payload):
            self.session.updated[self.tmdb_id] = payload

    class DummySession:
        def __init__(self):
            self.calls = 0
            self.updated = {}
            self.inserted = []

        def execute(self, stmt):
            self.calls += 1
            if self.calls == 1:
                return DummyResult([(1,)])
            return DummyResult([])

        def bulk_insert_mappings(self, model, payload):
            assert model is Item
            self.inserted.extend(payload)

        def query(self, model):
            assert model is Item
            return DummyQuery(self)

    items = [
        {
            "id": 1,
            "media_type": "movie",
            "title": "Existing",
            "overview": "",
            "genres": [],
            "poster_path": None,
        },
        {
            "id": 2,
            "media_type": "movie",
            "title": "New",
            "overview": "",
            "genres": [],
            "poster_path": None,
        },
    ]

    session = DummySession()
    mod._upsert_items(session, items)

    assert session.updated[1]["title"] == "Existing"
    assert session.inserted[0]["tmdb_id"] == 2


def test_upsert_items_no_ids_short_circuits():
    class DummySession:
        def execute(self, *args, **kwargs):
            raise AssertionError("should not be called for empty input")

    mod._upsert_items(DummySession(), [])


def test_fetch_and_upsert_collects_lists_and_commits(monkeypatch):
    captured = {"upserts": []}
    sessions = []

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

    class FakeSessionmaker:
        def __call__(self):
            sess = FakeSession()
            sessions.append(sess)
            return sess

    list_data = {
        ("movie", "popular"): [{"id": 1}],
        ("movie", "top_rated"): [{"id": 2}],
        ("movie", "trending"): [{"id": 3}],
        ("tv", "popular"): [{"id": 10}],
    }

    class FakeTMDBClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.iter_calls = []
            self.detail_calls = []
            self.closed = False

        async def iter_list(self, media, list_name, pages):
            self.iter_calls.append((media, list_name, pages))
            for item in list_data.get((media, list_name), []):
                yield item

        async def details(self, media, tmdb_id):
            self.detail_calls.append((media, tmdb_id))
            if tmdb_id == 3:
                raise RuntimeError("boom")
            return {"id": tmdb_id, "title": f"{media}-{tmdb_id}"}

        async def aclose(self):
            self.closed = True

    fake_client = FakeTMDBClient(mod.TMDB_API_KEY)

    monkeypatch.setattr(mod, "TMDBClient", lambda api_key: fake_client)
    monkeypatch.setattr(
        mod,
        "_upsert_items",
        lambda db, items: captured["upserts"].append((db, list(items))),
    )

    asyncio.run(mod._fetch_and_upsert(FakeSessionmaker(), pages=2))

    assert fake_client.iter_calls == [
        ("movie", "popular", 2),
        ("movie", "top_rated", 1),
        ("movie", "trending", 1),
        ("tv", "popular", 2),
        ("tv", "top_rated", 1),
        ("tv", "trending", 1),
    ]
    assert fake_client.closed is True
    assert len(captured["upserts"]) == 1
    assert len(sessions) == 1
    db, items = captured["upserts"][0]
    assert db is sessions[0]
    assert sessions[0].commits == 1
    assert sessions[0].closed is True
    assert {item["id"] for item in items} == {1, 2, 10}
    assert all(item["media_type"] in {"movie", "tv"} for item in items)


def test_run_invokes_asyncio(monkeypatch):
    class DummyConnection:
        def __init__(self, tracker):
            self.tracker = tracker

        def __enter__(self):
            self.tracker["enter"] += 1
            return self

        def __exit__(self, exc_type, exc, tb):
            self.tracker["exit"] += 1

    class DummyEngine:
        def __init__(self):
            self.connect_called = 0
            self.tracker = {"enter": 0, "exit": 0}

        def connect(self):
            self.connect_called += 1
            return DummyConnection(self.tracker)

    engine = DummyEngine()
    monkeypatch.setattr(mod, "get_engine", lambda: engine)
    monkeypatch.setattr(mod, "get_sessionmaker", lambda: "factory")

    called = {}

    def fake_fetch(sessionmaker, pages):
        called["sessionmaker"] = sessionmaker
        called["pages"] = pages
        return "FAKE_CORO"

    monkeypatch.setattr(mod, "_fetch_and_upsert", fake_fetch)

    def fake_asyncio_run(coro):
        called["coro"] = coro
        called["asyncio_run_calls"] = called.get("asyncio_run_calls", 0) + 1

    monkeypatch.setattr(mod.asyncio, "run", fake_asyncio_run)

    mod.run(pages=3)

    assert engine.connect_called == 1
    assert engine.tracker["enter"] == 1
    assert engine.tracker["exit"] == 1
    assert called["sessionmaker"] == "factory"
    assert called["pages"] == 3
    assert called["coro"] == "FAKE_CORO"
    assert called["asyncio_run_calls"] == 1
