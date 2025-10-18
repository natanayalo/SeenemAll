from __future__ import annotations

import asyncio

from etl.tmdb_client import TMDBClient


def test_throttle_respects_rate(monkeypatch):
    async def scenario():
        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        monkeypatch.setattr("etl.tmdb_client.asyncio.sleep", fake_sleep)
        monkeypatch.setattr("etl.tmdb_client.time.time", lambda: 0.0)

        client = TMDBClient("key", rate_per_sec=2.0)
        await client._throttle()
        await client.aclose()

        assert sleep_calls == [0.5]

    asyncio.run(scenario())


def test_get_appends_api_key_and_raises(monkeypatch):
    async def scenario():
        class DummyResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"ok": True}

        class DummyAsyncClient:
            def __init__(self, timeout):
                self.timeout = timeout
                self.calls = []
                self.closed = False

            async def get(self, url, params):
                self.calls.append((url, params))
                return DummyResponse()

            async def aclose(self):
                self.closed = True

        dummy_client = DummyAsyncClient(timeout=15.0)

        def fake_async_client(timeout):
            assert timeout == 15.0
            return dummy_client

        monkeypatch.setattr("etl.tmdb_client.httpx.AsyncClient", fake_async_client)
        monkeypatch.setattr("etl.tmdb_client.time.time", lambda: 10.0)

        client = TMDBClient("secret", rate_per_sec=100.0)
        data = await client._get("/foo", {"page": 3})

        assert data == {"ok": True}
        assert dummy_client.calls[0][0].endswith("/foo")
        assert dummy_client.calls[0][1]["api_key"] == "secret"
        await client.aclose()
        assert dummy_client.closed is True

    asyncio.run(scenario())


def test_iter_list_handles_trending(monkeypatch):
    async def scenario():
        client = TMDBClient("secret")

        async def fake_get(path, params):
            assert path == "/trending/movie/day"
            assert params == {"page": 1}
            return {"results": [{"id": 1}, {"id": 2}]}

        monkeypatch.setattr(client, "_get", fake_get)

        results = []
        async for item in client.iter_list("movie", "trending", pages=1):
            results.append(item)
        await client.aclose()

        assert [it["media_type"] for it in results] == ["movie", "movie"]

    asyncio.run(scenario())


def test_iter_list_handles_regular_lists(monkeypatch):
    async def scenario():
        client = TMDBClient("secret")

        async def fake_get(path, params):
            assert path == "/tv/popular"
            assert params == {"page": 1}
            return {"results": [{"id": 5}]}

        monkeypatch.setattr(client, "_get", fake_get)

        items = [item async for item in client.iter_list("tv", "popular", pages=1)]
        await client.aclose()

        assert items[0]["media_type"] == "tv"

    asyncio.run(scenario())


def test_details_delegates_to_get(monkeypatch):
    async def scenario():
        client = TMDBClient("secret")

        async def fake_get(path, params):
            assert path == "/movie/9"
            assert params == {
                "language": "en-US",
                "append_to_response": "release_dates",
            }
            return {"id": 9}

        monkeypatch.setattr(client, "_get", fake_get)
        data = await client.details("movie", 9)
        await client.aclose()

        assert data["id"] == 9

    asyncio.run(scenario())


def test_details_tv_includes_content_ratings(monkeypatch):
    async def scenario():
        client = TMDBClient("secret")

        async def fake_get(path, params):
            assert path == "/tv/42"
            assert params == {
                "language": "en-US",
                "append_to_response": "content_ratings",
            }
            return {"id": 42}

        monkeypatch.setattr(client, "_get", fake_get)
        data = await client.details("tv", 42)
        await client.aclose()

        assert data["id"] == 42

    asyncio.run(scenario())


def test_search_delegates_to_get(monkeypatch):
    async def scenario():
        client = TMDBClient("secret")

        async def fake_get(path, params):
            assert path == "/search/multi"
            assert params == {"query": "test query"}
            return {"results": [{"id": 1}]}

        monkeypatch.setattr(client, "_get", fake_get)
        data = await client.search("test query")
        await client.aclose()

        assert data["results"][0]["id"] == 1

    asyncio.run(scenario())


def test_search_with_media_type_delegates_to_get(monkeypatch):
    async def scenario():
        client = TMDBClient("secret")

        async def fake_get(path, params):
            assert path == "/search/movie"
            assert params == {"query": "test query"}
            return {"results": [{"id": 1}]}

        monkeypatch.setattr(client, "_get", fake_get)
        data = await client.search("test query", media_type="movie")
        await client.aclose()

        assert data["results"][0]["id"] == 1

    asyncio.run(scenario())
