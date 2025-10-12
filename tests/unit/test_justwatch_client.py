from __future__ import annotations

import asyncio
import pytest
import httpx
from typing import Any, Dict, List

from etl.justwatch_client import (
    JustWatchClient,
    _choose_best_match,
    flatten_offers,
)


class DummyClient(JustWatchClient):
    def __init__(self) -> None:
        super().__init__(country="US")
        self.calls: List[Dict[str, Any]] = []

    async def _graphql(self, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append({"query": query, "variables": variables})
        if "searchTitles" in query:
            return {
                "searchTitles": {
                    "edges": [
                        {
                            "node": {
                                "id": "tm42409",
                                "objectType": "MOVIE",
                                "objectId": 42409,
                                "content": {
                                    "title": "Fight Club",
                                    "originalReleaseYear": 1999,
                                },
                            }
                        },
                        {
                            "node": {
                                "id": "ts999",
                                "objectType": "SHOW",
                                "objectId": 999,
                                "content": {
                                    "title": "Fight Club Series",
                                    "originalReleaseYear": 2020,
                                },
                            }
                        },
                    ]
                }
            }
        if "TitleOffers" in query:
            return {
                "node": {
                    "__typename": "Movie",
                    "offers": [
                        {
                            "monetizationType": "FLATRATE",
                            "standardWebURL": "https://example.com/watch",
                            "preAffiliatedStandardWebURL": None,
                            "package": {
                                "packageId": 10,
                                "shortName": "nfx",
                                "clearName": "Netflix",
                            },
                        },
                        {
                            "monetizationType": "BUY",
                            "standardWebURL": "https://example.com/buy",
                            "preAffiliatedStandardWebURL": "app://buy",
                            "package": {
                                "packageId": 20,
                                "shortName": "amz",
                                "clearName": "Amazon",
                            },
                        },
                    ],
                }
            }
        return {}


def test_search_and_fetch_offers():
    async def scenario():
        client = DummyClient()
        match, offers = await client.resolve_offers_for_item(
            title="Fight Club", media_type="movie", release_year=1999
        )
        await client.aclose()
        return match, offers

    match, offers = asyncio.run(scenario())
    assert match is not None
    assert match["id"] == "tm42409"
    assert len(offers) == 2
    flattened = flatten_offers(offers)
    assert flattened[0]["service"] == "nfx"
    assert flattened[0]["offer_type"] == "FLATRATE"
    assert flattened[1]["deeplink"] == "app://buy"


def test_choose_best_match_prefers_year_match():
    candidates = [
        {
            "id": "a",
            "objectType": "MOVIE",
            "content": {"title": "Example", "originalReleaseYear": 1998},
        },
        {
            "id": "b",
            "objectType": "MOVIE",
            "content": {"title": "Example", "originalReleaseYear": 2000},
        },
    ]
    match = _choose_best_match("Example", "movie", 2000, candidates)
    assert match["id"] == "b"


def test_choose_best_match_handles_missing_year():
    candidates = [
        {
            "id": "a",
            "objectType": "SHOW",
            "content": {"title": "Example Show", "originalReleaseYear": None},
        }
    ]
    match = _choose_best_match("Example Show", "tv", None, candidates)
    assert match["id"] == "a"


def test_graphql_raises_on_error(monkeypatch):
    client = JustWatchClient(country="US")

    class DummyResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"errors": [{"message": "boom"}]}

    async def fake_post(url, json):
        return DummyResponse()

    monkeypatch.setattr(client._client, "post", fake_post)

    async def scenario():
        try:
            await client._graphql("query {}", {})
        finally:
            await client.aclose()

    with pytest.raises(Exception):
        asyncio.run(scenario())


def test_graphql_returns_payload(monkeypatch):
    client = JustWatchClient(country="US")

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"ok": True}}

    async def fake_post(url, json):
        return DummyResponse()

    monkeypatch.setattr(client._client, "post", fake_post)

    async def scenario():
        try:
            return await client._graphql("query {}", {})
        finally:
            await client.aclose()

    data = asyncio.run(scenario())
    assert data == {"ok": True}


def test_resolve_offers_handles_search_failure(monkeypatch):
    client = JustWatchClient(country="US")

    async def failing_graphql(query, variables):
        raise httpx.HTTPError("network down")

    monkeypatch.setattr(client, "_graphql", failing_graphql)

    async def scenario():
        match, offers = await client.resolve_offers_for_item(
            title="Missing", media_type="movie", release_year=2024
        )
        await client.aclose()
        return match, offers

    match, offers = asyncio.run(scenario())
    assert match is None
    assert offers == []


def test_resolve_offers_returns_none_without_match(monkeypatch):
    client = JustWatchClient(country="US")

    async def fake_search(title, media_type, limit=5):
        return []

    monkeypatch.setattr(client, "search_candidates", fake_search)

    async def scenario():
        result = await client.resolve_offers_for_item(
            title="Ghost", media_type="movie", release_year=2023
        )
        await client.aclose()
        return result

    match, offers = asyncio.run(scenario())
    assert match is None and offers == []


def test_resolve_offers_handles_fetch_failure(monkeypatch):
    client = JustWatchClient(country="US")

    async def fake_search(title, media_type, limit=5):
        return [
            {
                "id": "tm123",
                "objectType": "MOVIE",
                "content": {"title": title, "originalReleaseYear": 2020},
            }
        ]

    async def fake_fetch(node_id):
        raise httpx.HTTPError("fetch failed")

    monkeypatch.setattr(client, "search_candidates", fake_search)
    monkeypatch.setattr(client, "fetch_offers_by_id", fake_fetch)

    async def scenario():
        result = await client.resolve_offers_for_item(
            title="Example", media_type="movie", release_year=2020
        )
        await client.aclose()
        return result

    match, offers = asyncio.run(scenario())
    assert match is not None and offers == []
