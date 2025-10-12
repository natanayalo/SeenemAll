from __future__ import annotations

import asyncio
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
