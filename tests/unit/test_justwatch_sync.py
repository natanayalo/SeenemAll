from __future__ import annotations

import asyncio

from etl import justwatch_sync as mod


def test_normalise_offer_extracts_expected_fields():
    offer = {
        "package_short_name": "nfx",
        "monetization_type": "flatrate",
        "urls": {
            "deeplink_ios": "app://open",
            "standard_web": "https://netflix.example/title",
        },
    }

    normalised = mod._normalise_offer(offer)
    assert normalised["service"] == "nfx"
    assert normalised["offer_type"] == "FLATRATE"
    assert normalised["deeplink"] == "app://open"
    assert normalised["web_url"].startswith("https://netflix")


def test_prepare_offers_deduplicates_on_service_type_web():
    offers = [
        {
            "package_short_name": "disney",
            "monetization_type": "flatrate",
            "urls": {"standard_web": "https://disney.example"},
        },
        {
            "package_short_name": "disney",
            "monetization_type": "flatrate",
            "urls": {"standard_web": "https://disney.example"},
        },
        {
            "provider_id": 7,
            "monetization_type": "rent",
            "urls": {"web": "https://store.example"},
        },
    ]

    prepared = mod._prepare_offers(offers)
    assert len(prepared) == 2
    assert prepared[0]["service"] == "disney"
    assert prepared[1]["service"] == "7"
    assert prepared[1]["offer_type"] == "RENT"


def test_replace_availability_deletes_and_inserts():
    captured = {"delete": None, "payload": None}

    class DummySession:
        def execute(self, stmt):
            captured["delete"] = stmt

        def bulk_insert_mappings(self, model, payload):
            captured["model"] = model
            captured["payload"] = payload

    session = DummySession()
    offers = [
        {
            "service": "nfx",
            "offer_type": "FLATRATE",
            "deeplink": "app://open",
            "web_url": "https://netflix.example",
        }
    ]

    mod._replace_availability(session, item_id=5, country="US", offers=offers)

    assert captured["delete"] is not None
    assert captured["payload"][0]["item_id"] == 5
    assert captured["payload"][0]["service"] == "nfx"


def test_sync_availability_skips_failed_fetches(monkeypatch):
    items = [
        mod.ItemRow(1, 100, "movie", "Example Movie", 2020),
        mod.ItemRow(2, 200, "tv", "Example Show", 2021),
    ]

    monkeypatch.setattr(mod, "_fetch_catalog_items", lambda db, limit: items)

    async def fake_fetch_chunk(client, rows):
        assert rows == items[: mod.CHUNK_SIZE]
        return [
            {
                "match": {"id": "tm1"},
                "offers": [
                    {
                        "monetizationType": "FLATRATE",
                        "standardWebURL": "https://netflix.example",
                        "preAffiliatedStandardWebURL": None,
                        "package": {
                            "shortName": "nfx",
                            "packageId": 8,
                            "clearName": "",
                        },
                    }
                ],
            },
            RuntimeError("boom"),
        ]

    monkeypatch.setattr(mod, "_fetch_chunk", fake_fetch_chunk)

    replacements = []

    def fake_replace(db, item_id, country, offers):
        replacements.append((item_id, country, list(offers)))

    monkeypatch.setattr(mod, "_replace_availability", fake_replace)

    sessions = []

    class FakeSession:
        def __enter__(self):
            sessions.append(self)
            self.commits = 0
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def commit(self):
            self.commits += 1

    def fake_sessionmaker():
        return FakeSession()

    monkeypatch.setattr(mod, "get_sessionmaker", lambda: fake_sessionmaker)

    class DummyClient:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    dummy_client = DummyClient()
    monkeypatch.setattr(mod, "JustWatchClient", lambda **_: dummy_client)

    asyncio.run(mod._sync_availability(country="US", limit=10))

    assert replacements == [
        (
            1,
            "US",
            [
                {
                    "service": "nfx",
                    "offer_type": "FLATRATE",
                    "web_url": "https://netflix.example",
                    "deeplink": "https://netflix.example",
                }
            ],
        )
    ]
    # First session is for catalog read, second for upserts.
    assert sessions[1].commits == 1
    assert dummy_client.closed is True
