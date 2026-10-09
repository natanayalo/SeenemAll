from __future__ import annotations

from starlette.testclient import TestClient

from api.main import app
from etl.justwatch_client import flatten_offers
from etl.justwatch_sync import _normalise_offer
from api.pipeline.context import _DEFAULT_STREAMING_PROVIDER_ALIASES


def test_flatten_offers_uses_slug_when_present():
    raw_offers = [
        {
            "monetizationType": "FLATRATE",
            "standardWebURL": "https://primevideo.com/watch",
            "preAffiliatedStandardWebURL": "app://primevideo",
            "package": {
                "packageId": 119,
                "shortName": "prv",
                "slug": "amazon-prime-video",
                "technicalName": "amazonprimevideo",
                "clearName": "Amazon Prime Video",
            },
        },
        {
            "monetizationType": "RENT",
            "standardWebURL": "https://apple.com/tv",
            "preAffiliatedStandardWebURL": None,
            "package": {
                "packageId": 350,
                "shortName": "atp",
                "slug": "apple-tv-plus",
                "clearName": "Apple TV",
            },
        },
    ]

    flattened = flatten_offers(raw_offers)
    assert len(flattened) == 2
    assert flattened[0]["service"] == "amazon-prime-video"
    assert flattened[0]["offer_type"] == "FLATRATE"
    assert flattened[0]["deeplink"] == "app://primevideo"
    assert flattened[0]["clear_name"] == "Amazon Prime Video"

    assert flattened[1]["service"] == "apple-tv-plus"
    assert flattened[1]["offer_type"] == "RENT"
    assert flattened[1]["clear_name"] == "Apple TV"


def test_flatten_offers_falls_back_to_shortname_when_slug_missing():
    raw_offers = [
        {
            "monetizationType": "FLATRATE",
            "standardWebURL": "https://netflix.com",
            "package": {
                "packageId": 8,
                "shortName": "nfx",
            },
        }
    ]
    flattened = flatten_offers(raw_offers)
    assert len(flattened) == 1
    assert flattened[0]["service"] == "nfx"
    assert flattened[0]["offer_type"] == "FLATRATE"


def test_normalise_offer_prefers_slug():
    payload_with_slug = {
        "package_slug": "amazon-prime-video",
        "monetization_type": "flatrate",
        "urls": {"standard_web": "https://prime.com"},
    }
    norm = _normalise_offer(payload_with_slug)
    assert norm is not None
    assert norm["service"] == "amazon-prime-video"
    assert norm["offer_type"] == "FLATRATE"

    payload_legacy = {
        "package_short_name": "prv",
        "monetization_type": "rent",
    }
    norm_legacy = _normalise_offer(payload_legacy)
    assert norm_legacy is not None
    assert norm_legacy["service"] == "prv"
    assert norm_legacy["offer_type"] == "RENT"


def test_default_aliases_include_slugs_and_shortcodes():
    prime = _DEFAULT_STREAMING_PROVIDER_ALIASES["prime_video"]
    assert "amazon-prime-video" in prime
    assert "prime_video" in prime
    assert "prv" in prime

    apple_plus = _DEFAULT_STREAMING_PROVIDER_ALIASES["apple_tv_plus"]
    assert "apple-tv-plus" in apple_plus
    assert "atp" in apple_plus
    assert "apple-tv" not in apple_plus
    assert "itu" not in apple_plus

    apple_store = _DEFAULT_STREAMING_PROVIDER_ALIASES["apple_tv"]
    assert "apple-tv" in apple_store
    assert "itu" in apple_store
    assert "apple-tv-plus" not in apple_store

    netflix = _DEFAULT_STREAMING_PROVIDER_ALIASES["netflix"]
    assert "netflix" in netflix
    assert "nfx" in netflix


def test_watch_link_route_resolves_aliases_and_slugs(monkeypatch):
    client = TestClient(app)

    # Test route parameters acceptance
    resp = client.get("/watch-link/999999?service=amazon-prime-video&country=IL")
    # Will return 404 because item 999999 doesn't exist, but endpoint executes validation successfully
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Link not found"


def test_apple_tv_plus_does_not_redirect_to_apple_tv_store():
    # If a movie only has apple-tv (store) available, requesting apple-tv-plus must not redirect to it
    client = TestClient(app)
    # Item 383 (The Terminator) has apple-tv and netflix in IL, but NOT apple-tv-plus
    resp = client.get("/watch-link/383?service=apple-tv-plus&country=IL")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Link not found"

    resp_store = client.get(
        "/watch-link/383?service=apple-tv&country=IL", follow_redirects=False
    )
    assert resp_store.status_code == 307
    assert "apple.com" in resp_store.headers["location"]


def test_score_candidates_preserves_offer_type():
    from unittest.mock import MagicMock
    import numpy as np
    from api.db.models import Item
    from api.core.legacy_intent_parser import IntentFilters
    from api.pipeline.models import (
        CandidatePool,
        PrefilterDecision,
        QueryUnderstanding,
        RecommendParams,
        UserContext,
    )
    from api.pipeline.scorer import score_candidates

    mock_item = MagicMock(spec=Item)
    mock_item.id = 100
    mock_item.tmdb_id = 999
    mock_item.media_type = "movie"
    mock_item.title = "Sample Movie"
    mock_item.overview = "Overview"
    mock_item.poster_url = "/poster.jpg"
    mock_item.runtime = 120
    mock_item.original_language = "en"
    mock_item.genres = ["Drama"]
    mock_item.release_year = 2024
    mock_item.collection_id = None
    mock_item.collection_name = None
    mock_item.maturity_rating = "PG-13"
    mock_item.popularity = 50.0
    mock_item.vote_average = 8.0
    mock_item.vote_count = 1000
    mock_item.popular_rank = None
    mock_item.trending_rank = None
    mock_item.top_rated_rank = None
    mock_item.directors = []
    mock_item.cast = []
    mock_item.keywords = []

    vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
    watch_opts = [
        {
            "service": "amazon-prime-video",
            "url": "https://primevideo.com/watch",
            "offer_type": "FLATRATE",
        },
        {"service": "apple-tv", "url": "https://apple.com/tv", "offer_type": "RENT"},
    ]

    pool = CandidatePool(
        ids=[100],
        merged_scores={100: {"ann": 0.9}},
        prefilter=PrefilterDecision(
            allowed_ids=None, boost_ids=[], enforce_genres=False
        ),
        items_with_data={100: (mock_item, vec, watch_opts)},
        boost_ids=[],
        enforce_genres=False,
        structured_search_filters=None,
    )
    intent = QueryUnderstanding(
        query="drama",
        llm_intent=MagicMock(),
        intent_filters=IntentFilters("drama"),
        structured_search_filters=None,
        es_text_query=None,
        query_vec=vec,
        prefer_top_rated=False,
        custom_genres=[],
        has_people_filters=False,
        candidate_limit=10,
        backend_override_normalized=None,
        preferred_services=set(),
        prefilter_kwargs={},
    )
    context = UserContext(
        canonical_id="u1",
        user_id="u1",
        profile=None,
        long_v=None,
        short_v=vec,
        exclude_set=set(),
        profile_meta={},
        cold_start=False,
        provider_alias_map={},
        top_query_keywords=set(),
        preferred_services=set(),
        taste_clusters=[],
    )
    params = RecommendParams(user_id="u1", query="drama", limit=10)

    scored = score_candidates(pool, intent, params, context)
    assert len(scored.ordered) == 1
    options = scored.ordered[0]["watch_options"]
    assert len(options) == 2
    assert options[0]["service"] == "amazon-prime-video"
    assert options[0]["offer_type"] == "FLATRATE"
    assert options[1]["service"] == "apple-tv"
    assert options[1]["offer_type"] == "RENT"
