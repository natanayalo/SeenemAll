"""Unit tests for independent diversity policies, serendipity guarantees, and candidate budgeting."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from api.pipeline.diversity import (
    apply_diversity_policies,
    apply_serendipity_slot,
)
from api.pipeline.models import RecommendParams
from api.pipeline.runner import RecommendationPipeline


def test_diversity_policies_independent_controls():
    items = [
        {"id": 1, "collection_id": 100, "original_rank": 0},
        {"id": 2, "collection_id": 100, "original_rank": 1},
        {"id": 3, "collection_id": 100, "original_rank": 2},
        {"id": 4, "collection_id": 200, "original_rank": 3},
    ]

    # Test franchise_cap=False disables cap (all 3 collection 100 items kept)
    out_no_cap = apply_diversity_policies(
        items,
        serendipity_context=[],
        limit=10,
        diversify=True,
        boost_ids=[],
        mmr=False,
        franchise_cap=False,
        serendipity=False,
    )
    assert len(out_no_cap) == 4
    assert [x["id"] for x in out_no_cap] == [1, 2, 3, 4]

    # Test franchise_cap=True enforces cap=2 (only 2 items from collection 100)
    out_with_cap = apply_diversity_policies(
        items,
        serendipity_context=[],
        limit=10,
        diversify=True,
        boost_ids=[],
        mmr=False,
        franchise_cap=True,
        serendipity=False,
    )
    coll_100 = [x for x in out_with_cap if x["collection_id"] == 100]
    assert len(coll_100) == 2


def test_diversity_policies_serendipity_toggle():
    current = [
        {"id": 1, "original_rank": 0},
        {"id": 2, "original_rank": 1},
        {"id": 3, "original_rank": 2},
    ]
    pool = [
        {"id": 99, "original_rank": 10},
    ]

    # Serendipity disabled -> no long tail items inserted
    out_no_serendipity = apply_diversity_policies(
        current,
        serendipity_context=pool,
        limit=3,
        diversify=False,
        boost_ids=[],
        serendipity=False,
        presentation_limit=3,
    )
    ids = [x["id"] for x in out_no_serendipity[:3]]
    assert 99 not in ids

    # Serendipity enabled -> long tail item inserted
    out_with_serendipity = apply_diversity_policies(
        current,
        serendipity_context=pool,
        limit=3,
        diversify=False,
        boost_ids=[],
        serendipity=True,
        presentation_limit=3,
    )
    ids_serendipity = [x["id"] for x in out_with_serendipity[:3]]
    assert 99 in ids_serendipity


def test_recommend_params_budget_and_diversity_defaults():
    params = RecommendParams(user_id="u1")
    assert params.diversify is True
    assert params.mmr is True
    assert params.franchise_cap is True
    assert params.serendipity is True
    assert params.rerank_budget is None


def test_post_rerank_serendipity_guarantee():
    # If reranking placed only short-tail items in top 3, serendipity guarantees long-tail inclusion
    reranked = [
        {"id": 1, "original_rank": 0},
        {"id": 2, "original_rank": 1},
        {"id": 3, "original_rank": 2},
        {"id": 4, "original_rank": 3},
    ]
    serendipity_context = [
        {"id": 99, "original_rank": 15},
    ]

    out = apply_serendipity_slot(reranked, serendipity_context, limit=3)
    top_3_ids = [x["id"] for x in out[:3]]
    assert 99 in top_3_ids


def test_pipeline_candidate_budget_decoupling_at_limit_10():
    """Verify pipeline provides 25 candidates to reranker when limit=10 and rerank=True."""
    # Create 30 items
    items = [{"id": i, "title": f"Item {i}", "original_rank": i} for i in range(1, 31)]
    context = SimpleNamespace(profile_meta={}, cold_start=True)
    intent = SimpleNamespace(
        prefilter_kwargs={},
        prefer_top_rated=False,
        intent_filters=None,
        matched_collection_ids=(),
        collection_item_ids=(),
        is_chronological_requested=False,
    )
    pool = SimpleNamespace(
        ids=[i for i in range(1, 31)],
        prefilter=SimpleNamespace(allowed_ids=[]),
        boost_ids=[],
    )
    scored = SimpleNamespace(ordered=list(items), serendipity_context=list(items))

    observed_rerank_candidates = []

    def mock_rerank(candidates, *args, **kwargs):
        observed_rerank_candidates.extend(candidates)
        return list(candidates)

    with patch("api.pipeline.runner.load_user_context", return_value=context), patch(
        "api.pipeline.runner.resolve_query_intent", new=AsyncMock(return_value=intent)
    ), patch("api.pipeline.runner.retrieve_candidates", return_value=pool), patch(
        "api.pipeline.runner.score_candidates", return_value=scored
    ), patch(
        "api.pipeline.runner.rerank_candidates", side_effect=mock_rerank
    ):
        pipeline = RecommendationPipeline()
        result = asyncio.run(
            pipeline.run(
                MagicMock(),
                RecommendParams(query="test", limit=10, rerank=True, diversify=True),
                MagicMock(),
            )
        )
        # Reranker receives 25 candidates (decoupled budget), not 10!
        assert len(observed_rerank_candidates) == 25
        # Pipeline returns evaluated candidates with top 10 preserved
        assert len(result.items[:10]) == 10


def test_pipeline_franchise_cap_preservation_with_serendipity():
    """Verify post-rerank serendipity does NOT cause a capped franchise to grow from 2 to 3 items."""
    # Candidates where franchise 42 has 2 items in top-10, and candidate pool has 3rd item from franchise 42
    top_items = [
        {"id": 1, "collection_id": 42, "original_rank": 0},
        {"id": 2, "collection_id": 42, "original_rank": 1},
        {"id": 3, "collection_id": 100, "original_rank": 2},
        {"id": 4, "collection_id": 101, "original_rank": 3},
    ]
    # Pool has a long-tail item that also belongs to collection 42
    pool = [
        {"id": 99, "collection_id": 42, "original_rank": 20},
        {"id": 88, "collection_id": 200, "original_rank": 25},
    ]

    context = SimpleNamespace(profile_meta={}, cold_start=True)
    intent = SimpleNamespace(
        prefilter_kwargs={},
        prefer_top_rated=False,
        intent_filters=None,
        matched_collection_ids=(),
        collection_item_ids=(),
        is_chronological_requested=False,
    )
    cand_pool = SimpleNamespace(
        ids=[1, 2, 3, 4],
        prefilter=SimpleNamespace(allowed_ids=[]),
        boost_ids=[],
    )
    scored = SimpleNamespace(ordered=list(top_items), serendipity_context=list(pool))

    with patch("api.pipeline.runner.load_user_context", return_value=context), patch(
        "api.pipeline.runner.resolve_query_intent", new=AsyncMock(return_value=intent)
    ), patch("api.pipeline.runner.retrieve_candidates", return_value=cand_pool), patch(
        "api.pipeline.runner.score_candidates", return_value=scored
    ), patch(
        "api.pipeline.runner.rerank_candidates", return_value=list(top_items)
    ):
        pipeline = RecommendationPipeline()
        result = asyncio.run(
            pipeline.run(
                MagicMock(),
                RecommendParams(
                    query="test",
                    limit=4,
                    rerank=True,
                    diversify=True,
                    franchise_cap=True,
                    serendipity=True,
                ),
                MagicMock(),
            )
        )
        # Collection 42 must NOT exceed 2 items!
        coll_42_items = [it for it in result.items[:4] if it.get("collection_id") == 42]
        assert len(coll_42_items) <= 2
        # Serendipity can pick the long-tail item from collection 200, but not collection 42
        assert 99 not in [it["id"] for it in result.items[:4]]
        assert 88 in [it["id"] for it in result.items[:4]]


def test_pipeline_chronological_preservation_with_serendipity():
    """Verify post-rerank serendipity does NOT corrupt chronological sequences like [1970, 1980, 1990] into [1970, 1980, 1960]."""
    chronological_franchise_items = [
        {"id": 1, "collection_id": 10, "release_year": 1970, "original_rank": 0},
        {"id": 2, "collection_id": 10, "release_year": 1980, "original_rank": 1},
        {"id": 3, "collection_id": 10, "release_year": 1990, "original_rank": 2},
    ]
    # Pool has a long-tail item with release_year=1960
    pool = [
        {"id": 99, "collection_id": None, "release_year": 1960, "original_rank": 20},
    ]

    context = SimpleNamespace(profile_meta={}, cold_start=True)
    intent = SimpleNamespace(
        prefilter_kwargs={},
        prefer_top_rated=False,
        intent_filters=None,
        matched_collection_ids=(10,),
        collection_item_ids=(1, 2, 3),
        is_chronological_requested=True,
    )
    cand_pool = SimpleNamespace(
        ids=[1, 2, 3],
        prefilter=SimpleNamespace(allowed_ids=[]),
        boost_ids=[],
    )
    scored = SimpleNamespace(
        ordered=list(chronological_franchise_items),
        serendipity_context=list(pool),
    )

    with patch("api.pipeline.runner.load_user_context", return_value=context), patch(
        "api.pipeline.runner.resolve_query_intent", new=AsyncMock(return_value=intent)
    ), patch("api.pipeline.runner.retrieve_candidates", return_value=cand_pool), patch(
        "api.pipeline.runner.score_candidates", return_value=scored
    ), patch(
        "api.pipeline.runner.rerank_candidates",
        return_value=list(chronological_franchise_items),
    ):
        pipeline = RecommendationPipeline()
        result = asyncio.run(
            pipeline.run(
                MagicMock(),
                RecommendParams(
                    query="franchise in order",
                    limit=3,
                    rerank=True,
                    serendipity=True,
                ),
                MagicMock(),
            )
        )
        years = [it["release_year"] for it in result.items[:3]]
        # Strict chronological order [1970, 1980, 1990] is preserved!
        assert years == [1970, 1980, 1990]
        assert 99 not in [it["id"] for it in result.items[:3]]


def test_cursor_pagination_with_decoupled_budget():
    """Verify cursor pagination cleanly slices the 25-item candidate pool into limit=10 pages."""
    from api.pipeline.reranker import decode_cursor
    from api.routes.recommend import recommend

    # 25 candidates produced by decoupled budget
    items = [
        {
            "id": i,
            "tmdb_id": i,
            "title": f"Film {i}",
            "overview": "Overview",
            "release_year": 2000 + i,
            "genres": ["Action"],
            "explanation": "Because action",
        }
        for i in range(1, 26)
    ]

    async def mock_compute(req, p, db):
        return SimpleNamespace(items=list(items), debug_context={})

    with patch(
        "api.routes.recommend.get_or_compute_recommendations",
        new=AsyncMock(return_value=(list(items), {})),
    ):
        mock_req = MagicMock()
        # Page 1: limit=10, cursor=None
        p1 = asyncio.run(
            recommend(
                mock_req,
                RecommendParams(user_id="u1", limit=10),
                cursor=None,
                db=MagicMock(),
            )
        )
        assert len(p1["items"]) == 10
        assert p1["items"][0]["id"] == 1
        assert p1["items"][9]["id"] == 10
        assert "next_cursor" in p1
        cursor_1 = p1["next_cursor"]
        assert decode_cursor(cursor_1) == 10

        # Page 2: limit=10, cursor=cursor_1
        p2 = asyncio.run(
            recommend(
                mock_req,
                RecommendParams(user_id="u1", limit=10),
                cursor=cursor_1,
                db=MagicMock(),
            )
        )
        assert len(p2["items"]) == 10
        assert p2["items"][0]["id"] == 11
        assert p2["items"][9]["id"] == 20
        assert "next_cursor" in p2
        cursor_2 = p2["next_cursor"]
        assert decode_cursor(cursor_2) == 20

        # Page 3: limit=10, cursor=cursor_2
        p3 = asyncio.run(
            recommend(
                mock_req,
                RecommendParams(user_id="u1", limit=10),
                cursor=cursor_2,
                db=MagicMock(),
            )
        )
        assert len(p3["items"]) == 5
        assert p3["items"][0]["id"] == 21
        assert p3["items"][4]["id"] == 25
        assert "next_cursor" not in p3


def test_franchise_collection_media_type_default_movie():
    """Verify that when a franchise/collection is matched, media_type defaults to movie unless TV is requested."""
    from api.pipeline.intent.builder import resolve_query_intent
    from api.core.llm_parser import Intent

    from api.pipeline.models import UserContext

    mock_req = MagicMock()
    mock_req.app.state.entity_linker = None
    mock_db = MagicMock()
    user_context = UserContext(
        canonical_id="u1",
        user_id="u1",
        profile=None,
        long_v=None,
        short_v=None,
        exclude_set=set(),
        profile_meta={},
        cold_start=True,
        provider_alias_map={},
        top_query_keywords=set(),
    )

    # 1. Without TV keywords: media_type defaults to movie
    llm_intent = Intent(
        franchises=["Spider-Man: Spider-Verse Collection"],
        media_types=[],
        include_genres=["Animation"],
    )
    mock_matcher = MagicMock()
    mock_matcher.resolve_collections.return_value = [
        (573436, "Spider-Man: Spider-Verse Collection")
    ]
    mock_matcher.resolve_person.return_value = (None, [])

    with patch(
        "api.routes.recommend._parse_llm_intent",
        create=True,
        side_effect=lambda *a, **kw: llm_intent,
    ), patch(
        "api.routes.recommend.get_query_filters",
        create=True,
        return_value=SimpleNamespace(
            languages=[],
            keywords=[],
            media_types=[],
            matched_collections=[],
            genres=[],
            cast=[],
            directors=[],
            producers=[],
            writers=[],
            reference_titles=[],
        ),
    ), patch(
        "api.routes.recommend.get_filter_matcher",
        create=True,
        return_value=mock_matcher,
    ), patch(
        "api.pipeline.intent.builder.build_query_vector", return_value=[]
    ):
        mock_db.execute.return_value.all.return_value = [(1, 2018), (2, 2023)]

        params = RecommendParams(query="Spidr-man multiverse animated", limit=10)
        understanding = asyncio.run(
            resolve_query_intent(mock_req, params, user_context, mock_db)
        )

        assert understanding.structured_search_filters is not None
        assert "movie" in understanding.structured_search_filters.media_types
        assert "tv" not in understanding.structured_search_filters.media_types
        assert understanding.intent_filters.media_types == ["movie"]

    # 2. With explicit TV keyword: media_type does NOT default to movie
    llm_intent_tv = Intent(
        franchises=["Spider-Man: Spider-Verse Collection"],
        media_types=["tv"],
        include_genres=["Animation"],
    )
    with patch(
        "api.routes.recommend._parse_llm_intent",
        create=True,
        side_effect=lambda *a, **kw: llm_intent_tv,
    ), patch(
        "api.routes.recommend.get_query_filters",
        create=True,
        return_value=SimpleNamespace(
            languages=[],
            keywords=[],
            media_types=[],
            matched_collections=[],
            genres=[],
            cast=[],
            directors=[],
            producers=[],
            writers=[],
            reference_titles=[],
        ),
    ), patch(
        "api.routes.recommend.get_filter_matcher",
        create=True,
        return_value=mock_matcher,
    ), patch(
        "api.pipeline.intent.builder.build_query_vector", return_value=[]
    ):
        mock_db.execute.return_value.all.return_value = [(1, 2018), (2, 2023)]

        params = RecommendParams(query="Spider-Man animated tv show", limit=10)
        understanding = asyncio.run(
            resolve_query_intent(mock_req, params, user_context, mock_db)
        )

        assert understanding.structured_search_filters is not None
        assert "tv" in understanding.structured_search_filters.media_types


def test_serendipity_slot_chronological_and_franchise_cap():
    # Test that serendipity respects chronology:
    current = [
        {"id": 1, "release_year": 1980, "original_rank": 0, "collection_id": 10},
        {"id": 2, "release_year": 1990, "original_rank": 1, "collection_id": 10},
        {"id": 3, "release_year": 2000, "original_rank": 2, "collection_id": 20},
    ]
    # Replacement candidate has year 1970 (would break chronology if replacing idx 1 (between 1980 and 2000))
    # Candidate 2 has year 1995 (valid between 1980 and 2000)
    pool = [
        {"id": 98, "release_year": 1970, "original_rank": 10, "collection_id": 30},
        {"id": 99, "release_year": 1995, "original_rank": 11, "collection_id": 30},
    ]

    res = apply_serendipity_slot(
        current,
        candidate_pool=pool,
        limit=3,
        cap=2,
        enforce_franchise_cap=True,
        is_chronological=True,
    )
    # The valid replacement 99 (1995) should be used, not 98 (1970)
    res_ids = [x["id"] for x in res[:3]]
    assert 99 in res_ids
    assert 98 not in res_ids
    # Verify entire result does not violate franchise cap
    coll_counts = {}
    for item in res:
        cid = item.get("collection_id")
        if cid:
            coll_counts[cid] = coll_counts.get(cid, 0) + 1
            assert coll_counts[cid] <= 2


def test_diversity_policies_boost_ids_with_mmr():
    items = [
        {"id": 1, "score": 0.9, "original_rank": 0},
        {"id": 2, "score": 0.8, "original_rank": 1},
        {"id": 3, "score": 0.7, "original_rank": 2},
        {"id": 4, "score": 0.6, "original_rank": 3},
    ]
    # item 4 is boosted
    out = apply_diversity_policies(
        items,
        serendipity_context=[],
        limit=3,
        diversify=True,
        boost_ids=[4],
        mmr=True,
        franchise_cap=True,
        serendipity=False,
    )
    assert out[0]["id"] == 4
    assert len(out) == 3


def test_runner_serendipity_runs_once_when_rerank_false():
    runner = RecommendationPipeline()
    req = MagicMock()
    params = RecommendParams(query="test", limit=5, rerank=False, serendipity=True)
    collector = MagicMock()

    mock_pool = MagicMock()
    mock_pool.ids = [1, 2, 3]
    mock_pool.boost_ids = []
    mock_intent = MagicMock()
    mock_intent.intent_filters = MagicMock()
    mock_intent.matched_collection_ids = ()
    mock_intent.collection_item_ids = ()
    mock_intent.is_chronological_requested = False

    items = [{"id": i, "score": 1.0 - i * 0.1, "original_rank": i} for i in range(10)]
    mock_scored = SimpleNamespace(
        ordered=items, serendipity_context=[{"id": 99, "original_rank": 20}]
    )

    serendipity_call_count = 0

    def mock_serendipity_slot(curr, pool, limit, **kw):
        nonlocal serendipity_call_count
        serendipity_call_count += 1
        return curr

    with patch(
        "api.pipeline.runner.load_user_context", return_value=MagicMock()
    ), patch(
        "api.pipeline.runner.resolve_query_intent", return_value=mock_intent
    ), patch(
        "api.pipeline.runner.retrieve_candidates", return_value=mock_pool
    ), patch(
        "api.pipeline.runner.score_candidates", return_value=mock_scored
    ), patch(
        "api.pipeline.diversity.apply_serendipity_slot",
        side_effect=mock_serendipity_slot,
    ), patch(
        "api.routes.recommend._apply_serendipity_slot",
        create=True,
        side_effect=mock_serendipity_slot,
    ):
        res = asyncio.run(runner._run(req, params, MagicMock(), collector))
        assert len(res.items) > 0
        assert serendipity_call_count == 1


def test_serendipity_replacements_have_explanations_when_rerank_false():
    runner = RecommendationPipeline()
    req = MagicMock()
    params = RecommendParams(
        query="space action", limit=10, rerank=False, serendipity=True
    )
    collector = MagicMock()

    mock_pool = MagicMock()
    mock_pool.ids = list(range(20))
    mock_pool.boost_ids = []
    mock_intent = MagicMock()
    mock_intent.intent_filters = MagicMock()
    mock_intent.intent_filters.effective_genres.return_value = ["Action"]
    mock_intent.matched_collection_ids = ()
    mock_intent.collection_item_ids = ()
    mock_intent.is_chronological_requested = False

    items = [
        {
            "id": i,
            "title": f"Item {i}",
            "genres": ["Action"],
            "score": 1.0 - i * 0.05,
            "original_rank": i,
        }
        for i in range(10)
    ]
    # Serendipity context items lack explanation
    serendipity_items = [
        {
            "id": 99,
            "title": "Long Tail Star",
            "genres": ["Action"],
            "popularity": 1.0,
            "vote_count": 10,
            "original_rank": 50,
        }
    ]
    mock_scored = SimpleNamespace(ordered=items, serendipity_context=serendipity_items)

    with patch(
        "api.pipeline.runner.load_user_context", return_value=MagicMock()
    ), patch(
        "api.pipeline.runner.resolve_query_intent", return_value=mock_intent
    ), patch(
        "api.pipeline.runner.retrieve_candidates", return_value=mock_pool
    ), patch(
        "api.pipeline.runner.score_candidates", return_value=mock_scored
    ):
        res = asyncio.run(runner._run(req, params, MagicMock(), collector))
        assert len(res.items) >= 10
        for it in res.items[: params.limit]:
            assert "explanation" in it
            assert it["explanation"] is not None
            assert len(it["explanation"]) > 0


def test_runner_preserves_depth_when_rerank_budget_less_than_limit():
    runner = RecommendationPipeline()
    req = MagicMock()
    params = RecommendParams(
        query="test", limit=100, rerank_budget=10, rerank=False, serendipity=False
    )
    collector = MagicMock()

    mock_pool = MagicMock()
    mock_pool.ids = list(range(100))
    mock_pool.boost_ids = []
    mock_intent = MagicMock()
    mock_intent.intent_filters = MagicMock()
    mock_intent.intent_filters.effective_genres.return_value = []
    mock_intent.matched_collection_ids = ()
    mock_intent.collection_item_ids = ()
    mock_intent.is_chronological_requested = False

    items = [
        {"id": i, "title": f"Item {i}", "score": 1.0 - i * 0.005, "original_rank": i}
        for i in range(100)
    ]
    mock_scored = SimpleNamespace(ordered=items, serendipity_context=[])

    with patch(
        "api.pipeline.runner.load_user_context", return_value=MagicMock()
    ), patch(
        "api.pipeline.runner.resolve_query_intent", return_value=mock_intent
    ), patch(
        "api.pipeline.runner.retrieve_candidates", return_value=mock_pool
    ), patch(
        "api.pipeline.runner.score_candidates", return_value=mock_scored
    ):
        res = asyncio.run(runner._run(req, params, MagicMock(), collector))
        assert len(res.items) == 100
        for it in res.items:
            assert "explanation" in it
            assert it["explanation"] is not None
