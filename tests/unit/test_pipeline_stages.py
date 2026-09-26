from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest
from starlette.requests import Request

from api.routes import recommend as recommend_routes
from api.pipeline.context import (
    clear_recommend_cache_for_tests,
    clear_user_cache,
    get_cache_key,
    load_user_context,
)
from api.pipeline.diversity import (
    apply_diversity_policies,
    apply_franchise_cap,
    is_long_tail,
    serendipity_target,
)
from api.pipeline.intent import (
    append_weighted_text,
    build_rewrite_vector,
    intent_filters_from_llm,
    matches_keywords,
    merge_maturity_rating,
    parse_llm_intent,
    resolve_query_intent,
)
from api.pipeline.models import (
    ComputeResult,
    PrefilterDecision,
    RecommendParams,
    UserContext,
)
from api.pipeline.reranker import (
    decode_cursor,
    encode_cursor,
    format_presentation_items,
)
from api.pipeline.retriever import (
    ANNRetriever,
    BaseRetriever,
    ColdStartRetriever,
    CollaborativeGraphRetriever,
    TrendingPriorRetriever,
    filter_excluded_candidate_ids,
    ordered_unique,
    relax_filters_for_people,
)
from api.pipeline.runner import get_pipeline
from api.pipeline.scorer import (
    apply_mixer_scores,
    prioritize_boosted_items,
)


@pytest.fixture(autouse=True)
def _reset_caches():
    clear_recommend_cache_for_tests()
    yield
    clear_recommend_cache_for_tests()


def test_models_contracts():
    params = RecommendParams(user_id="u_test", limit=10)
    assert params.user_id == "u_test"
    assert params.limit == 10
    assert params.profile is None

    res = ComputeResult(items=[{"id": 1}], debug_context={"k": "v"})
    assert len(res.items) == 1
    assert res.debug_context["k"] == "v"

    decision = PrefilterDecision(allowed_ids=[1, 2], boost_ids=[1], enforce_genres=True)
    assert decision.allowed_ids == [1, 2]
    assert decision.enforce_genres is True


def test_context_load_and_caching(monkeypatch):
    mock_db = MagicMock()
    mock_state = (
        np.ones(10, dtype=np.float32),
        np.zeros(10, dtype=np.float32),
        [5],
        {"genre_prefs": {"Action": 1.0}},
    )
    monkeypatch.setattr(recommend_routes, "load_user_state", lambda db, uid: mock_state)
    monkeypatch.setattr("api.pipeline.context.get_streaming_alias_map", lambda db: {"nfx": {"netflix"}})
    monkeypatch.setattr("api.pipeline.context.get_top_query_keywords", lambda db: {"epic"})

    ctx = load_user_context(mock_db, "user_1", "profile_a")
    assert ctx.canonical_id == "user_1::profile_a"
    assert ctx.cold_start is False
    assert 5 in ctx.exclude_set
    assert ctx.provider_alias_map == {"nfx": {"netflix"}}

    params = RecommendParams(user_id="user_1", limit=10)
    key = get_cache_key("user_1::profile_a", params)
    assert "user_1::profile_a:" in key

    clear_user_cache("user_1::profile_a")


def test_intent_helpers(monkeypatch):
    monkeypatch.setattr(
        recommend_routes,
        "encode_texts",
        lambda texts: np.ones((len(texts), 384), dtype=np.float32),
    )
    assert matches_keywords("This is an epic movie", {"epic"}) is True
    assert matches_keywords("Boring film", {"epic"}) is False

    texts = []
    weights = []
    w = append_weighted_text("hello", None, 1.5, texts, weights)
    assert w == 1.5
    assert texts == ["hello"]
    assert weights == [1.5]

    w_zero = append_weighted_text("", None, 1.5, texts, weights)
    assert w_zero == 0.0

    vec = build_rewrite_vector("space adventure", "sci-fi battle", 1.0, 1.0)
    assert vec is not None
    assert np.isclose(np.linalg.norm(vec), 1.0)


@pytest.mark.asyncio
async def test_resolve_query_intent(monkeypatch):
    mock_request = MagicMock(spec=Request)
    mock_request.app.state.entity_linker = None
    mock_db = MagicMock()
    mock_db.execute.return_value.all.return_value = []
    monkeypatch.setattr(
        recommend_routes,
        "get_query_filters",
        lambda q: MagicMock(
            languages=(),
            keywords=(),
            genres=(),
            media_types=(),
            cast=(),
            directors=(),
            producers=(),
            writers=(),
            residual_text=q or "",
            reference_titles=(),
        ),
    )
    monkeypatch.setattr(
        recommend_routes,
        "encode_texts",
        lambda texts: np.ones((len(texts), 384), dtype=np.float32),
    )

    ctx = UserContext(
        canonical_id="u1",
        user_id="u1",
        profile=None,
        long_v=None,
        short_v=None,
        exclude_set=set(),
        profile_meta={},
        cold_start=True,
        provider_alias_map={},
        top_query_keywords={"best"},
    )
    params = RecommendParams(user_id="u1", query="best sci-fi", limit=10, classic_top_rated=True)
    intent = await resolve_query_intent(mock_request, params, ctx, mock_db)
    assert intent.query == "best sci-fi"
    assert intent.prefer_top_rated is True
    assert intent.candidate_limit >= 10


def test_retriever_helpers():
    assert ordered_unique([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]
    assert filter_excluded_candidate_ids([1, 2, 3, 4], {2, 4}) == [1, 3]


def test_scorer_and_mixer():
    candidates = [
        {"id": 1, "popularity": 10.0, "vote_count": 100, "ann_rank": 0, "source_scores": {}},
        {"id": 2, "popularity": 50.0, "vote_count": 500, "ann_rank": 1, "source_scores": {}},
    ]
    apply_mixer_scores(candidates)
    assert "retrieval_score" in candidates[0]
    assert candidates[0]["original_rank"] == 0
    assert candidates[1]["original_rank"] == 1

    reordered = prioritize_boosted_items(candidates, [2])
    assert reordered[0]["id"] == 2


def test_diversity_and_presentation():
    items = [
        {"id": 1, "collection_id": 100},
        {"id": 2, "collection_id": 100},
        {"id": 3, "collection_id": 100},
        {"id": 4, "collection_id": 200},
    ]
    capped = apply_franchise_cap(items, cap=2)
    assert len(capped) == 3
    assert [i["id"] for i in capped] == [1, 2, 4]

    assert serendipity_target(10) >= 1
    assert serendipity_target(2) == 0
    assert is_long_tail({"original_rank": 15}, limit=10) is True
    assert is_long_tail({"original_rank": 2}, limit=10) is False

    diversified = apply_diversity_policies(items, items, limit=4, diversify=True, boost_ids=[])
    assert len(diversified) <= 4

    cursor = encode_cursor(20)
    assert decode_cursor(cursor) == 20
    assert decode_cursor(None) == 0

    formatted = format_presentation_items([{"id": 1, "vector": np.zeros(5), "original_rank": 0}])
    assert "vector" not in formatted[0]
    assert "original_rank" not in formatted[0]


@pytest.mark.asyncio
async def test_pipeline_runner(monkeypatch):
    pipeline = get_pipeline()
    mock_request = MagicMock(spec=Request)
    mock_request.app.state.entity_linker = None
    mock_db = MagicMock()
    mock_db.execute.return_value.all.return_value = []
    mock_db.execute.return_value.scalars.return_value.all.return_value = []

    monkeypatch.setattr(
        recommend_routes,
        "ann_candidates",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        recommend_routes,
        "_cold_start_candidates",
        lambda db, intent, limit, allowlist, prefer_top_rated=False: [],
    )

    params = RecommendParams(user_id="u_cold", limit=5)
    res = await pipeline.run(mock_request, params, mock_db)
    assert isinstance(res, ComputeResult)
    assert res.items == []


def test_retriever_classes_adapter_interface(monkeypatch):
    mock_db = MagicMock()
    ctx = UserContext(
        canonical_id="u1",
        user_id="u1",
        profile=None,
        long_v=None,
        short_v=np.ones(384, dtype=np.float32),
        exclude_set={99},
        profile_meta={"neighbors": [{"user_id": "u2", "weight": 1.0}]},
        cold_start=False,
        provider_alias_map={},
        top_query_keywords=set(),
    )
    from api.pipeline.models import QueryUnderstanding
    from api.core.legacy_intent_parser import IntentFilters
    intent = QueryUnderstanding(
        query=None,
        llm_intent=MagicMock(),
        intent_filters=IntentFilters(""),
        structured_search_filters=None,
        es_text_query=None,
        rewrite_vec=None,
        prefer_top_rated=False,
        custom_genres=[],
        has_people_filters=False,
        candidate_limit=10,
        backend_override_normalized=None,
        preferred_services=set(),
        prefilter_kwargs={},
    )

    # BaseRetriever subclassing check
    assert issubclass(ColdStartRetriever, BaseRetriever)
    assert issubclass(CollaborativeGraphRetriever, BaseRetriever)
    assert issubclass(TrendingPriorRetriever, BaseRetriever)
    assert issubclass(ANNRetriever, BaseRetriever)

    # ColdStartRetriever
    monkeypatch.setattr(recommend_routes, "_cold_start_candidates", lambda db, it, lim, al, **kw: [1, 2, 3])
    cold_retriever = ColdStartRetriever()
    assert cold_retriever.retrieve(mock_db, ctx, intent, [1, 2]) == [1, 2, 3]

    # CollaborativeGraphRetriever
    monkeypatch.setattr(recommend_routes, "_collaborative_candidates", lambda db, neigh, exc, lim, allowed_ids=None, **kw: [(10, 0.9)])
    collab_retriever = CollaborativeGraphRetriever()
    assert collab_retriever.retrieve(mock_db, ctx, intent, None) == [(10, 0.9)]

    # TrendingPriorRetriever
    monkeypatch.setattr(recommend_routes, "_trending_prior_candidates", lambda db, it, exc, lim, allowed_ids=None, **kw: [(20, 0.8)])
    trending_retriever = TrendingPriorRetriever()
    assert trending_retriever.retrieve(mock_db, ctx, intent, None) == [(20, 0.8)]

    # ANNRetriever
    monkeypatch.setattr(recommend_routes, "ann_candidates", lambda *args, **kw: [100, 200])
    ann_retriever = ANNRetriever()
    ids, rewrite_used = ann_retriever.retrieve(mock_db, ctx, intent, None)
    assert ids == [100, 200]
    assert rewrite_used is False


def test_intent_parser_and_filters():
    assert parse_llm_intent(None, {}) is not None

    mock_intent = MagicMock()
    mock_intent.include_genres = ["action", "thriller"]
    mock_intent.runtime_minutes_min = 60
    mock_intent.runtime_minutes_max = 120
    mock_intent.maturity_rating_max = "PG-13"

    filters = intent_filters_from_llm("thriller query", mock_intent)
    assert filters.raw_query == "thriller query"
    assert "Action" in filters.genres or "action" in [g.lower() for g in filters.genres]
    assert filters.min_runtime == 60
    assert filters.max_runtime == 120
    assert filters.maturity_rating_max == "PG-13"

    # Maturity merging
    primary = MagicMock(maturity_rating_max=None)
    fallback = MagicMock(maturity_rating_max="R")
    merge_maturity_rating(primary, fallback)
    assert primary.maturity_rating_max == "R"


def test_prefilter_relaxation_and_helpers():
    from api.core.elasticsearch_search import SearchFilters
    filters = SearchFilters(
        include_item_ids=(1, 2),
        genres=("Action",),
        keywords=("heist",),
        cast=("Tom Cruise",),
    )
    relaxed = relax_filters_for_people(filters)
    assert relaxed is not None
    assert relaxed.cast == ("Tom Cruise",)
    assert relaxed.genres == ()  # genres relaxed

