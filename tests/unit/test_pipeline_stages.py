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
    build_query_vector,
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
    monkeypatch.setattr(
        "api.pipeline.context.get_streaming_alias_map", lambda db: {"nfx": {"netflix"}}
    )
    monkeypatch.setattr(
        "api.pipeline.context.get_top_query_keywords", lambda db: {"epic"}
    )

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

    vec = build_query_vector("space adventure")
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
    params = RecommendParams(
        user_id="u1", query="best sci-fi", limit=10, classic_top_rated=True
    )
    intent = await resolve_query_intent(mock_request, params, ctx, mock_db)
    assert intent.query == "best sci-fi"
    assert intent.prefer_top_rated is True
    assert intent.candidate_limit >= 10


@pytest.mark.asyncio
async def test_resolve_query_intent_with_roles(monkeypatch):
    from api.core.intent_parser import Intent

    mock_request = MagicMock(spec=Request)
    mock_request.app.state.entity_linker = None
    mock_db = MagicMock()
    mock_db.execute.return_value.all.return_value = []

    mock_llm_intent = Intent(
        include_directors=["Christopher Nolan"],
        include_actors=["Keanu Reeves"],
        include_producers=["Steven Spielberg"],
        include_writers=["Quentin Tarantino"],
        include_people=["Tom Cruise"],
        reference_titles=["Arrival"],
        franchises=["Star Wars"],
        languages=["fr"],
    )

    monkeypatch.setattr(
        recommend_routes,
        "_parse_llm_intent",
        lambda query, user_context, linked_entities: mock_llm_intent,
    )
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
        top_query_keywords=set(),
    )
    params = RecommendParams(
        user_id="u1",
        query="film by Nolan with Keanu",
        limit=10,
        use_llm_intent=True,
    )
    intent = await resolve_query_intent(mock_request, params, ctx, mock_db)
    assert intent.structured_search_filters is not None
    assert intent.structured_search_filters.directors == ("Christopher Nolan",)
    assert intent.structured_search_filters.cast == ("Keanu Reeves", "Tom Cruise")
    assert intent.structured_search_filters.producers == ("Steven Spielberg",)
    assert intent.structured_search_filters.writers == ("Quentin Tarantino",)
    assert "fr" in intent.structured_search_filters.languages
    assert "Arrival" in intent.intent_filters.reference_titles
    assert "Star Wars" in intent.intent_filters.reference_titles
    assert "Science Fiction" in intent.structured_search_filters.genres
    assert "Adventure" in intent.structured_search_filters.genres


@pytest.mark.asyncio
async def test_resolve_query_intent_chronological_detection(monkeypatch):
    mock_request = MagicMock(spec=Request)
    mock_request.app.state.entity_linker = None
    mock_db = MagicMock()
    mock_db.execute.return_value.all.return_value = [
        (100, 1977),
        (200, 2015),
    ]

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
            matched_collections=((10, "Star Wars Collection"),),
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
        top_query_keywords=set(),
    )

    # 1. Queries with explicit strong chronological cues
    for q in (
        "Star Wars chronological",
        "Star Wars in order",
        "release order Star Wars",
    ):
        params = RecommendParams(user_id="u1", query=q, limit=10)
        intent = await resolve_query_intent(mock_request, params, ctx, mock_db)
        assert intent.is_chronological_requested is True, f"Failed for {q}"

    # 2. Sequence cues without ranking intent (e.g. browsing a trilogy / saga)
    for q in ("The Lord of the Rings trilogy", "The Godfather saga"):
        params = RecommendParams(user_id="u1", query=q, limit=10)
        intent = await resolve_query_intent(mock_request, params, ctx, mock_db)
        assert intent.is_chronological_requested is True, f"Failed for {q}"

    # 3. Sequence cues WITH explicit ranking intent (ranking intent MUST override chronological)
    for q in (
        "best star wars movies",
        "best Mission Impossible series",
        "best films from the Daniel Craig era",
        "top Batman trilogy movies",
    ):
        params = RecommendParams(user_id="u1", query=q, limit=10)
        intent = await resolve_query_intent(mock_request, params, ctx, mock_db)
        assert intent.is_chronological_requested is False, f"Failed for {q}"

    # 4. Phrases containing 'order' without chronological intent (avoid false positives)
    params_hp = RecommendParams(
        user_id="u1", query="Harry Potter and the Order of the Phoenix", limit=10
    )
    intent_hp = await resolve_query_intent(mock_request, params_hp, ctx, mock_db)
    assert intent_hp.is_chronological_requested is False


def test_retriever_helpers():
    assert ordered_unique([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]
    assert filter_excluded_candidate_ids([1, 2, 3, 4], {2, 4}) == [1, 3]


def test_scorer_and_mixer():
    candidates = [
        {
            "id": 1,
            "popularity": 10.0,
            "vote_count": 100,
            "ann_rank": 0,
            "source_scores": {},
        },
        {
            "id": 2,
            "popularity": 50.0,
            "vote_count": 500,
            "ann_rank": 1,
            "source_scores": {},
        },
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

    exempt_capped = apply_franchise_cap(items, cap=2, exempt_collection_ids={100})
    assert len(exempt_capped) == 4
    assert [i["id"] for i in exempt_capped] == [1, 2, 3, 4]

    assert serendipity_target(10) >= 1
    assert serendipity_target(2) == 0
    assert is_long_tail({"original_rank": 15}, limit=10) is True
    assert is_long_tail({"original_rank": 2}, limit=10) is False

    diversified = apply_diversity_policies(
        items, items, limit=4, diversify=True, boost_ids=[]
    )
    assert len(diversified) <= 4

    cursor = encode_cursor(20)
    assert decode_cursor(cursor) == 20
    assert decode_cursor(None) == 0

    formatted = format_presentation_items(
        [{"id": 1, "vector": np.zeros(5), "original_rank": 0}]
    )
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


@pytest.mark.asyncio
async def test_pipeline_runner_with_rerank_params(monkeypatch):
    pipeline = get_pipeline()
    mock_request = MagicMock(spec=Request)
    mock_request.app.state.entity_linker = None
    mock_db = MagicMock()
    mock_db.execute.return_value.all.return_value = []
    mock_db.execute.return_value.scalars.return_value.all.return_value = []

    captured = {}

    def fake_rerank(
        ordered, intent, query, context, *, rerank=None, rerank_provider=None
    ):
        captured["rerank"] = rerank
        captured["rerank_provider"] = rerank_provider
        return [{"id": 1, "title": "Reranked"}]

    monkeypatch.setattr("api.pipeline.runner.rerank_candidates", fake_rerank)
    monkeypatch.setattr(recommend_routes, "ann_candidates", lambda *args, **kwargs: [1])
    mock_item = MagicMock()
    mock_item.id = 1
    mock_item.tmdb_id = 101
    mock_item.media_type = "movie"
    mock_item.title = "Item 1"
    mock_item.overview = "Overview"
    mock_item.genres = []
    mock_item.release_year = 2024
    mock_item.runtime = 100
    mock_item.original_language = "en"
    mock_item.collection_id = None
    mock_item.collection_name = None
    mock_item.poster_url = None
    mock_item.popularity = 10.0
    mock_item.vote_average = 8.0
    mock_item.vote_count = 100
    mock_item.popular_rank = None
    mock_item.trending_rank = None
    mock_item.top_rated_rank = None
    mock_item.directors = []
    mock_item.cast = []
    mock_item.keywords = []

    from api.pipeline.models import CandidatePool, PrefilterDecision

    def fake_retrieve(db, ctx, intent):
        return CandidatePool(
            ids=[1],
            merged_scores={1: {"ann": 0.9}},
            prefilter=PrefilterDecision(
                allowed_ids=None, boost_ids=[], enforce_genres=False
            ),
            items_with_data={1: (mock_item, np.ones(384, dtype=np.float32), [])},
            boost_ids=[],
            enforce_genres=False,
            structured_search_filters=None,
        )

    monkeypatch.setattr("api.pipeline.runner.retrieve_candidates", fake_retrieve)

    params = RecommendParams(
        user_id="u_test", limit=5, rerank=False, rerank_provider="cross_encoder"
    )
    res = await pipeline.run(mock_request, params, mock_db)
    assert captured.get("rerank") is False
    assert captured.get("rerank_provider") == "cross_encoder"
    assert len(res.items) == 1


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
        query_vec=None,
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
    monkeypatch.setattr(
        recommend_routes,
        "_cold_start_candidates",
        lambda db, it, lim, al, **kw: [1, 2, 3],
    )
    cold_retriever = ColdStartRetriever()
    assert cold_retriever.retrieve(mock_db, ctx, intent, [1, 2]) == [1, 2, 3]

    # CollaborativeGraphRetriever
    monkeypatch.setattr(
        recommend_routes,
        "_collaborative_candidates",
        lambda db, neigh, exc, lim, allowed_ids=None, **kw: [(10, 0.9)],
    )
    collab_retriever = CollaborativeGraphRetriever()
    assert collab_retriever.retrieve(mock_db, ctx, intent, None) == [(10, 0.9)]

    # TrendingPriorRetriever
    monkeypatch.setattr(
        recommend_routes,
        "_trending_prior_candidates",
        lambda db, it, exc, lim, allowed_ids=None, **kw: [(20, 0.8)],
    )
    trending_retriever = TrendingPriorRetriever()
    assert trending_retriever.retrieve(mock_db, ctx, intent, None) == [(20, 0.8)]

    # ANNRetriever
    monkeypatch.setattr(
        recommend_routes, "ann_candidates", lambda *args, **kw: [100, 200]
    )
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


@pytest.mark.asyncio
async def test_franchise_relevance_ordering_preserved_without_chronological_sort(
    monkeypatch,
):
    """Verify franchise items are promoted to top while strictly preserving reranker relevance ordering."""
    pipeline = get_pipeline()
    mock_request = MagicMock(spec=Request)
    mock_request.app.state.entity_linker = None
    mock_db = MagicMock()
    mock_db.execute.return_value.all.return_value = []
    mock_db.execute.return_value.scalars.return_value.all.return_value = []

    # Suppose reranker scores Star Wars 2015 higher than Star Wars 1977
    reranked_output = [
        {
            "id": 200,
            "collection_id": 10,
            "release_year": 2015,
            "title": "The Force Awakens",
        },
        {"id": 100, "collection_id": 10, "release_year": 1977, "title": "A New Hope"},
        {
            "id": 300,
            "collection_id": None,
            "release_year": 2020,
            "title": "Unrelated Movie",
        },
    ]
    monkeypatch.setattr(
        "api.pipeline.runner.rerank_candidates", lambda *a, **kw: list(reranked_output)
    )

    from api.pipeline.models import CandidatePool, PrefilterDecision, QueryUnderstanding
    from api.core.legacy_intent_parser import IntentFilters

    intent = QueryUnderstanding(
        query="best star wars movies",
        llm_intent=MagicMock(),
        intent_filters=IntentFilters("best star wars movies"),
        structured_search_filters=None,
        es_text_query=None,
        query_vec=None,
        prefer_top_rated=False,
        custom_genres=[],
        has_people_filters=False,
        candidate_limit=10,
        matched_collection_ids=[10],
        collection_item_ids=[100, 200],
    )

    mock_item_1 = MagicMock(
        id=100,
        tmdb_id=100,
        media_type="movie",
        title="A New Hope",
        overview="",
        genres=[],
        release_year=1977,
        runtime=120,
        original_language="en",
        collection_id=10,
        collection_name="Star Wars",
        poster_url=None,
        popularity=10.0,
        vote_average=8.6,
        vote_count=1000,
        popular_rank=None,
        trending_rank=None,
        top_rated_rank=None,
        directors=[],
        cast=[],
        keywords=[],
    )
    mock_item_2 = MagicMock(
        id=200,
        tmdb_id=200,
        media_type="movie",
        title="The Force Awakens",
        overview="",
        genres=[],
        release_year=2015,
        runtime=135,
        original_language="en",
        collection_id=10,
        collection_name="Star Wars",
        poster_url=None,
        popularity=12.0,
        vote_average=7.8,
        vote_count=1500,
        popular_rank=None,
        trending_rank=None,
        top_rated_rank=None,
        directors=[],
        cast=[],
        keywords=[],
    )
    mock_item_3 = MagicMock(
        id=300,
        tmdb_id=300,
        media_type="movie",
        title="Unrelated Movie",
        overview="",
        genres=[],
        release_year=2020,
        runtime=90,
        original_language="en",
        collection_id=None,
        collection_name=None,
        poster_url=None,
        popularity=5.0,
        vote_average=6.0,
        vote_count=100,
        popular_rank=None,
        trending_rank=None,
        top_rated_rank=None,
        directors=[],
        cast=[],
        keywords=[],
    )

    def fake_retrieve(db, ctx, it):
        return CandidatePool(
            ids=[200, 100, 300],
            merged_scores={
                200: {"ann": 0.9},
                100: {"ann": 0.8},
                300: {"ann": 0.5},
            },
            prefilter=PrefilterDecision(
                allowed_ids=None, boost_ids=[], enforce_genres=False
            ),
            items_with_data={
                100: (mock_item_1, np.ones(384, dtype=np.float32), []),
                200: (mock_item_2, np.ones(384, dtype=np.float32), []),
                300: (mock_item_3, np.ones(384, dtype=np.float32), []),
            },
            boost_ids=[100, 200],
            enforce_genres=False,
            structured_search_filters=None,
        )

    monkeypatch.setattr("api.pipeline.runner.retrieve_candidates", fake_retrieve)

    async def fake_resolve(req, p, u, db):
        return intent

    monkeypatch.setattr("api.pipeline.runner.resolve_query_intent", fake_resolve)

    params = RecommendParams(user_id="u_sw", query="best star wars movies", limit=5)
    res = await pipeline.run(mock_request, params, mock_db)

    # Franchise items must be at the front, but in reranker relevance order
    # (id=200 before id=100, NOT sorted by release_year 1977 before 2015)
    returned_ids = [it["id"] for it in res.items]
    assert returned_ids == [200, 100, 300]


@pytest.mark.asyncio
async def test_franchise_chronological_sort_when_requested(monkeypatch):
    """Verify franchise items are sorted in release-year order when is_chronological_requested is True."""
    pipeline = get_pipeline()
    mock_request = MagicMock(spec=Request)
    mock_request.app.state.entity_linker = None
    mock_db = MagicMock()
    mock_db.execute.return_value.all.return_value = []
    mock_db.execute.return_value.scalars.return_value.all.return_value = []

    # Suppose reranker scores Star Wars 2015 higher than Star Wars 1977
    reranked_output = [
        {
            "id": 200,
            "collection_id": 10,
            "release_year": 2015,
            "title": "The Force Awakens",
        },
        {"id": 100, "collection_id": 10, "release_year": 1977, "title": "A New Hope"},
        {
            "id": 300,
            "collection_id": None,
            "release_year": 2020,
            "title": "Unrelated Movie",
        },
    ]
    monkeypatch.setattr(
        "api.pipeline.runner.rerank_candidates", lambda *a, **kw: list(reranked_output)
    )

    from api.pipeline.models import CandidatePool, PrefilterDecision, QueryUnderstanding
    from api.core.legacy_intent_parser import IntentFilters

    intent = QueryUnderstanding(
        query="Star Wars chronological",
        llm_intent=MagicMock(),
        intent_filters=IntentFilters("Star Wars chronological"),
        structured_search_filters=None,
        es_text_query=None,
        query_vec=None,
        prefer_top_rated=False,
        custom_genres=[],
        has_people_filters=False,
        candidate_limit=10,
        matched_collection_ids=[10],
        collection_item_ids=[100, 200],
        is_chronological_requested=True,
    )

    mock_item_1 = MagicMock(
        id=100,
        tmdb_id=100,
        media_type="movie",
        title="A New Hope",
        overview="",
        genres=[],
        release_year=1977,
        runtime=120,
        original_language="en",
        collection_id=10,
        collection_name="Star Wars",
        poster_url=None,
        popularity=10.0,
        vote_average=8.6,
        vote_count=1000,
        popular_rank=None,
        trending_rank=None,
        top_rated_rank=None,
        directors=[],
        cast=[],
        keywords=[],
    )
    mock_item_2 = MagicMock(
        id=200,
        tmdb_id=200,
        media_type="movie",
        title="The Force Awakens",
        overview="",
        genres=[],
        release_year=2015,
        runtime=135,
        original_language="en",
        collection_id=10,
        collection_name="Star Wars",
        poster_url=None,
        popularity=12.0,
        vote_average=7.8,
        vote_count=1500,
        popular_rank=None,
        trending_rank=None,
        top_rated_rank=None,
        directors=[],
        cast=[],
        keywords=[],
    )
    mock_item_3 = MagicMock(
        id=300,
        tmdb_id=300,
        media_type="movie",
        title="Unrelated Movie",
        overview="",
        genres=[],
        release_year=2020,
        runtime=90,
        original_language="en",
        collection_id=None,
        collection_name=None,
        poster_url=None,
        popularity=5.0,
        vote_average=6.0,
        vote_count=100,
        popular_rank=None,
        trending_rank=None,
        top_rated_rank=None,
        directors=[],
        cast=[],
        keywords=[],
    )

    def fake_retrieve(db, ctx, it):
        return CandidatePool(
            ids=[200, 100, 300],
            merged_scores={
                200: {"ann": 0.9},
                100: {"ann": 0.8},
                300: {"ann": 0.5},
            },
            prefilter=PrefilterDecision(
                allowed_ids=None, boost_ids=[], enforce_genres=False
            ),
            items_with_data={
                100: (mock_item_1, np.ones(384, dtype=np.float32), []),
                200: (mock_item_2, np.ones(384, dtype=np.float32), []),
                300: (mock_item_3, np.ones(384, dtype=np.float32), []),
            },
            boost_ids=[100, 200],
            enforce_genres=False,
            structured_search_filters=None,
        )

    monkeypatch.setattr("api.pipeline.runner.retrieve_candidates", fake_retrieve)

    async def fake_resolve(req, p, u, db):
        return intent

    monkeypatch.setattr("api.pipeline.runner.resolve_query_intent", fake_resolve)

    params = RecommendParams(user_id="u_sw", query="Star Wars chronological", limit=5)
    res = await pipeline.run(mock_request, params, mock_db)

    # Franchise items must be sorted in release_year order (id=100 in 1977 before id=200 in 2015)
    returned_ids = [it["id"] for it in res.items]
    assert returned_ids == [100, 200, 300]


def test_franchise_collection_boosts_respect_provider_allowlist_in_fusion(monkeypatch):
    """Verify collection item IDs outside the prefilter allowlist are not boosted."""
    from api.pipeline.retriever.fusion import retrieve_candidates
    from api.pipeline.models import PrefilterDecision, QueryUnderstanding, UserContext
    from api.core.legacy_intent_parser import IntentFilters

    mock_db = MagicMock()
    mock_db.execute.return_value.scalars.return_value.all.return_value = []

    # Prefilter allows only items 10 and 20 (e.g. available on Netflix)
    fake_prefilter = PrefilterDecision(
        allowed_ids=[10, 20], boost_ids=[10], enforce_genres=False
    )
    monkeypatch.setattr(
        recommend_routes,
        "_prefilter_allowed_ids",
        lambda *a, **kw: fake_prefilter,
    )
    monkeypatch.setattr(
        "api.pipeline.retriever.fusion.prefilter_allowed_ids",
        lambda *a, **kw: fake_prefilter,
    )
    monkeypatch.setattr(
        "api.pipeline.retriever.fusion.ANNRetriever.retrieve",
        lambda self, db, ctx, intent, allowlist: ([10], False),
    )
    monkeypatch.setattr(
        "api.pipeline.retriever.fusion.collaborative_candidates",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        "api.pipeline.retriever.fusion.trending_prior_candidates",
        lambda *a, **kw: [],
    )

    ctx = UserContext(
        canonical_id="u1",
        user_id="u1",
        profile=None,
        long_v=None,
        short_v=np.ones(384, dtype=np.float32),
        exclude_set=set(),
        profile_meta={},
        cold_start=False,
        provider_alias_map={},
        top_query_keywords=set(),
    )

    # Intent has collection_item_ids containing 10, 20, 30, 40 (where 30, 40 are not in allowlist)
    intent = QueryUnderstanding(
        query="harry potter movies on netflix",
        llm_intent=MagicMock(),
        intent_filters=IntentFilters("harry potter movies on netflix"),
        structured_search_filters=None,
        es_text_query=None,
        query_vec=np.ones(384, dtype=np.float32),
        prefer_top_rated=False,
        custom_genres=[],
        has_people_filters=False,
        candidate_limit=10,
        preferred_services={"netflix"},
        matched_collection_ids=[1241],
        collection_item_ids=[10, 20, 30, 40],
    )

    pool = retrieve_candidates(mock_db, ctx, intent)

    # boost_ids in pool must NOT contain items 30 or 40
    assert 30 not in pool.boost_ids
    assert 40 not in pool.boost_ids
    assert pool.boost_ids == [10, 20]


def test_single_vector_people_fallback_preserves_blended_active_query_vec(monkeypatch):
    """Verify that in single-vector personalized mode, people-filter fallback uses blended q_vec rather than query_vec alone."""
    from api.pipeline.retriever.ann import ANNRetriever
    from api.pipeline.models import QueryUnderstanding, UserContext
    from api.core.legacy_intent_parser import IntentFilters
    from api.core.elasticsearch_search import SearchFilters

    mock_db = MagicMock()
    vec_calls = []

    def fake_ann_candidates(db, query_vec, exclude, **kw):
        vec_calls.append(np.array(query_vec, copy=True))
        # First call (with people filters) returns empty to trigger fallback
        if len(vec_calls) == 1:
            return []
        # Fallback call returns item 999
        return [999]

    retriever = ANNRetriever()
    short_v = np.zeros(384, dtype=np.float32)
    short_v[0] = 1.0  # Unit vector along axis 0 (user taste)

    query_v = np.zeros(384, dtype=np.float32)
    query_v[1] = 1.0  # Unit vector along axis 1 (query)

    ctx = UserContext(
        canonical_id="u1",
        user_id="u1",
        profile=None,
        long_v=None,
        short_v=short_v,
        exclude_set=set(),
        profile_meta={},
        cold_start=False,
        provider_alias_map={},
        top_query_keywords=set(),
    )

    search_filters = SearchFilters(
        cast=("Nonexistent Actor",),
        genres=("Action",),
    )

    intent = QueryUnderstanding(
        query="action with Nonexistent Actor",
        llm_intent=MagicMock(),
        intent_filters=IntentFilters("action with Nonexistent Actor"),
        structured_search_filters=search_filters,
        es_text_query=None,
        query_vec=query_v,
        prefer_top_rated=False,
        custom_genres=[],
        has_people_filters=True,
        candidate_limit=10,
    )

    monkeypatch.setattr(recommend_routes, "ann_candidates", fake_ann_candidates)
    monkeypatch.setattr(
        "api.pipeline.retriever.ann.ann_candidates", fake_ann_candidates
    )

    ids, rewrite_used = retriever.retrieve(mock_db, ctx, intent, allowlist=None)

    assert ids == [999]
    assert len(vec_calls) == 2

    # The first call uses the blended vector
    first_vec = vec_calls[0]
    # The fallback call MUST also use the blended vector (active_query_vec), NOT raw query_v!
    fallback_vec = vec_calls[1]

    np.testing.assert_allclose(fallback_vec, first_vec, rtol=1e-5, atol=1e-5)
    # Ensure it's not just query_v (taste contribution must be present)
    assert (
        fallback_vec[0] > 0.0
    ), "User taste vector contribution (axis 0) was lost in fallback vector!"
