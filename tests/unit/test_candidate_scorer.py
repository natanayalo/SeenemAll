from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np

from api.core.elasticsearch_search import SearchFilters
from api.core.legacy_intent_parser import IntentFilters
from api.db.models import Item
from api.pipeline.models import (
    CandidatePool,
    PrefilterDecision,
    QueryUnderstanding,
    RecommendParams,
    UserContext,
)
from api.pipeline.scorer import (
    _extract_cluster_vector,
    apply_mixer_scores,
    compute_bayesian_vote_score,
    compute_intent_overlap,
    compute_popularity_percentiles,
    compute_semantic_affinity,
    item_matches_people_filters,
    prioritize_boosted_items,
    score_candidates,
)


# --- 1. Bayesian Vote Quality Tests ---


def test_bayesian_vote_score_missing():
    assert compute_bayesian_vote_score(None, None) == 0.65
    assert compute_bayesian_vote_score(8.0, None) == 0.65
    assert compute_bayesian_vote_score(None, 100) == 0.65
    assert compute_bayesian_vote_score(8.0, 0) == 0.65
    assert compute_bayesian_vote_score(8.0, -5) == 0.65


def test_bayesian_vote_score_small_sample():
    # 1 vote with 10.0 is heavily damped toward prior mean (0.65)
    score_1_vote = compute_bayesian_vote_score(10.0, 1, min_votes=50.0, prior_mean=0.65)
    expected_1 = (1.0 / 51.0) * 1.0 + (50.0 / 51.0) * 0.65
    assert np.isclose(score_1_vote, expected_1)
    assert score_1_vote < 0.67  # not allowed to jump to 1.0!

    # 1 vote with 1.0 is damped toward 0.65
    score_low_1 = compute_bayesian_vote_score(1.0, 1, min_votes=50.0, prior_mean=0.65)
    expected_low = (1.0 / 51.0) * 0.1 + (50.0 / 51.0) * 0.65
    assert np.isclose(score_low_1, expected_low)
    assert score_low_1 > 0.63


def test_bayesian_vote_score_large_sample():
    # 10,000 votes with 8.5 retains high rating
    score_large = compute_bayesian_vote_score(
        8.5, 10000, min_votes=50.0, prior_mean=0.65
    )
    assert np.isclose(score_large, 0.85, atol=0.01)

    # 10,000 votes with 3.0 retains low rating
    score_low_large = compute_bayesian_vote_score(
        3.0, 10000, min_votes=50.0, prior_mean=0.65
    )
    assert np.isclose(score_low_large, 0.30, atol=0.01)


def test_bayesian_vote_score_bounds_and_custom():
    score = compute_bayesian_vote_score(12.0, 100, min_votes=10.0, prior_mean=0.5)
    assert 0.0 <= score <= 1.0

    # invalid types fallback to prior mean
    assert compute_bayesian_vote_score("invalid", "none") == 0.65  # type: ignore


# --- 2. Popularity Percentile Ranking Tests ---


def test_compute_popularity_percentiles_edge_cases():
    assert compute_popularity_percentiles([]) == []
    assert compute_popularity_percentiles([{"popularity": 100.0}]) == [0.5]
    assert compute_popularity_percentiles([{"popularity": None}]) == [0.5]

    # All items have equal popularity
    items_equal = [{"popularity": 25.0}, {"popularity": 25.0}, {"popularity": 25.0}]
    assert compute_popularity_percentiles(items_equal) == [0.5, 0.5, 0.5]


def test_compute_popularity_percentiles_distinct_and_ties():
    items = [{"popularity": 10.0}, {"popularity": 20.0}, {"popularity": 30.0}]
    pcts = compute_popularity_percentiles(items)
    assert pcts == [0.0, 0.5, 1.0]

    # Tied middle items
    items_tied = [
        {"popularity": 10.0},
        {"popularity": 20.0},
        {"popularity": 20.0},
        {"popularity": 30.0},
    ]
    pcts_tied = compute_popularity_percentiles(items_tied)
    assert pcts_tied[0] == 0.0
    assert pcts_tied[1] == 0.5
    assert pcts_tied[2] == 0.5
    assert pcts_tied[3] == 1.0


def test_compute_popularity_percentiles_mitigates_power_law():
    # Massive blockbuster (10,000) vs indie films (5, 10, 15)
    items = [
        {"popularity": 5.0},
        {"popularity": 10.0},
        {"popularity": 15.0},
        {"popularity": 10000.0},
    ]
    pcts = compute_popularity_percentiles(items)
    assert pcts[0] == 0.0
    assert np.isclose(pcts[1], 1.0 / 3.0)
    assert np.isclose(pcts[2], 2.0 / 3.0)
    assert pcts[3] == 1.0


# --- 3. Cluster Vector Extraction & Semantic Affinity Tests ---


def test_extract_cluster_vector():
    assert _extract_cluster_vector(None) is None
    assert _extract_cluster_vector({}) is None

    # Dict representation
    c_dict = {"centroid": [3.0, 4.0]}
    vec = _extract_cluster_vector(c_dict)
    assert vec is not None
    assert np.isclose(np.linalg.norm(vec), 1.0)
    assert np.allclose(vec, [0.6, 0.8])

    # Object representation
    obj = MagicMock()
    obj.centroid = [0.0, 5.0]
    vec_obj = _extract_cluster_vector(obj)
    assert vec_obj is not None
    assert np.isclose(np.linalg.norm(vec_obj), 1.0)
    assert np.allclose(vec_obj, [0.0, 1.0])


def test_compute_semantic_affinity_fallback():
    # When vector is missing, falls back strictly to reciprocal rank
    assert compute_semantic_affinity(None, ann_rank=0) == 1.0
    assert compute_semantic_affinity(None, ann_rank=1) == 0.5
    assert compute_semantic_affinity(np.zeros(5), ann_rank=0) == 1.0


def test_compute_semantic_affinity_with_vectors():
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # identical direction

    # With identical query vector
    aff_high = compute_semantic_affinity(v, query_vec=q, ann_rank=0)
    assert 0.8 <= aff_high <= 1.0

    # Orthogonal query vector
    q_ortho = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    aff_ortho = compute_semantic_affinity(v, query_vec=q_ortho, ann_rank=0)
    assert aff_ortho < aff_high

    # Taste clusters
    clusters = [{"centroid": [1.0, 0.0, 0.0]}, {"centroid": [0.0, 1.0, 0.0]}]
    aff_cluster = compute_semantic_affinity(v, taste_clusters=clusters, ann_rank=0)
    assert aff_cluster > 0.7

    # Single user vector
    user_v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    aff_user = compute_semantic_affinity(v, user_vec=user_v, ann_rank=0)
    assert aff_user > 0.7


# --- 4. Intent Overlap Tests ---


def test_compute_intent_overlap():
    assert compute_intent_overlap({"genres": ["Action"]}) == 1.0

    # Target genres match
    intent = MagicMock()
    intent.genres = ["Action", "Sci-Fi"]
    intent.keywords = []

    # Exact match
    item_full = {"genres": [{"name": "Action"}, {"name": "Sci-Fi"}]}
    assert compute_intent_overlap(item_full, intent_filters=intent) == 1.0

    # Partial match
    item_partial = {"genres": ["Action", "Comedy"]}
    assert compute_intent_overlap(item_partial, intent_filters=intent) == 0.5

    # Zero match
    item_zero = {"genres": ["Romance", "Drama"]}
    assert compute_intent_overlap(item_zero, intent_filters=intent) == 0.0

    # Keywords match in title or overview
    intent_kw = MagicMock()
    intent_kw.genres = []
    intent_kw.keywords = ["cyberpunk", "heist"]

    item_kw = {
        "title": "Neon Heist",
        "overview": "A cyberpunk thriller in Tokyo.",
        "keywords": [],
    }
    assert compute_intent_overlap(item_kw, intent_filters=intent_kw) == 1.0

    item_kw_partial = {
        "title": "Neon Heist",
        "overview": "A bank robbery.",
        "keywords": [],
    }
    assert compute_intent_overlap(item_kw_partial, intent_filters=intent_kw) == 0.5


# --- 5. Calibrated apply_mixer_scores Tests ---


def test_apply_mixer_scores_empty():
    candidates: list[dict] = []
    apply_mixer_scores(candidates)
    assert candidates == []


def test_apply_mixer_scores_legacy_contract():
    # Verify legacy mock dictionaries without vector or extra fields are handled cleanly
    candidates = [
        {
            "id": 1,
            "popularity": 10.0,
            "vote_count": 100,
            "vote_average": 8.0,
            "ann_rank": 0,
            "source_scores": {},
        },
        {
            "id": 2,
            "popularity": 50.0,
            "vote_count": 500,
            "vote_average": 6.5,
            "ann_rank": 1,
            "source_scores": {},
        },
    ]
    apply_mixer_scores(candidates)

    # Retrieval scores must be strictly bounded in (0, 1)
    for c in candidates:
        assert "retrieval_score" in c
        assert "score" in c
        assert "features" in c
        assert 0.0 < c["retrieval_score"] < 1.0
        assert c["score"] == c["retrieval_score"]

    # Candidate 0 has ann_rank 0 and higher vote quality -> original_rank 0
    assert candidates[0]["id"] == 1
    assert candidates[0]["original_rank"] == 0
    assert candidates[1]["id"] == 2
    assert candidates[1]["original_rank"] == 1

    # Features map verification
    f = candidates[0]["features"]
    assert "semantic_affinity" in f
    assert "collab_overlap" in f
    assert "popularity_percentile" in f
    assert "bayesian_vote" in f
    assert "novelty_score" in f
    assert "p_relevant" in f


def test_apply_mixer_scores_weight_overrides():
    candidates = [
        {
            "id": 1,
            "ann_rank": 0,
            "popularity": 5.0,
            "vote_count": 10,
            "source_scores": {"collab": 0.1},
        },
        {
            "id": 2,
            "ann_rank": 5,
            "popularity": 100.0,
            "vote_count": 1000,
            "source_scores": {"collab": 0.9},
        },
    ]

    # Override: purely collaborative
    apply_mixer_scores(
        candidates,
        ann_weight_override=0.0,
        collab_weight_override=1.0,
        trending_weight_override=0.0,
        popularity_weight_override=0.0,
        vote_weight_override=0.0,
        novelty_weight_override=0.0,
    )
    # Candidate 2 has 0.9 collab vs 0.1 collab -> must rank first
    assert candidates[0]["id"] == 2
    assert candidates[1]["id"] == 1

    # Override: purely ANN
    apply_mixer_scores(
        candidates,
        ann_weight_override=1.0,
        collab_weight_override=0.0,
        trending_weight_override=0.0,
        popularity_weight_override=0.0,
        vote_weight_override=0.0,
        novelty_weight_override=0.0,
    )
    # Candidate 1 has ann_rank 0 -> must rank first
    assert candidates[0]["id"] == 1
    assert candidates[1]["id"] == 2


def test_apply_mixer_scores_with_query_and_clusters():
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    candidates = [
        {
            "id": 1,
            "vector": v1,
            "ann_rank": 1,
            "popularity": 10.0,
            "genres": ["Sci-Fi"],
            "source_scores": {},
        },
        {
            "id": 2,
            "vector": v2,
            "ann_rank": 0,
            "popularity": 10.0,
            "genres": ["Comedy"],
            "source_scores": {},
        },
    ]

    intent = MagicMock()
    intent.genres = ["Sci-Fi"]
    intent.keywords = []

    apply_mixer_scores(
        candidates,
        query_vector=query_vec,
        intent_filters=intent,
        is_query=True,
    )

    # Candidate 1 aligns with query vector and genre -> receives higher relevance
    assert candidates[0]["id"] == 1
    assert (
        candidates[0]["features"]["semantic_affinity"]
        > candidates[1]["features"]["semantic_affinity"]
    )
    assert candidates[0]["features"]["intent_overlap"] == 1.0
    assert candidates[1]["features"]["intent_overlap"] == 0.0


# --- 6. Helper Functions & People Filters Tests ---


def test_prioritize_boosted_items():
    items = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
    assert prioritize_boosted_items([], [2]) == []
    assert prioritize_boosted_items(items, []) == items

    boosted = prioritize_boosted_items(items, [3, 1])
    assert [i["id"] for i in boosted] == [3, 1, 2, 4]


def test_item_matches_people_filters():
    mock_item = MagicMock(spec=Item)
    mock_item.cast = [{"name": "Tom Cruise"}, "Simon Pegg"]
    mock_item.directors = ["Christopher McQuarrie"]
    mock_item.producers = []
    mock_item.writers = []

    filters = SearchFilters(cast=("Tom Cruise",))
    assert item_matches_people_filters(mock_item, filters) is True

    filters_no_match = SearchFilters(cast=("Brad Pitt",))
    assert item_matches_people_filters(mock_item, filters_no_match) is False

    filters_dir = SearchFilters(directors=("Christopher McQuarrie",))
    assert item_matches_people_filters(mock_item, filters_dir) is True


# --- 7. End-to-End score_candidates Integration Test ---


def test_score_candidates_integration():
    mock_item = MagicMock(spec=Item)
    mock_item.id = 100
    mock_item.tmdb_id = 999
    mock_item.media_type = "movie"
    mock_item.title = "Interstellar"
    mock_item.overview = "A team of explorers travel through a wormhole in space."
    mock_item.poster_url = "/poster.jpg"
    mock_item.runtime = 169
    mock_item.original_language = "en"
    mock_item.genres = [{"id": 878, "name": "Sci-Fi"}, {"id": 18, "name": "Drama"}]
    mock_item.release_year = 2014
    mock_item.collection_id = None
    mock_item.collection_name = None
    mock_item.maturity_rating = "PG-13"
    mock_item.popularity = 85.0
    mock_item.vote_average = 8.4
    mock_item.vote_count = 32000
    mock_item.popular_rank = None
    mock_item.trending_rank = 10
    mock_item.top_rated_rank = 20
    mock_item.directors = ["Christopher Nolan"]
    mock_item.cast = ["Matthew McConaughey", "Anne Hathaway"]
    mock_item.keywords = ["space", "wormhole", "black hole"]

    vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
    watch_opts = [{"service": "netflix", "url": "https://netflix.com"}]

    pool = CandidatePool(
        ids=[100],
        merged_scores={100: {"ann": 0.9, "collab": 0.5}},
        prefilter=PrefilterDecision(
            allowed_ids=None, boost_ids=[], enforce_genres=False
        ),
        items_with_data={100: (mock_item, vec, watch_opts)},
        boost_ids=[],
        enforce_genres=False,
        structured_search_filters=None,
    )

    intent = QueryUnderstanding(
        query="epic space adventure",
        llm_intent=MagicMock(),
        intent_filters=IntentFilters("epic space adventure"),
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
        top_query_keywords={"space"},
        preferred_services=set(),
        taste_clusters=[],
    )

    params = RecommendParams(user_id="u1", query="epic space adventure", limit=10)

    scored = score_candidates(pool, intent, params, context)
    assert len(scored.ordered) == 1
    item = scored.ordered[0]
    assert item["id"] == 100
    assert 0.0 < item["retrieval_score"] < 1.0
    assert "features" in item
    assert item["features"]["semantic_affinity"] > 0.7
    assert item["features"]["bayesian_vote"] > 0.8
