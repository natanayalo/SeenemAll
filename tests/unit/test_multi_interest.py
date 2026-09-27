from __future__ import annotations

import numpy as np

from api.core.user_profile import (
    TasteCluster,
    UserVectorResult,
    _spherical_kmeans,
    cluster_user_tastes,
    compute_user_vector,
    upsert_user_vectors,
)
from api.core.user_utils import load_user_state
from api.db.models import User
from api.pipeline.models import QueryUnderstanding, UserContext
from api.pipeline.retriever.ann import (
    ANNRetriever,
    _fuse_candidate_rankings_rrf,
)
from tests.helpers import FakeSession


def test_user_vector_result_tuple_behavior():
    long_v = np.array([1.0, 0.0], dtype="float32")
    short_v = np.array([0.0, 1.0], dtype="float32")
    prefs = {"Action": 1.0}
    negs = [123]
    clusters = [
        TasteCluster(
            cluster_id=0,
            centroid=short_v,
            weight=1.0,
            size=1,
            top_genres=["Action"],
        )
    ]

    res = UserVectorResult(long_v, short_v, prefs, [], negs, taste_clusters=clusters)

    assert len(res) == 5
    assert isinstance(res, tuple)

    (
        unpacked_long,
        unpacked_short,
        unpacked_prefs,
        unpacked_neighbors,
        unpacked_negs,
    ) = res
    assert np.array_equal(unpacked_long, long_v)
    assert np.array_equal(unpacked_short, short_v)
    assert unpacked_prefs == prefs
    assert unpacked_neighbors == []
    assert unpacked_negs == negs

    assert res.taste_clusters == clusters
    assert np.array_equal(res.long_vec, long_v)
    assert np.array_equal(res.short_vec, short_v)
    assert res.genre_prefs == prefs
    assert res.neighbors == []
    assert res.negatives == negs


def test_spherical_kmeans_k1_single_vector():
    vecs = np.array([[1.0, 0.0, 0.0]], dtype="float32")
    weights = np.array([1.0], dtype="float32")
    centroids, labels, disp = _spherical_kmeans(vecs, weights, k=1)

    assert centroids.shape == (1, 3)
    assert np.allclose(centroids[0], [1.0, 0.0, 0.0])
    assert labels[0] == 0
    assert np.isclose(disp, 0.0)


def test_spherical_kmeans_separates_orthogonal_clusters():
    # 3 vectors along X axis, 3 vectors along Y axis
    x_vecs = np.tile([1.0, 0.0, 0.0], (3, 1))
    y_vecs = np.tile([0.0, 1.0, 0.0], (3, 1))
    vecs = np.vstack([x_vecs, y_vecs]).astype("float32")
    weights = np.ones(6, dtype="float32")

    centroids, labels, disp = _spherical_kmeans(vecs, weights, k=2, random_state=42)

    assert centroids.shape == (2, 3)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
    assert np.isclose(disp, 0.0, atol=1e-5)


def test_cluster_user_tastes_empty_and_single():
    assert cluster_user_tastes(np.zeros((0, 3)), np.zeros(0), []) == []

    v = np.array([[0.0, 1.0, 0.0]], dtype="float32")
    w = np.array([2.0], dtype="float32")
    genres = [[{"name": "Horror"}]]
    clusters = cluster_user_tastes(v, w, genres)

    assert len(clusters) == 1
    assert clusters[0].cluster_id == 0
    assert np.allclose(clusters[0].centroid, [0.0, 1.0, 0.0])
    assert clusters[0].weight == 1.0
    assert clusters[0].size == 1
    assert clusters[0].top_genres == ["Horror"]
    d = clusters[0].to_dict()
    assert d["weight"] == 1.0
    assert d["top_genres"] == ["Horror"]


def test_cluster_user_tastes_homogeneous_stays_k1():
    # 6 very similar vectors (cosine sim ~ 0.99)
    base = np.array([1.0, 0.1, 0.0], dtype="float32")
    base /= np.linalg.norm(base)
    vecs = np.tile(base, (6, 1)) + np.random.normal(0, 0.01, (6, 3)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    weights = np.ones(6, dtype="float32")
    genres = [[{"name": "Sci-Fi"}]] * 6

    clusters = cluster_user_tastes(vecs, weights, genres, max_k=3)
    assert len(clusters) == 1
    assert clusters[0].top_genres == ["Sci-Fi"]


def test_cluster_user_tastes_bimodal_creates_k2():
    # 5 Sci-Fi vectors along X, 5 Romance vectors along Y
    v1 = np.tile([1.0, 0.0, 0.0], (5, 1)).astype("float32")
    v2 = np.tile([0.0, 1.0, 0.0], (5, 1)).astype("float32")
    vecs = np.vstack([v1, v2])
    weights = np.ones(10, dtype="float32")
    genres = [[{"name": "Sci-Fi"}]] * 5 + [[{"name": "Romance"}]] * 5

    clusters = cluster_user_tastes(vecs, weights, genres, max_k=3)
    assert len(clusters) == 2
    top_genre_set = {c.top_genres[0] for c in clusters}
    assert top_genre_set == {"Sci-Fi", "Romance"}
    assert np.isclose(clusters[0].weight, 0.5, atol=0.05)
    assert np.isclose(clusters[1].weight, 0.5, atol=0.05)


def test_compute_user_vector_returns_taste_clusters():
    dim = 384
    v_action = [1.0] + [0.0] * (dim - 1)
    v_drama = [0.0, 1.0] + [0.0] * (dim - 2)

    history = [(i, 1.0, "watched") for i in range(1, 7)]
    embeddings = [(i, v_action if i <= 3 else v_drama) for i in range(1, 7)]
    items = [
        (i, [{"name": "Action"}] if i <= 3 else [{"name": "Drama"}])
        for i in range(1, 7)
    ]

    session = FakeSession(
        history_rows=history,
        embedding_vectors=embeddings,
        item_rows=items,
    )

    res = compute_user_vector(session, "u-multi")
    long_v, short_v, genre_prefs, neighbors, negatives = res

    assert len(res) == 5
    assert hasattr(res, "taste_clusters")
    clusters = res.taste_clusters
    assert len(clusters) == 2
    assert "Action" in genre_prefs and "Drama" in genre_prefs


def test_upsert_user_vectors_saves_clusters():
    dim = 384
    history = [(1, 1.0, "watched"), (2, 1.0, "watched")]
    embeddings = [
        (1, [1.0] + [0.0] * (dim - 1)),
        (2, [0.0, 1.0] + [0.0] * (dim - 2)),
    ]
    items = [(1, [{"name": "Sci-Fi"}]), (2, [{"name": "Comedy"}])]

    session = FakeSession(
        history_rows=history,
        embedding_vectors=embeddings,
        item_rows=items,
        user=None,
    )

    upsert_user_vectors(session, "user-new")
    assert len(session.added) == 1
    user = session.added[0]
    assert user.taste_clusters is not None
    assert isinstance(user.taste_clusters, list)
    assert len(user.taste_clusters) >= 1


def test_load_user_state_synthesizes_cluster_when_missing():
    user = User(
        user_id="legacy-user",
        short_vec=[0.0, 1.0, 0.0],
        genre_prefs={"Comedy": 1.0},
        taste_clusters=None,
    )
    session = FakeSession(history_rows=[(10, 1.0)], user=user)

    long_v, short_v, exclude, meta = load_user_state(session, "legacy-user")
    assert "taste_clusters" in meta
    clusters = meta["taste_clusters"]
    assert len(clusters) == 1
    assert clusters[0]["cluster_id"] == 0
    assert clusters[0]["weight"] == 1.0
    assert clusters[0]["top_genres"] == ["Comedy"]


def test_fuse_candidate_rankings_rrf():
    list1 = [101, 102, 103]
    list2 = [102, 104, 101]

    # Item 102 appears at rank 2 in list1 and rank 1 in list2 -> highest fused score
    fused = _fuse_candidate_rankings_rrf([list1, list2], weights=[1.0, 1.0], limit=3)
    assert fused[0] == 102
    assert set(fused) <= {101, 102, 103, 104}

    # Empty candidate lists
    assert _fuse_candidate_rankings_rrf([]) == []
    # Single list
    assert _fuse_candidate_rankings_rrf([[1, 2, 3]], limit=2) == [1, 2]


def test_ann_retriever_multi_cluster_browse_mode(monkeypatch):
    c1 = [1.0, 0.0, 0.0]
    c2 = [0.0, 1.0, 0.0]
    taste_clusters = [
        {"cluster_id": 0, "centroid": c1, "weight": 0.6},
        {"cluster_id": 1, "centroid": c2, "weight": 0.4},
    ]

    context = UserContext(
        canonical_id="u1",
        user_id="u1",
        profile=None,
        long_v=np.array(c1, dtype="float32"),
        short_v=np.array(c1, dtype="float32"),
        exclude_set=set(),
        profile_meta={"taste_clusters": taste_clusters},
        cold_start=False,
        provider_alias_map={},
        top_query_keywords=set(),
        taste_clusters=taste_clusters,
    )

    intent = QueryUnderstanding(
        query=None,
        llm_intent=None,  # type: ignore
        intent_filters=None,  # type: ignore
        structured_search_filters=None,
        es_text_query=None,
        query_vec=None,
        prefer_top_rated=False,
        custom_genres=[],
        has_people_filters=False,
        candidate_limit=10,
    )

    queries_run = []

    def mock_ann_candidates(db, vec, exclude, limit, **kwargs):
        queries_run.append(vec)
        if np.allclose(vec, c1):
            return [11, 12, 13]
        else:
            return [21, 22, 11]

    import api.routes.recommend as recommend_routes

    monkeypatch.setattr(recommend_routes, "ann_candidates", mock_ann_candidates)
    monkeypatch.setattr(
        "api.pipeline.retriever.ann.ann_candidates", mock_ann_candidates
    )
    retriever = ANNRetriever()
    ids, rewrite_used = retriever.retrieve(FakeSession(), context, intent, None)
    assert len(queries_run) == 2  # Ran for both clusters
    assert not rewrite_used
    assert 11 in ids  # In both clusters, should be top ranked
    assert ids[0] == 11


def test_ann_retriever_multi_cluster_query_selection(monkeypatch):
    # User has Sci-Fi cluster (X axis) and Romance cluster (Y axis)
    c_scifi = [1.0, 0.0, 0.0]
    c_romance = [0.0, 1.0, 0.0]
    taste_clusters = [
        {"cluster_id": 0, "centroid": c_scifi, "weight": 0.5},
        {"cluster_id": 1, "centroid": c_romance, "weight": 0.5},
    ]

    context = UserContext(
        canonical_id="u1",
        user_id="u1",
        profile=None,
        long_v=np.array(c_scifi, dtype="float32"),
        short_v=np.array(c_scifi, dtype="float32"),
        exclude_set=set(),
        profile_meta={"taste_clusters": taste_clusters},
        cold_start=False,
        provider_alias_map={},
        top_query_keywords=set(),
        taste_clusters=taste_clusters,
    )

    # Query aligns with Romance (Y axis)
    query_vec = np.array([0.1, 0.99, 0.0], dtype="float32")
    query_vec /= np.linalg.norm(query_vec)

    intent = QueryUnderstanding(
        query="romantic date night",
        llm_intent=None,  # type: ignore
        intent_filters=None,  # type: ignore
        structured_search_filters=None,
        es_text_query=None,
        query_vec=query_vec,
        prefer_top_rated=False,
        custom_genres=[],
        has_people_filters=False,
        candidate_limit=10,
    )

    query_vectors_used = []

    def mock_ann_candidates(db, vec, exclude, limit, **kwargs):
        query_vectors_used.append(vec)
        return [201, 202]

    import api.routes.recommend as recommend_routes

    monkeypatch.setattr(recommend_routes, "ann_candidates", mock_ann_candidates)
    monkeypatch.setattr(
        "api.pipeline.retriever.ann.ann_candidates", mock_ann_candidates
    )
    retriever = ANNRetriever()
    ids, rewrite_used = retriever.retrieve(FakeSession(), context, intent, None)
    assert len(query_vectors_used) == 1
    used_vec = query_vectors_used[0]
    # Blended vector should be aligned with Romance (Y > 0.8), NOT Sci-Fi (X < 0.2)
    assert used_vec[1] > 0.8
    assert used_vec[0] < 0.3
    assert ids == [201, 202]


def test_ann_retriever_cold_start_fallback(monkeypatch):
    context = UserContext(
        canonical_id="u-cold",
        user_id="u-cold",
        profile=None,
        long_v=None,
        short_v=None,
        exclude_set=set(),
        profile_meta={},
        cold_start=True,
        provider_alias_map={},
        top_query_keywords=set(),
        taste_clusters=[],
    )

    intent = QueryUnderstanding(
        query=None,
        llm_intent=None,  # type: ignore
        intent_filters=None,  # type: ignore
        structured_search_filters=None,
        es_text_query=None,
        query_vec=None,
        prefer_top_rated=False,
        custom_genres=[],
        has_people_filters=False,
        candidate_limit=10,
    )

    import api.routes.recommend as recommend_routes

    monkeypatch.setattr(
        recommend_routes, "_cold_start_candidates", lambda *args, **kwargs: [999]
    )
    monkeypatch.setattr(
        "api.pipeline.retriever.ann.cold_start_candidates",
        lambda *args, **kwargs: [999],
    )
    retriever = ANNRetriever()
    ids, rewrite_used = retriever.retrieve(FakeSession(), context, intent, None)
    assert ids == [999]
    assert not rewrite_used
