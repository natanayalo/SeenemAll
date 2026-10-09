from __future__ import annotations

import base64
import json
import pytest
from unittest.mock import MagicMock
from fastapi import HTTPException
from sqlalchemy.orm import Session

from api.pipeline.context import (
    get_catalog_metadata,
    get_streaming_alias_map,
    get_top_query_keywords,
    normalize_streaming_services,
    load_media_genres,
    load_user_context,
    get_cache_key,
    cache_get,
    cache_set,
    cache_remove_user,
    clear_user_cache,
    clear_recommend_cache_for_tests,
    get_or_compute_recommendations,
)
from api.pipeline.models import ComputeResult, RecommendParams, UserContext
from api.pipeline.reranker import (
    rerank_candidates,
    build_debug_snapshot,
    encode_cursor,
    decode_cursor,
    format_presentation_items,
    empty_response,
)


@pytest.fixture(autouse=True)
def isolate_pipeline_hooks(monkeypatch):
    # These tests exercise pipeline functions, not legacy route injection.
    monkeypatch.setattr("api.pipeline.context.get_hook", lambda name, default: default)
    monkeypatch.setattr("api.pipeline.reranker.get_hook", lambda name, default: default)


def test_get_catalog_metadata_branches():
    mock_db = MagicMock(spec=Session)

    # 1. scalar_one_or_none path
    mock_res_scalar = MagicMock()
    mock_res_scalar.scalar_one_or_none.return_value = {"key": "val"}
    mock_db.execute.return_value = mock_res_scalar
    assert get_catalog_metadata(mock_db, "test_key_1") == {"key": "val"}

    # 2. tuple/list row fallback
    mock_res_tuple = MagicMock(spec=["all"])
    mock_res_tuple.all.return_value = [("tuple_val",)]
    mock_db.execute.return_value = mock_res_tuple
    assert get_catalog_metadata(mock_db, "test_key_2") == "tuple_val"

    # 3. dict row fallback
    mock_res_dict = MagicMock(spec=["all"])
    mock_res_dict.all.return_value = [{"data": "dict_val"}]
    mock_db.execute.return_value = mock_res_dict
    assert get_catalog_metadata(mock_db, "test_key_3") == "dict_val"

    # 4. raw row fallback
    mock_res_raw = MagicMock(spec=["all"])
    mock_res_raw.all.return_value = ["raw_val"]
    mock_db.execute.return_value = mock_res_raw
    assert get_catalog_metadata(mock_db, "test_key_4") == "raw_val"

    # 5. Exception branch
    mock_db.execute.side_effect = RuntimeError("DB error")
    assert get_catalog_metadata(mock_db, "test_key_err") is None


def test_get_streaming_alias_map_and_keywords(monkeypatch):
    mock_db = MagicMock(spec=Session)

    # get_streaming_alias_map with dict data
    monkeypatch.setattr(
        "api.pipeline.context.get_catalog_metadata",
        lambda db, key: {
            "netflix": ["nfx", "netflix_us", 123],
            "hulu": "hlu",
            123: "invalid",
        },
    )
    alias_map = get_streaming_alias_map(mock_db)
    assert "netflix" in alias_map
    assert "nfx" in alias_map["netflix"]
    assert "hulu" in alias_map
    assert "hlu" in alias_map["hulu"]

    # fallback to default aliases
    monkeypatch.setattr(
        "api.pipeline.context.get_catalog_metadata",
        lambda db, key: None,
    )
    default_aliases = get_streaming_alias_map(mock_db)
    assert "netflix" in default_aliases

    # get_top_query_keywords
    monkeypatch.setattr(
        "api.pipeline.context.get_catalog_metadata",
        lambda db, key: ["Action", "Sci-Fi", "", 456],
    )
    kw = get_top_query_keywords(mock_db)
    assert "action" in kw
    assert "sci-fi" in kw

    # default keywords fallback
    monkeypatch.setattr(
        "api.pipeline.context.get_catalog_metadata",
        lambda db, key: None,
    )
    def_kw = get_top_query_keywords(mock_db)
    assert "best" in def_kw


def test_normalize_streaming_services():
    alias_map = {"netflix": {"netflix", "nfx"}, "hulu": {"hulu"}}
    assert normalize_streaming_services(None, alias_map) == set()
    assert normalize_streaming_services([], alias_map) == set()

    res = normalize_streaming_services(["NFX", "hulu", "unknown_service", "", 123], alias_map)  # type: ignore
    assert "netflix" in res
    assert "nfx" in res
    assert "hulu" in res
    assert "unknown_service" in res


def test_load_media_genres(monkeypatch):
    mock_db = MagicMock(spec=Session)

    # 1. DB rows
    row1 = MagicMock()
    row1.media_type = "movie"
    row1.genres = [{"name": "Action"}, {"name": "Comedy"}, {"name": None}]

    # row without attributes (index access)
    row2 = ("tv", [{"name": "Drama"}])
    mock_db.execute.return_value.all.return_value = [
        row1,
        row2,
        ("movie", "invalid_genres"),
    ]

    # clear cache
    import api.pipeline.context as ctx_mod

    ctx_mod._MEDIA_GENRE_CACHE = None
    ctx_mod._MEDIA_GENRE_CACHE_TS = None

    genres = load_media_genres(mock_db)
    assert "movie" in genres
    assert "Action" in genres["movie"]
    assert "Comedy" in genres["movie"]
    assert "tv" in genres
    assert "Drama" in genres["tv"]

    # Cache hit branch
    genres2 = load_media_genres(mock_db)
    assert genres2 is genres


def test_mask_preferences_cannot_restore_clusters_or_reranker_preferences(monkeypatch):
    from api.pipeline.scorer import compute_semantic_affinity
    from api.pipeline.models import UserContext

    metadata = {
        "taste_clusters": [{"centroid": [1.0, 0.0]}],
        "genre_prefs": {"Drama": 1.0},
        "neighbors": [{"user_id": "neighbor"}],
        "negative_items": [99],
    }
    monkeypatch.setattr(
        "api.pipeline.context.load_user_state",
        lambda *a: ([1, 0], [1, 0], [98, 99], metadata),
    )
    monkeypatch.setattr("api.pipeline.context.get_streaming_alias_map", lambda db: {})
    monkeypatch.setattr("api.pipeline.context.get_top_query_keywords", lambda db: set())
    context = load_user_context(MagicMock(), "persona", None, mask_preferences=True)
    assert isinstance(context, UserContext)
    assert context.active_taste_clusters == []
    assert context.long_v is context.short_v is None
    assert context.exclude_set == {98, 99}
    assert context.profile_meta == {"negative_items": [99]}
    assert metadata["taste_clusters"] and metadata["neighbors"]
    scores = [
        compute_semantic_affinity(
            v, taste_clusters=context.active_taste_clusters, ann_rank=1
        )
        for v in ([1, 0], [-1, 0])
    ]
    assert scores[0] == scores[1]


def test_context_cache_and_lifecycle(monkeypatch):
    clear_recommend_cache_for_tests()
    params = RecommendParams(user_id="u1", limit=10)
    key = get_cache_key("u1", params)
    assert cache_get(key) is None

    cache_set("u1", key, [{"id": 1}], debug_context={"debug": True})
    cached = cache_get(key)
    assert cached is not None
    assert cached["items"] == [{"id": 1}]
    assert cached["debug_context"] == {"debug": True}

    removed = cache_remove_user("u1")
    assert removed == 1
    assert cache_get(key) is None

    cache_set("u1", key, [{"id": 2}])
    clear_user_cache("u1")
    assert cache_get(key) is None

    # load_user_context branches
    mock_db = MagicMock(spec=Session)
    monkeypatch.setattr(
        "api.pipeline.context.load_user_state",
        lambda db, uid: (
            None,
            None,
            [99],
            {"genre_prefs": {}, "taste_clusters": ["c1"]},
        ),
    )
    monkeypatch.setattr("api.pipeline.context.get_streaming_alias_map", lambda db: {})
    monkeypatch.setattr("api.pipeline.context.get_top_query_keywords", lambda db: set())

    # Cold start unmasked
    u_ctx = load_user_context(
        mock_db, "user_abc", profile="kids", mask_preferences=False
    )
    assert u_ctx.canonical_id == "user_abc::kids"
    assert u_ctx.cold_start is True
    assert 99 in u_ctx.exclude_set
    assert u_ctx.taste_clusters == ["c1"]

    # Masked preferences
    u_ctx_masked = load_user_context(
        mock_db, "user_abc", profile="kids", mask_preferences=True
    )
    assert u_ctx_masked.cold_start is True
    assert u_ctx_masked.long_v is None
    assert u_ctx_masked.taste_clusters == []
    assert 99 in u_ctx_masked.exclude_set


@pytest.mark.asyncio
async def test_get_or_compute_recommendations():
    clear_recommend_cache_for_tests()
    mock_request = MagicMock()
    mock_db = MagicMock(spec=Session)

    called_count = 0

    async def compute_mock(req, p, db):
        nonlocal called_count
        called_count += 1
        return ComputeResult(items=[{"id": 100}], debug_context={"c": called_count})

    # 1. bypass_cache
    params_bypass = RecommendParams(user_id="u_b", limit=5, bypass_cache=True)
    items, debug = await get_or_compute_recommendations(
        mock_request, params_bypass, mock_db, "u_b", compute_mock
    )
    assert len(items) == 1
    assert called_count == 1

    # 2. compute miss then store
    params = RecommendParams(user_id="u_norm", limit=5)
    items2, debug2 = await get_or_compute_recommendations(
        mock_request, params, mock_db, "u_norm", compute_mock
    )
    assert len(items2) == 1
    assert called_count == 2

    # 3. cache hit
    items3, debug3 = await get_or_compute_recommendations(
        mock_request, params, mock_db, "u_norm", compute_mock
    )
    assert len(items3) == 1
    assert called_count == 2  # not incremented on cache hit


def test_reranker_candidates_and_branches(monkeypatch):
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
    mock_intent = MagicMock()
    mock_intent.effective_genres.return_value = []
    mock_intent.effective_languages.return_value = []
    mock_intent.effective_keywords.return_value = []
    mock_intent.effective_cast.return_value = []
    mock_intent.effective_directors.return_value = []
    ordered = [{"id": 1, "title": "A"}]

    # 1. rerank is False -> early bypass with default explanations
    res_bypass = rerank_candidates(ordered, mock_intent, "query", ctx, rerank=False)
    assert len(res_bypass) == 1
    assert "explanation" in res_bypass[0]

    # 2. rerank with invalid provider falls back to cross_encoder
    monkeypatch.setattr(
        "api.pipeline.reranker.rerank_with_explanations",
        lambda items, **kwargs: [
            {"id": items[0]["id"], "provider": kwargs.get("provider_override")}
        ],
    )
    res_invalid = rerank_candidates(
        ordered,
        mock_intent,
        "query",
        ctx,
        rerank=True,
        rerank_provider="non_existent_provider",
    )
    assert res_invalid[0]["provider"] == "cross_encoder"


def test_build_debug_snapshot():
    mock_db = MagicMock(spec=Session)
    row = MagicMock()
    row.id = 10
    row.tmdb_id = 999
    mock_db.execute.return_value.all.return_value = [row]

    snap = build_debug_snapshot(
        mock_db,
        allowlist=[10, 20],
        boost_ids=[10],
        prefer_top_rated=True,
        strict_filters=False,
        initial_candidates_count=100,
        post_filter_candidates_count=50,
        neighbors_count=5,
        cold_start=False,
        pipeline_ms=12.34,
        actual_inferences_performed=42,
        cache_hits=3,
    )
    assert snap["allowlist_len"] == 2
    assert snap["actual_inferences_performed"] == 42
    assert snap["cache_hits"] == 3
    assert snap["metrics"]["actual_inferences_performed"] == 42
    assert snap["metrics"]["cache_hits"] == 3
    assert snap["allowlist_tmdb_ids"] == [999, None]


def test_cursor_and_presentation():
    # encode / decode
    cur = encode_cursor(5)
    assert decode_cursor(cur) == 5
    assert decode_cursor(None) == 0

    with pytest.raises(HTTPException):
        decode_cursor("invalid_cursor_string!!!")

    with pytest.raises(HTTPException):
        # negative rank encoded
        neg_cur = base64.urlsafe_b64encode(
            json.dumps({"rank": -1}).encode("utf-8")
        ).decode("utf-8")
        decode_cursor(neg_cur)

    # format_presentation_items
    items = [
        {
            "id": 1,
            "title": "T",
            "original_rank": 0,
            "vector": [0.1],
            "ann_rank": 1,
            "retrieval_score": 0.9,
            "source_scores": {},
            "directors": ["Dir"],
            "cast": ["Actor"],
            "keywords": ["tag"],
        }
    ]
    formatted = format_presentation_items(items)
    assert formatted[0]["id"] == 1
    assert "vector" not in formatted[0]
    assert "original_rank" not in formatted[0]
    assert "directors" not in formatted[0]

    assert empty_response() == {"items": []}
