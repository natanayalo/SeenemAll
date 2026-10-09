"""Dedicated test suite to ensure >= 85% branch and statement coverage for api.pipeline.intent.builder."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from api.core.llm_parser import Intent
from api.pipeline.intent.builder import resolve_query_intent
from api.pipeline.models import RecommendParams, UserContext


@pytest.fixture
def base_context():
    return UserContext(
        canonical_id="test_user",
        user_id="test_user",
        profile=None,
        long_v=None,
        short_v=None,
        exclude_set=set(),
        profile_meta={},
        cold_start=False,
        provider_alias_map={"netflix": {"netflix"}, "disney": {"disney_plus"}},
        top_query_keywords={"masterpiece", "epic"},
    )


def test_ann_backend_override_validation(base_context):
    req = MagicMock(spec=Request)
    req.app.state.entity_linker = None
    db = MagicMock()

    # Invalid backend raises 400
    params_invalid = RecommendParams(
        query="movies", ann_backend_override="unsupported_engine"
    )
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(resolve_query_intent(req, params_invalid, base_context, db))
    assert exc_info.value.status_code == 400

    # Valid backend succeeds
    params_valid = RecommendParams(
        query="movies", ann_backend_override=" ELASTICSEARCH "
    )
    mock_qf = SimpleNamespace(
        languages=(),
        keywords=(),
        genres=(),
        media_types=(),
        cast=(),
        directors=(),
        producers=(),
        writers=(),
        residual_text="movies",
        reference_titles=(),
        matched_collections=(),
    )
    with patch(
        "api.routes.recommend._parse_llm_intent",
        create=True,
        side_effect=lambda *a, **kw: Intent(),
    ), patch(
        "api.routes.recommend.get_query_filters", create=True, return_value=mock_qf
    ), patch(
        "api.routes.recommend.get_filter_matcher", create=True, return_value=MagicMock()
    ), patch(
        "api.pipeline.intent.builder.parse_llm_intent", return_value=Intent()
    ), patch(
        "api.pipeline.intent.builder.get_query_filters", return_value=mock_qf
    ), patch(
        "api.pipeline.intent.builder.get_filter_matcher", return_value=MagicMock()
    ), patch(
        "api.pipeline.intent.builder.build_query_vector", return_value=[]
    ):
        res = asyncio.run(resolve_query_intent(req, params_valid, base_context, db))
        assert res.backend_override_normalized == "elasticsearch"


def test_entity_linker_interaction(base_context):
    req = MagicMock(spec=Request)
    mock_linker = MagicMock()
    mock_linker.link_entities = AsyncMock(return_value={"entities": ["Nolan"]})
    req.app.state.entity_linker = mock_linker
    db = MagicMock()

    params = RecommendParams(query="Inception Christopher Nolan")
    mock_qf = SimpleNamespace(
        languages=(),
        keywords=(),
        genres=(),
        media_types=(),
        cast=(),
        directors=(),
        producers=(),
        writers=(),
        residual_text="Inception Christopher Nolan",
        reference_titles=(),
        matched_collections=(),
    )
    with patch(
        "api.routes.recommend._parse_llm_intent",
        create=True,
        side_effect=lambda *a, **kw: Intent(),
    ), patch(
        "api.routes.recommend.get_query_filters", create=True, return_value=mock_qf
    ), patch(
        "api.routes.recommend.get_filter_matcher", create=True, return_value=MagicMock()
    ), patch(
        "api.pipeline.intent.builder.parse_llm_intent", return_value=Intent()
    ), patch(
        "api.pipeline.intent.builder.get_query_filters", return_value=mock_qf
    ), patch(
        "api.pipeline.intent.builder.get_filter_matcher", return_value=MagicMock()
    ), patch(
        "api.pipeline.intent.builder.build_query_vector", return_value=[]
    ):
        res = asyncio.run(resolve_query_intent(req, params, base_context, db))
        assert res is not None
        mock_linker.link_entities.assert_awaited_once_with(
            "Inception Christopher Nolan"
        )


def test_llm_intent_disabled_and_genre_override(base_context):
    req = MagicMock(spec=Request)
    req.app.state.entity_linker = None
    db = MagicMock()

    params = RecommendParams(
        query="action movies",
        use_llm_intent=False,
        genre_override="Horror, Sci-Fi",
    )

    mock_qf = SimpleNamespace(
        languages=(),
        keywords=(),
        genres=(),
        media_types=(),
        cast=(),
        directors=(),
        producers=(),
        writers=(),
        residual_text="action movies",
        reference_titles=(),
        matched_collections=(),
    )
    with patch(
        "api.routes.recommend.get_query_filters", create=True, return_value=mock_qf
    ), patch(
        "api.routes.recommend.get_filter_matcher", create=True, return_value=MagicMock()
    ), patch(
        "api.pipeline.intent.builder.get_query_filters", return_value=mock_qf
    ), patch(
        "api.pipeline.intent.builder.get_filter_matcher", return_value=MagicMock()
    ), patch(
        "api.pipeline.intent.builder.build_query_vector", return_value=[]
    ):
        res = asyncio.run(resolve_query_intent(req, params, base_context, db))
        assert "Horror" in res.intent_filters.genres
        assert "Science Fiction" in res.intent_filters.genres


def test_classic_top_rated_keyword_heuristic(base_context):
    req = MagicMock(spec=Request)
    req.app.state.entity_linker = None
    db = MagicMock()

    # Query matches top_query_keywords "epic" and has non-top keyword "superhero"
    params = RecommendParams(
        query="epic superhero movies",
        diversify=True,
    )

    mock_qf = SimpleNamespace(
        languages=(),
        keywords=("superhero",),
        genres=(),
        media_types=(),
        cast=(),
        directors=(),
        producers=(),
        writers=(),
        residual_text="epic superhero movies",
        reference_titles=(),
        matched_collections=(),
    )
    with patch(
        "api.routes.recommend._parse_llm_intent",
        create=True,
        side_effect=lambda *a, **kw: Intent(),
    ), patch(
        "api.routes.recommend.get_query_filters", create=True, return_value=mock_qf
    ), patch(
        "api.routes.recommend.get_filter_matcher", create=True, return_value=MagicMock()
    ), patch(
        "api.pipeline.intent.builder.parse_llm_intent", return_value=Intent()
    ), patch(
        "api.pipeline.intent.builder.get_query_filters", return_value=mock_qf
    ), patch(
        "api.pipeline.intent.matcher.get_top_query_keywords",
        return_value={"masterpiece", "epic"},
    ), patch(
        "api.pipeline.intent.builder.get_filter_matcher", return_value=MagicMock()
    ), patch(
        "api.pipeline.intent.builder.build_query_vector", return_value=[]
    ):
        res = asyncio.run(resolve_query_intent(req, params, base_context, db))
        assert res is not None
        # diversify set to False and mixer weights adjusted
        assert params.diversify is False
        assert params.mixer_vote_weight == 1.2
        assert params.mixer_ann_weight == 0.2


def test_strict_filters_and_reference_titles_dataclass(base_context):
    req = MagicMock(spec=Request)
    req.app.state.entity_linker = None
    db = MagicMock()

    params = RecommendParams(
        query="The Matrix",
        strict_filters=True,
    )

    llm_intent = Intent(
        reference_titles=["The Matrix Reloaded"],
        franchises=["The Matrix Collection"],
    )

    mock_matcher = MagicMock()
    mock_matcher.resolve_collections.return_value = [(101, "The Matrix Collection")]
    mock_matcher.resolve_person.return_value = (None, [])

    mock_qf = SimpleNamespace(
        languages=(),
        keywords=(),
        genres=(),
        media_types=(),
        cast=(),
        directors=(),
        producers=(),
        writers=(),
        residual_text="The Matrix",
        reference_titles=("The Matrix",),
        matched_collections=[(101, "The Matrix Collection")],
    )

    with patch(
        "api.routes.recommend._parse_llm_intent",
        create=True,
        side_effect=lambda *a, **kw: llm_intent,
    ), patch(
        "api.routes.recommend.get_query_filters", create=True, return_value=mock_qf
    ), patch(
        "api.routes.recommend.get_filter_matcher",
        create=True,
        return_value=mock_matcher,
    ), patch(
        "api.pipeline.intent.builder.parse_llm_intent", return_value=llm_intent
    ), patch(
        "api.pipeline.intent.builder.get_query_filters", return_value=mock_qf
    ), patch(
        "api.pipeline.intent.builder.get_filter_matcher", return_value=mock_matcher
    ), patch(
        "api.pipeline.intent.builder.strict_required_genres", return_value=["Action"]
    ), patch(
        "api.pipeline.intent.builder.build_query_vector", return_value=[]
    ):
        db.execute.return_value.all.return_value = [(10, 1999), (11, 2003)]
        res = asyncio.run(resolve_query_intent(req, params, base_context, db))
        assert "Action" in res.intent_filters.required_genres
        assert "The Matrix Reloaded" in res.intent_filters.reference_titles
        assert 10 in res.collection_item_ids


def test_person_role_reassignment_and_people_field(base_context):
    req = MagicMock(spec=Request)
    req.app.state.entity_linker = None
    db = MagicMock()

    params = RecommendParams(query="Christopher Nolan movies")

    llm_intent = Intent(
        include_actors=["Tom Hardy"],
        include_directors=["Christopher Nolan"],
        include_producers=["Emma Thomas"],
        include_writers=["Jonathan Nolan"],
        include_people=["Cillian Murphy"],
    )

    mock_matcher = MagicMock()

    def mock_resolve_person(name):
        if name == "Tom Hardy":
            return ("Tom Hardy", ["directors"])
        return (name, [])

    mock_matcher.resolve_person.side_effect = mock_resolve_person
    mock_matcher.resolve_collections.return_value = []
    mock_matcher.resolve_collection.return_value = None

    mock_qf = SimpleNamespace(
        languages=(),
        keywords=(),
        genres=(),
        media_types=(),
        cast=(),
        directors=(),
        producers=(),
        writers=(),
        residual_text="",
        reference_titles=(),
        matched_collections=(),
    )

    with patch(
        "api.routes.recommend._parse_llm_intent",
        create=True,
        side_effect=lambda *a, **kw: llm_intent,
    ), patch(
        "api.routes.recommend.get_query_filters", create=True, return_value=mock_qf
    ), patch(
        "api.routes.recommend.get_filter_matcher",
        create=True,
        return_value=mock_matcher,
    ), patch(
        "api.pipeline.intent.builder.parse_llm_intent", return_value=llm_intent
    ), patch(
        "api.pipeline.intent.builder.get_query_filters", return_value=mock_qf
    ), patch(
        "api.pipeline.intent.builder.get_filter_matcher", return_value=mock_matcher
    ), patch(
        "api.pipeline.intent.builder.build_query_vector", return_value=[]
    ):
        res = asyncio.run(resolve_query_intent(req, params, base_context, db))
        assert res.structured_search_filters is not None
        assert "Tom Hardy" in res.structured_search_filters.directors
        assert "Christopher Nolan" in res.structured_search_filters.directors
        assert "Emma Thomas" in res.structured_search_filters.producers
        assert "Jonathan Nolan" in res.structured_search_filters.writers
        assert "Cillian Murphy" in res.structured_search_filters.cast


def test_dark_knight_and_mcu_query_handling(base_context):
    req = MagicMock(spec=Request)
    req.app.state.entity_linker = None
    db = MagicMock()

    params_dk = RecommendParams(query="dark knight trilogy in chronological order")
    llm_intent_dk = Intent(
        year_min=2000,
        year_max=2015,
        exclude_genres=["Horror"],
        streaming_providers=["netflix"],
    )

    mock_matcher = MagicMock()
    mock_matcher.resolve_collections.return_value = []
    mock_matcher.resolve_collection.return_value = (263, "The Dark Knight Collection")
    mock_matcher.resolve_person.return_value = (None, [])

    mock_qf_dk = SimpleNamespace(
        languages=(),
        keywords=(),
        genres=(),
        media_types=(),
        cast=(),
        directors=(),
        producers=(),
        writers=(),
        residual_text="",
        reference_titles=(),
        matched_collections=[
            (263, "The Dark Knight Collection"),
            (999, "Other Batman"),
        ],
    )

    with patch(
        "api.routes.recommend._parse_llm_intent",
        create=True,
        side_effect=lambda *a, **kw: llm_intent_dk,
    ), patch(
        "api.routes.recommend.get_query_filters", create=True, return_value=mock_qf_dk
    ), patch(
        "api.routes.recommend.get_filter_matcher",
        create=True,
        return_value=mock_matcher,
    ), patch(
        "api.pipeline.intent.builder.parse_llm_intent", return_value=llm_intent_dk
    ), patch(
        "api.pipeline.intent.builder.get_query_filters", return_value=mock_qf_dk
    ), patch(
        "api.pipeline.intent.builder.get_filter_matcher", return_value=mock_matcher
    ), patch(
        "api.pipeline.intent.builder.build_query_vector", return_value=[]
    ):
        mock_row = MagicMock()
        mock_row.id = 155
        mock_row.release_year = 2008
        db.execute.return_value.all.return_value = [mock_row]

        res = asyncio.run(resolve_query_intent(req, params_dk, base_context, db))
        assert res.matched_collection_ids == [263]
        assert res.collection_item_ids == [155]
        assert res.is_chronological_requested is True

    # MCU query handling
    params_mcu = RecommendParams(query="mcu marvel timeline release order")
    mock_qf_mcu = SimpleNamespace(
        languages=(),
        keywords=(),
        genres=(),
        media_types=(),
        cast=(),
        directors=(),
        producers=(),
        writers=(),
        residual_text="",
        reference_titles=(),
        matched_collections=(),
    )
    with patch(
        "api.routes.recommend._parse_llm_intent",
        create=True,
        side_effect=lambda *a, **kw: Intent(year_min=2008, year_max=2024),
    ), patch(
        "api.routes.recommend.get_query_filters", create=True, return_value=mock_qf_mcu
    ), patch(
        "api.routes.recommend.get_filter_matcher",
        create=True,
        return_value=mock_matcher,
    ), patch(
        "api.pipeline.intent.builder.parse_llm_intent",
        return_value=Intent(year_min=2008, year_max=2024),
    ), patch(
        "api.pipeline.intent.builder.get_query_filters", return_value=mock_qf_mcu
    ), patch(
        "api.pipeline.intent.builder.get_filter_matcher", return_value=mock_matcher
    ), patch(
        "api.pipeline.intent.builder.build_query_vector", return_value=[]
    ):
        row_mcu = MagicMock()
        row_mcu.id = 1726
        row_mcu.collection_id = 86311
        db.execute.return_value.all.return_value = [row_mcu]

        res_mcu = asyncio.run(resolve_query_intent(req, params_mcu, base_context, db))
        assert 1726 in res_mcu.collection_item_ids
        assert 86311 in res_mcu.matched_collection_ids
        assert res_mcu.is_chronological_requested is True


def test_runtime_and_empty_query(base_context):
    req = MagicMock(spec=Request)
    req.app.state.entity_linker = None
    db = MagicMock()

    # Empty query
    params_empty = RecommendParams(query=None)
    mock_qf_empty = SimpleNamespace(
        languages=(),
        keywords=(),
        genres=(),
        media_types=(),
        cast=(),
        directors=(),
        producers=(),
        writers=(),
        residual_text="",
        reference_titles=(),
        matched_collections=(),
    )
    with patch(
        "api.routes.recommend._parse_llm_intent",
        create=True,
        side_effect=lambda *a, **kw: Intent(),
    ), patch(
        "api.routes.recommend.get_query_filters",
        create=True,
        return_value=mock_qf_empty,
    ), patch(
        "api.routes.recommend.get_filter_matcher", create=True, return_value=MagicMock()
    ), patch(
        "api.pipeline.intent.builder.parse_llm_intent", return_value=Intent()
    ), patch(
        "api.pipeline.intent.builder.get_query_filters", return_value=mock_qf_empty
    ), patch(
        "api.pipeline.intent.builder.get_filter_matcher", return_value=MagicMock()
    ), patch(
        "api.pipeline.intent.builder.build_query_vector", return_value=[]
    ):
        res_empty = asyncio.run(
            resolve_query_intent(req, params_empty, base_context, db)
        )
        assert res_empty.query_vec == []

    # Runtime filters
    params_runtime = RecommendParams(query="short films under 90 minutes")
    llm_intent_rt = Intent(
        runtime_minutes_min=30,
        runtime_minutes_max=90,
    )
    mock_qf_rt = SimpleNamespace(
        languages=(),
        keywords=(),
        genres=(),
        media_types=(),
        cast=(),
        directors=(),
        producers=(),
        writers=(),
        residual_text="",
        reference_titles=(),
        matched_collections=(),
    )
    with patch(
        "api.routes.recommend._parse_llm_intent",
        create=True,
        side_effect=lambda *a, **kw: llm_intent_rt,
    ), patch(
        "api.routes.recommend.get_query_filters", create=True, return_value=mock_qf_rt
    ), patch(
        "api.routes.recommend.get_filter_matcher", create=True, return_value=MagicMock()
    ), patch(
        "api.pipeline.intent.builder.parse_llm_intent", return_value=llm_intent_rt
    ), patch(
        "api.pipeline.intent.builder.get_query_filters", return_value=mock_qf_rt
    ), patch(
        "api.pipeline.intent.builder.get_filter_matcher", return_value=MagicMock()
    ), patch(
        "api.pipeline.intent.builder.build_query_vector", return_value=[]
    ):
        res_rt = asyncio.run(
            resolve_query_intent(req, params_runtime, base_context, db)
        )
        assert res_rt.structured_search_filters is not None
        assert res_rt.structured_search_filters.runtime_gte == 30
        assert res_rt.structured_search_filters.runtime_lte == 90
