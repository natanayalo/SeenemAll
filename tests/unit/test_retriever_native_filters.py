from unittest.mock import MagicMock
import pytest
from starlette.requests import Request

from api.core.elasticsearch_search import SearchFilters
from api.core.legacy_intent_parser import IntentFilters
from api.pipeline.retriever.cold_start import cold_start_candidates
from api.pipeline.retriever.collaborative import collaborative_candidates
from api.pipeline.retriever.trending import trending_prior_candidates


class _Result:
    def all(self):
        return []

    def scalars(self):
        return self


class _Session:
    def __init__(self):
        self.statements = []

    def execute(self, statement, *args, **kwargs):
        self.statements.append(statement)
        return _Result()


def _filters():
    return SearchFilters(
        media_types=("movie",),
        genres=("Crime", "Mystery"),
        strict_genres=True,
        release_year_gte=1990,
        release_year_lte=2020,
        runtime_gte=80,
        runtime_lte=140,
        providers=("netflix", "prime"),
        exclude_item_ids=("99",),
    )


def _assert_native_sql(stmt):
    sql = str(stmt).lower()
    assert "items.release_year >=" in sql
    assert "items.release_year <=" in sql
    assert "items.runtime >=" in sql
    assert "availability" in sql and "availability.country" in sql
    assert "items.id not in" in sql
    assert "items.genres" in sql
    assert " and " in sql


def test_collaborative_candidates_apply_metadata_filters_in_sql():
    db = _Session()
    collaborative_candidates(
        db,
        [{"user_id": "neighbor", "weight": 1.0}],
        [],
        10,
        allowed_ids=None,
        search_filters=_filters(),
    )
    _assert_native_sql(db.statements[0])


def test_trending_candidates_apply_metadata_filters_in_sql():
    db = _Session()
    trending_prior_candidates(
        db,
        IntentFilters(raw_query=""),
        [],
        10,
        allowed_ids=None,
        search_filters=_filters(),
    )
    _assert_native_sql(db.statements[0])


def test_cold_start_candidates_apply_metadata_filters_in_sql():
    db = _Session()
    cold_start_candidates(
        db,
        IntentFilters(raw_query=""),
        10,
        allowlist=None,
        search_filters=_filters(),
    )
    _assert_native_sql(db.statements[0])


@pytest.mark.asyncio
async def test_resolve_query_intent_franchise_provider_sql(monkeypatch):
    from api.pipeline.intent.builder import resolve_query_intent
    from api.pipeline.models import RecommendParams, UserContext
    from api.core.llm_parser import Intent

    db = _Session()
    req = MagicMock(spec=Request)
    req.app.state.entity_linker = None

    fake_intent = Intent(
        franchises=["Star Wars"],
        streaming_providers=["netflix"],
    )
    monkeypatch.setattr(
        "api.pipeline.intent.builder.parse_llm_intent",
        lambda q, ctx, le: fake_intent,
    )
    from api.routes import recommend as recommend_routes

    monkeypatch.setattr(
        recommend_routes,
        "_parse_llm_intent",
        lambda q, ctx, le: fake_intent,
    )

    fake_matcher = MagicMock()
    fake_matcher.resolve_collections.return_value = [(10, "Star Wars Collection")]
    fake_matcher.resolve_collection.return_value = (10, "Star Wars Collection")
    fake_matcher.resolve_person.return_value = (None, ())
    monkeypatch.setattr(
        recommend_routes,
        "get_filter_matcher",
        lambda: fake_matcher,
    )
    monkeypatch.setattr(
        "api.pipeline.intent.builder.get_filter_matcher",
        lambda: fake_matcher,
    )

    query_filter_mock = MagicMock(
        matched_collections=[(10, "Star Wars Collection")],
        reference_titles=(),
        genres=(),
        media_types=(),
        keywords=(),
        cast=(),
        directors=(),
        producers=(),
        writers=(),
        languages=(),
    )
    monkeypatch.setattr(
        recommend_routes,
        "get_query_filters",
        lambda q: query_filter_mock,
    )
    monkeypatch.setattr(
        "api.pipeline.intent.builder.get_query_filters",
        lambda q: query_filter_mock,
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
        provider_alias_map={"netflix": {"netflix"}},
        top_query_keywords=set(),
    )
    params = RecommendParams(
        user_id="u1",
        query="Star Wars on netflix",
        limit=10,
        use_llm_intent=True,
    )

    understanding = await resolve_query_intent(req, params, ctx, db)
    assert 10 in understanding.matched_collection_ids

    # Find the collection statement executed on db
    coll_stmts = [
        str(s).lower() for s in db.statements if "items.collection_id" in str(s).lower()
    ]
    assert len(coll_stmts) >= 1
    sql = coll_stmts[0]
    assert "availability" in sql
    assert "availability.service" in sql
    assert "availability.country" in sql
