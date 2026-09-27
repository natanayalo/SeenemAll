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
