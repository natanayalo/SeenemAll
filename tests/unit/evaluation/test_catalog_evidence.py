"""Regressions for namespace collisions and dropped judge evidence."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from evaluation.evidence import (
    build_item_evidence,
    catalog_metadata_from_item,
    index_catalog_metadata,
    load_catalog_metadata,
    pool_item_evidence,
)
from evaluation.judge.qualification import JudgeQualificationRunner
from evaluation.models import ItemEvidence, TypedId


def movie(**overrides):
    return dict(
        tmdb_id=42,
        media_type="movie",
        title="Movie 42",
        overview="A real synopsis.",
        genres=[{"name": "Drama"}],
        keywords=[{"name": "space"}],
        cast=[{"name": "Actor A"}, "Actor B", {"id": 3}, "Actor A"],
        directors=[{"name": "Director A"}],
        producers=[{"name": "Producer A"}],
        writers=[{"name": "Writer A"}],
        release_year=2001,
        runtime=95,
        original_language="en",
        maturity_rating="PG-13",
        collection_id=9,
        collection_name="Example Collection",
        **overrides,
    )


def case(ids):
    return SimpleNamespace(family_id="entity", golden_set=None, golden_ids=ids)


def test_annotation_cannot_replace_catalog_facts_or_supply_missing_catalog(monkeypatch):
    catalog = index_catalog_metadata([movie()])
    monkeypatch.setattr(
        "evaluation.judge.qualification.load_catalog_metadata", lambda: catalog
    )
    references = [
        {"id": 42, "title": "Wrong movie", "synopsis": "Wrong plot", "runtime": 999}
    ]
    ev = JudgeQualificationRunner().build_pilot_items_for_family(
        "entity", count=1, catalog_cases=[case(references)]
    )[0]
    assert (
        ev.title == "Movie 42"
        and ev.synopsis == "A real synopsis."
        and ev.runtime == 95
    )
    with pytest.raises(ValueError, match="No real catalog items"):
        JudgeQualificationRunner().build_pilot_items_for_family(
            "entity",
            catalog_cases=[
                case([{"id": 99, "title": "Made up", "synopsis": "Invented"}])
            ],
        )


def test_pilot_samples_unique_items_and_prioritizes_family(monkeypatch):
    catalog = index_catalog_metadata([{**movie(), "tmdb_id": i} for i in range(42, 65)])
    monkeypatch.setattr(
        "evaluation.judge.qualification.load_catalog_metadata", lambda: catalog
    )
    cases = [
        case([42, 43]),
        SimpleNamespace(
            family_id="other", golden_set=None, golden_ids=list(range(44, 65))
        ),
    ]
    items = JudgeQualificationRunner().build_pilot_items_for_family(
        "entity", count=20, catalog_cases=cases
    )
    assert [ev.typed_id.id for ev in items[:2]] == [42, 43]
    assert len(items) == len({str(ev.typed_id) for ev in items}) == 20


def test_movie_and_tv_with_same_tmdb_id_never_share_evidence(monkeypatch):
    film = movie()
    series = {**film, "media_type": "tv", "title": "Series 42", "cast": ["TV Actor"]}
    indexed = index_catalog_metadata([film, series])
    monkeypatch.setattr(
        "evaluation.judge.qualification.load_catalog_metadata", lambda: indexed
    )
    runner = JudgeQualificationRunner()
    items = runner.build_pilot_items_for_family(
        "entity", count=2, catalog_cases=[case([42, "tv:42"])]
    )
    by_id = {str(item.typed_id): item for item in items}
    assert by_id["movie:42"].title == "Movie 42"
    assert by_id["tv:42"].title == "Series 42"
    assert by_id["tv:42"].cast == ["TV Actor"]
    assert all(item.media_type == item.typed_id.media_type for item in items)


def test_explicit_tv_dictionary_is_not_forced_to_movie(monkeypatch):
    series = {**movie(), "media_type": "tv", "title": "Series"}
    monkeypatch.setattr(
        "evaluation.judge.qualification.load_catalog_metadata",
        lambda: index_catalog_metadata([series]),
    )
    items = JudgeQualificationRunner().build_pilot_items_for_family(
        "entity", count=1, catalog_cases=[case([{"id": 42, "media_type": "tv"}])]
    )
    assert str(items[0].typed_id) == "tv:42"
    assert items[0].media_type == "tv"


def test_wrong_namespace_is_not_substituted_for_a_missing_movie(monkeypatch):
    indexed = index_catalog_metadata([{**movie(), "media_type": "tv"}])
    monkeypatch.setattr(
        "evaluation.judge.qualification.load_catalog_metadata", lambda: indexed
    )
    with pytest.raises(ValueError, match="No real catalog items"):
        JudgeQualificationRunner().build_pilot_items_for_family(
            "entity", count=1, catalog_cases=[case([42, "bad:id", None])]
        )


def test_pilot_and_pool_preserve_credits_rating_and_franchise(monkeypatch):
    catalog = index_catalog_metadata([movie()])
    monkeypatch.setattr(
        "evaluation.judge.qualification.load_catalog_metadata", lambda: catalog
    )
    pilot = JudgeQualificationRunner().build_pilot_items_for_family(
        "entity", count=1, catalog_cases=[case([42])]
    )[0]
    pool = pool_item_evidence(
        TypedId("movie", 42),
        {
            "id": 42,
            "score": 99,
            "rank": 1,
            "popularity": 999,
            "explanation": "producer says great",
        },
        catalog,
    )
    assert pilot.content_hash() == pool.content_hash()
    text = pilot.to_evidence_text()
    for fact in [
        "Actor A",
        "Actor B",
        "Directors: Director A",
        "Producer: Producer A",
        "Writer: Writer A",
        "Maturity Rating: PG-13",
        "Collection: Example Collection",
        "Keywords: space",
    ]:
        assert fact in text
    assert pilot.cast == ["Actor A", "Actor B"]
    assert not {"cast", "directors", "maturity_rating", "collection"} & set(
        pilot.missing_fields
    )
    assert "producer says great" not in text
    assert "999" not in text


def test_sparse_pool_evidence_is_explicitly_missing_not_fabricated():
    evidence = pool_item_evidence(TypedId("tv", 42), {}, {})
    assert evidence.media_type == "tv"
    assert evidence.synopsis is None
    assert {"synopsis", "cast", "directors", "maturity_rating", "collection"} <= set(
        evidence.missing_fields
    )
    assert "[Not Provided]" in evidence.to_evidence_text()


@pytest.mark.parametrize(
    "field,value",
    [
        ("directors", ["Other director"]),
        ("maturity_rating", "R"),
        ("collection_id", 10),
        ("collection_name", "Other collection"),
        ("cast", ["Other actor"]),
    ],
)
def test_changed_semantic_facts_change_cache_identity(field, value):
    original = movie()
    first = build_item_evidence(TypedId("movie", 42), original)
    changed = build_item_evidence(TypedId("movie", 42), {**original, field: value})
    assert first.content_hash() != changed.content_hash()


def test_all_credits_are_presented_not_silently_clipped():
    row = movie()
    row.update(
        cast=[f"Actor {i}" for i in range(12)],
        crew=[f"Crew {i}" for i in range(7)],
        keywords=[f"Tag {i}" for i in range(17)],
    )
    text = build_item_evidence(TypedId("movie", 42), row).to_evidence_text()
    assert "Actor 11" in text and "Crew 6" in text and "Tag 16" in text


def test_item_evidence_defaults_to_typed_media_and_rejects_conflicts():
    assert ItemEvidence(TypedId("tv", 42), "Series").media_type == "tv"
    with pytest.raises(ValueError, match="typed ID"):
        ItemEvidence(TypedId("tv", 42), "Series", media_type="movie")
    with pytest.raises(ValueError, match="media type mismatch"):
        build_item_evidence(TypedId("movie", 42), {"media_type": "tv"})


def test_legacy_numeric_snapshot_respects_row_media_type(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"42": {**movie(), "media_type": "tv"}}))
    indexed = load_catalog_metadata(path)
    assert "tv:42" in indexed and "movie:42" not in indexed


@pytest.mark.parametrize(
    "data",
    [
        None,
        {"42": None},
        {"42": {"id": 42}},
        {"movie:42": {"id": 43, "media_type": "movie"}},
        {"movie:42": {"id": 42, "media_type": "tv"}},
        [movie(), movie()],
    ],
)
def test_corrupt_catalogs_fail_without_changing_source(tmp_path, data):
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_catalog_metadata(path)


def test_typed_key_without_redundant_id_is_supported():
    assert (
        index_catalog_metadata({"tv:42": {"title": "Series"}})["tv:42"]["tmdb_id"] == 42
    )


def test_missing_explicit_snapshot_never_falls_back_to_db(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_catalog_metadata(tmp_path / "missing.json")


def test_db_fallback_preserves_both_namespaces_and_semantic_fields(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "evaluation.evidence.LEGACY_CATALOG_PATH", tmp_path / "missing-legacy.json"
    )
    item = SimpleNamespace(**movie(), popularity=999)
    tv = SimpleNamespace(**{**movie(), "media_type": "tv"})
    db = MagicMock()
    db.query.return_value.order_by.return_value.all.return_value = [item, tv]
    session = MagicMock()
    session.return_value.__enter__.return_value = db
    monkeypatch.setattr("api.db.session.get_sessionmaker", lambda: session)
    catalog = load_catalog_metadata()
    assert set(catalog) == {"movie:42", "tv:42"}
    assert catalog["movie:42"]["directors"] == item.directors
    assert catalog["movie:42"]["maturity_rating"] == "PG-13"
    assert "popularity" not in catalog_metadata_from_item(item)


def test_empty_or_unhydrated_candidates_cannot_become_pilot_evidence(monkeypatch):
    catalog = index_catalog_metadata([{**movie(), "overview": ""}])
    monkeypatch.setattr(
        "evaluation.judge.qualification.load_catalog_metadata", lambda: catalog
    )
    with pytest.raises(ValueError, match="No real catalog items"):
        JudgeQualificationRunner().build_pilot_items_for_family(
            "entity", catalog_cases=[case([42])]
        )


def test_dataset_loader_preserves_typed_golds_through_hydration(tmp_path, monkeypatch):
    from evaluation.datasets import load_evaluation_cases

    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "query": "actor films",
                    "family_id": "entity",
                    "golden_set": [{"id": 42, "media_type": "tv"}, "movie:42"],
                },
                {
                    "query": "TV drama",
                    "family_id": "entity",
                    "constraints": {"media_type": "tv"},
                    "golden_ids": [42],
                },
            ]
        )
    )
    cases = load_evaluation_cases(path=path)
    assert cases[0].golden_ids == [42, 42]
    assert cases[1].golden_set == ["tv:42"]
    assert cases[0].to_dict()["golden_set"][0]["media_type"] == "tv"
    catalog = index_catalog_metadata(
        [movie(), {**movie(), "media_type": "tv", "title": "TV Series"}]
    )
    monkeypatch.setattr(
        "evaluation.judge.qualification.load_catalog_metadata", lambda: catalog
    )
    items = JudgeQualificationRunner().build_pilot_items_for_family(
        "entity", count=2, catalog_cases=cases
    )
    assert {str(item.typed_id) for item in items} == {"movie:42", "tv:42"}


@pytest.mark.parametrize("shape", ["typed", "list", "legacy"])
def test_movielens_positives_cannot_use_tv_metadata_or_tv_only_ids(tmp_path, shape):
    from evaluation.datasets import MovieLens20MLoader

    loader = MovieLens20MLoader()
    interactions = tmp_path / "ratings.json"
    interactions.write_text(
        json.dumps(
            {
                "users": [
                    {
                        "user_id": "test-user",
                        "train_items": [
                            {
                                "tmdb_id": i,
                                "timestamp": loader.TRAIN_CUTOFF - 1,
                                "rating": 4,
                            }
                            for i in range(100, 120)
                        ],
                        "test_items": [
                            {
                                "tmdb_id": i,
                                "timestamp": loader.VAL_CUTOFF + 1,
                                "rating": 4.5,
                            }
                            for i in range(42, 49)
                        ],
                    }
                ]
            }
        )
    )
    rows = [
        {"id": 42, "media_type": "movie", "release_year": 2020},
        {"id": 42, "media_type": "tv", "release_year": 2000},
        *[
            {"id": i, "media_type": "movie", "release_year": 2010}
            for i in range(43, 48)
        ],
        {"id": 48, "media_type": "tv", "release_year": 2010},
    ]
    data = (
        rows
        if shape == "list"
        else {
            (
                f"{row['media_type']}:{row['id']}"
                if shape == "typed"
                else str(row["id"])
            ): row
            for row in rows
        }
    )
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(data))
    entries = loader.load_as_public_entries(interactions, path)
    assert len(entries) == 1
    assert entries[0]["golden_ids"] == [43, 44, 45, 46, 47]


def test_movielens_default_prefers_corrected_frozen_catalog(tmp_path, monkeypatch):
    from evaluation import datasets

    monkeypatch.setattr(datasets, "FIXTURES_DIR", tmp_path)
    (tmp_path / "catalog_evidence_v2.2.json").write_text(
        json.dumps({"movie:42": {"release_year": 2010}})
    )
    # If the legacy file is accidentally selected this malformed file fails.
    (tmp_path / "catalog_metadata.json").write_text("invalid JSON")
    (tmp_path / "movielens_sample.json").write_text(json.dumps({"users": []}))
    assert datasets.MovieLens20MLoader().load_as_public_entries() == []
