"""Unit tests for dataset loaders, splits, and external anchors."""

from collections import Counter
import json
import pytest

from evaluation.datasets import (
    MovieLens20MLoader,
    TagGenomeLoader,
    load_evaluation_cases,
    load_public_dataset,
)


def test_product_dev_and_reg_splits():
    dev_cases = load_evaluation_cases(track="product", split="dev")
    reg_cases = load_evaluation_cases(track="product", split="regression")

    assert len(dev_cases) >= 50
    assert len(reg_cases) >= 50

    dev_families = {c.family_id for c in dev_cases}
    reg_families = {c.family_id for c in reg_cases}
    assert len(dev_families) >= 50
    assert len(reg_families) >= 50

    dev_slices = Counter(c.slice_tags[0] for c in dev_cases)
    reg_slices = Counter(c.slice_tags[0] for c in reg_cases)

    # Critical slices minimums (>= 10 each)
    for critical_slice in ("vibe", "franchise", "constraint", "entity"):
        assert (
            dev_slices[critical_slice] >= 10
        ), f"Dev slice {critical_slice} has {dev_slices[critical_slice]} < 10"
        assert (
            reg_slices[critical_slice] >= 10
        ), f"Reg slice {critical_slice} has {reg_slices[critical_slice]} < 10"


def test_cold_start_cases():
    cold_cases = load_evaluation_cases(track="cold_start", split="dev")
    assert len(cold_cases) == 12
    for c in cold_cases:
        assert c.track == "cold_start" or "cold_start" in c.slice_tags


def test_movielens_20m_loader():
    loader = MovieLens20MLoader()
    # Rating mappings
    assert loader.map_rating_to_grade(5.0) == 3
    assert loader.map_rating_to_grade(4.5) == 3
    assert loader.map_rating_to_grade(4.0) == 2
    assert loader.map_rating_to_grade(3.5) == 1
    assert loader.map_rating_to_grade(3.0) == 1
    assert loader.map_rating_to_grade(2.5) == 0
    assert loader.map_rating_to_grade(1.0) == 0

    # Missing dataset must raise FileNotFoundError rather than silently fabricating fake users
    with pytest.raises(FileNotFoundError):
        load_public_dataset("movielens20m")


def test_tag_genome_loader(tmp_path):
    # Missing dataset must raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_public_dataset("tag_genome")

    # With valid continuous score genome file
    sample_genome = tmp_path / "tag_genome.json"
    genome_data = {
        "cyberpunk": {"603": 0.95, "335984": 0.88, "100": 0.20},
        "atmospheric": {"157336": 0.92, "27205": 0.85},
    }
    with sample_genome.open("w", encoding="utf-8") as fp:
        json.dump(genome_data, fp)

    loader = TagGenomeLoader(genome_path=sample_genome)
    entries = loader.load_as_public_entries(tags=["cyberpunk", "atmospheric"])
    assert len(entries) == 2
    cyber_entry = next(e for e in entries if e["query"] == "cyberpunk")
    # Only items with score >= 0.5 are included in golden_ids
    assert set(cyber_entry["golden_ids"]) == {603, 335984}

    with pytest.raises(NotImplementedError):
        load_public_dataset("unknown_dataset_xyz")


def test_legacy_cases_and_fallback_paths(tmp_path):
    import json
    from evaluation.datasets import _load_legacy_cases

    # 1. Nonexistent split returns empty list when no legacy file
    cases_empty = load_evaluation_cases(track="nonexistent", split="none")
    assert isinstance(cases_empty, list)

    # 2. _load_legacy_cases
    legacy_file = tmp_path / "legacy_eval.json"
    legacy_data = [
        {
            "query": "film noir",
            "category": "vibe",
            "golden_set": [{"id": 10}, {"id": 20}],
        },
        {
            "query": "cold start query",
            "category": "cold_start",
            "golden_set": [{"id": 30}],
        },
    ]
    with legacy_file.open("w", encoding="utf-8") as fp:
        json.dump(legacy_data, fp)

    loaded_prod = _load_legacy_cases(legacy_file, track="product")
    assert len(loaded_prod) == 1
    assert loaded_prod[0].query == "film noir"
    assert loaded_prod[0].golden_ids == [10, 20]

    loaded_cold = _load_legacy_cases(legacy_file, track="cold_start")
    assert len(loaded_cold) == 1
    assert loaded_cold[0].query == "cold start query"


def test_movielens_loader_with_custom_file(tmp_path):
    import json

    loader = MovieLens20MLoader()
    sample_file = tmp_path / "custom_ml.json"
    # User 999 has >= 20 train interactions and >= 5 test items in test period
    train_ts = 1380000000  # 2013 (< TRAIN_CUTOFF)
    test_ts = 1420000000  # Dec 2014 (between VAL_CUTOFF and TEST_CUTOFF)

    data = {
        "users": [
            {
                "user_id": 999,
                "train_items": [
                    {"tmdb_id": 100 + i, "timestamp": train_ts} for i in range(25)
                ],
                "test_items": [
                    {
                        "tmdb_id": 1,
                        "rating": 4.5,
                        "timestamp": test_ts,
                        "release_year": 2010,
                    },
                    {
                        "tmdb_id": 2,
                        "rating": 5.0,
                        "timestamp": test_ts,
                        "release_year": 2011,
                    },
                    {
                        "tmdb_id": 3,
                        "rating": 4.0,
                        "timestamp": test_ts,
                        "release_year": 2012,
                    },
                    {
                        "tmdb_id": 4,
                        "rating": 4.0,
                        "timestamp": test_ts,
                        "release_year": 2013,
                    },
                    {
                        "tmdb_id": 5,
                        "rating": 4.0,
                        "timestamp": test_ts,
                        "release_year": 2008,
                    },
                    {
                        "tmdb_id": 6,
                        "rating": 2.0,
                        "timestamp": test_ts,
                        "release_year": 2009,
                    },  # negative
                    {
                        "tmdb_id": 7,
                        "rating": 5.0,
                        "timestamp": test_ts,
                        "release_year": 2015,
                    },  # post-2013 catalog excluded
                ],
            },
            {
                "user_id": 998,
                "train_items": [
                    {"tmdb_id": 200 + i, "timestamp": train_ts} for i in range(25)
                ],
                "test_items": [
                    {
                        "tmdb_id": 1,
                        "rating": 5.0,
                        "timestamp": test_ts,
                        "release_year": 2010,
                    },  # only 1 positive (< 5)
                ],
            },
        ]
    }
    with sample_file.open("w", encoding="utf-8") as fp:
        json.dump(data, fp)

    catalog_data = {
        str(i): {
            "id": i,
            "title": f"Movie {i}",
            "release_year": 2015 if i == 7 else 2010,
        }
        for i in range(1, 10)
    }
    sample_cat = tmp_path / "custom_catalog.json"
    with sample_cat.open("w", encoding="utf-8") as fp:
        json.dump(catalog_data, fp)

    entries = loader.load_as_public_entries(
        interactions_path=sample_file,
        movie_catalog_path=sample_cat,
    )
    assert len(entries) == 1
    assert set(entries[0]["golden_ids"]) == {1, 2, 3, 4, 5}

    # Missing catalog raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        loader.load_as_public_entries(
            interactions_path=sample_file,
            movie_catalog_path=tmp_path / "nonexistent_cat.json",
        )

    # Empty catalog raises ValueError
    empty_cat = tmp_path / "empty_catalog.json"
    with empty_cat.open("w", encoding="utf-8") as fp:
        json.dump({}, fp)
    with pytest.raises(ValueError):
        loader.load_as_public_entries(
            interactions_path=sample_file,
            movie_catalog_path=empty_cat,
        )
