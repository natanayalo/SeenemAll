import json

from evaluation.catalog_references import reconcile_reference


def catalog():
    return {
        "movie:1": {
            "title": "Actual Movie",
            "overview": "Plot",
            "tmdb_id": 1,
            "media_type": "movie",
        },
        "tv:2": {
            "title": "Actual Series",
            "overview": "Plot",
            "tmdb_id": 2,
            "media_type": "tv",
        },
    }


def test_identity_resolution_is_exact_unique_and_typed():
    rows = catalog()
    rows["movie:4"] = {"title": None}  # An unhydrated row cannot crash resolution.
    reference = {"id": 1, "tmdb_id": 1, "title": " ACTUAL   Series "}
    result, reason = reconcile_reference(reference, rows)
    assert result["id"] == 2 and result["media_type"] == "tv"
    assert result["tmdb_id"] == 2
    assert reference["id"] == 1 and reason == "resolved_by_unique_title"
    assert (
        reconcile_reference({"id": 1, "title": "Actual Movie"}, rows)[1] == "unchanged"
    )
    assert reconcile_reference(1, rows) == (1, "unchanged")
    assert reconcile_reference(99, rows) == (None, "missing_catalog")
    assert reconcile_reference({"id": 1, "title": "Imaginary"}, rows) == (
        None,
        "unresolved_title",
    )
    rows["movie:3"] = {"title": "Actual Series"}
    assert reconcile_reference(reference, rows) == (None, "ambiguous_title")


def test_regenerating_splits_cannot_reintroduce_wrong_title_pairs(
    tmp_path, monkeypatch
):
    from evaluation import build_dataset_splits as builder

    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "catalog_evidence_v2.2.json").write_text(json.dumps(catalog()))
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            [
                {
                    "query": "series",
                    "category": "vibe",
                    "golden_set": [
                        {"id": 1, "title": "Actual Series"},
                        {"id": 1, "title": "Missing title"},
                    ],
                }
            ]
        )
    )
    monkeypatch.setattr(builder, "FIXTURES_DIR", fixtures)
    monkeypatch.setattr(builder, "OUTPUT_DIR", tmp_path / "datasets")
    monkeypatch.setattr(builder, "EVAL_SET_PATH", source)
    builder.build_splits()
    case = json.loads((tmp_path / "datasets/product_dev.json").read_text())[0]
    assert case["golden_set"][0]["media_type"] == "tv"
    assert len(case["golden_set"]) == len(case["excluded_golden_references"]) == 1
