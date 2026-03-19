import json
from unittest.mock import mock_open, patch

from evaluation import create_split_manifest
from evaluation import evaluate


def test_filter_evaluation_entries_by_distribution_bucket() -> None:
    evaluation_set = [
        {
            "case_id": "c1",
            "query": "family adventure movies",
            "slice_tags": ["query", "movie"],
            "distribution_bucket": "constrained_single_filter",
        },
        {
            "case_id": "c2",
            "query": "crime tv",
            "slice_tags": ["query", "tv"],
            "distribution_bucket": "semantic_query_tv",
        },
    ]

    filtered = evaluate.filter_evaluation_entries(
        evaluation_set,
        distribution_buckets=["constrained_single_filter"],
    )

    assert [entry["case_id"] for entry in filtered] == ["c1"]


def test_filter_evaluation_entries_by_slice_tag() -> None:
    evaluation_set = [
        {
            "case_id": "c1",
            "query": "feel good comedies",
            "slice_tags": ["warm_start", "query"],
            "distribution_bucket": "constrained_single_filter",
        },
        {
            "case_id": "c2",
            "query": None,
            "slice_tags": ["cold_start"],
            "distribution_bucket": "cold_start_catalog",
        },
    ]

    filtered = evaluate.filter_evaluation_entries(
        evaluation_set,
        slice_tags=["no_query"],
    )

    assert [entry["case_id"] for entry in filtered] == ["c2"]


def test_filter_evaluation_entries_by_case_id() -> None:
    evaluation_set = [
        {
            "case_id": "m2_case_037",
            "query": "feel-good movies on netflix with short runtime",
            "slice_tags": ["warm_start", "movie"],
            "distribution_bucket": "constrained_multi_filter",
        },
        {
            "case_id": "m2_case_045",
            "query": "family-friendly fantasy adventure movies",
            "slice_tags": ["warm_start", "movie"],
            "distribution_bucket": "constrained_single_filter",
        },
    ]

    filtered = evaluate.filter_evaluation_entries(
        evaluation_set,
        case_ids=["m2_case_045"],
    )

    assert [entry["case_id"] for entry in filtered] == ["m2_case_045"]


def test_filter_evaluation_entries_by_split() -> None:
    evaluation_set = [
        {
            "case_id": "c1",
            "query": "feel good comedies",
            "slice_tags": ["warm_start", "query"],
            "distribution_bucket": "constrained_single_filter",
        },
        {
            "case_id": "c2",
            "query": "crime tv",
            "slice_tags": ["warm_start", "query"],
            "distribution_bucket": "semantic_query_tv",
        },
    ]

    filtered = evaluate.filter_evaluation_entries(
        evaluation_set,
        split_assignments={"c1": "dev", "c2": "holdout"},
        splits=["holdout"],
    )

    assert [entry["case_id"] for entry in filtered] == ["c2"]


def test_load_split_manifest_accepts_wrapped_format() -> None:
    with patch("evaluation.evaluate.Path.exists", return_value=True):
        with patch(
            "builtins.open",
            mock_open(
                read_data=json.dumps(
                    {"meta": {"seed": 42}, "splits": {"c1": "Dev", "c2": "holdout"}}
                )
            ),
        ):
            assignments = evaluate.load_split_manifest("evaluation/split_manifest.json")

    assert assignments == {"c1": "dev", "c2": "holdout"}


def test_validate_split_assignments_rejects_missing_case() -> None:
    evaluation_set = [{"case_id": "c1"}, {"case_id": "c2"}]

    try:
        evaluate._validate_split_assignments(
            evaluation_set,
            {"c1": "dev"},
        )
    except ValueError as exc:
        assert "c2" in str(exc)
    else:
        raise AssertionError("Expected split assignment validation to fail.")


def test_build_split_manifest_stratifies_by_distribution_bucket() -> None:
    evaluation_set = [
        {"case_id": "m1", "distribution_bucket": "movies"},
        {"case_id": "m2", "distribution_bucket": "movies"},
        {"case_id": "m3", "distribution_bucket": "movies"},
        {"case_id": "t1", "distribution_bucket": "tv"},
        {"case_id": "t2", "distribution_bucket": "tv"},
        {"case_id": "t3", "distribution_bucket": "tv"},
    ]

    manifest = create_split_manifest.build_split_manifest(
        evaluation_set,
        holdout_ratio=0.34,
        seed=42,
        dev_name="dev",
        holdout_name="holdout",
    )

    assert manifest["summary"]["overall"] == {"dev": 4, "holdout": 2}
    assert manifest["summary"]["by_distribution_bucket"]["movies"]["holdout"] == 1
    assert manifest["summary"]["by_distribution_bucket"]["tv"]["holdout"] == 1
    assert set(manifest["splits"].values()) == {"dev", "holdout"}


def test_build_progress_line_includes_eta(monkeypatch) -> None:
    monkeypatch.setattr(evaluate, "_utc_timestamp", lambda: "12:34:56Z")

    line = evaluate._build_progress_line(
        completed_requests=12,
        total_requests=48,
        elapsed_s=120.0,
        last_latency_ms=1875.0,
    )

    assert line == (
        "[12:34:56Z] Progress 12/48 (25.0%) | elapsed 02:00 | eta 06:00 | last 1875ms"
    )


def test_build_eval_plan_excludes_non_applicable_entries() -> None:
    evaluation_set = [
        {"case_id": "c1", "genre_override": "comedy"},
        {"case_id": "c2"},
    ]
    param_specs = [
        {"name": "default", "mode": "static", "params": {"use_llm_intent": True}},
        {"name": "genre_override", "mode": "genre_override_if_present", "params": {}},
    ]

    plan = evaluate._build_eval_plan(evaluation_set, param_specs)

    assert len(plan) == 2
    assert [entry["case_id"] for _, entry, _ in plan[0][1]] == ["c1", "c2"]
    assert [entry["case_id"] for _, entry, _ in plan[1][1]] == ["c1"]


def test_format_compact_summary_table_includes_delta_columns() -> None:
    summary = {
        "_meta": {"k": 10},
        "default": {
            "overall": {
                "count": 72,
                "ndcg@10": 0.0675,
                "map": 0.0372,
                "hit_rate@10": 0.2639,
                "latency_ms_avg": 412.34,
            },
            "by_slice": {},
        },
        "ann_only": {
            "overall": {
                "count": 72,
                "ndcg@10": 0.0410,
                "map": 0.0195,
                "hit_rate@10": 0.1800,
                "latency_ms_avg": 398.12,
            },
            "by_slice": {},
            "delta_vs_baseline": {
                "overall": {
                    "ndcg@10": -0.0265,
                    "map": -0.0177,
                }
            },
        },
    }

    lines = evaluate._format_compact_summary_table(summary, 10)

    assert "scenario" in lines[0]
    assert "dNDCG" in lines[0]
    assert "default" in lines[1]
    assert "ann_only" in lines[2]
    assert "-0.0265" in lines[2]
    assert "-0.0177" in lines[2]
