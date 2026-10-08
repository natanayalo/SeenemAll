"""Production reference capture and compatibility regressions."""

import json
from unittest.mock import MagicMock

import pytest

from evaluation import baseline_v2 as baseline
from evaluation import evaluate
from evaluation.judge.consensus import JudgmentCache
from evaluation.judge.stub import StubJudgeAdapter
from evaluation.models import (
    DeterministicConstraint,
    EvaluationStatus,
    TestCase,
)
from evaluation.trace import EvaluationTrace


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    qualification = tmp_path / "evaluation/.judge_qualification_ollama.json"
    qualification.parent.mkdir()
    qualification.write_text(json.dumps({"reports": {"test-judge": {}}}))
    judge = StubJudgeAdapter("test-judge", fixed_grade=2)
    cases = [
        TestCase(
            case_id="case-1",
            family_id="family-1",
            track="product",
            split="full",
            task="search",
            slice_tags=["constraint"],
            query="journey under 90 minutes",
            constraints=DeterministicConstraint(max_runtime=90),
            golden_set=["movie:999"],
            canonical_sequence=["movie:1"],
        )
    ]
    items = [
        {
            "tmdb_id": 1,
            "media_type": "movie",
            "title": "Journey",
            "overview": "A journey.",
            "runtime": 120,
        }
    ]
    catalog = {
        "movie:1": items[0],
        "movie:999": {**items[0], "tmdb_id": 999, "runtime": 80},
    }
    runner = MagicMock()
    runner.run_case.side_effect = lambda case, **kw: (
        items,
        EvaluationTrace(case.query, case.user_id),
    )
    monkeypatch.setattr(baseline, "load_evaluation_cases", lambda **kw: cases)
    monkeypatch.setattr(baseline, "load_catalog_metadata", lambda: catalog)
    monkeypatch.setattr(
        baseline, "discover_ollama_judges", lambda: {"bespoke-nimble-9b": judge}
    )
    monkeypatch.setattr(baseline, "qualification_record_matches", lambda *a: True)
    monkeypatch.setattr(baseline, "EvaluationRunner", lambda **kw: runner)
    monkeypatch.setattr(baseline, "code_identity", lambda: {"revision": "test"})
    monkeypatch.setattr(
        baseline, "index_identity", lambda backend: {"checksum": "index-test"}
    )
    monkeypatch.setattr(JudgmentCache, "save", lambda self: None)
    latency = MagicMock()
    latency.benchmark_paired_latency.return_value = {
        "passed": True,
        "error_count": 0,
        "total_measurements": 400,
    }
    monkeypatch.setattr(baseline, "LatencyHarness", lambda *a: latency)
    args = evaluate.parse_args(
        [
            "--v2",
            "--split",
            "full",
            "--config",
            "default",
            "--backend",
            "elasticsearch",
            "--save-v2-baseline",
            str(tmp_path / "baseline.json"),
            "--v2-report",
            str(tmp_path / "report.json"),
        ]
    )
    return args, judge, cases, items, runner, latency


def test_capture_defective_quality_is_a_valid_reference(setup):
    args, judge, cases, items, runner, latency = setup
    assert evaluate.run_evaluation_v2(args) == 0
    saved = baseline.load_reference(args.save_v2_baseline, cases, judge, args)
    assert saved["reference_valid"]
    assert saved["per_query_results"][0]["constraint_violations"]
    assert saved["quality"]["ndcg_at_k"] == 0
    assert saved["quality"]["known_positive_recall_100"] == 0
    assert saved["per_query_results"][0]["qrels"] == {"movie:1": 0.0, "movie:999": 2.0}
    assert runner.run_case.call_count == 1
    assert (
        latency.benchmark_paired_latency.call_args.args[1]
        == latency.benchmark_paired_latency.call_args.args[2]
    )
    with pytest.raises(ValueError, match="immutable"):
        baseline.capture_baseline(args)


@pytest.mark.parametrize(
    "change",
    [
        "backend",
        "gain_mode",
        "k",
        "dataset",
        "catalog",
        "judge",
        "checksum",
        "case_coverage",
        "index",
    ],
)
def test_reference_rejects_incompatible_or_tampered_contract(
    setup, monkeypatch, change
):
    args, judge, cases, *_ = setup
    baseline.capture_baseline(args)
    if change in ("backend", "gain_mode", "k"):
        setattr(args, change, "other")
    elif change == "dataset":
        cases[0].query = "a different query"
    elif change == "catalog":
        monkeypatch.setattr(baseline, "load_catalog_metadata", lambda: {})
    elif change == "judge":
        judge.checkpoint_revision = "changed"
    elif change == "index":
        monkeypatch.setattr(
            baseline, "index_identity", lambda backend: {"checksum": "changed"}
        )
    else:
        saved = json.loads(args.save_v2_baseline.read_text())
        saved["per_query_results"] = []
        if change == "case_coverage":
            saved.pop("snapshot_sha256")
            saved["snapshot_sha256"] = baseline.fingerprint(saved)
        args.save_v2_baseline.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        baseline.load_reference(args.save_v2_baseline, cases, judge, args)


@pytest.mark.parametrize(
    "problem",
    [
        "errors",
        "fallbacks",
        "empty",
        "duplicates",
        "judge_failed",
        "latency",
        "index",
    ],
)
def test_incomplete_execution_cannot_write_reference(setup, monkeypatch, problem):
    args, judge, cases, items, runner, latency = setup
    if problem in ("errors", "fallbacks"):
        trace = EvaluationTrace(cases[0].query, cases[0].user_id)
        getattr(trace, problem).append("broken")
        runner.run_case.side_effect = lambda *a, **kw: (items, trace)
    elif problem == "empty":
        items.clear()
    elif problem == "duplicates":
        items.append(dict(items[0]))
    elif problem == "latency":
        latency.benchmark_paired_latency.return_value = {
            "error_count": 1,
            "errors": ["cache hit"],
        }
    elif problem == "index":
        calls = iter([{"checksum": "start"}, {"checksum": "changed"}])
        monkeypatch.setattr(baseline, "index_identity", lambda backend: next(calls))
    else:

        def inference(*a):
            if problem == "judge_failed":
                raise RuntimeError("service failure")
            return {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0}, False, "insufficient"

        monkeypatch.setattr(judge, "_run_inference", inference)
    assert baseline.capture_baseline(args) in (
        int(EvaluationStatus.INVALID),
        int(EvaluationStatus.INCONCLUSIVE),
    )
    assert not args.save_v2_baseline.exists()
    report = json.loads(args.v2_report.read_text())
    assert not report["reference_valid"]


@pytest.mark.parametrize(
    "option,value",
    [
        ("split", "dev"),
        ("track", "cold_start"),
        ("judge_config", "stub"),
        ("judgment_mode", "consensus"),
        ("allow_stub_judges_for_testing", True),
        ("private_eval", True),
        ("config", "missing"),
        ("config", "genre_override"),
        ("baseline_judge_workers", 0),
        ("baseline_judge_workers", 5),
    ],
)
def test_capture_restrictions(setup, option, value):
    args, *_ = setup
    setattr(args, option, value)
    with pytest.raises(ValueError):
        baseline.capture_baseline(args)


def test_successful_abstentions_remain_explicit_in_provisional_baseline(
    setup, monkeypatch
):
    args, judge, cases, *_ = setup
    monkeypatch.setattr(
        judge,
        "_run_inference",
        lambda *a: ({0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0}, False, "insufficient"),
    )
    assert baseline.capture_baseline(args) == 0
    saved = baseline.load_reference(args.save_v2_baseline, cases, judge, args)
    assert not saved["judgments_complete"]
    assert saved["quality_status"] == "inconclusive"
    assert saved["unresolved_judgments"] == 2
    assert saved["per_query_results"][0]["qrels"] == {}


def test_missing_dataset_and_qualification(setup, monkeypatch):
    args, judge, *_ = setup
    monkeypatch.setattr(baseline, "qualification_record_matches", lambda *a: False)
    with pytest.raises(ValueError, match="qualified"):
        baseline.capture_baseline(args)
    monkeypatch.setattr(baseline, "load_evaluation_cases", lambda **kw: [])
    with pytest.raises(ValueError, match="empty"):
        baseline.capture_baseline(args)


def test_code_identity_hashes_tracked_diff(monkeypatch):
    monkeypatch.setattr(
        baseline.subprocess,
        "check_output",
        lambda cmd, **kw: "abc\n" if kw else b"diff",
    )
    assert baseline.code_identity()["revision"] == "abc"


def test_comparison_uses_frozen_ranks_and_rejects_legacy_reference(setup, monkeypatch):
    args, judge, cases, items, runner, _ = setup
    baseline.capture_baseline(args)
    args.save_v2_baseline = None
    args.baseline_file = args.v2_report.parent / "baseline.json"
    args.config = None
    args.judge_config = "stub"
    args.allow_stub_judges_for_testing = True
    monkeypatch.setattr(evaluate, "StubJudgeAdapter", lambda *a, **kw: judge)
    monkeypatch.setattr(evaluate, "discover_ollama_judges", lambda: {})
    monkeypatch.setattr(evaluate, "load_evaluation_cases", lambda **kw: cases)
    monkeypatch.setattr(evaluate, "EvaluationRunner", lambda **kw: runner)
    monkeypatch.setattr(
        evaluate, "load_catalog_metadata", baseline.load_catalog_metadata
    )
    runner.reset_mock()
    evaluate.run_evaluation_v2(args)
    assert runner.run_case.call_count == 1
    report = json.loads(args.v2_report.read_text())
    assert report["baseline_source"] == str(args.baseline_file)
    assert report["per_query_results"][0]["base_recall_100"] == 0
    args.baseline_file.write_text(json.dumps({"config": "default", "metrics": {}}))
    assert evaluate.run_evaluation_v2(args) == int(EvaluationStatus.INVALID)


def test_cli_save_dispatch(setup, monkeypatch):
    args, *_ = setup
    monkeypatch.setattr(baseline, "capture_baseline", lambda args: 0)
    with pytest.raises(SystemExit) as result:
        evaluate.main(["--save-v2-baseline", str(args.save_v2_baseline)])
    assert result.value.code == 0


def test_full_reference_can_serve_unchanged_subset(setup):
    args, judge, cases, *_ = setup
    import copy

    second = copy.deepcopy(cases[0])
    second.case_id = "case-2"
    second.family_id = "family-2"
    second.query = "another journey"
    cases.append(second)
    baseline.capture_baseline(args)
    loaded = baseline.load_reference(args.save_v2_baseline, cases[:1], judge, args)
    assert loaded["query_count"] == 2
