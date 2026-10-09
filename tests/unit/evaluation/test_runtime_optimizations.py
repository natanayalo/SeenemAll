"""Runtime improvements must preserve cache identity, unknowns and promotion gates."""

import json
from pathlib import Path
from threading import Barrier
from unittest.mock import MagicMock

import pytest

from evaluation import evaluate
from evaluation.judge.consensus import (
    ConsensusJudgeEngine,
    JudgmentCache,
    PoolAdjudicator,
)
from evaluation.judge.stub import StubJudgeAdapter
from evaluation.judge.throughput import benchmark_workers, main as benchmark_main
from evaluation.models import (
    DeterministicConstraint,
    EvaluationStatus,
    ItemEvidence,
    JudgeInput,
    TestCase,
    TypedId,
)
from evaluation.preflight import check_preflight, quick_dev_cases
from evaluation.trace import EvaluationTrace


def item(identifier=1):
    return ItemEvidence(
        TypedId("movie", identifier), "Journey", "A journey.", runtime=80
    )


def case(identifier=1, tags=None):
    return TestCase(
        str(identifier),
        str(identifier),
        "product",
        "dev",
        "search",
        tags or ["vibe"],
        f"journey {identifier}",
    )


def test_checkpoint_replays_every_completed_append_and_compacts(tmp_path):
    judge = StubJudgeAdapter()
    path = tmp_path / "cache.json"
    cache = JudgmentCache(path, checkpoint_every=3)
    for identifier in (1, 2):
        inp = JudgeInput("journey", item(identifier))
        cache.set(inp, judge, judge.judge_pair(inp).to_dict())
    assert not path.exists()
    restored = JudgmentCache(path)
    assert len(restored._cache) == 2
    restored.flush()
    assert path.exists() and not restored.journal_path.exists()
    inp = JudgeInput("journey", item(3))
    restored.set(inp, judge, judge.judge_pair(inp).to_dict())
    assert len(JudgmentCache(path)._cache) == 3
    restored.flush()
    before = path.read_bytes()
    restored.flush()
    assert path.read_bytes() == before


def test_torn_journal_tail_is_repaired_before_next_append(tmp_path):
    judge = StubJudgeAdapter()
    inp = JudgeInput("journey", item())
    cache = JudgmentCache(tmp_path / "cache.json")
    cache.set(inp, judge, judge.judge_pair(inp).to_dict())
    with cache.journal_path.open("ab") as fp:
        fp.write(b'{"key": "torn')
    restored = JudgmentCache(cache.path)
    assert restored.get(inp, judge)["grade"] == judge.judge_pair(inp).grade
    assert restored.journal_path.read_bytes().endswith(b"\n")
    restored.set(
        JudgeInput("different", item()), judge, judge.judge_pair(inp).to_dict()
    )
    assert len(JudgmentCache(cache.path)._cache) == 2
    restored.journal_path.write_bytes(b"invalid complete record\n")
    with pytest.raises(ValueError):
        JudgmentCache(cache.path)


def test_failed_atomic_checkpoint_keeps_previous_snapshot_and_recoverable_journal(
    tmp_path, monkeypatch
):
    import evaluation.judge.consensus as consensus

    judge = StubJudgeAdapter()
    cache = JudgmentCache(tmp_path / "cache.json", checkpoint_every=1)
    inp = JudgeInput("journey", item())
    cache.set(inp, judge, judge.judge_pair(inp).to_dict())
    before = cache.path.read_bytes()
    monkeypatch.setattr(
        consensus.os, "replace", MagicMock(side_effect=OSError("locked"))
    )
    changed = JudgeInput("different", item())
    with pytest.raises(OSError):
        cache.set(changed, judge, judge.judge_pair(changed).to_dict())
    assert cache.path.read_bytes() == before
    assert JudgmentCache(cache.path).get(changed, judge)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "cache.json",
        "cache.json.journal",
    ]


def test_serialization_failure_keeps_checkpoint_and_journal(tmp_path, monkeypatch):
    import evaluation.judge.consensus as consensus

    cache = JudgmentCache(tmp_path / "cache.json")
    monkeypatch.setattr(
        consensus.json, "dump", MagicMock(side_effect=ValueError("bad"))
    )
    with pytest.raises(ValueError):
        cache.save()
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError):
        JudgmentCache(tmp_path / "bad.json", checkpoint_every=0)


def test_parallel_pool_only_grades_missing_pairs_and_preserves_unknowns(tmp_path):
    judge = StubJudgeAdapter(fixed_sufficiency=False)
    barrier = Barrier(2)
    original = judge.judge_pair

    def concurrent(inp):
        barrier.wait(timeout=5)
        return original(inp)

    judge.judge_pair = concurrent
    engine = ConsensusJudgeEngine(
        primary_judge=judge,
        cache=JudgmentCache(tmp_path / "cache.json"),
        qrels_dir=tmp_path / "qrels",
    )
    engine.seed_query_qrels("journey", {"movie:1": 3.0})
    qrels, records, changed = engine.label_pool(
        "journey", [item(1), item(2)], mode="single_judge", workers=2
    )
    assert changed and qrels == {} and all(r.grade is None for r in records)
    assert engine.adjudicator.inference_calls == 2
    judge.judge_pair = MagicMock(side_effect=AssertionError("must reuse"))
    engine.label_pool("journey", [item(1), item(2)], mode="single_judge", workers=4)
    assert engine.adjudicator.inference_calls == 2
    with pytest.raises(ValueError):
        engine.adjudicator.warm_pool("journey", [], 5)


def test_parallel_consensus_does_not_spend_tie_breaker_calls_when_models_agree(
    tmp_path,
):
    primary = StubJudgeAdapter("primary")
    secondary = StubJudgeAdapter("secondary")
    tie = StubJudgeAdapter("tie")
    tie.judge_pair = MagicMock(side_effect=AssertionError("not needed"))
    adjudicator = PoolAdjudicator(
        primary, secondary, tie, JudgmentCache(tmp_path / "cache.json")
    )
    adjudicator.warm_pool("journey", [item(), item()], 2, "consensus")
    assert adjudicator.inference_calls == 2
    record = adjudicator.adjudicate_pair("journey", item(), mode="consensus")
    assert record.grade == 3 and record.consensus_model_count == 2


@pytest.mark.parametrize("change", ["query", "evidence", "model", "prompt"])
def test_warm_pool_cannot_reuse_changed_judgment_identity(tmp_path, change):
    judge = StubJudgeAdapter()
    adjudicator = PoolAdjudicator(judge, cache=JudgmentCache(tmp_path / "cache.json"))
    ev = item()
    adjudicator.warm_pool("journey", [ev], 1)
    query = "journey"
    if change == "query":
        query = "another journey"
    elif change == "evidence":
        ev.synopsis = "Changed evidence"
    elif change == "model":
        judge.checkpoint_revision = "another"
    else:
        judge.get_prompt_hash = lambda: "another"
    adjudicator.warm_pool(query, [ev], 1)
    assert adjudicator.inference_calls == 2


def preflight(cases=None, **changes):
    values = dict(
        execution_failures=0,
        unexpected_fallbacks=0,
        duplicates=False,
        empty_outputs=0,
        hard_violations=0,
        disliked_violations=0,
        chronology_violations=0,
        missing_canonical_items=0,
        completeness=[1.0],
        statistical_promotion=False,
    )
    values.update(changes)
    return check_preflight(cases or [case()], **values)


@pytest.mark.parametrize(
    "field",
    [
        "execution_failures",
        "unexpected_fallbacks",
        "duplicates",
        "empty_outputs",
        "hard_violations",
        "disliked_violations",
        "chronology_violations",
        "missing_canonical_items",
    ],
)
def test_preflight_failures_never_pass(field):
    assert not preflight(**{field: 1}).passed


def test_preflight_sample_and_completeness_keep_existing_requirements():
    assert preflight().passed
    assert not preflight(completeness=[0.94]).passed
    assert preflight(completeness=[]).passed
    assert preflight(statistical_promotion=True).status == EvaluationStatus.INCONCLUSIVE
    cases = [case(i, ["vibe", "entity", "franchise", "constraint"]) for i in range(50)]
    assert preflight(cases, statistical_promotion=True).passed


def test_quick_selection_is_fixed_diverse_and_never_duplicate_families():
    from evaluation.datasets import load_evaluation_cases

    cases = load_evaluation_cases(split="dev")
    selected = quick_dev_cases(cases)
    assert len(selected) == 14
    assert [c.case_id for c in selected] == [
        c.case_id for c in quick_dev_cases(list(reversed(cases)))
    ]
    assert len({c.family_id for c in selected}) == 14
    assert {t for c in selected for t in c.slice_tags} == {
        "vibe",
        "entity",
        "franchise",
        "constraint",
        "typo",
        "multi_constraint",
    }
    assert quick_dev_cases([]) == []


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cases = [case()]
    cases[0].eligible_catalog_count = 1
    film = {
        "tmdb_id": 1,
        "media_type": "movie",
        "title": "Journey",
        "overview": "A journey",
        "runtime": 80,
    }
    runner = MagicMock()
    runner.run_case.return_value = ([film], EvaluationTrace("journey", None))
    monkeypatch.setattr(evaluate, "load_evaluation_cases", lambda **kwargs: cases)
    monkeypatch.setattr(evaluate, "load_catalog_metadata", lambda: {"movie:1": film})
    monkeypatch.setattr(evaluate, "EvaluationRunner", lambda **kwargs: runner)
    monkeypatch.setattr(evaluate, "discover_ollama_judges", lambda: {})
    args = evaluate.parse_args(
        ["--v2", "--judge-config", "stub", "--v2-report", str(tmp_path / "report.json")]
    )
    return args, runner, cases


def test_preflight_rejects_without_any_judge_calls_and_can_collect_full_diagnostics(
    comparison, monkeypatch
):
    args, runner, _ = comparison
    runner.run_case.return_value = ([], EvaluationTrace("journey", None))
    spy = MagicMock(side_effect=AssertionError("should not grade"))
    monkeypatch.setattr(evaluate.ConsensusJudgeEngine, "label_pool", spy)
    assert evaluate.run_evaluation_v2(args) == EvaluationStatus.INVALID
    report = json.loads(args.v2_report.read_text())
    assert report["phase"] == "preflight" and not report["quality_evaluated"]
    assert (
        report["performance"]["judge_calls"] == 0 and not report["promotion_eligible"]
    )
    args.continue_on_preflight_failure = True
    with pytest.raises(AssertionError, match="should not grade"):
        evaluate.run_evaluation_v2(args)


def test_full_comparison_reuses_cached_grades_with_parallel_workers(comparison):
    args, _, _ = comparison
    args.judge_workers = 2
    evaluate.run_evaluation_v2(args)
    first = json.loads(args.v2_report.read_text())
    assert first["performance"]["judge_calls"] == 1
    evaluate.run_evaluation_v2(args)
    second = json.loads(args.v2_report.read_text())
    assert second["performance"]["judge_calls"] == 0
    assert first["per_query_results"] == second["per_query_results"]
    args.quick_dev = True
    evaluate.run_evaluation_v2(args)
    assert not json.loads(args.v2_report.read_text())["promotion_eligible"]


def test_preflight_reports_actionable_constraint_failure_before_grading(comparison):
    args, runner, cases = comparison
    cases[0].constraints = DeterministicConstraint(max_runtime=60)
    assert evaluate.run_evaluation_v2(args) == EvaluationStatus.FAIL
    report = json.loads(args.v2_report.read_text())
    assert report["performance"]["judge_calls"] == 0
    violation = report["per_query_results"][0]["constraint_violations"][0]
    assert violation["typed_id"] == "movie:1"
    assert "runtime_too_long" in violation["reasons"][0]
    assert runner.run_case.call_count == 2


def test_private_preflight_returns_redacted_submission_without_public_report(
    comparison, monkeypatch
):
    args, runner, _ = comparison
    args.private_eval = True
    args.allow_stub_judges_for_testing = True
    private_dir = args.v2_report.parent / "private"
    private_dir.mkdir()
    (private_dir / "holdout_cases.json").write_text("[]")
    harness = MagicMock()
    harness.benchmark_dir = private_dir
    harness.submit_candidate.return_value = {"exit_code": int(EvaluationStatus.INVALID)}
    monkeypatch.setattr(evaluate, "PrivateBenchmarkHarness", lambda: harness)
    runner.run_case.return_value = ([], EvaluationTrace("journey", None))
    assert evaluate.run_evaluation_v2(args) == EvaluationStatus.INVALID
    assert not args.v2_report.exists()
    assert "journey" not in json.dumps(harness.submit_candidate.call_args.kwargs)


@pytest.mark.parametrize(
    "flag",
    [
        "--split=full",
        "--track=cold_start",
        "--private-eval",
        "--qualify-judges",
        "--latency-benchmark",
        "--save-v2-baseline=x.json",
        "--personalization-test",
        "--hardware-inspect",
        "--rescore-only",
    ],
)
def test_quick_mode_rejects_promotion_or_unrelated_operations(flag):
    args = evaluate.parse_args(["--v2", "--quick-dev", flag])
    assert evaluate.run_evaluation_v2(args) == EvaluationStatus.INVALID


def test_worker_benchmark_is_uncached_and_preserves_qualification():
    judge = StubJudgeAdapter()
    result = benchmark_workers(judge, [JudgeInput("journey", item())])
    assert result["native_requests"] == 7 and result["distinct_pairs"] == 1
    assert not result["judgment_cache_used"] and not result["qualification_changed"]
    assert all(row["eligible"] for row in result["measurements"])
    assert len(result["raw_outputs"]) == 3
    with pytest.raises(ValueError):
        benchmark_workers(judge, [])
    judge.is_available = lambda: False
    with pytest.raises(ValueError, match="unavailable"):
        benchmark_workers(judge, [JudgeInput("journey", item())])


def test_benchmark_runtime_changes_or_failures_cannot_recommend_parallelism():
    judge = StubJudgeAdapter()
    judge.is_available = MagicMock(side_effect=[True, False])
    result = benchmark_workers(judge, [JudgeInput("journey", item())])
    assert (
        not result["artifact_verified_after_run"] and result["recommended_workers"] == 1
    )
    judge.is_available = lambda: True
    judge._run_inference = MagicMock(side_effect=RuntimeError("failed"))
    with pytest.raises(ValueError, match="warmup"):
        benchmark_workers(judge, [JudgeInput("journey", item())])


def test_benchmark_cli_checks_qualification_and_writes_diagnostic(
    tmp_path, monkeypatch
):
    import evaluation.judge.throughput as throughput

    monkeypatch.chdir(tmp_path)
    path = Path("evaluation/.judge_qualification_ollama.json")
    path.parent.mkdir()
    path.write_text(json.dumps({"reports": {}}))
    judge = StubJudgeAdapter()
    monkeypatch.setattr(
        throughput, "discover_ollama_judges", lambda: {"bespoke-nimble-9b": judge}
    )
    monkeypatch.setattr(throughput, "qualification_record_matches", lambda *a: False)
    with pytest.raises(ValueError, match="qualified"):
        benchmark_main(["--output", "out.json"])
    monkeypatch.setattr(throughput, "qualification_record_matches", lambda *a: True)
    cases = [case(1), case(2)]
    cases[0].golden_ids = [1]
    monkeypatch.setattr(throughput, "load_evaluation_cases", lambda **kw: cases)
    monkeypatch.setattr(throughput, "load_catalog_metadata", lambda: {})
    benchmark_main(["--output", "out.json"])
    assert json.loads(Path("out.json").read_text())["distinct_pairs"] == 1
