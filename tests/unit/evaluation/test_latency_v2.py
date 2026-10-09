"""Unit tests for recommendation latency harness and judge throughput reporting."""

from unittest.mock import MagicMock, patch

from evaluation.judge.stub import StubJudgeAdapter
from evaluation.latency import LatencyHarness, benchmark_judge_throughput
from evaluation.models import ItemEvidence, JudgeInput, TypedId
from evaluation.runner import EvaluationRunner


def test_latency_harness_abba_baab_and_gates():
    runner = EvaluationRunner(in_process=True)
    harness = LatencyHarness(runner=runner, warmup_count=1, repetition_count=1)

    mock_runner = MagicMock()
    from tests.unit.evaluation.test_latency_contract import measured_trace

    mock_runner.execute_query.return_value = ([{"tmdb_id": 1}], measured_trace())
    harness.runner = mock_runner

    curr_time = 0.0

    def fake_perf_counter():
        nonlocal curr_time
        curr_time += 0.010
        return curr_time

    with patch("time.perf_counter", side_effect=fake_perf_counter):
        report = harness.benchmark_paired_latency(
            queries=["q1", "q2"],
            baseline_params={"param": "base"},
            candidate_params={"param": "cand"},
        )
    assert report["passed"] is True
    assert report["median_paired_ratio"] <= 1.10
    assert report["p95_ratio"] <= 1.15
    assert report["total_measurements"] > 0


def test_latency_harness_fails_on_request_error():
    runner = EvaluationRunner(in_process=True)
    harness = LatencyHarness(runner=runner, warmup_count=1, repetition_count=1)

    mock_runner = MagicMock()
    mock_trace = MagicMock()
    mock_trace.errors = ["HTTP 500 internal server error"]
    mock_runner.execute_query.return_value = ([], mock_trace)
    harness.runner = mock_runner

    report = harness.benchmark_paired_latency(
        queries=["q1"],
        baseline_params={"param": "base"},
        candidate_params={"param": "cand"},
    )
    assert report["passed"] is False
    assert report["error_count"] > 0


def test_judge_throughput_reporting():
    judge = StubJudgeAdapter()
    ev1 = ItemEvidence(typed_id=TypedId("movie", 1), title="M1", synopsis="S1")
    ev2 = ItemEvidence(typed_id=TypedId("movie", 2), title="M2", synopsis="S2")
    sample_inputs = [
        JudgeInput(query="test query 1", evidence=ev1),
        JudgeInput(query="test query 2", evidence=ev2),
    ]

    report = benchmark_judge_throughput(judge, sample_inputs)
    assert report["sample_count"] == 2
    assert report["first_pass_items_per_sec"] > 0.0
    assert report["second_pass_items_per_sec"] > 0.0
