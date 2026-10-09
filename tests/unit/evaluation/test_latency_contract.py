from unittest.mock import MagicMock

import pytest

from evaluation.latency import LatencyHarness
from evaluation.trace import EvaluationTrace


def measured_trace(enabled=True, provider="cross_encoder", latency=10.0):
    trace = EvaluationTrace(query="query", user_id="test")
    trace.timings_ms["total_latency_ms"] = latency
    trace.inference_counts = {
        "actual_inferences_performed": int(enabled),
        "cache_hits": 0,
    }
    trace.inference_providers = {
        provider: {"successful_calls": int(enabled), "scored_items": 1, "failures": 0}
    }
    return trace


@pytest.mark.parametrize(
    "baseline,candidate", [(False, True), (False, False), (True, True)]
)
def test_valid_configurations_and_abba_order(baseline, candidate):
    runner = MagicMock()
    calls = []

    def execute(query, params, **kwargs):
        calls.append(params["name"])
        assert kwargs["bypass_cache"] is True
        return [{"id": 1}], measured_trace(
            params["rerank"], params.get("rerank_provider", "cross_encoder")
        )

    runner.execute_query.side_effect = execute
    harness = LatencyHarness(runner, warmup_count=0, repetition_count=1)
    report = harness.benchmark_paired_latency(
        ["query"],
        {"name": "A", "rerank": baseline},
        {"name": "B", "rerank": candidate, "rerank_provider": "small"},
    )
    assert report["passed"]
    assert calls == list("ABBABAAB")
    assert report["actual_inferences_performed"] == 4 * (baseline + candidate)
    assert report["baseline_p50_ms"] == report["candidate_p50_ms"] == 10


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "wrong_provider",
        "cached",
        "fallback",
        "failure",
        "empty",
        "timing",
        "counter",
        "mock",
        "unexpected",
    ],
)
def test_reject_invalid_measurements(defect):
    trace = measured_trace()
    items = [{"id": 1}]
    params = {"rerank": True}
    if defect == "missing":
        trace.inference_counts["actual_inferences_performed"] = 0
    elif defect == "wrong_provider":
        trace.inference_providers = {"small": {"successful_calls": 1}}
    elif defect == "cached":
        trace.inference_counts["cache_hits"] = 1
    elif defect == "fallback":
        trace.fallbacks = ["baseline ordering"]
    elif defect == "failure":
        trace.inference_providers["cross_encoder"]["failures"] = 1
    elif defect == "empty":
        items = []
    elif defect == "timing":
        trace.timings_ms = {}
    elif defect == "counter":
        trace.inference_counts["actual_inferences_performed"] = True
    elif defect == "mock":
        trace.inference_counts = MagicMock()
    elif defect == "unexpected":
        params["rerank"] = False
    runner = MagicMock()
    runner.execute_query.return_value = items, trace
    report = LatencyHarness(
        runner, warmup_count=0, repetition_count=1
    ).benchmark_paired_latency(["query"], params, params)
    assert not report["passed"] and report["errors"]


def test_empty_queries_and_invalid_repetition_count():
    with pytest.raises(ValueError):
        LatencyHarness(MagicMock(), repetition_count=0)
    assert not LatencyHarness(MagicMock()).benchmark_paired_latency([], {}, {})[
        "passed"
    ]
