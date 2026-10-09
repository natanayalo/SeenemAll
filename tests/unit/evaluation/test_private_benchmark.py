"""Unit tests for Private Promotion Benchmark harness (Milestone 3)."""

from pathlib import Path
from evaluation.private_benchmark import PrivateBenchmarkHarness


def test_private_benchmark_harness_init(tmp_path: Path):
    harness = PrivateBenchmarkHarness(benchmark_dir=tmp_path)
    assert harness.benchmark_dir == tmp_path
    assert harness._ledger["attempt_count"] == 0
    assert not harness._ledger["refresh_required"]


def test_private_benchmark_submission_and_redaction(tmp_path: Path):
    harness = PrivateBenchmarkHarness(benchmark_dir=tmp_path)
    gate_result = {
        "status": "PASS",
        "exit_code": 0,
        "passed": True,
        "checks": [{"name": "nDCG Gate", "passed": True, "details": "lift = +0.02"}],
        "reasons": ["All gates passed"],
        "confidential_raw_queries": ["private query 1", "private query 2"],
        "confidential_labels": [1, 2, 3],
    }

    export = harness.submit_candidate("candidate_v1.0_commit123", gate_result)
    assert export["status"] == "PASS"
    assert export["passed"] is True
    assert export["submission_index"] == 1
    assert export["refresh_required"] is False
    # Verify confidential data is strictly not present in export
    assert "confidential_raw_queries" not in export
    assert "confidential_labels" not in export
    assert len(export["summary_checks"]) == 1


def test_private_benchmark_reject_duplicate(tmp_path: Path):
    harness = PrivateBenchmarkHarness(benchmark_dir=tmp_path)
    gate_result = {"status": "PASS", "exit_code": 0, "passed": True}

    res1 = harness.submit_candidate("candidate_unique_id", gate_result)
    assert res1["status"] == "PASS"

    res2 = harness.submit_candidate("candidate_unique_id", gate_result)
    assert res2["status"] == "REJECTED"
    assert "advance candidate revision" in res2["error"]


def test_private_benchmark_refresh_trigger_after_ten(tmp_path: Path):
    harness = PrivateBenchmarkHarness(benchmark_dir=tmp_path)
    gate_result = {"status": "PASS", "exit_code": 0, "passed": True}

    for i in range(9):
        export = harness.submit_candidate(f"candidate_rev_{i}", gate_result)
        assert export["refresh_required"] is False

    # 10th submission triggers refresh
    export_10 = harness.submit_candidate("candidate_rev_9", gate_result)
    assert export_10["refresh_required"] is True

    # Check persistence in a new instance with the same dir
    harness2 = PrivateBenchmarkHarness(benchmark_dir=tmp_path)
    assert harness2._ledger["attempt_count"] == 10
    assert harness2._ledger["refresh_required"] is True
