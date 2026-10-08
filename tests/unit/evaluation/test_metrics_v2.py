"""Unit tests for v2 metrics, gain transformations, bootstrapping, and comparison gates."""

import pytest

from evaluation.metrics import (
    bootstrap_paired_family_deltas,
    calculate_average_precision,
    calculate_completeness,
    calculate_coverage,
    calculate_fill_rate,
    calculate_known_positive_recall_at_k,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    check_for_duplicates,
    compute_gain,
    evaluate_comparison_gates,
)
from evaluation.models import EvaluationStatus, GainMode


def test_gain_modes():
    # Graded exponential: 2^r - 1
    assert compute_gain(0.0, GainMode.GRADED_EXPONENTIAL) == 0.0
    assert compute_gain(1.0, GainMode.GRADED_EXPONENTIAL) == 1.0
    assert compute_gain(2.0, GainMode.GRADED_EXPONENTIAL) == 3.0
    assert compute_gain(3.0, GainMode.GRADED_EXPONENTIAL) == 7.0

    # Continuous identity: r
    assert compute_gain(0.0, GainMode.CONTINUOUS_IDENTITY) == 0.0
    assert compute_gain(0.85, GainMode.CONTINUOUS_IDENTITY) == 0.85
    assert compute_gain(1.0, GainMode.CONTINUOUS_IDENTITY) == 1.0


def test_ndcg_at_k_both_gain_modes():
    qrels = {
        "movie:1": 3.0,
        "movie:2": 2.0,
        "movie:3": 1.0,
        "movie:4": 0.0,
    }
    # Perfect order
    rec_perfect = ["movie:1", "movie:2", "movie:3"]
    ndcg_exp = calculate_ndcg_at_k(
        rec_perfect, qrels, k=3, gain_mode=GainMode.GRADED_EXPONENTIAL
    )
    assert pytest.approx(ndcg_exp, abs=1e-4) == 1.0

    ndcg_ident = calculate_ndcg_at_k(
        rec_perfect, qrels, k=3, gain_mode=GainMode.CONTINUOUS_IDENTITY
    )
    assert pytest.approx(ndcg_ident, abs=1e-4) == 1.0

    # Inverted order
    rec_inv = ["movie:3", "movie:2", "movie:1"]
    ndcg_inv = calculate_ndcg_at_k(
        rec_inv, qrels, k=3, gain_mode=GainMode.GRADED_EXPONENTIAL
    )
    assert 0.0 < ndcg_inv < 1.0

    # Empty cases
    assert calculate_ndcg_at_k([], qrels, k=3) == 0.0
    assert calculate_ndcg_at_k(rec_perfect, {}, k=3) == 0.0
    assert calculate_ndcg_at_k(rec_perfect, qrels, k=0) == 0.0


def test_precision_map_and_known_positive_recall():
    qrels = {
        "movie:10": 3.0,  # positive
        "movie:20": 2.0,  # positive
        "movie:30": 1.0,  # non-positive
        "movie:40": 0.0,  # non-positive
        "movie:50": 3.0,  # positive (not in rec list)
    }
    rec = ["movie:10", "movie:30", "movie:20", "movie:40"]

    # Precision@2: 1/2 = 0.5; Precision@4: 2/4 = 0.5
    assert calculate_precision_at_k(rec, qrels, k=2) == 0.5
    assert calculate_precision_at_k(rec, qrels, k=4) == 0.5

    # Recall@4: 2 positives out of 3 total in qrels = 2/3
    assert (
        pytest.approx(calculate_known_positive_recall_at_k(rec, qrels, k=4), abs=1e-4)
        == 2 / 3
    )

    # MAP: hits at rank 1 (P=1/1) and rank 3 (P=2/3) -> (1 + 2/3) / 3 = 5/9
    assert (
        pytest.approx(calculate_average_precision(rec, qrels), abs=1e-4)
        == (1.0 + 2 / 3) / 3
    )


def test_coverage_fill_rate_and_completeness():
    qrels = {"movie:1": 2.0, "movie:2": 3.0}
    rec = ["movie:1", "movie:2", "movie:99"]

    # Coverage: 2 judged out of 3 unique = 2/3
    assert pytest.approx(calculate_coverage(rec, qrels, k=3), abs=1e-4) == 2 / 3
    assert calculate_coverage([], qrels, k=3) == 0.0

    # Fill rate: 3 items for K=5 -> 3/5
    assert calculate_fill_rate(rec, k=5) == 0.6

    # Completeness: 3 unique items when catalog has 4 eligible -> 3/3 = 1.0 (min(3, 4) = 3)
    assert calculate_completeness(rec, eligible_catalog_count=3, k=3) == 1.0


def test_duplicates_detection():
    assert check_for_duplicates(["movie:1", "movie:2", "movie:1"], k=3) is True
    assert check_for_duplicates(["movie:1", "movie:2", "movie:3"], k=3) is False


def test_bootstrap_paired_family_deltas():
    deltas = [0.05, 0.02, -0.01, 0.04, 0.03, -0.02, 0.01, 0.06] * 10
    mean_val, lower_ci, upper_ci = bootstrap_paired_family_deltas(
        deltas, n_resamples=1000, seed=42
    )
    assert lower_ci <= mean_val <= upper_ci
    assert -0.05 < mean_val < 0.10


def test_comparison_gates_pass_and_fail():
    # 1. Successful passing comparison
    fam_deltas = {f"fam_{i}": 0.02 for i in range(50)}
    slices = {
        "vibe": [0.02] * 12,
        "franchise": [0.02] * 12,
        "constraint": [0.02] * 12,
        "entity": [0.02] * 12,
    }
    pass_res = evaluate_comparison_gates(
        family_ndcg_deltas=fam_deltas,
        slice_family_deltas=slices,
        baseline_recall_100=0.80,
        candidate_recall_100=0.81,
        hard_constraint_violations=0,
        disliked_violations=0,
        exploratory_coverages=[0.95] * 50,
        authoritative_unresolved_count=0,
        execution_failures=0,
        unexpected_fallbacks=0,
        duplicate_outputs_detected=False,
        is_statistical_promotion=True,
    )
    assert pass_res.passed is True
    assert pass_res.status == EvaluationStatus.PASS

    # 2. Hard constraint violation -> FAIL
    fail_res = evaluate_comparison_gates(
        family_ndcg_deltas=fam_deltas,
        slice_family_deltas=slices,
        baseline_recall_100=0.80,
        candidate_recall_100=0.81,
        hard_constraint_violations=1,
        disliked_violations=0,
        exploratory_coverages=[0.95] * 50,
        authoritative_unresolved_count=0,
        execution_failures=0,
        unexpected_fallbacks=0,
        duplicate_outputs_detected=False,
        is_statistical_promotion=True,
    )
    assert fail_res.passed is False
    assert fail_res.status == EvaluationStatus.FAIL

    # 3. Execution failure / duplicates -> INVALID
    inv_res = evaluate_comparison_gates(
        family_ndcg_deltas=fam_deltas,
        slice_family_deltas=slices,
        baseline_recall_100=0.80,
        candidate_recall_100=0.81,
        hard_constraint_violations=0,
        disliked_violations=0,
        exploratory_coverages=[0.95] * 50,
        authoritative_unresolved_count=0,
        execution_failures=1,
        unexpected_fallbacks=0,
        duplicate_outputs_detected=True,
        is_statistical_promotion=True,
    )
    assert inv_res.status == EvaluationStatus.INVALID

    # 4. Unresolved judgments in top 10 -> INCONCLUSIVE
    inconc_res = evaluate_comparison_gates(
        family_ndcg_deltas=fam_deltas,
        slice_family_deltas=slices,
        baseline_recall_100=0.80,
        candidate_recall_100=0.81,
        hard_constraint_violations=0,
        disliked_violations=0,
        exploratory_coverages=[0.95] * 50,
        authoritative_unresolved_count=2,  # unresolved items
        execution_failures=0,
        unexpected_fallbacks=0,
        duplicate_outputs_detected=False,
        is_statistical_promotion=True,
    )
    assert inconc_res.status == EvaluationStatus.INCONCLUSIVE
