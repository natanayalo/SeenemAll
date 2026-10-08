"""Evaluation metrics, gain transformations, bootstrapping, and comparison gates."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from evaluation.models import (
    ComparisonGateResult,
    EvaluationStatus,
    GainMode,
    GateCheckResult,
    RubricGrade,
    TypedId,
)


def compute_gain(relevance: float, gain_mode: GainMode) -> float:
    """Transform raw relevance grade into metric gain.

    - graded_exponential: 2^relevance - 1
    - continuous_identity: relevance
    """
    if gain_mode == GainMode.GRADED_EXPONENTIAL:
        return float(math.pow(2.0, relevance) - 1.0)
    if gain_mode == GainMode.CONTINUOUS_IDENTITY:
        return float(relevance)
    raise ValueError(f"Unsupported GainMode: {gain_mode}")


def calculate_dcg_at_k(
    relevance_scores: Sequence[float],
    k: int,
    gain_mode: GainMode = GainMode.GRADED_EXPONENTIAL,
) -> float:
    """Compute Discounted Cumulative Gain up to rank K."""
    if k <= 0 or not relevance_scores:
        return 0.0
    dcg = 0.0
    for idx, rel in enumerate(relevance_scores[:k]):
        if rel > 0:
            gain = compute_gain(rel, gain_mode)
            # Rank is 1-indexed, discount is log2(idx + 2)
            dcg += gain / math.log2(idx + 2)
    return dcg


def calculate_ndcg_at_k(
    recommended_items: Sequence[Any],
    qrels: Dict[str, float],
    k: int = 10,
    gain_mode: GainMode = GainMode.GRADED_EXPONENTIAL,
    unjudged_as_zero: bool = True,
    all_eligible_qrels: Optional[Dict[str, float]] = None,
) -> float:
    """Compute Normalized Discounted Cumulative Gain (nDCG@K).

    Uses identical qrels and gain transformation consistently for DCG and IDCG.
    If unjudged_as_zero is True, unjudged items yield 0.0 gain.
    """
    if k <= 0 or not recommended_items:
        return 0.0

    # Extract typed IDs for recommended items
    typed_keys: List[str] = []
    for it in recommended_items:
        try:
            typed_keys.append(str(TypedId.parse(it)))
        except Exception:
            typed_keys.append(str(it))

    # Actual gains
    relevance_scores: List[float] = []
    for key in typed_keys[:k]:
        if key in qrels:
            relevance_scores.append(float(qrels[key]))
        elif unjudged_as_zero:
            relevance_scores.append(0.0)
        else:
            relevance_scores.append(0.0)

    dcg = calculate_dcg_at_k(relevance_scores, k=k, gain_mode=gain_mode)

    # Ideal DCG: pool of best known relevant items
    universe = all_eligible_qrels if all_eligible_qrels is not None else qrels
    all_gains = sorted([float(r) for r in universe.values() if r > 0], reverse=True)
    if not all_gains:
        return 0.0

    idcg = calculate_dcg_at_k(all_gains, k=k, gain_mode=gain_mode)
    return (dcg / idcg) if idcg > 0.0 else 0.0


def calculate_precision_at_k(
    recommended_items: Sequence[Any],
    qrels: Dict[str, float],
    k: int = 10,
) -> float:
    """Compute Precision@K where grades >= 2 are considered positive hits."""
    if k <= 0 or not recommended_items:
        return 0.0

    hits = 0
    for it in recommended_items[:k]:
        key = str(TypedId.parse(it))
        grade = qrels.get(key, 0.0)
        if RubricGrade.is_positive(int(grade)):
            hits += 1
    return hits / k


def calculate_average_precision(
    recommended_items: Sequence[Any],
    qrels: Dict[str, float],
    total_positives: Optional[int] = None,
) -> float:
    """Compute Average Precision where grades >= 2 are considered positive hits."""
    if not recommended_items:
        return 0.0

    positive_keys = {k for k, g in qrels.items() if RubricGrade.is_positive(int(g))}
    denom = total_positives if total_positives is not None else len(positive_keys)
    if denom <= 0:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for idx, it in enumerate(recommended_items, start=1):
        key = str(TypedId.parse(it))
        if key in positive_keys:
            hits += 1
            precision_sum += hits / idx
    return precision_sum / denom


def calculate_known_positive_recall_at_k(
    recommended_items: Sequence[Any],
    qrels: Dict[str, float],
    k: int = 50,
    total_positives: Optional[int] = None,
) -> float:
    """Compute Known-Positive Recall@K (e.g. Recall@50 or Recall@100)."""
    if k <= 0 or not recommended_items:
        return 0.0

    positive_keys = {k for k, g in qrels.items() if RubricGrade.is_positive(int(g))}
    denom = total_positives if total_positives is not None else len(positive_keys)
    if denom <= 0:
        return 0.0

    seen_positives: Set[str] = set()
    for it in recommended_items[:k]:
        key = str(TypedId.parse(it))
        if key in positive_keys:
            seen_positives.add(key)
    return len(seen_positives) / denom


def calculate_coverage(
    recommended_items: Sequence[Any],
    qrels: Dict[str, float],
    k: int = 10,
) -> float:
    """Compute judgment coverage in top-K: judged unique top-K / returned unique top-K.

    Returns 0.0 if empty.
    """
    top_items = [str(TypedId.parse(it)) for it in recommended_items[:k]]
    if not top_items:
        return 0.0

    unique_top = list(dict.fromkeys(top_items))
    if not unique_top:
        return 0.0

    judged_count = sum(1 for item in unique_top if item in qrels)
    return judged_count / len(unique_top)


def calculate_fill_rate(
    recommended_items: Sequence[Any],
    k: int = 10,
) -> float:
    """Compute fill rate: min(unique_result_count, K) / K."""
    if k <= 0:
        return 0.0
    unique_items = {str(TypedId.parse(it)) for it in recommended_items}
    return min(len(unique_items), k) / k


def calculate_completeness(
    recommended_items: Sequence[Any],
    eligible_catalog_count: int,
    k: int = 10,
) -> float:
    """Compute completeness: min(K, eligible_catalog_count) / K."""
    if k <= 0 or eligible_catalog_count <= 0:
        return 0.0
    target_count = min(k, eligible_catalog_count)
    unique_items = {str(TypedId.parse(it)) for it in recommended_items[:k]}
    return len(unique_items) / target_count


def check_for_duplicates(recommended_items: Sequence[Any], k: int = 10) -> bool:
    """Check if top-K recommendations contain duplicate items.

    Duplicate output invalidates an authoritative run.
    """
    keys = [str(TypedId.parse(it)) for it in recommended_items[:k]]
    return len(keys) != len(set(keys))


def bootstrap_paired_family_deltas(
    family_deltas: Sequence[float],
    n_resamples: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Compute mean delta and bootstrap confidence interval [lower_ci, upper_ci].

    Seed is fixed to 42 for deterministic reproducibility.
    """
    if not family_deltas:
        return (0.0, 0.0, 0.0)

    arr = np.asarray(family_deltas, dtype=float)
    mean_val = float(np.mean(arr))
    if len(arr) == 1:
        return (mean_val, mean_val, mean_val)

    rng = np.random.default_rng(seed)
    n = len(arr)
    # Generate bootstrap samples
    indices = rng.integers(0, n, size=(n_resamples, n))
    bootstrap_means = np.mean(arr[indices], axis=1)

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    lower_ci = float(np.percentile(bootstrap_means, lower_pct))
    upper_ci = float(np.percentile(bootstrap_means, upper_pct))
    return (mean_val, lower_ci, upper_ci)


def evaluate_comparison_gates(
    family_ndcg_deltas: Dict[str, float],
    slice_family_deltas: Dict[str, List[float]],
    baseline_recall_100: float,
    candidate_recall_100: float,
    hard_constraint_violations: int,
    disliked_violations: int,
    exploratory_coverages: List[float],
    authoritative_unresolved_count: int,
    execution_failures: int,
    unexpected_fallbacks: int,
    duplicate_outputs_detected: bool,
    is_statistical_promotion: bool = False,
    critical_slices: Sequence[str] = ("vibe", "franchise", "constraint", "entity"),
    completeness_scores: Optional[Sequence[float]] = None,
    chronology_violations: int = 0,
    missing_canonical_items: int = 0,
    empty_output_cases: int = 0,
) -> ComparisonGateResult:
    """Evaluate candidate against baseline across all comparison gates.

    Enforces:
      - Execution: 0 failed cases, 0 unexpected fallbacks, no duplicate output, valid non-empty output
      - Judgment coverage: No unresolved top-K or recall references (otherwise INCONCLUSIVE)
      - Sample sizes: >= 50 independent families (statistical promotion), >= 10 per critical slice
      - Non-regression: Paired nDCG@10 delta CI lower bound >= -0.01
      - Critical slices: Mean delta >= -0.03 for each slice
      - Merged known-positive Recall@100: Decline <= 0.01 absolute
      - Hard constraints and known dislikes: 0 violations
      - Completeness: Mean candidate completeness >= 0.95
      - Chronology: 0 canonical sequence inversion violations
      - Exploratory coverage: >= 0.90 for each nonempty case
    """
    checks: List[GateCheckResult] = []
    reasons: List[str] = []

    # 1. Execution & Validity Check
    exec_passed = (
        execution_failures == 0
        and unexpected_fallbacks == 0
        and not duplicate_outputs_detected
        and empty_output_cases == 0
    )
    checks.append(
        GateCheckResult(
            name="execution_integrity",
            passed=exec_passed,
            observed={
                "failures": execution_failures,
                "fallbacks": unexpected_fallbacks,
                "duplicates": duplicate_outputs_detected,
                "empty_cases": empty_output_cases,
            },
            threshold={
                "failures": 0,
                "fallbacks": 0,
                "duplicates": False,
                "empty_cases": 0,
            },
            details="Execution must have zero failures, zero unexpected fallbacks, no empty outputs, and no duplicate items.",
        )
    )
    if not exec_passed:
        reasons.append(
            "Execution integrity check failed (invalid run or empty candidate output)."
        )
        return ComparisonGateResult(
            status=EvaluationStatus.INVALID,
            passed=False,
            checks=checks,
            family_count=len(family_ndcg_deltas),
            slice_counts={k: len(v) for k, v in slice_family_deltas.items()},
            reasons=reasons,
        )

    # 2. Authoritative Judgment Coverage
    judg_passed = authoritative_unresolved_count == 0
    checks.append(
        GateCheckResult(
            name="authoritative_judgment_coverage",
            passed=judg_passed,
            observed=authoritative_unresolved_count,
            threshold=0,
            details="No unresolved items permitted in either system's top-K or declared recall references.",
        )
    )
    if not judg_passed:
        reasons.append(
            f"{authoritative_unresolved_count} unresolved items in top ten; judgment inconclusive."
        )

    # 3. Sample Size Sufficiency
    total_families = len(family_ndcg_deltas)
    slice_counts = {k: len(v) for k, v in slice_family_deltas.items()}
    sample_size_passed = True
    if is_statistical_promotion:
        if total_families < 50:
            sample_size_passed = False
            reasons.append(
                f"Insufficient family sample size for statistical promotion ({total_families} < 50)."
            )
        for sl in critical_slices:
            cnt = slice_counts.get(sl, 0)
            if cnt < 10:
                sample_size_passed = False
                reasons.append(
                    f"Critical slice '{sl}' has fewer than 10 families ({cnt} < 10)."
                )

    checks.append(
        GateCheckResult(
            name="sample_size_sufficiency",
            passed=sample_size_passed,
            observed={"total_families": total_families, "slice_counts": slice_counts},
            threshold={
                "min_total": 50 if is_statistical_promotion else 1,
                "min_slice": 10 if is_statistical_promotion else 0,
            },
            details="Statistical promotion requires >= 50 independent families and >= 10 per critical slice.",
        )
    )

    # 4. Bootstrap CI Non-regression
    deltas_list = list(family_ndcg_deltas.values())
    mean_delta, ci_lower, ci_upper = bootstrap_paired_family_deltas(
        deltas_list, n_resamples=10000, seed=42
    )

    non_reg_passed = ci_lower >= -0.01
    checks.append(
        GateCheckResult(
            name="ndcg_non_regression_gate",
            passed=non_reg_passed,
            observed={
                "mean_delta": round(mean_delta, 4),
                "ci_lower": round(ci_lower, 4),
                "ci_upper": round(ci_upper, 4),
            },
            threshold={"ci_lower_bound": -0.01},
            details="Paired nDCG@10 delta CI lower bound must be >= -0.01.",
        )
    )
    if not non_reg_passed:
        reasons.append(
            f"nDCG@10 non-regression failed (CI lower {ci_lower:.4f} < -0.01)."
        )

    # 5. Critical Slices Non-regression
    slice_passed = True
    for sl in critical_slices:
        s_deltas = slice_family_deltas.get(sl, [])
        if s_deltas:
            s_mean = float(np.mean(s_deltas))
            if s_mean < -0.03:
                slice_passed = False
                reasons.append(
                    f"Critical slice '{sl}' mean delta regressed ({s_mean:.4f} < -0.03)."
                )
    checks.append(
        GateCheckResult(
            name="critical_slices_gate",
            passed=slice_passed,
            observed={
                sl: round(float(np.mean(slice_family_deltas[sl])), 4)
                for sl in critical_slices
                if sl in slice_family_deltas and slice_family_deltas[sl]
            },
            threshold={"min_mean_delta": -0.03},
            details="Critical product slices mean delta must be >= -0.03.",
        )
    )

    # 6. Known-Positive Recall@100 Decline & Valid Output
    recall_decline = baseline_recall_100 - candidate_recall_100
    valid_recall = not (baseline_recall_100 > 0.0 and candidate_recall_100 == 0.0)
    recall_passed = (recall_decline <= 0.01) and valid_recall
    checks.append(
        GateCheckResult(
            name="known_positive_recall_100_gate",
            passed=recall_passed,
            observed={
                "recall_decline": round(recall_decline, 4),
                "baseline_recall": round(baseline_recall_100, 4),
                "candidate_recall": round(candidate_recall_100, 4),
            },
            threshold=0.01,
            details="Merged known-positive Recall@100 decline must be <= 0.01 absolute, with non-zero output.",
        )
    )
    if not recall_passed:
        reasons.append(
            f"Known-positive Recall@100 declined by {recall_decline:.4f} (> 0.01) or returned zero recall."
        )

    # 7. Hard Constraints & Known Dislikes (Zero Violations)
    total_violations = hard_constraint_violations + disliked_violations
    viol_passed = total_violations == 0
    checks.append(
        GateCheckResult(
            name="zero_constraint_violations_gate",
            passed=viol_passed,
            observed={
                "hard_constraints": hard_constraint_violations,
                "dislikes": disliked_violations,
            },
            threshold={"total_violations": 0},
            details="Hard constraints and known dislikes must have zero violations.",
        )
    )
    if not viol_passed:
        reasons.append(
            f"Hard constraint or dislike violations detected ({total_violations} violations)."
        )

    # 8. Completeness Gate (>= 0.95)
    mean_comp = float(np.mean(completeness_scores)) if completeness_scores else 1.0
    comp_passed = mean_comp >= 0.95
    checks.append(
        GateCheckResult(
            name="completeness_gate",
            passed=comp_passed,
            observed=round(mean_comp, 4),
            threshold=0.95,
            details="Candidate completeness must be >= 0.95 across queries.",
        )
    )
    if not comp_passed:
        reasons.append(f"Completeness fell below 0.95 ({mean_comp:.4f} < 0.95).")

    # 9. Canonical Sequence Chronology Gate (0 inversions and 0 missing items)
    chrono_passed = (chronology_violations == 0) and (missing_canonical_items == 0)
    checks.append(
        GateCheckResult(
            name="canonical_sequence_chronology_gate",
            passed=chrono_passed,
            observed={
                "inversions": chronology_violations,
                "missing_items": missing_canonical_items,
            },
            threshold={"inversions": 0, "missing_items": 0},
            details="Canonical sequences must have zero chronological inversions and zero missing prefix items.",
        )
    )
    if not chrono_passed:
        reasons.append(
            f"Canonical chronology violations detected ({chronology_violations} inversions, {missing_canonical_items} missing items)."
        )

    # 10. Exploratory Coverage (>= 0.90)
    exp_passed = (
        all(cov >= 0.90 for cov in exploratory_coverages)
        if exploratory_coverages
        else True
    )
    checks.append(
        GateCheckResult(
            name="exploratory_coverage_gate",
            passed=exp_passed,
            observed=(
                round(float(np.mean(exploratory_coverages)), 4)
                if exploratory_coverages
                else 1.0
            ),
            threshold=0.90,
            details="Exploratory coverage must be >= 0.90 for each nonempty case.",
        )
    )
    if not exp_passed:
        reasons.append("Exploratory coverage fell below 0.90 for one or more cases.")

    # Determine final status
    if not judg_passed or not sample_size_passed:
        final_status = EvaluationStatus.INCONCLUSIVE
        overall_passed = False
    elif all(c.passed for c in checks):
        final_status = EvaluationStatus.PASS
        overall_passed = True
    else:
        final_status = EvaluationStatus.FAIL
        overall_passed = False

    return ComparisonGateResult(
        status=final_status,
        passed=overall_passed,
        checks=checks,
        family_count=total_families,
        slice_counts=slice_counts,
        reasons=reasons,
    )
