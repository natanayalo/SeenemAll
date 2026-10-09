"""Cheap candidate checks and a fixed development-only diagnostic selection."""

from __future__ import annotations

from typing import Any, Sequence

from evaluation.models import ComparisonGateResult, EvaluationStatus, GateCheckResult


def quick_dev_cases(cases: Sequence[Any]) -> list[Any]:
    """Select three independent families per main slice and one per small slice."""
    selected = []
    families: set[str] = set()
    for tag, count in (
        ("franchise", 3),
        ("vibe", 3),
        ("constraint", 3),
        ("entity", 3),
        ("typo", 1),
        ("multi_constraint", 1),
    ):
        for case in sorted(cases, key=lambda row: row.case_id):
            if tag in case.slice_tags and case.family_id not in families:
                selected.append(case)
                families.add(case.family_id)
                count -= 1
                if count == 0:
                    break
    return selected


def check_preflight(
    cases: Sequence[Any],
    *,
    execution_failures: int,
    unexpected_fallbacks: int,
    duplicates: bool,
    empty_outputs: int,
    hard_violations: int,
    disliked_violations: int,
    chronology_violations: int,
    missing_canonical_items: int,
    completeness: Sequence[float],
    statistical_promotion: bool,
) -> ComparisonGateResult:
    """A pass only permits grading to start; it never authorizes promotion."""
    family_count = len({case.family_id for case in cases})
    slice_counts = {
        tag: len({case.family_id for case in cases if tag in case.slice_tags})
        for tag in sorted({tag for case in cases for tag in case.slice_tags})
    }
    integrity = (
        execution_failures + unexpected_fallbacks + empty_outputs + int(duplicates)
    )
    violations = (
        hard_violations
        + disliked_violations
        + chronology_violations
        + missing_canonical_items
    )
    mean_completeness = sum(completeness) / len(completeness) if completeness else 1.0
    sample_ok = not statistical_promotion or (
        family_count >= 50
        and all(
            slice_counts.get(tag, 0) >= 10
            for tag in ("franchise", "vibe", "constraint", "entity")
        )
    )
    checks = [
        GateCheckResult(
            "preflight_execution",
            integrity == 0,
            integrity,
            0,
            "No failures, fallbacks, duplicates or empty candidate outputs.",
        ),
        GateCheckResult(
            "preflight_constraints_and_order",
            violations == 0,
            violations,
            0,
            "No hard constraint, dislike or canonical order violations.",
        ),
        GateCheckResult(
            "preflight_completeness", mean_completeness >= 0.95, mean_completeness, 0.95
        ),
        GateCheckResult(
            "preflight_sample_size",
            sample_ok,
            {"families": family_count, "slices": slice_counts},
            {"families": 50, "per_critical_slice": 10} if statistical_promotion else {},
        ),
    ]
    passed = all(check.passed for check in checks)
    reasons = [check.name for check in checks if not check.passed]
    if statistical_promotion:
        if family_count < 50:
            reasons.append(
                f"Insufficient family sample size for statistical promotion ({family_count} < 50)."
            )
        reasons.extend(
            f"Critical slice '{tag}' has fewer than 10 families ({slice_counts.get(tag, 0)} < 10)."
            for tag in ("franchise", "vibe", "constraint", "entity")
            if slice_counts.get(tag, 0) < 10
        )
    return ComparisonGateResult(
        status=(
            EvaluationStatus.INVALID
            if integrity
            else (
                EvaluationStatus.INCONCLUSIVE
                if not sample_ok
                else EvaluationStatus.PASS if passed else EvaluationStatus.FAIL
            )
        ),
        passed=passed,
        checks=checks,
        family_count=family_count,
        slice_counts=slice_counts,
        reasons=reasons,
    )
