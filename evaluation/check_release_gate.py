"""Offline release-gate checker for recommendation evaluation summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_summary(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_metric_rows(path: Path, scenario: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if str(row.get("params_name", "")) == scenario]

    by_case: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("eval_case_id", "")).strip()
        if not case_id:
            raise ValueError(
                "CSV is missing required 'eval_case_id'. "
                "Re-run baseline/candidate with the current evaluator."
            )
        by_case[case_id] = row
    return by_case


def _parse_slice_tags(row: Dict[str, Any]) -> set[str]:
    raw = str(row.get("slice_tags", ""))
    return {chunk.strip() for chunk in raw.split("|") if chunk.strip()}


def _safe_float(row: Dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _bootstrap_mean_ci(values: List[float], *, samples: int, seed: int) -> Dict[str, float]:
    if not values:
        return {"low": 0.0, "high": 0.0}
    if len(values) == 1:
        singleton = float(values[0])
        return {"low": singleton, "high": singleton}
    rng = random.Random(seed)
    n = len(values)
    means: List[float] = []
    for _ in range(samples):
        sampled = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sampled) / n)
    means.sort()
    low_idx = max(0, int(math.floor((len(means) - 1) * 0.025)))
    high_idx = min(len(means) - 1, int(math.ceil((len(means) - 1) * 0.975)))
    return {"low": float(means[low_idx]), "high": float(means[high_idx])}


def _paired_deltas(
    *,
    baseline_rows: Dict[str, Dict[str, Any]],
    candidate_rows: Dict[str, Dict[str, Any]],
    metric_key: str,
    slice_name: str | None = None,
) -> List[float]:
    deltas: List[float] = []
    common_case_ids = sorted(set(baseline_rows.keys()) & set(candidate_rows.keys()))
    for case_id in common_case_ids:
        baseline_row = baseline_rows[case_id]
        candidate_row = candidate_rows[case_id]
        if slice_name:
            base_tags = _parse_slice_tags(baseline_row)
            cand_tags = _parse_slice_tags(candidate_row)
            if slice_name not in base_tags or slice_name not in cand_tags:
                continue
        deltas.append(_safe_float(candidate_row, metric_key) - _safe_float(baseline_row, metric_key))
    return deltas


def _paired_significance_check(
    *,
    scope: str,
    metric_name: str,
    allowed_drop: float,
    deltas: List[float],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Tuple[bool, str, Dict[str, Any]]:
    if not deltas:
        message = f"[SKIP] {scope} {metric_name}: no paired rows available."
        return True, message, {"count": 0}

    observed_mean = sum(deltas) / len(deltas)
    ci95 = _bootstrap_mean_ci(
        deltas, samples=max(int(bootstrap_samples), 1), seed=int(bootstrap_seed)
    )
    # Fail only when even optimistic bound is below allowed threshold.
    passed = not (observed_mean < -allowed_drop and ci95["high"] < -allowed_drop)
    status = "PASS" if passed else "FAIL"
    message = (
        f"[{status}] {scope} {metric_name} paired delta: "
        f"mean={observed_mean:+.4f} ci95=[{ci95['low']:+.4f}, {ci95['high']:+.4f}] "
        f"allowed_drop={allowed_drop:.4f} n={len(deltas)}"
    )
    return passed, message, {"count": len(deltas), "mean_delta": observed_mean, "ci95": ci95}


def _find_metric_key(payload: Dict[str, Any], prefix: str) -> str:
    for key, value in payload.items():
        if key.startswith(prefix) and not key.endswith("_ci95") and isinstance(
            value, (float, int)
        ):
            return key
    raise KeyError(f"Could not find metric starting with '{prefix}' in payload.")


def _check_metric(
    *,
    scope: str,
    metric_name: str,
    candidate_value: float,
    baseline_value: float,
    allowed_drop: float,
) -> Tuple[bool, str]:
    diff = candidate_value - baseline_value
    passed = diff >= -allowed_drop
    status = "PASS" if passed else "FAIL"
    message = (
        f"[{status}] {scope} {metric_name}: "
        f"candidate={candidate_value:.4f} baseline={baseline_value:.4f} "
        f"delta={diff:+.4f} allowed_drop={allowed_drop:.4f}"
    )
    return passed, message


def main() -> None:
    parser = argparse.ArgumentParser(description="Check offline release gate.")
    parser.add_argument(
        "--baseline-summary",
        required=True,
        help="Path to baseline evaluation_summary JSON.",
    )
    parser.add_argument(
        "--candidate-summary",
        required=True,
        help="Path to candidate evaluation_summary JSON.",
    )
    parser.add_argument(
        "--baseline-scenario",
        default="default",
        help="Scenario key in baseline summary (default: default).",
    )
    parser.add_argument(
        "--candidate-scenario",
        default="default",
        help="Scenario key in candidate summary (default: default).",
    )
    parser.add_argument(
        "--max-drop-map",
        type=float,
        default=0.02,
        help="Maximum allowed MAP drop for overall metrics.",
    )
    parser.add_argument(
        "--max-drop-ndcg",
        type=float,
        default=0.02,
        help="Maximum allowed nDCG drop for overall metrics.",
    )
    parser.add_argument(
        "--max-drop-hit-rate",
        type=float,
        default=0.03,
        help="Maximum allowed HitRate drop for overall metrics.",
    )
    parser.add_argument(
        "--slice-max-drop-map",
        type=float,
        default=0.05,
        help="Maximum allowed MAP drop per critical slice.",
    )
    parser.add_argument(
        "--slice-max-drop-ndcg",
        type=float,
        default=0.05,
        help="Maximum allowed nDCG drop per critical slice.",
    )
    parser.add_argument(
        "--slice-max-drop-hit-rate",
        type=float,
        default=0.07,
        help="Maximum allowed HitRate drop per critical slice.",
    )
    parser.add_argument(
        "--critical-slices",
        default="cold_start,no_query,provider_filtered,maturity_capped,runtime_capped,movie,tv,query,warm_start",
        help="Comma-separated slice names to enforce.",
    )
    parser.add_argument(
        "--min-slice-count",
        type=float,
        default=1.0,
        help="Skip slice checks if either baseline/candidate count is below this.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path for detailed gate report JSON.",
    )
    parser.add_argument(
        "--baseline-csv",
        default="",
        help="Optional baseline detailed CSV for paired-significance checks.",
    )
    parser.add_argument(
        "--candidate-csv",
        default="",
        help="Optional candidate detailed CSV for paired-significance checks.",
    )
    parser.add_argument(
        "--paired-bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap samples used for paired-significance CI checks.",
    )
    parser.add_argument(
        "--paired-bootstrap-seed",
        type=int,
        default=17,
        help="Seed used for paired-significance CI checks.",
    )
    parser.add_argument(
        "--significance-mode",
        choices=("off", "strict"),
        default="off",
        help="Use paired-significance checks when CSV inputs are provided.",
    )
    args = parser.parse_args()

    baseline_summary = _load_summary(Path(args.baseline_summary))
    candidate_summary = _load_summary(Path(args.candidate_summary))

    if args.baseline_scenario not in baseline_summary:
        raise KeyError(f"Scenario '{args.baseline_scenario}' missing from baseline.")
    if args.candidate_scenario not in candidate_summary:
        raise KeyError(f"Scenario '{args.candidate_scenario}' missing from candidate.")

    baseline_payload = baseline_summary[args.baseline_scenario]
    candidate_payload = candidate_summary[args.candidate_scenario]

    baseline_overall = baseline_payload["overall"]
    candidate_overall = candidate_payload["overall"]
    map_key = "map"
    ndcg_key = _find_metric_key(baseline_overall, "ndcg@")
    hit_rate_key = _find_metric_key(baseline_overall, "hit_rate@")

    checks: List[Dict[str, Any]] = []
    failures: List[str] = []

    for metric_name, allowed_drop in (
        (map_key, float(args.max_drop_map)),
        (ndcg_key, float(args.max_drop_ndcg)),
        (hit_rate_key, float(args.max_drop_hit_rate)),
    ):
        passed, message = _check_metric(
            scope="overall",
            metric_name=metric_name,
            candidate_value=float(candidate_overall.get(metric_name, 0.0)),
            baseline_value=float(baseline_overall.get(metric_name, 0.0)),
            allowed_drop=allowed_drop,
        )
        checks.append({"scope": "overall", "metric": metric_name, "passed": passed, "message": message})
        print(message)
        if not passed:
            failures.append(message)

    baseline_slices = baseline_payload.get("by_slice", {})
    candidate_slices = candidate_payload.get("by_slice", {})
    critical_slices = [name.strip() for name in args.critical_slices.split(",") if name.strip()]

    for slice_name in critical_slices:
        if slice_name not in baseline_slices or slice_name not in candidate_slices:
            note = f"[SKIP] slice {slice_name}: missing from baseline or candidate summary."
            checks.append({"scope": f"slice:{slice_name}", "passed": True, "message": note})
            print(note)
            continue

        baseline_slice = baseline_slices[slice_name]
        candidate_slice = candidate_slices[slice_name]
        baseline_count = float(baseline_slice.get("count", 0.0))
        candidate_count = float(candidate_slice.get("count", 0.0))
        if baseline_count < args.min_slice_count or candidate_count < args.min_slice_count:
            note = (
                f"[SKIP] slice {slice_name}: count too low "
                f"(baseline={baseline_count:.1f}, candidate={candidate_count:.1f})."
            )
            checks.append({"scope": f"slice:{slice_name}", "passed": True, "message": note})
            print(note)
            continue

        for metric_name, allowed_drop in (
            (map_key, float(args.slice_max_drop_map)),
            (ndcg_key, float(args.slice_max_drop_ndcg)),
            (hit_rate_key, float(args.slice_max_drop_hit_rate)),
        ):
            passed, message = _check_metric(
                scope=f"slice:{slice_name}",
                metric_name=metric_name,
                candidate_value=float(candidate_slice.get(metric_name, 0.0)),
                baseline_value=float(baseline_slice.get(metric_name, 0.0)),
                allowed_drop=allowed_drop,
            )
            checks.append(
                {
                    "scope": f"slice:{slice_name}",
                    "metric": metric_name,
                    "passed": passed,
                    "message": message,
                }
            )
            print(message)
            if not passed:
                failures.append(message)

    significance_results: Dict[str, Any] = {"enabled": False}
    if args.significance_mode != "off":
        if not args.baseline_csv or not args.candidate_csv:
            note = (
                "[SKIP] significance checks requested but --baseline-csv/--candidate-csv "
                "were not both provided."
            )
            checks.append({"scope": "significance", "passed": True, "message": note})
            print(note)
        else:
            baseline_metric_rows = _load_metric_rows(
                Path(args.baseline_csv), args.baseline_scenario
            )
            candidate_metric_rows = _load_metric_rows(
                Path(args.candidate_csv), args.candidate_scenario
            )
            significance_results = {
                "enabled": True,
                "mode": args.significance_mode,
                "overall": {},
                "by_slice": {},
            }

            metric_to_column = {
                map_key: "average_precision",
                ndcg_key: ndcg_key,
                hit_rate_key: hit_rate_key,
            }
            metric_to_drop = {
                map_key: float(args.max_drop_map),
                ndcg_key: float(args.max_drop_ndcg),
                hit_rate_key: float(args.max_drop_hit_rate),
            }

            for metric_name, csv_metric_key in metric_to_column.items():
                deltas = _paired_deltas(
                    baseline_rows=baseline_metric_rows,
                    candidate_rows=candidate_metric_rows,
                    metric_key=csv_metric_key,
                    slice_name=None,
                )
                passed, message, details = _paired_significance_check(
                    scope="overall",
                    metric_name=metric_name,
                    allowed_drop=metric_to_drop[metric_name],
                    deltas=deltas,
                    bootstrap_samples=int(args.paired_bootstrap_samples),
                    bootstrap_seed=int(args.paired_bootstrap_seed)
                    + abs(hash(f"overall:{metric_name}")) % 100000,
                )
                checks.append(
                    {
                        "scope": "significance:overall",
                        "metric": metric_name,
                        "passed": passed,
                        "message": message,
                    }
                )
                significance_results["overall"][metric_name] = details
                print(message)
                if not passed:
                    failures.append(message)

            slice_metric_to_drop = {
                map_key: float(args.slice_max_drop_map),
                ndcg_key: float(args.slice_max_drop_ndcg),
                hit_rate_key: float(args.slice_max_drop_hit_rate),
            }

            for slice_name in critical_slices:
                significance_results["by_slice"][slice_name] = {}
                for metric_name, csv_metric_key in metric_to_column.items():
                    deltas = _paired_deltas(
                        baseline_rows=baseline_metric_rows,
                        candidate_rows=candidate_metric_rows,
                        metric_key=csv_metric_key,
                        slice_name=slice_name,
                    )
                    if len(deltas) < int(args.min_slice_count):
                        note = (
                            f"[SKIP] significance slice:{slice_name} {metric_name}: "
                            f"paired sample too small (n={len(deltas)})."
                        )
                        checks.append(
                            {
                                "scope": f"significance:slice:{slice_name}",
                                "metric": metric_name,
                                "passed": True,
                                "message": note,
                            }
                        )
                        significance_results["by_slice"][slice_name][metric_name] = {
                            "count": len(deltas),
                            "skipped": True,
                        }
                        print(note)
                        continue

                    passed, message, details = _paired_significance_check(
                        scope=f"slice:{slice_name}",
                        metric_name=metric_name,
                        allowed_drop=slice_metric_to_drop[metric_name],
                        deltas=deltas,
                        bootstrap_samples=int(args.paired_bootstrap_samples),
                        bootstrap_seed=int(args.paired_bootstrap_seed)
                        + abs(hash(f"slice:{slice_name}:{metric_name}")) % 100000,
                    )
                    checks.append(
                        {
                            "scope": f"significance:slice:{slice_name}",
                            "metric": metric_name,
                            "passed": passed,
                            "message": message,
                        }
                    )
                    significance_results["by_slice"][slice_name][metric_name] = details
                    print(message)
                    if not passed:
                        failures.append(message)

    report = {
        "baseline_summary": str(Path(args.baseline_summary)),
        "candidate_summary": str(Path(args.candidate_summary)),
        "baseline_scenario": args.baseline_scenario,
        "candidate_scenario": args.candidate_scenario,
        "passed": len(failures) == 0,
        "checks": checks,
        "failures": failures,
        "thresholds": {
            "overall": {
                "map": args.max_drop_map,
                "ndcg": args.max_drop_ndcg,
                "hit_rate": args.max_drop_hit_rate,
            },
            "slice": {
                "map": args.slice_max_drop_map,
                "ndcg": args.slice_max_drop_ndcg,
                "hit_rate": args.slice_max_drop_hit_rate,
            },
        },
        "significance": significance_results,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Gate report saved to {output_path}")

    if failures:
        print(f"\nRelease gate FAILED with {len(failures)} violation(s).")
        sys.exit(1)

    print("\nRelease gate PASSED.")


if __name__ == "__main__":
    main()
