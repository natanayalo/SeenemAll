"""Create a deterministic dev/holdout split manifest for evaluation cases."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

try:
    from evaluation.evaluate import load_evaluation_set
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from evaluation.evaluate import load_evaluation_set


def _case_id(entry: Dict[str, Any], index: int) -> str:
    raw = entry.get("case_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return f"case_{index:03d}"


def _bucket(entry: Dict[str, Any]) -> str:
    raw = entry.get("distribution_bucket")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return "unbucketed"


def _holdout_count(group_size: int, holdout_ratio: float) -> int:
    if group_size <= 1 or holdout_ratio <= 0:
        return 0
    raw_count = int(round(group_size * holdout_ratio))
    raw_count = max(raw_count, 1)
    return min(raw_count, group_size - 1)


def build_split_manifest(
    evaluation_set: List[Dict[str, Any]],
    *,
    holdout_ratio: float,
    seed: int,
    dev_name: str,
    holdout_name: str,
) -> Dict[str, Any]:
    grouped_case_ids: Dict[str, List[str]] = defaultdict(list)
    for index, entry in enumerate(evaluation_set):
        grouped_case_ids[_bucket(entry)].append(_case_id(entry, index))

    assignments: Dict[str, str] = {}
    by_bucket: Dict[str, Dict[str, int]] = {}

    for bucket_name in sorted(grouped_case_ids):
        case_ids = sorted(grouped_case_ids[bucket_name])
        rng = random.Random(f"{seed}:{bucket_name}")
        rng.shuffle(case_ids)

        holdout_count = _holdout_count(len(case_ids), holdout_ratio)
        holdout_case_ids = set(case_ids[:holdout_count])
        bucket_counts = {dev_name: 0, holdout_name: 0}

        for case_id in case_ids:
            split_name = holdout_name if case_id in holdout_case_ids else dev_name
            assignments[case_id] = split_name
            bucket_counts[split_name] += 1

        by_bucket[bucket_name] = bucket_counts

    total_counts = {dev_name: 0, holdout_name: 0}
    for split_name in assignments.values():
        total_counts[split_name] += 1

    return {
        "meta": {
            "seed": seed,
            "holdout_ratio": holdout_ratio,
            "strategy": "stratified_by_distribution_bucket",
            "dev_split": dev_name,
            "holdout_split": holdout_name,
            "total_cases": len(assignments),
        },
        "summary": {
            "overall": total_counts,
            "by_distribution_bucket": by_bucket,
        },
        "splits": assignments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic evaluation split manifest."
    )
    parser.add_argument(
        "--evaluation-set",
        default="evaluation/evaluation_set_v2.json",
        help="Path to the evaluation set JSON.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/evaluation_split_manifest_v1.json",
        help="Output path for the generated split manifest JSON.",
    )
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="Fraction of each distribution bucket assigned to holdout.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for shuffling within each bucket.",
    )
    parser.add_argument(
        "--dev-name",
        default="dev",
        help="Label to use for the tuning split.",
    )
    parser.add_argument(
        "--holdout-name",
        default="holdout",
        help="Label to use for the untouched evaluation split.",
    )
    args = parser.parse_args()

    holdout_ratio = max(0.0, min(float(args.holdout_ratio), 0.9))
    evaluation_set = load_evaluation_set(args.evaluation_set)
    manifest = build_split_manifest(
        evaluation_set,
        holdout_ratio=holdout_ratio,
        seed=int(args.seed),
        dev_name=str(args.dev_name).strip() or "dev",
        holdout_name=str(args.holdout_name).strip() or "holdout",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    overall = manifest["summary"]["overall"]
    print(
        "Created split manifest: "
        f"{output_path} | dev={overall[manifest['meta']['dev_split']]} "
        f"| holdout={overall[manifest['meta']['holdout_split']]}"
    )


if __name__ == "__main__":
    main()
