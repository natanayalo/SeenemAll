"""Check evaluation-set distribution against v2 target quotas."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_BUCKET = "semantic_query_movie"
CONSTRAINT_TAGS = {
    "provider_filtered",
    "runtime_capped",
    "maturity_capped",
    "language_or_region",
    "year_or_decade",
}


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalized_tags(entry: Dict[str, Any]) -> set[str]:
    raw = entry.get("slice_tags", [])
    if not isinstance(raw, list):
        return set()
    return {str(tag).strip() for tag in raw if str(tag).strip()}


def _infer_bucket(entry: Dict[str, Any]) -> str:
    explicit = entry.get("distribution_bucket")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    tags = _normalized_tags(entry)
    query = entry.get("query")
    has_query = query not in (None, "")

    if not has_query:
        if "cold_start" in tags:
            return "no_query_cold_start"
        return "no_query_warm_start"

    if "edge_adversarial" in tags:
        return "edge_adversarial"
    if "profile_persona" in tags:
        return "profile_persona"

    constraint_count = sum(1 for tag in CONSTRAINT_TAGS if tag in tags)
    if constraint_count >= 2:
        return "constrained_multi_filter"
    if constraint_count == 1:
        return "constrained_single_filter"

    has_movie = "movie" in tags
    has_tv = "tv" in tags
    if has_movie and not has_tv:
        return "semantic_query_movie"
    if has_tv and not has_movie:
        return "semantic_query_tv"
    if has_movie and has_tv:
        return "constrained_multi_filter"
    return DEFAULT_BUCKET


def _slice_counter(evaluation_set: List[Dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for entry in evaluation_set:
        tags = _normalized_tags(entry)
        has_movie = "movie" in tags
        has_tv = "tv" in tags
        if has_movie and has_tv:
            tags = set(tags)
            tags.add("mixed")
        for tag in tags:
            counts[tag] += 1
    return counts


def _milestone_spec(
    targets: Dict[str, Any],
    milestone: str,
) -> Dict[str, Any]:
    if milestone == "v2":
        return {
            "name": "Final",
            "target_total_cases": int(targets.get("target_total_cases", 0)),
            "bucket_targets": dict(targets.get("bucket_targets", {})),
            "minimum_slice_counts": dict(targets.get("minimum_slice_counts", {})),
        }

    milestones = targets.get("milestones", {})
    if not isinstance(milestones, dict) or milestone not in milestones:
        raise KeyError(f"Milestone '{milestone}' not found in targets file.")

    selected = milestones[milestone]
    if not isinstance(selected, dict):
        raise ValueError(f"Milestone '{milestone}' must be an object.")
    return {
        "name": str(selected.get("name", milestone)),
        "target_total_cases": int(selected.get("target_total_cases", 0)),
        "bucket_targets": dict(selected.get("bucket_targets", {})),
        "minimum_slice_counts": dict(selected.get("minimum_slice_counts", {})),
    }


def _deficit(actual: int, target: int) -> int:
    return max(target - actual, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check evaluation set distribution targets.")
    parser.add_argument(
        "--evaluation-set",
        default="evaluation/evaluation_set.json",
        help="Path to evaluation set JSON.",
    )
    parser.add_argument(
        "--targets",
        default="evaluation/evaluation_set_v2_targets.json",
        help="Path to v2 targets JSON.",
    )
    parser.add_argument(
        "--milestone",
        default="m1",
        help="Target milestone key (m1, m2, v2).",
    )
    parser.add_argument(
        "--report-json",
        default="",
        help="Optional report JSON output path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when quotas are not met.",
    )
    args = parser.parse_args()

    evaluation_path = Path(args.evaluation_set)
    targets_path = Path(args.targets)
    evaluation_set = _load_json(evaluation_path)
    targets = _load_json(targets_path)

    if not isinstance(evaluation_set, list):
        raise ValueError("Evaluation set must be a JSON list.")
    if not isinstance(targets, dict):
        raise ValueError("Targets file must be a JSON object.")

    milestone = _milestone_spec(targets, args.milestone)
    total_cases = len(evaluation_set)
    bucket_counts: Counter[str] = Counter()
    for entry in evaluation_set:
        if not isinstance(entry, dict):
            continue
        bucket_counts[_infer_bucket(entry)] += 1
    slice_counts = _slice_counter([entry for entry in evaluation_set if isinstance(entry, dict)])

    total_target = int(milestone["target_total_cases"])
    total_deficit = _deficit(total_cases, total_target)

    bucket_deficits: Dict[str, int] = {}
    for bucket, target_count in milestone["bucket_targets"].items():
        bucket_deficits[str(bucket)] = _deficit(
            int(bucket_counts.get(str(bucket), 0)),
            int(target_count),
        )

    slice_deficits: Dict[str, int] = {}
    for slice_name, target_count in milestone["minimum_slice_counts"].items():
        slice_deficits[str(slice_name)] = _deficit(
            int(slice_counts.get(str(slice_name), 0)),
            int(target_count),
        )

    passed = (
        total_deficit == 0
        and all(value == 0 for value in bucket_deficits.values())
        and all(value == 0 for value in slice_deficits.values())
    )

    print(f"Milestone: {args.milestone} ({milestone['name']})")
    print(f"Total cases: {total_cases} / {total_target}")
    print("Bucket counts:")
    for bucket in sorted(set(bucket_counts.keys()) | set(milestone["bucket_targets"].keys())):
        actual = int(bucket_counts.get(bucket, 0))
        target = int(milestone["bucket_targets"].get(bucket, 0))
        deficit = _deficit(actual, target)
        print(f"  {bucket}: {actual} / {target} (deficit={deficit})")

    print("Slice counts:")
    for slice_name in sorted(milestone["minimum_slice_counts"].keys()):
        actual = int(slice_counts.get(slice_name, 0))
        target = int(milestone["minimum_slice_counts"].get(slice_name, 0))
        deficit = _deficit(actual, target)
        print(f"  {slice_name}: {actual} / {target} (deficit={deficit})")

    report = {
        "evaluation_set": str(evaluation_path),
        "targets": str(targets_path),
        "milestone": args.milestone,
        "milestone_name": milestone["name"],
        "passed": passed,
        "total_cases": total_cases,
        "target_total_cases": total_target,
        "total_deficit": total_deficit,
        "bucket_counts": dict(bucket_counts),
        "bucket_targets": milestone["bucket_targets"],
        "bucket_deficits": bucket_deficits,
        "slice_counts": dict(slice_counts),
        "slice_targets": milestone["minimum_slice_counts"],
        "slice_deficits": slice_deficits,
    }

    if args.report_json:
        output_path = Path(args.report_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Report written to {output_path}")

    if args.strict and not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
