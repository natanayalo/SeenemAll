"""Audit quality and health characteristics of an evaluation set."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

CONSTRAINT_TAGS = {
    "provider_filtered",
    "runtime_capped",
    "maturity_capped",
    "language_or_region",
    "year_or_decade",
}

RecKey = Tuple[int, str]


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalized_tags(entry: Dict[str, Any]) -> List[str]:
    raw = entry.get("slice_tags", [])
    if not isinstance(raw, list):
        return []
    cleaned: List[str] = []
    seen: set[str] = set()
    for tag in raw:
        value = str(tag).strip()
        if not value or value in seen:
            continue
        cleaned.append(value)
        seen.add(value)
    return cleaned


def _infer_bucket(entry: Dict[str, Any]) -> str:
    explicit = entry.get("distribution_bucket")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    tags = set(_normalized_tags(entry))
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
    return "semantic_query_movie"


def _eval_case_id(entry: Dict[str, Any], index: int) -> str:
    raw = entry.get("case_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return f"case_{index:03d}"


def _to_key(row: Dict[str, Any]) -> RecKey | None:
    try:
        tmdb_id = int(row.get("id"))
    except (TypeError, ValueError):
        return None
    media_type = str(row.get("media_type", "")).strip().lower()
    if media_type not in {"movie", "tv"}:
        return None
    return tmdb_id, media_type


def _jaccard(a: set[RecKey], b: set[RecKey]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _token(key: RecKey) -> str:
    return f"{key[0]}:{key[1]}"


def _build_markdown_report(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines: List[str] = []
    lines.append("# Evaluation Set Audit")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Total cases: {summary['total_cases']}")
    lines.append(f"- Query cases: {summary['query_cases']}")
    lines.append(f"- No-query cases: {summary['no_query_cases']}")
    lines.append(
        f"- Golden size avg/median/min/max: "
        f"{summary['golden_avg']:.2f} / {summary['golden_median']:.2f} / "
        f"{summary['golden_min']} / {summary['golden_max']}"
    )
    lines.append(
        f"- Cases with negatives: {summary['cases_with_negatives']} "
        f"({summary['negative_coverage_rate']:.1%})"
    )
    lines.append(
        f"- Missing case_id: {summary['missing_case_id_count']} | "
        f"Missing bucket: {summary['missing_distribution_bucket_count']}"
    )

    lines.append("")
    lines.append("## Bucket Health")
    for bucket, stats in report["by_bucket"].items():
        lines.append(
            f"- `{bucket}`: count={stats['count']}, "
            f"avg_golden={stats['avg_golden']:.2f}, "
            f"negative_coverage={stats['negative_coverage_rate']:.1%}"
        )

    weak_cases = report["weak_golden_cases"]
    lines.append("")
    lines.append(f"## Weak Golden Cases ({len(weak_cases)})")
    for item in weak_cases[:20]:
        lines.append(
            f"- `{item['case_id']}` golden={item['golden_count']} "
            f"bucket={item['bucket']} query={item['query']}"
        )

    overlap_pairs = report["high_overlap_pairs"]
    lines.append("")
    lines.append(f"## High Overlap Pairs ({len(overlap_pairs)})")
    for pair in overlap_pairs[:20]:
        lines.append(
            f"- `{pair['left_case_id']}` vs `{pair['right_case_id']}` "
            f"jaccard={pair['jaccard']:.3f} "
            f"shared={pair['intersection_count']}/{pair['union_count']}"
        )

    frequent_labels = report["frequent_golden_labels"]
    lines.append("")
    lines.append(f"## Frequent Golden Labels ({len(frequent_labels)})")
    for row in frequent_labels[:20]:
        lines.append(f"- `{row['item']}` appears in {row['count']} cases")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit evaluation-set quality.")
    parser.add_argument(
        "--evaluation-set",
        default="evaluation/evaluation_set_v2.json",
        help="Path to evaluation set JSON.",
    )
    parser.add_argument(
        "--min-golden",
        type=int,
        default=4,
        help="Threshold below which a case is marked as weak.",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.6,
        help="Pairwise golden-set Jaccard threshold for overlap warnings.",
    )
    parser.add_argument(
        "--max-overlap-pairs",
        type=int,
        default=30,
        help="Maximum overlap pairs to include in output.",
    )
    parser.add_argument(
        "--frequent-label-threshold",
        type=int,
        default=8,
        help="Minimum number of cases for a golden label to be flagged as frequent.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional JSON report output path.",
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="Optional Markdown report output path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero when metadata is missing or any case is below --min-golden."
        ),
    )
    args = parser.parse_args()

    payload = _load_json(Path(args.evaluation_set))
    if not isinstance(payload, list):
        raise ValueError("Evaluation set must be a JSON list.")

    bucket_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    golden_sizes: List[int] = []
    cases_with_negatives = 0
    query_cases = 0
    no_query_cases = 0
    missing_case_id: List[str] = []
    missing_bucket: List[str] = []
    weak_cases: List[Dict[str, Any]] = []
    label_usage: Counter[RecKey] = Counter()
    case_golden_sets: List[Dict[str, Any]] = []

    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            continue
        case_id = _eval_case_id(entry, index)
        query = entry.get("query")
        display_query = str(query) if query not in (None, "") else "<NO_QUERY>"
        has_query = query not in (None, "")
        if has_query:
            query_cases += 1
        else:
            no_query_cases += 1

        if not (
            isinstance(entry.get("case_id"), str) and str(entry["case_id"]).strip()
        ):
            missing_case_id.append(case_id)
        if not (
            isinstance(entry.get("distribution_bucket"), str)
            and str(entry["distribution_bucket"]).strip()
        ):
            missing_bucket.append(case_id)

        bucket = _infer_bucket(entry)
        golden_raw = entry.get("golden_set", [])
        negative_raw = entry.get("negative_set", [])

        golden_keys: set[RecKey] = set()
        if isinstance(golden_raw, list):
            for row in golden_raw:
                if not isinstance(row, dict):
                    continue
                key = _to_key(row)
                if key is None:
                    continue
                golden_keys.add(key)
                label_usage[key] += 1

        negative_count = len(negative_raw) if isinstance(negative_raw, list) else 0
        if negative_count > 0:
            cases_with_negatives += 1

        golden_count = len(golden_keys)
        golden_sizes.append(golden_count)

        row = {
            "case_id": case_id,
            "bucket": bucket,
            "query": display_query,
            "golden_count": golden_count,
            "negative_count": negative_count,
            "slice_tags": _normalized_tags(entry),
        }
        bucket_rows[bucket].append(row)
        case_golden_sets.append({"case_id": case_id, "golden_keys": golden_keys})

        if golden_count < args.min_golden:
            weak_cases.append(row)

    by_bucket: Dict[str, Dict[str, Any]] = {}
    for bucket, rows in sorted(bucket_rows.items()):
        golden_counts = [int(row["golden_count"]) for row in rows]
        with_negative = sum(1 for row in rows if int(row["negative_count"]) > 0)
        by_bucket[bucket] = {
            "count": len(rows),
            "avg_golden": (
                sum(golden_counts) / len(golden_counts) if golden_counts else 0.0
            ),
            "min_golden": min(golden_counts) if golden_counts else 0,
            "max_golden": max(golden_counts) if golden_counts else 0,
            "negative_coverage_rate": with_negative / len(rows) if rows else 0.0,
        }

    overlap_pairs: List[Dict[str, Any]] = []
    for left_index in range(len(case_golden_sets)):
        left = case_golden_sets[left_index]
        for right_index in range(left_index + 1, len(case_golden_sets)):
            right = case_golden_sets[right_index]
            jaccard = _jaccard(left["golden_keys"], right["golden_keys"])
            if jaccard < args.overlap_threshold:
                continue
            intersection = left["golden_keys"] & right["golden_keys"]
            union = left["golden_keys"] | right["golden_keys"]
            overlap_pairs.append(
                {
                    "left_case_id": left["case_id"],
                    "right_case_id": right["case_id"],
                    "jaccard": round(jaccard, 4),
                    "intersection_count": len(intersection),
                    "union_count": len(union),
                    "shared_items": sorted(_token(item) for item in intersection)[:12],
                }
            )
    overlap_pairs.sort(key=lambda item: item["jaccard"], reverse=True)
    overlap_pairs = overlap_pairs[: max(0, int(args.max_overlap_pairs))]

    frequent_labels = [
        {"item": _token(item), "count": count}
        for item, count in label_usage.most_common()
        if count >= args.frequent_label_threshold
    ]

    report: Dict[str, Any] = {
        "evaluation_set": args.evaluation_set,
        "summary": {
            "total_cases": len(payload),
            "query_cases": query_cases,
            "no_query_cases": no_query_cases,
            "golden_avg": (
                (sum(golden_sizes) / len(golden_sizes)) if golden_sizes else 0.0
            ),
            "golden_median": statistics.median(golden_sizes) if golden_sizes else 0.0,
            "golden_min": min(golden_sizes) if golden_sizes else 0,
            "golden_max": max(golden_sizes) if golden_sizes else 0,
            "cases_with_negatives": cases_with_negatives,
            "negative_coverage_rate": (
                cases_with_negatives / len(payload) if payload else 0.0
            ),
            "missing_case_id_count": len(missing_case_id),
            "missing_distribution_bucket_count": len(missing_bucket),
        },
        "by_bucket": by_bucket,
        "weak_golden_cases": sorted(
            weak_cases, key=lambda row: (int(row["golden_count"]), row["case_id"])
        ),
        "high_overlap_pairs": overlap_pairs,
        "frequent_golden_labels": frequent_labels,
        "metadata_gaps": {
            "missing_case_id": missing_case_id,
            "missing_distribution_bucket": missing_bucket,
        },
    }

    summary = report["summary"]
    print(f"Total cases: {summary['total_cases']}")
    print(
        "Golden size avg/median/min/max: "
        f"{summary['golden_avg']:.2f}/{summary['golden_median']:.2f}/"
        f"{summary['golden_min']}/{summary['golden_max']}"
    )
    print(
        "Negative coverage: "
        f"{summary['cases_with_negatives']}/{summary['total_cases']} "
        f"({summary['negative_coverage_rate']:.1%})"
    )
    print(
        f"Weak cases (<{args.min_golden} goldens): {len(report['weak_golden_cases'])}"
    )
    print(
        f"High-overlap pairs (>= {args.overlap_threshold:.2f}): "
        f"{len(report['high_overlap_pairs'])}"
    )
    print(
        f"Missing metadata: case_id={summary['missing_case_id_count']} "
        f"bucket={summary['missing_distribution_bucket_count']}"
    )

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Wrote JSON report to {output_json}")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        with open(output_md, "w", encoding="utf-8") as handle:
            handle.write(_build_markdown_report(report))
        print(f"Wrote Markdown report to {output_md}")

    should_fail = args.strict and (
        summary["missing_case_id_count"] > 0
        or summary["missing_distribution_bucket_count"] > 0
        or len(report["weak_golden_cases"]) > 0
    )
    if should_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
