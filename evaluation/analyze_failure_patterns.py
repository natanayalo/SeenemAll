"""Analyze offline evaluation failures and rank improvement opportunities."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

CONSTRAINT_TAGS = {
    "provider_filtered",
    "runtime_capped",
    "maturity_capped",
    "language_or_region",
    "year_or_decade",
}


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_tags(raw: str) -> List[str]:
    return [tag.strip() for tag in str(raw).split("|") if tag.strip()]


def _infer_bucket(meta: Dict[str, Any], query: str, tags: List[str]) -> str:
    explicit = meta.get("distribution_bucket")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    tag_set = set(tags)
    has_query = query not in ("", "<NO_QUERY>")
    if not has_query:
        if "cold_start" in tag_set:
            return "no_query_cold_start"
        return "no_query_warm_start"

    if "edge_adversarial" in tag_set:
        return "edge_adversarial"
    if "profile_persona" in tag_set:
        return "profile_persona"

    constraint_count = sum(1 for tag in CONSTRAINT_TAGS if tag in tag_set)
    if constraint_count >= 2:
        return "constrained_multi_filter"
    if constraint_count == 1:
        return "constrained_single_filter"

    has_movie = "movie" in tag_set
    has_tv = "tv" in tag_set
    if has_movie and not has_tv:
        return "semantic_query_movie"
    if has_tv and not has_movie:
        return "semantic_query_tv"
    if has_movie and has_tv:
        return "constrained_multi_filter"
    return "semantic_query_movie"


def _load_results(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_case_meta(path: Path) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        evaluation_set = json.load(handle)
    by_case: Dict[str, Dict[str, Any]] = {}
    for index, entry in enumerate(evaluation_set):
        case_id = str(entry.get("case_id") or f"case_{index:03d}")
        by_case[case_id] = entry
    return by_case


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "count": 0.0,
            "zero_hit_count": 0.0,
            "zero_hit_rate": 0.0,
            "zero_recall_count": 0.0,
            "zero_recall_rate": 0.0,
            "avg_map": 0.0,
            "avg_ndcg": 0.0,
            "avg_recall": 0.0,
            "avg_hit_rate": 0.0,
        }

    count = float(len(rows))
    zero_hit_count = sum(1 for row in rows if row["hit_rate@10"] <= 0.0)
    zero_recall_count = sum(1 for row in rows if row["recall@10"] <= 0.0)
    return {
        "count": count,
        "zero_hit_count": float(zero_hit_count),
        "zero_hit_rate": zero_hit_count / count,
        "zero_recall_count": float(zero_recall_count),
        "zero_recall_rate": zero_recall_count / count,
        "avg_map": mean(row["average_precision"] for row in rows),
        "avg_ndcg": mean(row["ndcg@10"] for row in rows),
        "avg_recall": mean(row["recall@10"] for row in rows),
        "avg_hit_rate": mean(row["hit_rate@10"] for row in rows),
    }


def _sort_group_items(grouped: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for name, rows in grouped.items():
        stats = _summary(rows)
        ranked.append(
            {
                "name": name,
                **stats,
            }
        )
    ranked.sort(
        key=lambda item: (
            -item["zero_hit_rate"],
            -item["zero_recall_rate"],
            item["avg_ndcg"],
            -item["count"],
        )
    )
    return ranked


def _render_group_table(
    title: str, rows: List[Dict[str, Any]], limit: int
) -> List[str]:
    lines = [f"## {title}", ""]
    lines.append(
        "| Name | Count | Zero-hit % | Zero-recall % | Avg nDCG@10 | Avg MAP |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows[:limit]:
        lines.append(
            "| "
            f"{row['name']} | {int(row['count'])} | "
            f"{row['zero_hit_rate'] * 100:.1f}% | {row['zero_recall_rate'] * 100:.1f}% | "
            f"{row['avg_ndcg']:.4f} | {row['avg_map']:.4f} |"
        )
    lines.append("")
    return lines


def _render_case_table(title: str, rows: List[Dict[str, Any]], limit: int) -> List[str]:
    lines = [f"## {title}", ""]
    lines.append(
        "| Case | Bucket | Query | hit@10 | recall@10 | nDCG@10 | MAP | Slices |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    for row in rows[:limit]:
        lines.append(
            "| "
            f"{row['eval_case_id']} | {row['distribution_bucket']} | "
            f"{row['query']} | {row['hit_rate@10']:.2f} | {row['recall@10']:.2f} | "
            f"{row['ndcg@10']:.4f} | {row['average_precision']:.4f} | "
            f"{','.join(row['slice_tags'])} |"
        )
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank recommendation failure patterns from evaluator outputs."
    )
    parser.add_argument(
        "--results-csv",
        default="evaluation/artifacts/gate/candidate_results.csv",
        help="Path to evaluation results CSV.",
    )
    parser.add_argument(
        "--evaluation-set",
        default="evaluation/evaluation_set_v2.json",
        help="Path to evaluation set JSON used for case metadata.",
    )
    parser.add_argument(
        "--scenario",
        default="default",
        help="Scenario name to analyze (params_name column).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top rows for bucket/slice/case sections.",
    )
    parser.add_argument(
        "--output-json",
        default="evaluation/artifacts/gate/failure_patterns_report.json",
        help="Path to output JSON report.",
    )
    parser.add_argument(
        "--output-md",
        default="evaluation/artifacts/gate/failure_patterns_report.md",
        help="Path to output markdown report.",
    )
    args = parser.parse_args()

    results = _load_results(Path(args.results_csv))
    case_meta = _load_case_meta(Path(args.evaluation_set))

    scenario_rows_raw = [
        row for row in results if str(row.get("params_name")) == args.scenario
    ]
    if not scenario_rows_raw:
        raise ValueError(f"No rows found for scenario '{args.scenario}'.")

    by_case_all_scenarios: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in results:
        case_id = str(row.get("eval_case_id", "")).strip()
        if case_id:
            by_case_all_scenarios[case_id].append(row)

    scenario_rows: List[Dict[str, Any]] = []
    for row in scenario_rows_raw:
        case_id = str(row["eval_case_id"])
        meta = case_meta.get(case_id, {})
        query = str(row.get("query", ""))
        tags = _parse_tags(str(row.get("slice_tags", "")))
        scenario_rows.append(
            {
                "params_name": str(row.get("params_name", "")),
                "eval_case_id": case_id,
                "query": query,
                "slice_tags": tags,
                "distribution_bucket": _infer_bucket(meta, query, tags),
                "precision@10": _to_float(row.get("precision@10")),
                "recall@10": _to_float(row.get("recall@10")),
                "average_precision": _to_float(row.get("average_precision")),
                "ndcg@10": _to_float(row.get("ndcg@10")),
                "hit_rate@10": _to_float(row.get("hit_rate@10")),
                "mrr@10": _to_float(row.get("mrr@10")),
                "negative_rate@10": _to_float(row.get("negative_rate@10")),
                "negative_hit_rate@10": _to_float(row.get("negative_hit_rate@10")),
            }
        )

    overall = _summary(scenario_rows)

    bucket_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    slice_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scenario_rows:
        bucket_groups[row["distribution_bucket"]].append(row)
        for tag in row["slice_tags"]:
            slice_groups[tag].append(row)

    bucket_ranked = _sort_group_items(bucket_groups)
    slice_ranked = _sort_group_items(slice_groups)

    zero_hit_cases = [row for row in scenario_rows if row["hit_rate@10"] <= 0.0]
    zero_hit_cases.sort(
        key=lambda row: (
            row["hit_rate@10"],
            row["recall@10"],
            row["ndcg@10"],
            row["average_precision"],
            row["eval_case_id"],
        )
    )

    recoverable_misses: List[Dict[str, Any]] = []
    for row in zero_hit_cases:
        case_id = row["eval_case_id"]
        alt_rows = []
        for alt in by_case_all_scenarios.get(case_id, []):
            if str(alt.get("params_name")) == args.scenario:
                continue
            alt_hit = _to_float(alt.get("hit_rate@10"))
            if alt_hit <= 0.0:
                continue
            alt_rows.append(
                {
                    "params_name": str(alt.get("params_name")),
                    "hit_rate@10": alt_hit,
                    "recall@10": _to_float(alt.get("recall@10")),
                    "average_precision": _to_float(alt.get("average_precision")),
                    "ndcg@10": _to_float(alt.get("ndcg@10")),
                }
            )
        if not alt_rows:
            continue
        best_alt = max(
            alt_rows,
            key=lambda alt: (
                alt["hit_rate@10"],
                alt["ndcg@10"],
                alt["average_precision"],
                alt["recall@10"],
            ),
        )
        recoverable_misses.append(
            {
                "eval_case_id": case_id,
                "query": row["query"],
                "distribution_bucket": row["distribution_bucket"],
                "slice_tags": row["slice_tags"],
                "best_alt": best_alt,
            }
        )

    recoverable_misses.sort(
        key=lambda item: (
            -item["best_alt"]["hit_rate@10"],
            -item["best_alt"]["ndcg@10"],
            item["eval_case_id"],
        )
    )

    report = {
        "scenario": args.scenario,
        "results_csv": args.results_csv,
        "evaluation_set": args.evaluation_set,
        "overall": overall,
        "zero_hit_case_count": len(zero_hit_cases),
        "recoverable_zero_hit_case_count": len(recoverable_misses),
        "bucket_ranked": bucket_ranked,
        "slice_ranked": slice_ranked,
        "top_zero_hit_cases": zero_hit_cases[: args.top],
        "top_recoverable_misses": recoverable_misses[: args.top],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    md_lines: List[str] = []
    md_lines.append(f"# Failure Pattern Report ({args.scenario})")
    md_lines.append("")
    md_lines.append(f"- Cases: {int(overall['count'])}")
    md_lines.append(
        f"- Zero-hit cases: {len(zero_hit_cases)} ({overall['zero_hit_rate'] * 100:.1f}%)"
    )
    md_lines.append(
        "- Recoverable zero-hit cases (other scenarios hit): "
        f"{len(recoverable_misses)}"
    )
    md_lines.append(
        f"- Avg nDCG@10: {overall['avg_ndcg']:.4f}, Avg MAP: {overall['avg_map']:.4f}"
    )
    md_lines.append("")

    md_lines.extend(_render_group_table("Top Failure Buckets", bucket_ranked, args.top))
    md_lines.extend(_render_group_table("Top Failure Slices", slice_ranked, args.top))
    md_lines.extend(_render_case_table("Top Zero-hit Cases", zero_hit_cases, args.top))

    recoverable_display = []
    for item in recoverable_misses:
        recoverable_display.append(
            {
                "eval_case_id": item["eval_case_id"],
                "distribution_bucket": item["distribution_bucket"],
                "query": item["query"],
                "slice_tags": item["slice_tags"],
                "hit_rate@10": item["best_alt"]["hit_rate@10"],
                "recall@10": item["best_alt"]["recall@10"],
                "ndcg@10": item["best_alt"]["ndcg@10"],
                "average_precision": item["best_alt"]["average_precision"],
                "best_scenario": item["best_alt"]["params_name"],
            }
        )
    md_lines.append("## Top Recoverable Misses")
    md_lines.append("")
    md_lines.append(
        "| Case | Bucket | Query | Best scenario | hit@10 | recall@10 | nDCG@10 | MAP | Slices |"
    )
    md_lines.append("|---|---|---|---|---:|---:|---:|---:|---|")
    for row in recoverable_display[: args.top]:
        md_lines.append(
            "| "
            f"{row['eval_case_id']} | {row['distribution_bucket']} | {row['query']} | "
            f"{row['best_scenario']} | {row['hit_rate@10']:.2f} | "
            f"{row['recall@10']:.2f} | {row['ndcg@10']:.4f} | "
            f"{row['average_precision']:.4f} | {','.join(row['slice_tags'])} |"
        )
    md_lines.append("")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote JSON report: {output_json}")
    print(f"Wrote Markdown report: {output_md}")
    print(
        f"Scenario '{args.scenario}': "
        f"{len(zero_hit_cases)} zero-hit cases, {len(recoverable_misses)} recoverable."
    )


if __name__ == "__main__":
    main()
