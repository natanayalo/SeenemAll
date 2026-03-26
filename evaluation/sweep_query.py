"""Parameter sweep helper for a single evaluation query.

This script fetches the golden set for a query defined in
`evaluation/evaluation_set.json`, runs the /recommend API across a grid of
parameter overrides, and writes a CSV report with precision/recall/MAP/nDCG
for each combination.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, cast

try:
    import httpx

    HAVE_HTTPX = True
except ImportError:  # pragma: no cover - fallback messaging for missing dep
    httpx = None
    HAVE_HTTPX = False

import csv

from evaluation.evaluate import (
    calculate_average_precision,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    normalise_golden_ids,
)

DEFAULT_SET_PATH = Path("evaluation/evaluation_set.json")
DEFAULT_BASE_URL = os.environ.get(
    "EVAL_RECOMMEND_URL", "http://localhost:8000/recommend"
)

GridOptions = List[float] | List[bool]
OverrideParams = Dict[str, float | bool]


def load_entry(path: Path, query: str) -> Dict[str, Any]:
    """Return the JSON entry for the provided query."""
    try:
        with path.open("r", encoding="utf-8") as src:
            data = json.load(src)
    except FileNotFoundError as exc:
        raise SystemExit(f"Evaluation set not found at {path}") from exc
    except json.JSONDecodeError as exc:  # pragma: no cover - invalid input
        raise SystemExit(f"Failed to parse evaluation set {path}: {exc}") from exc

    for entry in data:
        if entry.get("query") == query:
            return entry
    raise SystemExit(f"Query '{query}' not found in {path}.")


def slugify(text: str) -> str:
    sanitized = "".join(c if c.isalnum() else "_" for c in text.lower())
    return "_".join(filter(None, sanitized.split("_"))) or "query"


def iter_param_grid(args: argparse.Namespace) -> Iterable[Tuple[OverrideParams, str]]:
    """Produce parameter combinations and a readable label."""

    def _as_float_list(values: Sequence[float | int]) -> List[float]:
        return [float(v) for v in values]

    classic_raw = cast(Sequence[str], getattr(args, "classic_top_rated_options", []))
    bool_options = [value.strip().lower() for value in classic_raw]
    classic_options = [
        value in {"1", "true", "yes"} for value in bool_options or ["false", "true"]
    ]

    strict_options = [
        value.strip().lower() in {"1", "true", "yes"}
        for value in getattr(args, "strict_filters_options", ["false"])
    ]

    grid_def: List[Tuple[str, GridOptions]] = [
        ("ann_weight_override", _as_float_list(cast(Sequence[float], args.ann_weight))),
        (
            "rewrite_weight_override",
            _as_float_list(cast(Sequence[float], args.rewrite_weight)),
        ),
        (
            "mixer_ann_weight",
            _as_float_list(cast(Sequence[float], args.mixer_ann_weight)),
        ),
        (
            "mixer_trending_weight",
            _as_float_list(cast(Sequence[float], args.mixer_trending_weight)),
        ),
        (
            "mixer_vote_weight",
            _as_float_list(cast(Sequence[float], args.mixer_vote_weight)),
        ),
        (
            "mixer_popularity_weight",
            _as_float_list(cast(Sequence[float], args.mixer_popularity_weight)),
        ),
        (
            "mixer_collab_weight",
            _as_float_list(cast(Sequence[float], args.mixer_collab_weight)),
        ),
        (
            "mixer_novelty_weight",
            _as_float_list(cast(Sequence[float], args.mixer_novelty_weight)),
        ),
        ("classic_top_rated", classic_options),
        ("strict_filters", strict_options),
    ]
    grid_values: List[GridOptions] = [options for _, options in grid_def]
    total = 1
    for values in grid_values:
        total *= max(1, len(values))
    if total > args.max_combinations:
        raise SystemExit(
            f"Parameter grid has {total} combinations, exceeding the "
            f"--max-combinations limit ({args.max_combinations}). "
            "Narrow the input ranges."
        )

    for combo in itertools.product(*grid_values):
        params: OverrideParams = {
            name: value for (name, _), value in zip(grid_def, combo)
        }
        label_parts = [f"{key}={value}" for key, value in params.items()]
        yield params, ", ".join(label_parts)


def fetch_recommendations(
    client: httpx.Client,
    base_url: str,
    query: str,
    user_id: str,
    limit: int,
    use_llm_intent: bool,
    overrides: Dict[str, Any],
    serendipity_ratio: float | None = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "user_id": user_id,
        "query": query,
        "limit": limit,
        "use_llm_intent": str(use_llm_intent).lower(),
        "debug": "true",
    }
    if serendipity_ratio is not None:
        params["serendipity_ratio"] = serendipity_ratio

    for key, value in overrides.items():
        if isinstance(value, bool):
            params[key] = "true" if value else "false"
        else:
            params[key] = value
    response = client.get(base_url, params=params)
    response.raise_for_status()
    return response.json()


def compute_metrics(
    recommended_ids: Sequence[int], golden_ids: Sequence[int], k: int
) -> Dict[str, float]:
    return {
        "precision": calculate_precision_at_k(recommended_ids, golden_ids, k),
        "recall": calculate_recall_at_k(recommended_ids, golden_ids, k),
        "map": calculate_average_precision(recommended_ids, golden_ids),
        "ndcg": calculate_ndcg_at_k(recommended_ids, golden_ids, k),
    }


def format_item_list(items: Sequence[int]) -> str:
    return " ".join(str(item) for item in items)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def format_override_summary(overrides: Dict[str, Any]) -> str:
    fields = [
        ("ann", "ann_weight_override"),
        ("rew", "rewrite_weight_override"),
        ("mix", "mixer_ann_weight"),
        ("trend", "mixer_trending_weight"),
        ("vote", "mixer_vote_weight"),
        ("pop", "mixer_popularity_weight"),
        ("collab", "mixer_collab_weight"),
        ("nov", "mixer_novelty_weight"),
        ("classic", "classic_top_rated"),
        ("strict", "strict_filters"),
    ]
    parts = []
    for label, key in fields:
        value = overrides.get(key)
        if isinstance(value, bool):
            parts.append(f"{label}={'Y' if value else 'N'}")
        elif value is not None:
            parts.append(f"{label}={float(value):.2f}")
    return " ".join(parts)


def write_markdown_report(
    output_path: Path,
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    golden_ids: Sequence[int],
) -> None:
    """Write a human-readable summary and conclusions report in Markdown format."""
    if not rows:
        return

    summary_path = output_path.with_name(output_path.stem + "_summary.md")
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"# Sweep Summary Report for Query: `{args.query}`\n\n")
        f.write(f"*Golden set size: {len(golden_ids)}*\n\n")

        f.write("## Summary of Best Performing Parameters\n\n")

        def write_best_metric(
            metric_name: str, metric_key: str, higher_is_better: bool
        ) -> None:
            best_score = max(row[metric_key] for row in rows)
            best_rows = [row for row in rows if row[metric_key] == best_score]

            f.write(f"### Best {metric_name}\n")
            f.write(f"- **{metric_name}:** {best_score:.4f}\n")

            if len(best_rows) == 1:
                best_row = best_rows[0]
                f.write(f"- **Parameters:** `{format_override_summary(best_row)}`\n")
                f.write(
                    f"- **Metrics:** P={best_row['precision']:.3f}, R={best_row['recall']:.3f}, "
                    f"MAP={best_row['map']:.3f}, nDCG={best_row['ndcg']:.3f}\n\n"
                )
            else:
                f.write(
                    f"- **Found {len(best_rows)} combinations with the highest score:**\n"
                )
                for i, best_row in enumerate(best_rows, 1):
                    f.write(
                        f"  {i}. **Parameters:** `{format_override_summary(best_row)}`\n"
                    )
                    f.write(
                        f"     - **Metrics:** P={best_row['precision']:.3f}, R={best_row['recall']:.3f}, "
                        f"MAP={best_row['map']:.3f}, nDCG={best_row['ndcg']:.3f}\n"
                    )
                f.write("\n")

        write_best_metric("MAP", "map", True)
        write_best_metric("nDCG", "ndcg", True)
        write_best_metric("Precision", "precision", True)
        write_best_metric("Recall", "recall", True)

        f.write("## Top Combinations by Metric\n\n")
        metrics_to_report = {
            "MAP": "map",
            "nDCG": "ndcg",
            "Precision": "precision",
            "Recall": "recall",
        }
        for metric_name, metric_key in metrics_to_report.items():
            sorted_rows = sorted(rows, key=lambda r: r[metric_key], reverse=True)

            if sorted_rows:
                highest_score = sorted_rows[0][metric_key]
                top_rows = [
                    row for row in sorted_rows if row[metric_key] == highest_score
                ]
            else:
                top_rows = []

            f.write(f"### Top {len(top_rows)} by {metric_name}\n\n")
            for idx, row in enumerate(top_rows, start=1):
                summary = format_override_summary(row)
                f.write(
                    f"{idx}. **{metric_name}={row[metric_key]:.3f}** (P={row['precision']:.3f}, "
                    f"R={row['recall']:.3f}, MAP={row['map']:.3f}, nDCG={row['ndcg']:.3f})\n"
                )
                f.write(f"    - **Params:** `{summary}`\n")
            f.write("\n")

        f.write("## Parameter Impact Analysis\n\n")
        param_impact: Dict[str, Dict[Any, Dict[str, float]]] = {}
        for row in rows:
            for key, value in row.items():
                is_param = (
                    key.endswith("_override")
                    or key.startswith("mixer_")
                    or key in ["classic_top_rated", "strict_filters"]
                )
                if is_param:
                    if key not in param_impact:
                        param_impact[key] = {}
                    if value not in param_impact[key]:
                        param_impact[key][value] = {"map": 0.0, "count": 0}
                    param_impact[key][value]["map"] += row["map"]
                    param_impact[key][value]["count"] += 1

        for param, values in sorted(param_impact.items()):
            f.write(f"### `{param}`\n\n")
            f.write("| Value | Average MAP | Combinations |\n")
            f.write("|---|---|---|\n")
            for value, data in sorted(values.items(), key=lambda item: str(item[0])):
                avg_map = data["map"] / data["count"]
                f.write(f"| {value} | {avg_map:.4f} | {int(data['count'])} |\n")
            f.write("\n")

        f.write("## Key Observations\n\n")
        top_rows_by_map = sorted(
            rows, key=lambda r: (r["map"], r["ndcg"], r["precision"]), reverse=True
        )[:5]
        param_counts: Dict[str, int] = {}
        for row in top_rows_by_map:
            for key, value in row.items():
                is_param = (
                    key.endswith("_override")
                    or key.startswith("mixer_")
                    or key in ["classic_top_rated", "strict_filters"]
                )
                if is_param and value not in [0.0, False]:
                    param_counts[key] = param_counts.get(key, 0) + 1

        if param_counts:
            f.write(
                "The following parameters frequently appeared with non-default "
                f"values in the top {len(top_rows_by_map)} results (by MAP):\n\n"
            )
            for param, count in sorted(
                param_counts.items(), key=lambda item: item[1], reverse=True
            ):
                f.write(
                    f"- `{param}` (in {count}/{len(top_rows_by_map)} top results)\n"
                )
        else:
            f.write(
                "No single parameter stood out in the top results. "
                "Performance seems to be a mix of factors.\n"
            )
        f.write("\n")

        f.write("## Full Results\n\n")
        f.write(
            "The full results for all combinations are in the accompanying CSV file. "
            "Here are the top 10 rows (by MAP):\n\n"
        )
        f.write("| Params | MAP | nDCG | Precision | Recall | Hits | Hit Titles |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        top_10_rows_by_map = sorted(
            rows, key=lambda r: (r["map"], r["ndcg"], r["precision"]), reverse=True
        )[:10]
        for row in top_10_rows_by_map:
            summary = format_override_summary(row)
            f.write(
                f"| `{summary}` | {row['map']:.3f} | {row['ndcg']:.3f} | "
                f"{row['precision']:.3f} | {row['recall']:.3f} | {row['hits']} | "
                f"{row['hit_titles']} |\n"
            )

    print(f"Saved markdown summary to {summary_path}")


def print_summary_report(
    rows: List[Dict[str, Any]], args: argparse.Namespace, golden_ids: Sequence[int]
) -> None:
    """Print a summary of the sweep results to the console."""
    if not rows:
        return

    print("\n--- Sweep Summary ---")
    print(f"Query: '{args.query}'")
    print(f"Golden set size: {len(golden_ids)}")

    def print_best_metric(
        metric_name: str, metric_key: str, higher_is_better: bool
    ) -> None:
        best_score = max(row[metric_key] for row in rows)
        best_rows = [row for row in rows if row[metric_key] == best_score]

        print(f"\nBest {metric_name}:")
        print(f"  {metric_name} = {best_score:.4f}")

        if len(best_rows) == 1:
            best_row = best_rows[0]
            print(f"  Params: {format_override_summary(best_row)}")
            print(
                f"  Metrics: P={best_row['precision']:.3f}, R={best_row['recall']:.3f}, "
                f"MAP={best_row['map']:.3f}, nDCG={best_row['ndcg']:.3f}"
            )
        else:
            print(f"  Found {len(best_rows)} combinations with the highest score:")
            for i, best_row in enumerate(best_rows, 1):
                print(f"  {i}. Params: {format_override_summary(best_row)}")
                print(
                    f"     Metrics: P={best_row['precision']:.3f}, R={best_row['recall']:.3f}, "
                    f"MAP={best_row['map']:.3f}, nDCG={best_row['ndcg']:.3f}"
                )

    print_best_metric("MAP", "map", True)
    print_best_metric("nDCG", "ndcg", True)
    print_best_metric("Precision", "precision", True)
    print_best_metric("Recall", "recall", True)

    sorted_rows = sorted(
        rows, key=lambda r: (r["map"], r["ndcg"], r["precision"]), reverse=True
    )
    if len(sorted_rows) > 5:
        fifth_score = sorted_rows[4]["map"]
        top_rows = [row for row in sorted_rows if row["map"] >= fifth_score]
    else:
        top_rows = sorted_rows

    print(f"\n--- Top {len(top_rows)} Combinations by MAP ---")
    for idx, row in enumerate(top_rows, start=1):
        summary = format_override_summary(row)
        print(
            f"  {idx}. MAP={row['map']:.3f}, P={row['precision']:.3f}, "
            f"R={row['recall']:.3f}, nDCG={row['ndcg']:.3f}"
        )
        print(f"     Params: {summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep recommendation parameters for a single evaluation query."
    )
    parser.add_argument(
        "--query", required=True, help="Query text from evaluation_set.json"
    )
    parser.add_argument(
        "--set",
        type=Path,
        default=DEFAULT_SET_PATH,
        help="Path to evaluation set JSON (default: evaluation/evaluation_set.json).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"/recommend endpoint to call (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument("--user-id", default="u_eval", help="User id for API calls.")
    parser.add_argument(
        "--limit", type=int, default=10, help="Number of recs to fetch."
    )
    parser.add_argument(
        "--serendipity-ratio",
        type=float,
        default=0.0,
        help="Override SERENDIPITY_RATIO for sweep requests (default disables).",
    )
    parser.add_argument(
        "--use-llm-intent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable the LLM intent parser (default: enabled).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout for each request (seconds).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path (default: evaluation/sweep_<query>.csv).",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "md", "both"],
        default="both",
        help="Format for the output report (default: both).",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=1024,
        help="Safety limit for the total number of parameter combinations.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information for each iteration.",
    )

    def add_float_grid_arg(name: str, default: Sequence[float]) -> None:
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            nargs="+",
            type=float,
            default=list(default),
            help=f"Values to try for {name.replace('_', ' ')}.",
        )

    add_float_grid_arg("ann_weight", (0.0, 1.0))
    add_float_grid_arg("rewrite_weight", (0.0, 1.0))
    add_float_grid_arg("mixer_ann_weight", (0.0, 1.0))
    add_float_grid_arg("mixer_trending_weight", (0.0, 1.0))
    add_float_grid_arg("mixer_vote_weight", (0.0, 1.0))
    add_float_grid_arg("mixer_popularity_weight", (0.0, 1.0))
    add_float_grid_arg("mixer_collab_weight", (0.0,))
    add_float_grid_arg("mixer_novelty_weight", (0.0,))
    parser.add_argument(
        "--classic-top-rated-options",
        nargs="+",
        default=["false", "true"],
        help="Boolean values to try for classic_top_rated (e.g., false true).",
    )
    parser.add_argument(
        "--strict-filters-options",
        nargs="+",
        default=["false", "true"],
        help="Boolean values to try for strict_filters (default: false true).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not HAVE_HTTPX:
        raise SystemExit(
            "httpx is required for evaluation sweeps. "
            "Install project requirements (pip install -r requirements.txt)."
        )
    entry = load_entry(args.set, args.query)
    golden_ids = normalise_golden_ids(entry)
    if not golden_ids:
        raise SystemExit(f"Query '{args.query}' has no golden ids.")
    golden_lookup = {
        row["id"]: row.get("title")
        for row in entry.get("golden_set", [])
        if "id" in row
    }
    slug = slugify(args.query)
    output_path = args.output
    if output_path is None:
        output_path = Path("evaluation") / f"sweep_{slug}.csv"
    ensure_parent_dir(output_path)

    print(f"Sweeping parameters for '{args.query}' (golden items: {len(golden_ids)}).")
    combos = list(iter_param_grid(args))
    print(f"Total combinations: {len(combos)}")
    rows: List[Dict[str, Any]] = []
    timeout = httpx.Timeout(args.timeout)
    total = len(combos)
    with httpx.Client(timeout=timeout) as client:
        for idx, (overrides, label) in enumerate(combos, start=1):
            if not args.verbose:
                print(f"Running sweep: {idx}/{total}", end="\r")
            try:
                payload = fetch_recommendations(
                    client,
                    args.base_url,
                    args.query,
                    args.user_id,
                    args.limit,
                    args.use_llm_intent,
                    overrides,
                    args.serendipity_ratio,
                )
            except httpx.HTTPError as exc:
                print(f"\n[error] API call failed for {label}: {exc}")
                continue
            items = payload.get("items", [])
            debug_info = payload.get("debug") or {}
            recommended_ids: List[int] = []
            for item in items:
                tmdb_id = item.get("tmdb_id")
                if isinstance(tmdb_id, int):
                    recommended_ids.append(tmdb_id)
            metrics = compute_metrics(recommended_ids, golden_ids, args.limit)
            hits = [rid for rid in recommended_ids if rid in golden_lookup]
            allowlist_tmdb = [
                rid
                for rid in debug_info.get("allowlist_tmdb_ids") or []
                if isinstance(rid, int)
            ]
            boost_tmdb = [
                rid
                for rid in debug_info.get("boost_tmdb_ids") or []
                if isinstance(rid, int)
            ]
            allowlist_hits = sum(1 for rid in allowlist_tmdb if rid in golden_lookup)
            boost_hits = sum(1 for rid in boost_tmdb if rid in golden_lookup)
            rows.append(
                {
                    **overrides,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "map": metrics["map"],
                    "ndcg": metrics["ndcg"],
                    "hits": len(hits),
                    "allowlist_len": debug_info.get("allowlist_len"),
                    "boost_len": debug_info.get("boost_len"),
                    "allowlist_hits": allowlist_hits,
                    "boost_hits": boost_hits,
                    "debug_classic_top_rated": debug_info.get("classic_top_rated"),
                    "debug_strict_filters": debug_info.get("strict_filters"),
                    "hit_titles": "; ".join(
                        f"{rid}:{golden_lookup.get(rid, '')}" for rid in hits
                    ),
                    "items": format_item_list(recommended_ids),
                }
            )
            if args.verbose:
                hit_display = (
                    ", ".join(golden_lookup.get(rid, str(rid)) for rid in hits)
                    or "none"
                )
                summary = format_override_summary(overrides)
                print(
                    f"[{idx:02}/{total:02}] {summary} "
                    f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                    f"MAP={metrics['map']:.3f} nDCG={metrics['ndcg']:.3f} "
                    f"hits={hit_display} allow_hits={allowlist_hits} boost_hits={boost_hits}"
                )
    if not args.verbose:
        print("\nSweep complete.")

    if not rows:
        raise SystemExit("No successful API calls; report not written.")

    if args.output_format in ["csv", "both"]:
        headers = list(rows[0].keys())
        with output_path.open("w", newline="", encoding="utf-8") as dest:
            writer = csv.DictWriter(dest, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved sweep CSV report to {output_path}")

    if args.output_format in ["md", "both"]:
        write_markdown_report(output_path, rows, args, golden_ids)

    print_summary_report(rows, args, golden_ids)


if __name__ == "__main__":
    main()
