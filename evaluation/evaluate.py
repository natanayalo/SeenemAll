"""Evaluation CLI for Seen'emAll ranking quality."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None

try:
    import httpx

    HAVE_HTTPX = True
except ImportError:  # pragma: no cover - missing dependency feedback
    httpx = None
    HAVE_HTTPX = False

from evaluation.datasets import load_public_dataset
from evaluation.resolver import fetch_titles_for_tmdb_ids, resolve_titles_to_tmdb_ids

EVIDENTLY_MODE: Optional[str]

try:
    from evidently.metric_preset import RankingPreset
    from evidently.report import Report

    HAVE_EVIDENTLY = True
    EVIDENTLY_MODE = "ranking_preset"
except Exception:  # pragma: no cover - match newer Evidently builds
    try:
        from evidently import Report as _EvidentlyReport  # noqa: F401
        from evidently.legacy.metric_preset import RecsysPreset
        from evidently.legacy.pipeline.column_mapping import ColumnMapping
        from evidently.legacy.pipeline.column_mapping import RecomType
        from evidently.legacy.pipeline.column_mapping import TaskType
        from evidently.legacy.report import Report as LegacyReport

        HAVE_EVIDENTLY = True
        EVIDENTLY_MODE = "legacy_recsys"
    except Exception:  # pragma: no cover - Evidently unavailable
        HAVE_EVIDENTLY = False
        EVIDENTLY_MODE = None


DEFAULT_SET_PATH = Path("evaluation/evaluation_set.json")
DEFAULT_TITLES_SET_PATH = Path("evaluation/evaluation_set.titles.json")
DEFAULT_RESULTS_PATH = Path("evaluation/evaluation_results.csv")
DEFAULT_SCORES_PATH = Path("evaluation/evaluation_scores.csv")
DEFAULT_REPORT_PATH = Path("evaluation/report.html")
DEFAULT_DSN = os.environ.get(
    "EVAL_DB_DSN", "postgresql+psycopg2://app:app@localhost:5432/reco"
)


@dataclass
class EvaluationEntry:
    """Single evaluation query definition."""

    query: str
    golden_ids: List[int]
    raw: Dict[str, Any]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_id_based_entries(path: Path) -> List[EvaluationEntry]:
    data = load_json(path)
    entries: List[EvaluationEntry] = []

    for item in data:
        try:
            golden = normalise_golden_ids(item)
        except KeyError as exc:
            print(f"Skipping entry for query '{item.get('query', '<unknown>')}': {exc}")
            continue
        if not golden:
            print(f"Skipping entry for query '{item.get('query')}' - no golden ids.")
            continue
        entries.append(
            EvaluationEntry(
                query=item["query"],
                golden_ids=golden,
                raw=item,
            )
        )
    return entries


def load_title_based_entries(path: Path, dsn: str) -> List[EvaluationEntry]:
    data = load_json(path)
    entries: List[EvaluationEntry] = []

    for item in data:
        titles = item.get("golden_titles") or []
        if isinstance(titles, (str, bytes)) or not isinstance(titles, Sequence):
            print(
                f"Skipping query '{item.get('query', '<unknown>')}' - "
                "golden_titles must be a list."
            )
            continue
        structured_titles = [t for t in titles if isinstance(t, dict)]
        if len(structured_titles) != len(titles):
            print(
                f"Warning: some titles for query '{item.get('query')}' "
                "are not objects and will be ignored."
            )

        resolved = resolve_titles_to_tmdb_ids(structured_titles, dsn)
        if not resolved:
            print(
                f"Warning: could not resolve titles for query '{item.get('query')}'. "
                "Entry will be skipped."
            )
            continue

        cloned_item = dict(item)
        cloned_item["golden_ids"] = resolved
        entries.append(
            EvaluationEntry(
                query=item["query"],
                golden_ids=resolved,
                raw=cloned_item,
            )
        )
    return entries


def normalise_golden_ids(entry: Dict[str, Any]) -> List[int]:
    if "golden_set" in entry:
        ids_iter: Iterable[Any] = entry["golden_set"]
        extracted = []
        for row in ids_iter:
            if isinstance(row, dict) and "id" in row:
                try:
                    extracted.append(int(row["id"]))
                except (TypeError, ValueError):
                    continue
        return extracted
    if "golden_ids" in entry:
        ids_iter = entry["golden_ids"]
        result: List[int] = []
        for gid in ids_iter:
            try:
                result.append(int(gid))
            except (TypeError, ValueError):
                continue
        return result
    raise KeyError("Entry must include 'golden_set' or 'golden_ids'.")


def call_recommendation_api(query: str, params: Dict[str, Any]) -> List[int]:
    if not HAVE_HTTPX:
        raise RuntimeError(
            "httpx is not installed. Install project requirements before running evaluations."
        )
    base_url = "http://localhost:8000/recommend"
    all_params = {"user_id": "u1", "query": query, **params}
    try:
        response = httpx.get(base_url, params=all_params, timeout=15.0)
        response.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        print(f"API request failed for query '{query}': {exc}")
        return []
    data = response.json()
    return [item["tmdb_id"] for item in data.get("items", []) if "tmdb_id" in item]


def calculate_precision_at_k(
    recommended: Sequence[int], golden: Sequence[int], k: int
) -> float:
    if not recommended or k <= 0:
        return 0.0
    return len(set(recommended[:k]) & set(golden)) / k


def calculate_recall_at_k(
    recommended: Sequence[int], golden: Sequence[int], k: int
) -> float:
    if not golden:
        return 0.0
    return len(set(recommended[:k]) & set(golden)) / len(golden)


def calculate_average_precision(
    recommended: Sequence[int], golden: Sequence[int]
) -> float:
    if not golden:
        return 0.0
    hits = 0
    precision_sum = 0.0
    golden_set = set(golden)
    for index, rec_id in enumerate(recommended, start=1):
        if rec_id in golden_set:
            hits += 1
            precision_sum += hits / index
    return precision_sum / len(golden)


def calculate_ndcg_at_k(
    recommended: Sequence[int], golden: Sequence[int], k: int
) -> float:
    if k <= 0:
        return 0.0
    golden_set = set(golden)
    dcg = 0.0
    for idx, rec_id in enumerate(recommended[:k]):
        if rec_id in golden_set:
            dcg += 1 / math.log2(idx + 2)
    ideal = 0.0
    for idx in range(min(k, len(golden))):
        ideal += 1 / math.log2(idx + 2)
    return dcg / ideal if ideal > 0 else 0.0


def default_param_grid() -> (
    Dict[str, Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]]
):
    return {
        "default": lambda entry: {"use_llm_intent": True},
        "ann_only": lambda entry: {
            "mixer_ann_weight": 1.2,
            "mixer_collab_weight": 0.0,
            "mixer_trending_weight": 0.0,
            "mixer_popularity_weight": 0.0,
            "mixer_vote_weight": 0.0,
            "mixer_novelty_weight": 0.0,
            "diversify": False,
            "use_llm_intent": False,
        },
        "collab_boost": lambda entry: {
            "mixer_ann_weight": 0.4,
            "mixer_collab_weight": 0.8,
            "mixer_trending_weight": 0.2,
            "use_llm_intent": False,
        },
        "popularity_boost": lambda entry: {
            "mixer_ann_weight": 0.3,
            "mixer_collab_weight": 0.2,
            "mixer_trending_weight": 0.2,
            "mixer_popularity_weight": 1.0,
            "mixer_vote_weight": 0.6,
            "use_llm_intent": False,
        },
        "classic_top_rated": lambda entry: {
            "mixer_ann_weight": 0.4,
            "mixer_collab_weight": 0.2,
            "mixer_trending_weight": 0.0,
            "mixer_popularity_weight": 0.0,
            "mixer_vote_weight": 1.2,
            "mixer_novelty_weight": 0.0,
            "diversify": False,
            "use_llm_intent": True,
            "classic_top_rated": True,
        },
        "no_diversify": lambda entry: {"diversify": False, "use_llm_intent": False},
        "genre_override": lambda entry: (
            {"genre_override": entry.get("genre_override"), "use_llm_intent": False}
            if entry.get("genre_override")
            else None
        ),
        "strict": lambda entry: {
            "use_llm_intent": True,
            "strict_filters": True,
            "genre_override": entry.get("genre_override"),
        },
    }


def evaluate_entries(
    entries: Sequence[EvaluationEntry],
    k: int,
    backends: Sequence[str],
    param_grid: Dict[str, Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for backend in backends:
        print(f"=== Evaluating backend: {backend} ===")
        for params_name, params_fn in param_grid.items():
            total_precision = 0.0
            total_recall = 0.0
            total_ap = 0.0
            total_ndcg = 0.0
            processed = 0

            for idx, entry in enumerate(entries):
                params = params_fn(entry.raw)
                if params is None:
                    continue
                payload = dict(params)
                payload["ann_backend_override"] = backend

                recommended_ids = call_recommendation_api(entry.query, payload)
                if not recommended_ids:
                    print(f"  Warning: no recommendations for query '{entry.query}'.")

                metrics = {
                    "precision": calculate_precision_at_k(
                        recommended_ids, entry.golden_ids, k
                    ),
                    "recall": calculate_recall_at_k(
                        recommended_ids, entry.golden_ids, k
                    ),
                    "map": calculate_average_precision(
                        recommended_ids, entry.golden_ids
                    ),
                    "ndcg": calculate_ndcg_at_k(recommended_ids, entry.golden_ids, k),
                }

                query_id = generate_query_id(entry.query, idx)
                for rank, rec_id in enumerate(recommended_ids[:k], start=1):
                    rows.append(
                        {
                            "backend": backend,
                            "params_name": params_name,
                            "query": entry.query,
                            "query_id": query_id,
                            "rank": rank,
                            "item_id": rec_id,
                            "relevant": int(rec_id in entry.golden_ids),
                        }
                    )

                total_precision += metrics["precision"]
                total_recall += metrics["recall"]
                total_ap += metrics["map"]
                total_ndcg += metrics["ndcg"]
                processed += 1

                summary_rows.append(
                    {
                        "backend": backend,
                        "params_name": params_name,
                        "query": entry.query,
                        "query_id": query_id,
                        f"precision@{k}": metrics["precision"],
                        f"recall@{k}": metrics["recall"],
                        "average_precision": metrics["map"],
                        f"ndcg@{k}": metrics["ndcg"],
                    }
                )

            if processed == 0:
                print(f"  No entries processed for parameter set '{params_name}'.\n")
                continue

            print(
                f"\n  Average Precision@{k}: {total_precision / processed:.4f}\n"
                f"  Average Recall@{k}: {total_recall / processed:.4f}\n"
                f"  Mean Average Precision (MAP): {total_ap / processed:.4f}\n"
                f"  Average nDCG@{k}: {total_ndcg / processed:.4f}\n"
            )

    return rows, summary_rows


def generate_query_id(query: str, index: int) -> str:
    digest = hashlib.md5(
        query.encode("utf-8")
    ).hexdigest()  # noqa: S324 - not for crypto
    return f"q{index:04d}_{digest[:8]}"


def save_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    if not rows:
        print("No evaluation rows to save.")
        return

    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError as exc:
        print(
            f"Unable to write CSV to {path}: {exc}. "
            "Check file permissions or rerun with a writable destination."
        )
        return
    print(f"Evaluation rows saved to {path}")


def save_scores_csv(rows: Sequence[Dict[str, Any]], path: Path, k: int) -> None:
    if not rows:
        print("No evaluation scores to save.")
        return

    field_order = [
        "backend",
        "params_name",
        "query",
        "query_id",
        f"precision@{k}",
        f"recall@{k}",
        "average_precision",
        f"ndcg@{k}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=field_order)
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError as exc:
        print(
            f"Unable to write scores CSV to {path}: {exc}. "
            "Check file permissions or rerun with a writable destination."
        )
        return
    print(f"Evaluation scores saved to {path}")


def maybe_generate_report(
    rows: Sequence[Dict[str, Any]], path: Path, k: int, dsn: str
) -> None:
    if not rows:
        print("No data available for Evidently report.")
        return
    if not HAVE_EVIDENTLY:
        print(
            "Evidently not installed; skipping HTML report. See evaluation/requirements.txt"
        )
        return
    if pd is None:
        print("Pandas not installed; skipping HTML report.")
        return

    df = pd.DataFrame(rows)
    try:
        _run_evidently_report(df, path, k, dsn)
    except PermissionError as exc:
        print(
            f"Unable to write Evidently report to {path}: {exc}. "
            "Check file permissions and rerun once corrected."
        )
    except Exception as exc:  # pragma: no cover - Evidently runtime failure
        print(f"Failed to build Evidently report: {exc}")
    else:
        print(f"Evidently ranking report saved to {path}")


def _run_evidently_report(df: "pd.DataFrame", path: Path, k: int, dsn: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    title_map = fetch_titles_for_tmdb_ids(df["item_id"].unique(), dsn)
    df["item_title"] = df["item_id"].map(title_map)
    df["item_display"] = df["item_title"].fillna(df["item_id"].astype(str))
    df["query"] = df["query"].astype(str)
    df["backend"] = df["backend"].astype(str)
    df["params_name"] = df["params_name"].astype(str)

    def _user_label(row: "pd.Series") -> str:
        return f"{row['query']} [{row['backend']}/{row['params_name']}]"

    df["user_label"] = df.apply(_user_label, axis=1)

    if EVIDENTLY_MODE == "ranking_preset":
        report_df = df[
            [
                "user_label",
                "rank",
                "relevant",
                "item_display",
                "item_title",
                "item_id",
                "query",
                "backend",
                "params_name",
            ]
        ].copy()
        report_df = report_df.rename(
            columns={
                "user_label": "user_id",
                "item_id": "item_tmdb_id",
            }
        )
        report_df["item_id"] = report_df["item_display"]
        report = Report(metrics=[RankingPreset()])
        report.run(reference_data=None, current_data=report_df)
        report.save_html(str(path))
        inject_report_explanations(path)
        return
    if EVIDENTLY_MODE == "legacy_recsys":
        report_df = df[
            [
                "user_label",
                "item_display",
                "rank",
                "relevant",
                "item_title",
                "item_id",
                "query",
                "backend",
                "params_name",
            ]
        ].copy()
        report_df = report_df.rename(
            columns={
                "user_label": "user_id",
                "rank": "prediction",
                "relevant": "target",
                "item_id": "item_tmdb_id",
            }
        )
        report_df["item_id"] = report_df["item_display"]
        report_df["prediction"] = report_df["prediction"].astype(int)
        report_df["target"] = report_df["target"].astype(int)
        column_mapping = ColumnMapping()
        column_mapping.user_id = "user_id"
        column_mapping.item_id = "item_id"
        column_mapping.prediction = "prediction"
        column_mapping.target = "target"
        column_mapping.recommendations_type = RecomType.RANK
        column_mapping.task = TaskType.RECOMMENDER_SYSTEMS
        display_features = [
            "item_title",
            "item_tmdb_id",
            "query",
            "backend",
            "params_name",
        ]
        legacy_report = LegacyReport(
            metrics=[RecsysPreset(k=k, display_features=display_features)]
        )
        legacy_report.run(
            reference_data=None, current_data=report_df, column_mapping=column_mapping
        )
        legacy_report.save_html(str(path))
        inject_report_explanations(path)
        return
    raise RuntimeError("Evidently available but unsupported configuration detected.")


def print_summary(rows: Sequence[Dict[str, Any]], k: int) -> None:
    if not rows:
        print("No evaluation summaries to report.")
        return

    from collections import defaultdict

    grouped: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
        lambda: {"precision": 0.0, "recall": 0.0, "map": 0.0, "ndcg": 0.0, "count": 0}
    )
    backend_totals: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"precision": 0.0, "recall": 0.0, "map": 0.0, "ndcg": 0.0, "count": 0}
    )

    for row in rows:
        backend = row["backend"]
        params_name = row["params_name"]
        grouped_key = (backend, params_name)
        metrics = {
            "precision": float(row[f"precision@{k}"]),
            "recall": float(row[f"recall@{k}"]),
            "map": float(row["average_precision"]),
            "ndcg": float(row[f"ndcg@{k}"]),
        }
        grouped[grouped_key]["precision"] += metrics["precision"]
        grouped[grouped_key]["recall"] += metrics["recall"]
        grouped[grouped_key]["map"] += metrics["map"]
        grouped[grouped_key]["ndcg"] += metrics["ndcg"]
        grouped[grouped_key]["count"] += 1

        backend_totals[backend]["precision"] += metrics["precision"]
        backend_totals[backend]["recall"] += metrics["recall"]
        backend_totals[backend]["map"] += metrics["map"]
        backend_totals[backend]["ndcg"] += metrics["ndcg"]
        backend_totals[backend]["count"] += 1

    print("=== Backend Summary ===")
    for backend, totals in sorted(backend_totals.items()):
        count = max(totals["count"], 1)
        print(
            f"  {backend:13} "
            f"P@{k}: {totals['precision']/count:.3f} | "
            f"R@{k}: {totals['recall']/count:.3f} | "
            f"MAP: {totals['map']/count:.3f} | "
            f"nDCG@{k}: {totals['ndcg']/count:.3f}"
        )

    print("\n=== Backend x Parameters ===")
    for (backend, params_name), totals in sorted(grouped.items()):
        count = max(totals["count"], 1)
        print(
            f"  {backend:13} {params_name:15} "
            f"P@{k}: {totals['precision']/count:.3f} | "
            f"R@{k}: {totals['recall']/count:.3f} | "
            f"MAP: {totals['map']/count:.3f} | "
            f"nDCG@{k}: {totals['ndcg']/count:.3f}"
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Seen'emAll recommender.")
    parser.add_argument(
        "--k", type=int, default=10, help="Cut-off for ranking metrics."
    )
    parser.add_argument(
        "--resolve-titles",
        action="store_true",
        help="Resolve evaluation titles to TMDB ids before running.",
    )
    parser.add_argument(
        "--dataset",
        choices=["none", "movielens20m"],
        default="none",
        help="Optional public dataset to evaluate against.",
    )
    parser.add_argument(
        "--set",
        type=Path,
        default=DEFAULT_SET_PATH,
        help="Path to ID-based evaluation set.",
    )
    parser.add_argument(
        "--titles-set",
        type=Path,
        default=DEFAULT_TITLES_SET_PATH,
        help="Path to title-based evaluation set.",
    )
    parser.add_argument(
        "--dsn",
        type=str,
        default=DEFAULT_DSN,
        help="Database DSN for resolving titles.",
    )
    return parser.parse_args(argv)


def load_entries_from_args(args: argparse.Namespace) -> List[EvaluationEntry]:
    if args.dataset != "none":
        try:
            dataset_entries = load_public_dataset(args.dataset)
        except NotImplementedError as exc:
            print(str(exc))
            raise SystemExit(1) from exc

        entries = [
            EvaluationEntry(
                query=item["query"], golden_ids=item["golden_ids"], raw=dict(item)
            )
            for item in dataset_entries
        ]
        if not entries:
            print("Dataset produced no entries; nothing to evaluate.")
        return entries

    title_path: Path = args.titles_set
    if args.resolve_titles and title_path.exists():
        print(f"Resolving titles from {title_path} using DSN {args.dsn!r}")
        return load_title_based_entries(title_path, args.dsn)

    id_path: Path = args.set
    if id_path.exists():
        print(f"Loading ID-based evaluation set from {id_path}")
        return load_id_based_entries(id_path)

    print("No evaluation set found.")
    return []


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if not HAVE_HTTPX:
        print(
            "httpx not installed; please install project requirements to run evaluations."
        )
        return

    entries = load_entries_from_args(args)
    if not entries:
        print("No evaluation entries available. Exiting.")
        return

    param_grid = default_param_grid()
    backend_variants = ["elasticsearch", "pgvector"]

    per_rank_rows, per_query_summary = evaluate_entries(
        entries=entries,
        k=args.k,
        backends=backend_variants,
        param_grid=param_grid,
    )

    save_csv(per_rank_rows, DEFAULT_RESULTS_PATH)
    save_scores_csv(per_query_summary, DEFAULT_SCORES_PATH, args.k)
    maybe_generate_report(per_rank_rows, DEFAULT_REPORT_PATH, args.k, args.dsn)
    print_summary(per_query_summary, args.k)


def inject_report_explanations(path: Path) -> None:
    try:
        html = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return

    marker = "data-report-explainer"
    if marker in html:
        return

    explainer = """
    <section data-report-explainer style="padding:16px 24px; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:#f5f7fb; border-bottom:1px solid #d9deeb; color:#1a1f36; line-height:1.5;">
      <h1 style="margin-top:0; font-size:1.4rem;">How to Read This Report</h1>
      <p style="margin-bottom:12px;">This Evidently dashboard summarises how each recommendation configuration performs against the evaluation set. Use the quick guide below while you explore the widgets.</p>
      <ul style="margin:0 0 12px 20px; padding:0;">
        <li style="margin-bottom:6px;"><strong>Precision@K / Recall@K</strong>: Precision shows how many of the top-K recommendations were relevant; Recall shows how much of the gold set we recovered in the first K slots.</li>
        <li style="margin-bottom:6px;"><strong>MAP</strong>: Mean Average Precision rewards correctly ordering relevant items earlier in the list.</li>
        <li style="margin-bottom:6px;"><strong>nDCG@K</strong>: Normalised Discounted Cumulative Gain emphasises ranking quality by weighting hits near the top more heavily.</li>
        <li style="margin-bottom:6px;"><strong>Personalisation (top-K)</strong>: Measures catalogue diversity; higher values mean different users see more distinct items. The accompanying “Top 10 popular items” table lists the most common recommendations (now shown with resolved titles).</li>
        <li style="margin-bottom:6px;"><strong>Score Distribution</strong>: Shows how ranking scores are distributed for the current run (reference data is empty when we only evaluate the current system).</li>
        <li style="margin-bottom:6px;"><strong>Recommendation Cases Table</strong>: Breaks down sample queries (rendered as pseudo user IDs) with the items they received so you can inspect individual results and the resolved titles.</li>
      </ul>
      <p style="margin:0;">Hover charts for exact values, and use the parameter labels (e.g. <code>[backend/params]</code>) to compare strategies side by side.</p>
    </section>
    """

    insertion_point = "<body>"
    if insertion_point in html:
        html = html.replace(insertion_point, insertion_point + explainer, 1)
    else:
        html = explainer + html

    path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
