"""Evaluation CLI and regression gate for Seen'emAll ranking quality."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

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
from evaluation.resolver import (
    fetch_embeddings_for_tmdb_ids,
    fetch_titles_for_tmdb_ids,
    resolve_titles_to_tmdb_ids,
)

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
DEFAULT_AB_REPORT_PATH = Path("evaluation/ab_comparison_report.json")
DEFAULT_BASELINE_PATH = Path("evaluation/baseline.json")
DEFAULT_REPORT_PATH = Path("evaluation/report.html")
DEFAULT_DSN = (
    os.environ.get("EVAL_DB_DSN")
    or os.environ.get("DATABASE_URL")
    or "postgresql+psycopg2://app:app@localhost:5432/reco"
)


@dataclass
class EvaluationEntry:
    """Single evaluation query definition."""

    query: str
    golden_ids: List[int]
    raw: Dict[str, Any]
    category: str = "general"
    constraints: Optional[Dict[str, Any]] = None
    user_id: str = "u1"


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
                category=item.get("category", "general"),
                constraints=item.get("constraints"),
                user_id=item.get("user_id", "u1"),
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
                category=item.get("category", "general"),
                constraints=item.get("constraints"),
                user_id=item.get("user_id", "u1"),
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


# --- In-Process Test Client Fallback ---


def _call_in_process(
    params: Dict[str, Any],
    headers: Dict[str, str],
    app_instance: Any = None,
) -> List[Dict[str, Any]]:
    """Fallback helper to call recommendation endpoint directly via FastAPI TestClient."""
    try:
        from starlette.testclient import TestClient

        if app_instance is None:
            from api.main import app as main_app

            app_instance = main_app
        client = TestClient(app_instance)
        resp = client.get("/recommend", params=params, headers=headers)
        if resp.status_code == 200:
            return resp.json().get("items", [])
        print(
            f"In-process request failed with status {resp.status_code}: {resp.text[:200]}"
        )
        return []
    except Exception as exc:
        print(f"In-process invocation error: {exc}")
        return []


_HTTP_CLIENT: Optional[Any] = None


def _get_http_client(timeout_val: float) -> Any:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or getattr(_HTTP_CLIENT, "is_closed", True):
        _HTTP_CLIENT = httpx.Client(timeout=timeout_val)
    return _HTTP_CLIENT


def call_recommendation_api_items(
    query: str,
    params: Dict[str, Any],
    api_key: Optional[str] = None,
    user_id: str = "u1",
    in_process: bool = False,
    base_url: str = "http://localhost:8000/recommend",
    app_instance: Any = None,
) -> List[Dict[str, Any]]:
    """Call recommendation endpoint and return raw item dictionaries."""
    all_params = {"user_id": user_id, "query": query, **params}
    headers: Dict[str, str] = {}
    key = api_key or os.environ.get("API_AUTH_KEY")
    if key:
        headers["X-API-Key"] = key

    if in_process:
        return _call_in_process(all_params, headers, app_instance)

    if not HAVE_HTTPX:
        return _call_in_process(all_params, headers, app_instance)

    timeout_val = float(os.environ.get("EVAL_HTTP_TIMEOUT", "90.0"))
    try:
        client = _get_http_client(timeout_val)
        response = client.get(base_url, params=all_params, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("items", [])
    except (httpx.ConnectError, httpx.ConnectTimeout):
        # Auto-fallback to in-process execution when live server is not running
        return _call_in_process(all_params, headers, app_instance)
    except httpx.ReadTimeout:
        print(f"Warning: request timed out ({timeout_val}s) for query '{query}'.")
        return []
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        print(f"API request failed for query '{query}': {exc}")
        return []


def call_recommendation_api(
    query: str,
    params: Dict[str, Any],
    api_key: Optional[str] = None,
    user_id: str = "u1",
    in_process: bool = False,
    base_url: str = "http://localhost:8000/recommend",
) -> List[int]:
    """Call recommendation endpoint and return TMDB ID list (backward compatible)."""
    items = call_recommendation_api_items(
        query=query,
        params=params,
        api_key=api_key,
        user_id=user_id,
        in_process=in_process,
        base_url=base_url,
    )
    return [item["tmdb_id"] for item in items if "tmdb_id" in item]


# --- Relevance Metrics ---


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
    if k <= 0 or not golden:
        return 0.0
    golden_set = set(golden)
    dcg = 0.0
    for idx, rec_id in enumerate(recommended[:k]):
        if rec_id in golden_set:
            dcg += 1.0 / math.log2(idx + 2)
    ideal = 0.0
    for idx in range(min(k, len(golden))):
        ideal += 1.0 / math.log2(idx + 2)
    return (dcg / ideal) if ideal > 0 else 0.0


# --- Diversity & Intent Metrics ---


def calculate_intra_list_diversity(
    embeddings: Sequence[Sequence[float]] | np.ndarray,
) -> float:
    """Calculate Intra-List Diversity (ILD) as the mean pairwise cosine distance."""
    if embeddings is None or len(embeddings) < 2:
        return 0.0

    arr = np.asarray(embeddings, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normed = arr / norms

    sim_matrix = np.clip(np.dot(normed, normed.T), -1.0, 1.0)
    k = arr.shape[0]
    triu_indices = np.triu_indices(k, k=1)
    pairwise_distances = 1.0 - sim_matrix[triu_indices]

    return float(np.mean(pairwise_distances))


def calculate_catalog_coverage(
    all_recommended_ids: Sequence[int],
    total_catalog_size: Optional[int] = None,
) -> Dict[str, float]:
    """Calculate unique coverage and popularity distribution Gini coefficient."""
    if not all_recommended_ids:
        return {
            "unique_count": 0.0,
            "coverage_ratio": 0.0,
            "gini_coefficient": 0.0,
        }

    unique_items = set(all_recommended_ids)
    unique_count = len(unique_items)
    coverage_ratio = (
        float(unique_count) / float(total_catalog_size)
        if total_catalog_size and total_catalog_size > 0
        else 0.0
    )

    counts = np.array(list(Counter(all_recommended_ids).values()), dtype=float)
    if len(counts) <= 1:
        gini = 0.0
    else:
        diff_matrix = np.abs(counts[:, None] - counts[None, :])
        gini = float(np.sum(diff_matrix) / (2.0 * len(counts) * np.sum(counts)))

    return {
        "unique_count": float(unique_count),
        "coverage_ratio": float(coverage_ratio),
        "gini_coefficient": float(gini),
    }


def calculate_intent_alignment(
    recommended_items: Sequence[Dict[str, Any]],
    constraints: Optional[Dict[str, Any]],
) -> Optional[float]:
    """Calculate fraction of query constraints satisfied across recommended items."""
    if not constraints or not recommended_items:
        return None

    constraint_keys = [
        k
        for k in constraints.keys()
        if k
        in (
            "media_type",
            "genres",
            "max_runtime",
            "min_runtime",
            "year_gte",
            "year_lte",
            "language",
            "providers",
        )
    ]
    if not constraint_keys:
        return None

    item_scores: List[float] = []

    for item in recommended_items:
        satisfied = 0
        total = len(constraint_keys)

        for key in constraint_keys:
            target = constraints[key]
            if key == "media_type":
                if item.get("media_type") == target:
                    satisfied += 1
            elif key == "genres":
                target_genres = [
                    g.lower()
                    for g in (target if isinstance(target, list) else [target])
                ]
                item_genres: List[str] = []
                raw_genres = item.get("genres")
                if isinstance(raw_genres, list):
                    for g in raw_genres:
                        if isinstance(g, dict) and "name" in g:
                            item_genres.append(str(g["name"]).lower())
                        elif isinstance(g, str):
                            item_genres.append(g.lower())
                elif isinstance(raw_genres, str):
                    item_genres.append(raw_genres.lower())
                if any(tg in item_genres for tg in target_genres):
                    satisfied += 1
            elif key == "max_runtime":
                rt = item.get("runtime")
                if rt is not None and rt <= int(target):
                    satisfied += 1
            elif key == "min_runtime":
                rt = item.get("runtime")
                if rt is not None and rt >= int(target):
                    satisfied += 1
            elif key == "year_gte":
                yr = item.get("release_year")
                if yr is not None and yr >= int(target):
                    satisfied += 1
            elif key == "year_lte":
                yr = item.get("release_year")
                if yr is not None and yr <= int(target):
                    satisfied += 1
            elif key == "language":
                lang = item.get("original_language")
                if lang and str(lang).lower() == str(target).lower():
                    satisfied += 1
            elif key == "providers":
                target_providers = [
                    p.lower()
                    for p in (target if isinstance(target, list) else [target])
                ]
                options = item.get("watch_options") or []
                item_providers = [
                    str(opt.get("service") or "").lower()
                    for opt in options
                    if isinstance(opt, dict)
                ]
                if any(
                    any(tp in ip for ip in item_providers) for tp in target_providers
                ):
                    satisfied += 1

        item_scores.append(satisfied / total)

    return float(np.mean(item_scores)) if item_scores else 0.0


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
            "rerank": False,
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
            "ann_weight_override": 0.4,
            "rewrite_weight_override": 0.2,
            "mixer_ann_weight": 0.3,
            "mixer_collab_weight": 0.2,
            "mixer_trending_weight": 0.0,
            "mixer_popularity_weight": 0.0,
            "mixer_vote_weight": 0.6,
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


# --- Evaluation Engine ---


def evaluate_entries(
    entries: Sequence[EvaluationEntry],
    k: int = 10,
    backends: Sequence[str] = ("pgvector",),
    param_grid: Optional[
        Dict[str, Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]]
    ] = None,
    dsn: str = DEFAULT_DSN,
    in_process: bool = False,
    total_catalog_size: Optional[int] = None,
) -> Tuple[
    List[Dict[str, Any]], List[Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Any]]
]:
    """Execute evaluation over all queries and return detailed rows, per-query metrics, and aggregates."""
    if param_grid is None:
        param_grid = default_param_grid()

    rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    aggregated_results: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for backend in backends:
        print(f"=== Evaluating backend: {backend} ===")
        for params_name, params_fn in param_grid.items():
            total_precision = 0.0
            total_recall = 0.0
            total_ap = 0.0
            total_ndcg = 0.0
            total_ild = 0.0
            total_intent = 0.0
            intent_query_count = 0
            processed = 0

            all_recommended_ids: List[int] = []
            category_metrics: Dict[str, Dict[str, float]] = defaultdict(
                lambda: {
                    "precision": 0.0,
                    "recall": 0.0,
                    "map": 0.0,
                    "ndcg": 0.0,
                    "ild": 0.0,
                    "intent": 0.0,
                    "intent_count": 0,
                    "count": 0,
                }
            )

            latencies: List[float] = []

            for idx, entry in enumerate(entries):
                params = params_fn(entry.raw)
                if params is None:
                    continue
                payload = dict(params)
                payload["ann_backend_override"] = backend

                t_start = time.perf_counter()
                items = call_recommendation_api_items(
                    query=entry.query,
                    params=payload,
                    user_id=entry.user_id,
                    in_process=in_process,
                )
                latency_ms = (time.perf_counter() - t_start) * 1000.0
                latencies.append(latency_ms)

                recommended_ids = [it["tmdb_id"] for it in items if "tmdb_id" in it]
                all_recommended_ids.extend(recommended_ids[:k])

                precision = calculate_precision_at_k(
                    recommended_ids, entry.golden_ids, k
                )
                recall = calculate_recall_at_k(recommended_ids, entry.golden_ids, k)
                avg_prec = calculate_average_precision(
                    recommended_ids, entry.golden_ids
                )
                ndcg = calculate_ndcg_at_k(recommended_ids, entry.golden_ids, k)

                # Intra-List Diversity
                top_items = items[:k]
                embeddings_map = fetch_embeddings_for_tmdb_ids(
                    [it["tmdb_id"] for it in top_items if "tmdb_id" in it], dsn
                )
                vectors: List[List[float]] = []
                for it in top_items:
                    tid = it.get("tmdb_id")
                    if tid in embeddings_map:
                        vectors.append(embeddings_map[tid])
                    elif "vector" in it and isinstance(it["vector"], (list, tuple)):
                        vectors.append(list(it["vector"]))

                ild = (
                    calculate_intra_list_diversity(vectors)
                    if len(vectors) >= 2
                    else 0.0
                )

                # Intent Alignment
                intent_score = calculate_intent_alignment(top_items, entry.constraints)

                query_id = generate_query_id(entry.query, idx)
                for rank, rec_id in enumerate(recommended_ids[:k], start=1):
                    rows.append(
                        {
                            "backend": backend,
                            "params_name": params_name,
                            "query": entry.query,
                            "query_id": query_id,
                            "category": entry.category,
                            "rank": rank,
                            "item_id": rec_id,
                            "relevant": int(rec_id in entry.golden_ids),
                        }
                    )

                total_precision += precision
                total_recall += recall
                total_ap += avg_prec
                total_ndcg += ndcg
                total_ild += ild

                if intent_score is not None:
                    total_intent += intent_score
                    intent_query_count += 1

                processed += 1

                cat_stats = category_metrics[entry.category]
                cat_stats["precision"] += precision
                cat_stats["recall"] += recall
                cat_stats["map"] += avg_prec
                cat_stats["ndcg"] += ndcg
                cat_stats["ild"] += ild
                cat_stats["count"] += 1
                if intent_score is not None:
                    cat_stats["intent"] += intent_score
                    cat_stats["intent_count"] += 1

                summary_rows.append(
                    {
                        "backend": backend,
                        "params_name": params_name,
                        "query": entry.query,
                        "query_id": query_id,
                        "category": entry.category,
                        f"precision@{k}": precision,
                        f"recall@{k}": recall,
                        "average_precision": avg_prec,
                        f"ndcg@{k}": ndcg,
                        f"ild@{k}": ild,
                        "intent_alignment": (
                            intent_score if intent_score is not None else ""
                        ),
                        "latency_ms": round(latency_ms, 2),
                    }
                )

            if processed == 0:
                print(f"  No entries processed for parameter set '{params_name}'.\n")
                continue

            coverage_stats = calculate_catalog_coverage(
                all_recommended_ids, total_catalog_size
            )

            agg = {
                "precision": total_precision / processed,
                "recall": total_recall / processed,
                "map": total_ap / processed,
                "ndcg": total_ndcg / processed,
                "ild": total_ild / processed,
                "unique_count": coverage_stats["unique_count"],
                "coverage_ratio": coverage_stats["coverage_ratio"],
                "gini": coverage_stats["gini_coefficient"],
                "intent_alignment": (
                    total_intent / intent_query_count if intent_query_count > 0 else 0.0
                ),
                "latency_mean": float(np.mean(latencies)) if latencies else 0.0,
                "latency_p50": (
                    float(np.percentile(latencies, 50)) if latencies else 0.0
                ),
                "latency_p95": (
                    float(np.percentile(latencies, 95)) if latencies else 0.0
                ),
                "count": processed,
                "category_metrics": dict(category_metrics),
            }
            aggregated_results[(backend, params_name)] = agg

            print(
                f"\n  --- Results for [{params_name}] ---"
                f"\n  Average Precision@{k}: {agg['precision']:.4f}"
                f"\n  Average Recall@{k}:    {agg['recall']:.4f}"
                f"\n  Mean Average Prec(MAP):{agg['map']:.4f}"
                f"\n  Average nDCG@{k}:       {agg['ndcg']:.4f}"
                f"\n  Intra-List Div (ILD):  {agg['ild']:.4f}"
                f"\n  Unique Items Rec'd:    {int(float(str(agg['unique_count'])))}"
                f"\n  Gini Coefficient:      {agg['gini']:.4f}"
                f"\n  Intent Adherence:      {agg['intent_alignment']:.4f}"
                f"\n  Latency Mean:          {agg['latency_mean']:.1f} ms"
                f"\n  Latency P50:           {agg['latency_p50']:.1f} ms"
                f"\n  Latency P95:           {agg['latency_p95']:.1f} ms\n"
            )

    return rows, summary_rows, aggregated_results


def generate_query_id(query: str, index: int) -> str:
    digest = hashlib.md5(query.encode("utf-8")).hexdigest()  # noqa: S324
    return f"q{index:04d}_{digest[:8]}"


# --- Baseline Snapshot Storage & Counterfactual A/B Comparison ---


def save_baseline_snapshot(
    aggregated: Dict[Tuple[str, str], Dict[str, Any]],
    per_query_summary: Sequence[Dict[str, Any]],
    path: Path,
    k: int,
    backend: str,
    config: str,
) -> None:
    """Save an evaluation run as a persistent baseline for future A/B comparisons."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "config": config,
        "k": k,
        "aggregates": aggregated.get((backend, config), {}),
        "per_query_scores": list(per_query_summary),
    }
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    print(f"Baseline snapshot saved to {path}")


def load_baseline_snapshot(path: Path) -> Dict[str, Any]:
    """Load a persistent baseline snapshot from JSON."""
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def run_ab_comparison(
    entries: Sequence[EvaluationEntry],
    k: int = 10,
    baseline_param: str = "ann_only",
    candidate_param: str = "default",
    backend: str = "pgvector",
    dsn: str = DEFAULT_DSN,
    in_process: bool = False,
    report_path: Optional[Path] = DEFAULT_AB_REPORT_PATH,
    baseline_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run side-by-side benchmark comparing baseline vs candidate configuration."""
    grid = default_param_grid()

    if baseline_file and baseline_file.exists():
        snapshot = load_baseline_snapshot(baseline_file)
        base = snapshot.get("aggregates", {})
        baseline_param = snapshot.get("config", baseline_param)
        print(f"Loaded stored baseline from {baseline_file} (config: {baseline_param})")
        selected_grid = {candidate_param: grid[candidate_param]}
    else:
        selected_grid = {
            baseline_param: grid[baseline_param],
            candidate_param: grid[candidate_param],
        }

    _, _, aggregates = evaluate_entries(
        entries=entries,
        k=k,
        backends=[backend],
        param_grid=selected_grid,
        dsn=dsn,
        in_process=in_process,
    )

    if not (baseline_file and baseline_file.exists()):
        base = aggregates.get((backend, baseline_param), {})
    cand = aggregates.get((backend, candidate_param), {})

    def _lift(c_val: float, b_val: float) -> Tuple[float, float]:
        delta = c_val - b_val
        lift_pct = (delta / b_val * 100.0) if b_val > 0 else 0.0
        return delta, lift_pct

    metrics_to_compare = [
        ("nDCG@" + str(k), "ndcg"),
        ("MAP", "map"),
        ("Precision@" + str(k), "precision"),
        ("Recall@" + str(k), "recall"),
        ("Intra-List Diversity", "ild"),
        ("Unique Items Rec'd", "unique_count"),
        ("Intent Adherence", "intent_alignment"),
        ("Latency Mean (ms)", "latency_mean"),
        ("Latency P50 (ms)", "latency_p50"),
        ("Latency P95 (ms)", "latency_p95"),
    ]

    report: Dict[str, Any] = {
        "backend": backend,
        "baseline_param": baseline_param,
        "candidate_param": candidate_param,
        "k": k,
        "comparisons": {},
    }

    print("\n" + "=" * 76)
    print("                    COUNTERFACTUAL A/B BENCHMARK HARNESS")
    print(
        f"             Baseline: [{baseline_param}]  vs  Candidate: [{candidate_param}]"
    )
    print("=" * 76)
    print(
        f"  {'Metric':<24} {'Baseline':>10} {'Candidate':>11} {'Delta':>11} {'% Lift':>12}"
    )
    print("  " + "-" * 72)

    for label, key in metrics_to_compare:
        b_val = float(base.get(key, 0.0))
        c_val = float(cand.get(key, 0.0))
        delta, lift_pct = _lift(c_val, b_val)
        report["comparisons"][key] = {
            "metric": label,
            "baseline": b_val,
            "candidate": c_val,
            "delta": delta,
            "lift_percent": lift_pct,
        }
        sign = "+" if delta >= 0 else ""
        if "latency" in key:
            fmt_b = f"{b_val:.1f}ms"
            fmt_c = f"{c_val:.1f}ms"
            fmt_d = f"{sign}{delta:.1f}ms"
        elif key == "unique_count":
            fmt_b = f"{int(b_val)}"
            fmt_c = f"{int(c_val)}"
            fmt_d = f"{sign}{int(delta)}"
        else:
            fmt_b = f"{b_val:.4f}"
            fmt_c = f"{c_val:.4f}"
            fmt_d = f"{sign}{delta:.4f}"
        print(
            f"  {label:<24} {fmt_b:>10} {fmt_c:>11} {fmt_d:>11} {sign + f'{lift_pct:.2f}%':>12}"
        )

    print("=" * 76 + "\n")

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)
        print(f"A/B comparison report saved to {report_path}")

    return report


# --- Regression Gate ---


def run_regression_gate(
    entries: Sequence[EvaluationEntry],
    k: int = 10,
    min_ndcg: float = 0.72,
    min_ild: float = 0.45,
    backend: str = "pgvector",
    candidate_param: str = "default",
    dsn: str = DEFAULT_DSN,
    in_process: bool = False,
) -> bool:
    """Evaluate candidate pipeline and assert relevance and diversity quality thresholds."""
    grid = default_param_grid()
    selected_grid = {candidate_param: grid[candidate_param]}

    _, _, aggregates = evaluate_entries(
        entries=entries,
        k=k,
        backends=[backend],
        param_grid=selected_grid,
        dsn=dsn,
        in_process=in_process,
    )

    result = aggregates.get((backend, candidate_param), {})
    obs_ndcg = float(result.get("ndcg", 0.0))
    obs_ild = float(result.get("ild", 0.0))

    pass_ndcg = obs_ndcg >= min_ndcg
    pass_ild = obs_ild >= min_ild
    all_passed = pass_ndcg and pass_ild

    status_str = "PASSED" if all_passed else "FAILED"
    banner_char = "=" if all_passed else "!"

    print("\n" + banner_char * 76)
    print(f"               AUTOMATED REGRESSION GATE: {status_str}")
    print(banner_char * 76)
    print(f"  {'Metric':<18} {'Observed':>12} {'Threshold':>12} {'Status':>16}")
    print("  " + "-" * 62)
    print(
        f"  {'nDCG@' + str(k):<18} {obs_ndcg:>12.4f} {min_ndcg:>12.4f} "
        f"{'PASSED' if pass_ndcg else 'FAILED':>16}"
    )
    print(
        f"  {'ILD@' + str(k):<18} {obs_ild:>12.4f} {min_ild:>12.4f} "
        f"{'PASSED' if pass_ild else 'FAILED':>16}"
    )
    print(banner_char * 76 + "\n")

    return all_passed


# --- CSV and Reporting ---


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
        print(f"Unable to write CSV to {path}: {exc}.")
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
        "category",
        f"precision@{k}",
        f"recall@{k}",
        "average_precision",
        f"ndcg@{k}",
        f"ild@{k}",
        "intent_alignment",
        "latency_ms",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=field_order)
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError as exc:
        print(f"Unable to write scores CSV to {path}: {exc}.")
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
        print(f"Unable to write Evidently report to {path}: {exc}.")
    except Exception as exc:  # pragma: no cover
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
        <li style="margin-bottom:6px;"><strong>Intra-List Diversity (ILD)</strong>: Mean pairwise distance between recommended embeddings.</li>
        <li style="margin-bottom:6px;"><strong>Personalisation (top-K)</strong>: Measures catalogue diversity; higher values mean different users see more distinct items.</li>
      </ul>
    </section>
    """

    insertion_point = "<body>"
    if insertion_point in html:
        html = html.replace(insertion_point, insertion_point + explainer, 1)
    else:
        html = explainer + html

    path.write_text(html, encoding="utf-8")


def print_summary(
    aggregated_results: Dict[Tuple[str, str], Dict[str, Any]], k: int
) -> None:
    if not aggregated_results:
        print("No evaluation summaries to report.")
        return

    print("=== Recommendation Quality Summary ===")
    for (backend, params_name), totals in sorted(aggregated_results.items()):
        print(
            f"  {backend:13} {params_name:15} "
            f"P@{k}: {totals['precision']:.3f} | "
            f"R@{k}: {totals['recall']:.3f} | "
            f"MAP: {totals['map']:.3f} | "
            f"nDCG@{k}: {totals['ndcg']:.3f} | "
            f"ILD: {totals['ild']:.3f} | "
            f"Unique: {int(totals['unique_count']):3d} | "
            f"Intent: {totals['intent_alignment']:.3f}"
        )

        cat_metrics = totals.get("category_metrics")
        if cat_metrics:
            print("    Category Breakdown:")
            for cat, cstats in sorted(cat_metrics.items()):
                c_cnt = max(cstats["count"], 1)
                c_int_cnt = max(cstats["intent_count"], 1)
                print(
                    f"      {cat:<14} (n={cstats['count']:2d}) "
                    f"nDCG@{k}: {cstats['ndcg']/c_cnt:.3f} | "
                    f"ILD: {cstats['ild']/c_cnt:.3f} | "
                    f"MAP: {cstats['map']/c_cnt:.3f} | "
                    f"Intent: {cstats['intent']/c_int_cnt:.3f}"
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
        help="Database DSN for resolving titles and embeddings.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run automated regression gate asserting nDCG@10 and ILD thresholds.",
    )
    parser.add_argument(
        "--min-ndcg",
        type=float,
        default=0.72,
        help="Minimum required nDCG@10 for the regression gate (default: 0.72).",
    )
    parser.add_argument(
        "--min-ild",
        type=float,
        default=0.45,
        help="Minimum required Intra-List Diversity for regression gate (default: 0.45).",
    )
    parser.add_argument(
        "--ab-compare",
        action="store_true",
        help="Run counterfactual A/B evaluation comparing baseline vs candidate.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="ann_only",
        help="Baseline parameter configuration for A/B comparison (default: ann_only).",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default="default",
        help="Candidate parameter configuration for A/B or benchmark (default: default).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="pgvector",
        help="ANN backend to evaluate ('pgvector' or 'elasticsearch').",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Force in-process execution using FastAPI TestClient instead of HTTP.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter evaluation entries by category (e.g. franchise, vibe, constraint, cold_start).",
    )
    parser.add_argument(
        "--config",
        "--params",
        dest="config",
        type=str,
        default=None,
        help="Run only a specific parameter configuration from grid (e.g. ann_only, default).",
    )
    parser.add_argument(
        "--save-baseline",
        type=Path,
        nargs="?",
        const=DEFAULT_BASELINE_PATH,
        default=None,
        help="Save evaluation results as a persistent baseline JSON file (default: evaluation/baseline.json).",
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=None,
        help="Path to stored baseline JSON file to use for A/B comparison (avoids re-running baseline queries).",
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
        return entries

    title_path: Path = args.titles_set
    if args.resolve_titles and title_path.exists():
        print(f"Resolving titles from {title_path} using DSN {args.dsn!r}")
        entries = load_title_based_entries(title_path, args.dsn)
    else:
        id_path: Path = args.set
        if id_path.exists():
            print(f"Loading ID-based evaluation set from {id_path}")
            entries = load_id_based_entries(id_path)
        else:
            print("No evaluation set found.")
            return []

    if args.category:
        entries = [e for e in entries if e.category.lower() == args.category.lower()]
        print(f"Filtered to {len(entries)} entries in category '{args.category}'")

    return entries


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    entries = load_entries_from_args(args)
    if not entries:
        print("No evaluation entries available. Exiting.")
        return

    # 1. Regression Gate Mode
    if args.benchmark:
        passed = run_regression_gate(
            entries=entries,
            k=args.k,
            min_ndcg=args.min_ndcg,
            min_ild=args.min_ild,
            backend=args.backend,
            candidate_param=args.candidate,
            dsn=args.dsn,
            in_process=args.in_process,
        )
        if not passed:
            sys.exit(1)
        return

    # 2. Counterfactual A/B Comparison Mode
    if args.ab_compare:
        run_ab_comparison(
            entries=entries,
            k=args.k,
            baseline_param=args.baseline,
            candidate_param=args.candidate,
            backend=args.backend,
            dsn=args.dsn,
            in_process=args.in_process,
            baseline_file=args.baseline_file,
        )
        return

    # 3. Standard Multi-grid Evaluation Mode
    param_grid = default_param_grid()
    selected_cfg = args.config or (
        args.candidate if args.candidate != "default" else None
    )
    if selected_cfg:
        if selected_cfg in param_grid:
            param_grid = {selected_cfg: param_grid[selected_cfg]}
        else:
            print(
                f"Unknown configuration '{selected_cfg}'. Available: {list(param_grid.keys())}"
            )
            return
    backend_variants = [args.backend] if args.backend else ["elasticsearch", "pgvector"]

    per_rank_rows, per_query_summary, aggregated = evaluate_entries(
        entries=entries,
        k=args.k,
        backends=backend_variants,
        param_grid=param_grid,
        dsn=args.dsn,
        in_process=args.in_process,
    )

    save_csv(per_rank_rows, DEFAULT_RESULTS_PATH)
    save_scores_csv(per_query_summary, DEFAULT_SCORES_PATH, args.k)
    maybe_generate_report(per_rank_rows, DEFAULT_REPORT_PATH, args.k, args.dsn)
    if args.save_baseline:
        target_cfg = selected_cfg or list(param_grid.keys())[0]
        target_backend = backend_variants[0]
        save_baseline_snapshot(
            aggregated=aggregated,
            per_query_summary=per_query_summary,
            path=args.save_baseline,
            k=args.k,
            backend=target_backend,
            config=target_cfg,
        )
    print_summary(aggregated, args.k)


if __name__ == "__main__":
    main()
