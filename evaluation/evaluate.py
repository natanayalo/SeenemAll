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

from evaluation.datasets import load_evaluation_cases, load_public_dataset
from evaluation.evidence import load_catalog_metadata, pool_item_evidence
from evaluation.deterministic import (
    check_canonical_order,
    check_deterministic_constraints,
)
from evaluation.judge.base import LocalJudgeAdapter
from evaluation.judge.consensus import (
    ConsensusJudgeEngine,
    JudgmentCache,
    PoolAdjudicator,
)
from evaluation.judge.ollama import discover_ollama_judges
from evaluation.judge.qualification import (
    generate_judge_control_cases as generate_factual_control_cases,
    qualification_record_matches,
    JudgeQualificationRunner,
    inspect_local_hardware,
)
from evaluation.judge.stub import StubJudgeAdapter
from evaluation.latency import LatencyHarness
from evaluation.metrics import (
    calculate_completeness,
    calculate_coverage,
    calculate_known_positive_recall_at_k,
    check_for_duplicates,
    evaluate_comparison_gates,
)
from evaluation.models import (
    EvaluationStatus,
    GainMode,
    TypedId,
)
from evaluation.personalization import PersonalizationHarness
from evaluation.private_benchmark import PrivateBenchmarkHarness
from evaluation.runner import EvaluationRunner, IndexArtifactVerifier
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
    recommended: Sequence[Any],
    golden: Any,
    k: int = 10,
    gain_mode: Optional[GainMode] = None,
    **kwargs: Any,
) -> float:
    if isinstance(golden, dict) or gain_mode is not None:
        from evaluation.metrics import calculate_ndcg_at_k as v2_ndcg

        return v2_ndcg(
            recommended_items=recommended,
            qrels=golden,
            k=k,
            gain_mode=gain_mode or GainMode.GRADED_EXPONENTIAL,
            **kwargs,
        )
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
        "default": lambda entry: {
            "use_llm_intent": True,
            "rerank": True,
            "rerank_provider": "cross_encoder",
        },
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
        "ann_cross_encoder": lambda entry: {
            "mixer_ann_weight": 1.2,
            "mixer_collab_weight": 0.0,
            "mixer_trending_weight": 0.0,
            "mixer_popularity_weight": 0.0,
            "mixer_vote_weight": 0.0,
            "mixer_novelty_weight": 0.0,
            "diversify": False,
            "use_llm_intent": False,
            "rerank": True,
            "rerank_provider": "cross_encoder",
        },
        "cross_encoder": lambda entry: {
            "use_llm_intent": True,
            "rerank": True,
            "rerank_provider": "cross_encoder",
        },
        "baseline_no_rerank": lambda entry: {
            "use_llm_intent": True,
            "rerank": False,
        },
        "calibrated_mixer": lambda entry: {
            "use_llm_intent": True,
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

                if "seed_history" in entry.raw or "train_items" in entry.raw:
                    history_items = (
                        entry.raw.get("seed_history")
                        or entry.raw.get("train_items")
                        or []
                    )
                    if history_items:
                        from api.db.session import get_sessionmaker
                        from evaluation.personalization import seed_persona_fixtures

                        SessionLocal = get_sessionmaker()
                        with SessionLocal() as db_s:
                            persona_dict = {
                                "persona_id": entry.user_id,
                                "seed_history": history_items,
                                "known_negatives": entry.raw.get("known_negatives", []),
                            }
                            # Verify seeded user state; fail explicitly if seeding fails
                            seed_persona_fixtures(db_s, persona_dict, verify=True)

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
                golden_scores = entry.raw.get("golden_scores")
                if golden_scores and isinstance(golden_scores, dict):
                    qrels = {
                        str(TypedId("movie", int(tid))): float(sc)
                        for tid, sc in golden_scores.items()
                    }
                    ndcg = calculate_ndcg_at_k(
                        recommended_ids,
                        qrels,
                        k,
                        gain_mode=GainMode.CONTINUOUS_IDENTITY,
                    )
                else:
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
    allow_regression: bool = False,
) -> bool:
    """Save an evaluation run as a persistent baseline for future A/B comparisons.

    Ensures that the baseline is only updated if candidate metrics have not
    regressed on nDCG@K compared to the existing baseline, or if allow_regression is True.
    """
    new_agg = aggregated.get((backend, config), {})
    if not new_agg:
        print(
            f"Warning: no aggregates found for ({backend}, {config}). Baseline not saved."
        )
        return False

    if path.exists() and not allow_regression:
        try:
            existing = load_baseline_snapshot(path)
            old_agg = existing.get("aggregates", {})
            old_ndcg = float(old_agg.get("ndcg", 0.0))
            new_ndcg = float(new_agg.get("ndcg", 0.0))
            if new_ndcg < old_ndcg:
                print(
                    f"\n[BLOCKED] Refusing to overwrite baseline at {path}:\n"
                    f"Candidate regressed on nDCG@{k}: {new_ndcg:.4f} < {old_ndcg:.4f}.\n"
                    f"Baseline should only be stored if it has not regressed or is explicitly accepted.\n"
                    f"Pass --accept-baseline to force update if this regression is intentional.\n"
                )
                return False
        except Exception as exc:
            print(
                f"Note: Could not compare against existing baseline at {path} ({exc}); proceeding."
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "config": config,
        "k": k,
        "aggregates": new_agg,
        "per_query_scores": list(per_query_summary),
    }
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")
    print(f"Baseline snapshot saved to {path}")
    return True


def load_baseline_snapshot(path: Path) -> Dict[str, Any]:
    """Load a persistent baseline snapshot from JSON."""
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _ab_quality_gate(
    baseline: Dict[str, Any], candidate: Dict[str, Any]
) -> Dict[str, bool]:
    """Require latency improvement without ranking-metric regression."""
    return {
        "latency_mean_improved": float(candidate.get("latency_mean", 0.0))
        < float(baseline.get("latency_mean", 0.0)),
        "latency_p95_improved": float(candidate.get("latency_p95", 0.0))
        < float(baseline.get("latency_p95", 0.0)),
        "ndcg_non_regression": float(candidate.get("ndcg", 0.0))
        >= float(baseline.get("ndcg", 0.0)),
        "map_non_regression": float(candidate.get("map", 0.0))
        >= float(baseline.get("map", 0.0)),
    }


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

    gate_enabled = bool(baseline_file and baseline_file.exists())
    gate_checks = _ab_quality_gate(base, cand) if gate_enabled else {}
    report["quality_gate"] = {
        "enabled": gate_enabled,
        "passed": all(gate_checks.values()),
        "checks": gate_checks,
    }
    print(
        "A/B quality gate: "
        + (
            "PASS"
            if report["quality_gate"]["passed"]
            else ("FAIL" if gate_enabled else "SKIPPED (no stored baseline)")
        )
        + (
            " (mean/P95 latency must improve; nDCG@K/MAP must not regress)."
            if gate_enabled
            else "."
        )
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
        choices=["none", "movielens20m", "tag_genome"],
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
        "--ab-test",
        dest="ab_compare",
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
    parser.add_argument(
        "--accept-baseline",
        "--force-baseline",
        action="store_true",
        help="Allow saving baseline even if candidate metrics regressed against the existing baseline.",
    )
    # Evaluation Suite v2 Flags
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Enable Seen'emAll Evaluation Suite v2 (automated judging, reproducible gates).",
    )
    parser.add_argument(
        "--track",
        type=str,
        default="product",
        choices=["product", "cold_start", "personalization", "anchor"],
        help="Evaluation track ('product', 'cold_start', 'personalization', 'anchor').",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "regression", "full"],
        help="Evaluation split ('dev', 'regression', 'full').",
    )
    parser.add_argument(
        "--judgment-mode",
        type=str,
        default="single_judge",
        choices=["consensus", "single_judge"],
        help="Use the qualified Nimble single judge; consensus is reserved for explicit test panels.",
    )
    parser.add_argument(
        "--judge-config",
        type=str,
        default="nimble",
        choices=["nimble", "stub"],
        help="Nimble via Ollama, or a deterministic stub for automated tests.",
    )
    parser.add_argument(
        "--allow-stub-judges-for-testing",
        action="store_true",
        help="Allow stub judges in authoritative/consensus mode (strictly for automated testing).",
    )
    parser.add_argument(
        "--gain-mode",
        type=str,
        default="graded_exponential",
        choices=["graded_exponential", "continuous_identity"],
        help="Relevance gain formulation for DCG/nDCG.",
    )
    parser.add_argument(
        "--hardware-inspect",
        action="store_true",
        help="Inspect local hardware devices (GPU, NPU, CPU, RAM) and runtime availability.",
    )
    parser.add_argument(
        "--judge-runtime",
        choices=["ollama"],
        default="ollama",
        help="Use the installed, pinned Ollama Nimble model (no downloads).",
    )
    parser.add_argument(
        "--qualify-judges",
        action="store_true",
        help="Run Milestone 0 local judge qualification pilot and report acceptance.",
    )
    parser.add_argument(
        "--option-order-diagnostic",
        action="store_true",
        help="Also measure reversed grading options; this diagnostic does not gate qualification.",
    )
    parser.add_argument(
        "--latency-benchmark",
        action="store_true",
        help="Run warm-model query-cache-cold latency benchmark using ABBA BAAB repetition pattern.",
    )
    parser.add_argument(
        "--personalization-test",
        action="store_true",
        help="Run synthetic personalization behavioral gate benchmark.",
    )
    parser.add_argument(
        "--private-eval",
        action="store_true",
        help="Run private holdout benchmark with confidential output redaction.",
    )
    parser.add_argument(
        "--rescore-only",
        action="store_true",
        help="Rescore existing ranked outputs against updated qrels without re-running retrieval.",
    )
    parser.add_argument(
        "--verify-index",
        action="store_true",
        help="Verify Elasticsearch index artifact matches expected frozen settings/UUID before running.",
    )
    parser.add_argument(
        "--expected-index-checksum",
        type=str,
        default=None,
        help="Expected sha256 checksum of frozen Elasticsearch index artifact.",
    )
    parser.add_argument(
        "--v2-report",
        type=Path,
        default=Path("evaluation/v2_comparison_report.json"),
        help="Output path for v2 comparison report JSON.",
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
                query=item["query"],
                golden_ids=item["golden_ids"],
                raw=dict(item),
                user_id=str(item.get("user_id") or "u1"),
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


def _comparison_params(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Resolve both systems with the explicitly selected retrieval backend."""
    grid = default_param_grid()
    base_params = grid.get(args.baseline, lambda r: {})({}) or {}
    cand_params = grid.get(args.candidate, lambda r: {})({}) or {}
    for params in (base_params, cand_params):
        params["ann_backend_override"] = args.backend
    return base_params, cand_params


def run_evaluation_v2(args: argparse.Namespace) -> int:
    """Execute Evaluation Suite v2 workflow."""
    # 1. Hardware Inspection Mode
    if args.hardware_inspect:
        hw_info = inspect_local_hardware()
        print("\n" + "=" * 70)
        print("          LOCAL HARDWARE & RUNTIME INSPECTION REPORT")
        print("=" * 70)
        print(f"  Detected Devices:        {', '.join(hw_info['devices'])}")
        print(f"  System RAM:              {hw_info['ram_gb']} GB")
        print(
            f"  Intel Arc GPU (16GB):    {'Available' if hw_info['gpu_available'] else 'Not Detected'}"
        )
        print(
            f"  Intel NPU:               {'Available' if hw_info['npu_available'] else 'Not Detected'}"
        )
        print(f"  OpenVINO Devices:        {hw_info['openvino_devices']}")
        print(f"  PyTorch Version:         {hw_info.get('pytorch_version', 'N/A')}")
        print(
            f"  Transformers Version:    {hw_info.get('transformers_version', 'N/A')}"
        )
        print(
            f"  Local Ollama Daemon:     {'Active' if hw_info['ollama_available'] else 'Offline / Inactive'}"
        )
        print("=" * 70 + "\n")
        return 0

    # 2. Local Judge Qualification Pilot (Milestone 0)
    if args.qualify_judges:
        runner = JudgeQualificationRunner()
        candidates: Dict[str, LocalJudgeAdapter] = dict(discover_ollama_judges())
        pilot_families = [
            "fam_vibe_noir",
            "fam_vibe_cozy",
            "fam_fr_bttf",
            "fam_fr_matrix",
            "fam_ent_nolan",
            "fam_ent_spielberg",
            "fam_c_short_action",
            "fam_c_french_romance",
            "fam_vibe_psych",
            "fam_vibe_sports",
            "fam_fr_toy_story",
            "fam_fr_alien",
            "fam_ent_tarantino",
            "fam_ent_villeneuve",
            "fam_c_80s_scifi",
            "fam_c_anim_under_100",
            "fam_vibe_whimsical",
            "fam_vibe_apocalyptic",
            "fam_fr_indy",
            "fam_c_short_comedy",
        ]
        controls = generate_factual_control_cases()
        catalog_cases = load_evaluation_cases(split="dev") + load_evaluation_cases(
            split="regression"
        )
        if catalog_cases:
            pilot_families = list(dict.fromkeys(c.family_id for c in catalog_cases))[
                :20
            ]
        reports: Dict[str, Dict[str, Any]] = {}
        print("\n" + "=" * 76)
        print("          MILESTONE 0: LOCAL JUDGE QUALIFICATION PILOT")
        print("=" * 76)
        for name, judge in candidates.items():
            print(f"Evaluating candidate [{name}] ({judge.runtime})...")
            rep = runner.run_candidate_pilot(
                judge,
                query_families=pilot_families,
                control_cases=controls,
                catalog_cases=catalog_cases,
                run_option_order_diagnostic=getattr(
                    args, "option_order_diagnostic", False
                ),
            )
            reports[name] = rep
            status_str = (
                "QUALIFIED" if rep["qualified"] else "DISQUALIFIED / UNQUALIFIED"
            )
            print(
                f"  -> {status_str}: Repeatability={rep['repeatability_rate']:.1%}, "
                f"Option-order Diagnostic={rep['option_permutation_rate']:.1%} "
                f"({rep.get('permutation_tests', 0)} tests), "
                f"Control Accuracy={rep['control_accuracy']:.1%}, "
                f"Failures={rep['execution_failures']}"
            )

        primary, secondary, tie_breaker, panel_mode = runner.select_judge_panel(
            reports, candidates
        )
        print("\n" + "-" * 76)
        print("  Selected Local Judge Panel:")
        print(f"    Primary Judge:     {primary.model_name if primary else 'None'}")
        print(f"    Secondary Judge:   {secondary.model_name if secondary else 'None'}")
        print(
            f"    Tie-Breaker Judge: {tie_breaker.model_name if tie_breaker else 'None'}"
        )
        print(f"    Operation Mode:    {panel_mode}")
        print("=" * 76 + "\n")

        qual_file = Path("evaluation/.judge_qualification_ollama.json")
        qual_file.parent.mkdir(parents=True, exist_ok=True)
        qual_data = {
            "primary": primary.model_name if primary else None,
            "secondary": secondary.model_name if secondary else None,
            "tie_breaker": tie_breaker.model_name if tie_breaker else None,
            "panel_mode": panel_mode,
            "reports": reports,
        }
        with qual_file.open("w", encoding="utf-8") as fp:
            json.dump(qual_data, fp, indent=2)
        print(f"Saved judge qualification results to {qual_file}\n")
        return 0

    # 3. Latency Benchmark Mode
    if args.latency_benchmark:
        exec_runner = EvaluationRunner(in_process=True)
        harness = LatencyHarness(runner=exec_runner, warmup_count=5, repetition_count=5)
        test_queries = [
            "Star Wars chronological",
            "mind-bending sci-fi",
            "classic movies",
            "action movies under 90 minutes",
            "French romance",
            "Christopher Nolan films",
            "Harry Potter movies",
            "animated family movies",
            "feel good comedy",
            "neo noir",
        ]
        base_p, cand_p = _comparison_params(args)
        print("\n" + "=" * 76)
        print("          WARM-MODEL QUERY-CACHE-COLD LATENCY BENCHMARK")
        print("=" * 76)
        lat_res = harness.benchmark_paired_latency(test_queries, base_p, cand_p)
        print(f"  Baseline P50:            {lat_res.get('baseline_p50_ms', 0)} ms")
        print(f"  Baseline P95:            {lat_res.get('baseline_p95_ms', 0)} ms")
        print(f"  Candidate P50:           {lat_res.get('candidate_p50_ms', 0)} ms")
        print(f"  Candidate P95:           {lat_res.get('candidate_p95_ms', 0)} ms")
        print(
            f"  Median Paired Ratio:     {lat_res.get('median_paired_ratio', 1.0):.4f} (gate: <= 1.10 -> {'PASS' if lat_res.get('median_paired_ratio_pass') else 'FAIL'})"
        )
        print(
            f"  Candidate P95 Ratio:     {lat_res.get('p95_ratio', 1.0):.4f} (gate: <= 1.15 -> {'PASS' if lat_res.get('p95_ratio_pass') else 'FAIL'})"
        )
        gate_status = "PASS" if lat_res.get("passed") else "FAIL"
        print(f"\n  Latency Gate Result:     {gate_status}")
        print("=" * 76 + "\n")
        return 0 if lat_res.get("passed") else 1

    # 4. Synthetic Personalization Behavioral Gate Mode
    if args.personalization_test:
        from api.db.session import get_sessionmaker

        exec_runner = EvaluationRunner(in_process=True)
        pers_harness = PersonalizationHarness(
            runner=exec_runner,
            db_session_factory=get_sessionmaker(),
            verify_seeding=True,
        )
        base_p, cand_p = _comparison_params(args)
        print("\n" + "=" * 76)
        print("          SYNTHETIC PERSONALIZATION BEHAVIORAL GATES")
        print("=" * 76)
        pers_res = pers_harness.run_personalization_benchmark(
            k=args.k, baseline_params=base_p, candidate_params=cand_p
        )
        print(
            f"  Mean Personalization Lift:    {pers_res.get('mean_personalization_lift', 0):+.4f} (gate: >= 0 -> {'PASS' if pers_res.get('mean_lift_pass') else 'FAIL'})"
        )
        print(
            f"  Known-Disliked Violations:    {pers_res.get('total_disliked_violations', 0)} (gate: 0 -> {'PASS' if pers_res.get('disliked_pass') else 'FAIL'})"
        )
        print(
            f"  Max Persona nDCG Decline:     {pers_res.get('max_persona_ndcg_decline', 0):.4f} (gate: <= 0.03 -> {'PASS' if pers_res.get('decline_pass') else 'FAIL'})"
        )
        print("  Persona Details:")
        for pr in pers_res.get("persona_results", []):
            print(
                f"    - {pr['persona_key']}: Cand nDCG={pr['candidate_ndcg']:.3f}, Masked Base={pr['masked_baseline_ndcg']:.3f}, Lift={pr['total_personalization_system_lift']:+.3f}, Disliked Violations={pr['disliked_violations_count']}"
            )
        gate_status = "PASS" if pers_res.get("passed") else "FAIL"
        print(f"\n  Personalization Gate Result:  {gate_status}")
        print("=" * 76 + "\n")
        return 0 if pers_res.get("passed") else 1

    # Verify index artifact if requested
    if getattr(args, "verify_index", False) or getattr(
        args, "expected_index_checksum", None
    ):
        expected_chk = getattr(args, "expected_index_checksum", None)
        manifest_file = Path("evaluation/fixtures/frozen_index_manifest.json")
        if not expected_chk and manifest_file.exists():
            try:
                with manifest_file.open("r", encoding="utf-8") as fp:
                    m_data = json.load(fp)
                    expected_chk = m_data.get("checksum") or m_data.get(
                        "expected_checksum"
                    )
            except Exception:
                pass

        if not expected_chk:
            print(
                "\nError: Index artifact verification requested but no expected checksum provided."
            )
            print(
                "Specify --expected-index-checksum <hash> or provide 'evaluation/fixtures/frozen_index_manifest.json'."
            )
            return int(EvaluationStatus.INVALID)

        verifier = IndexArtifactVerifier(expected_checksum=expected_chk)
        try:
            from api.core.elasticsearch_client import get_elasticsearch_client

            es = get_elasticsearch_client()
            if es is None:
                print(
                    "\nError: Elasticsearch client is unavailable. Cannot verify frozen index artifact."
                )
                return int(EvaluationStatus.INVALID)

            meta = verifier.fetch_live_index_metadata(es)
            if not verifier.verify_reuse(meta):
                print(
                    "\nError: Index artifact verification failed: live index does not match expected frozen artifact!"
                )
                return int(EvaluationStatus.INVALID)
        except Exception as exc:
            print(f"\nIndex artifact verification error: {exc}")
            return int(EvaluationStatus.INVALID)

    # 5. Private Holdout Benchmark Mode
    priv_harness = None
    if args.private_eval:
        priv_harness = PrivateBenchmarkHarness()
        holdout_path = priv_harness.benchmark_dir / "holdout_cases.json"
        if not holdout_path.exists():
            print(f"\nError: Private holdout dataset not found at '{holdout_path}'.")
            print("Private promotion requires an actual isolated holdout dataset.")
            print("Missing holdout files produce an explicit invalid decision.")
            return int(EvaluationStatus.INVALID)

        cases = load_evaluation_cases(path=holdout_path)
        if not cases:
            print(
                f"\nError: Private holdout dataset at '{holdout_path}' contains no valid test cases."
            )
            return int(EvaluationStatus.INVALID)
    else:
        # 6. Core v2 Automated Judging & Comparison Gate Runner
        cases = load_evaluation_cases(track=args.track, split=args.split)
        if not cases:
            print(
                f"No test cases found for track={args.track!r}, split={args.split!r}."
            )
            return int(EvaluationStatus.INVALID)

    print("\n" + "=" * 76)
    print("          SEEN'EMALL EVALUATION SUITE v2 — LOCAL AUTOMATED JUDGING")
    print(
        f"          Track: [{args.track}]  |  Split: [{args.split}]  |  Cases: {len(cases)}"
    )
    print(f"          Baseline: [{args.baseline}]  vs  Candidate: [{args.candidate}]")
    print(
        f"          Judgment Mode: [{args.judgment_mode}]  |  Gain Mode: [{args.gain_mode}]"
    )
    print("=" * 76)

    # Check authoritative run restrictions on stub judges
    is_authoritative = (
        args.judgment_mode == "consensus"
        or args.private_eval
        or args.split in ("regression", "full")
    )
    if is_authoritative and args.judge_config == "stub":
        if not getattr(args, "allow_stub_judges_for_testing", False):
            print(
                "\nError: Stub judges are strictly rejected in authoritative evaluation runs."
            )
            print(
                "Consensus mode, regression/full splits, and private promotion require a qualified local model panel."
            )
            print("Use the qualified Nimble model (--judge-config nimble).")
            return int(EvaluationStatus.INVALID)

    # Check qualification records in authoritative runs
    ollama_candidates = discover_ollama_judges()
    ollama_panel_names = []
    if ollama_candidates is not None and args.judge_config != "stub":
        primary_name = "bespoke-nimble-9b"
        if args.judge_config != "nimble":
            print(f"Error: Unsupported judge configuration: {args.judge_config}")
            return int(EvaluationStatus.INVALID)
        if args.judgment_mode != "single_judge":
            print(
                "Error: Nimble is qualified as a single judge; no production consensus panel is configured."
            )
            return int(EvaluationStatus.INVALID)
        ollama_panel_names = [primary_name] + [
            name for name in ollama_candidates if name != primary_name
        ]
    qual_file = Path("evaluation/.judge_qualification_ollama.json")
    if is_authoritative and not getattr(args, "allow_stub_judges_for_testing", False):
        if not qual_file.exists():
            print(
                f"\nError: Missing qualification record file '{qual_file}' for authoritative evaluation."
            )
            print(
                "Consensus mode, regression/full splits, and private promotion require a qualified local model panel."
            )
            print(
                "Run 'python -m evaluation.evaluate --qualify-judges' before running authoritative evaluation."
            )
            return int(EvaluationStatus.INVALID)

        try:
            with qual_file.open("r", encoding="utf-8") as fp:
                loaded_qual_data = json.load(fp)
            if not isinstance(loaded_qual_data, dict):
                raise ValueError("Malformed qualification file: expected JSON object")
            raw_reports = loaded_qual_data.get("reports")
            if not isinstance(raw_reports, dict):
                raise ValueError(
                    "Malformed qualification file: missing or invalid 'reports' mapping"
                )
        except Exception as exc:
            print(
                f"\nError: Qualification record file '{qual_file}' is invalid or malformed: {exc}"
            )
            return int(EvaluationStatus.INVALID)

        reports_dict: Dict[str, Any] = raw_reports
        panel_model_names = ollama_panel_names[:1]

        for model_name in panel_model_names:
            if model_name not in reports_dict:
                print(
                    f"\nError: Panel judge '{model_name}' has no qualification report in {qual_file}."
                )
                return int(EvaluationStatus.INVALID)
            report_entry = reports_dict[model_name]
            current_judge = ollama_candidates[model_name]
            if not qualification_record_matches(current_judge, report_entry):
                print(
                    f"Error: Stale or unqualified contract for {model_name}; "
                    "rerun --qualify-judges."
                )
                return int(EvaluationStatus.INVALID)

    # Initialize Judges
    p_judge: LocalJudgeAdapter
    s_judge: Optional[LocalJudgeAdapter]
    t_judge: Optional[LocalJudgeAdapter]

    if args.judge_config == "stub":
        p_judge = StubJudgeAdapter("stub-primary", fixed_grade=2)
        s_judge = StubJudgeAdapter("stub-secondary", fixed_grade=2)
        t_judge = StubJudgeAdapter("stub-tie", fixed_grade=2)
    else:
        p_judge = ollama_candidates["bespoke-nimble-9b"]
        s_judge = None
        t_judge = None

    if priv_harness is not None:
        priv_cache = JudgmentCache(priv_harness.benchmark_dir / ".judgment_cache.json")
        adjudicator = PoolAdjudicator(
            primary_judge=p_judge,
            secondary_judge=s_judge if args.judgment_mode == "consensus" else None,
            tie_breaker_judge=t_judge if args.judgment_mode == "consensus" else None,
            cache=priv_cache,
        )
        engine = ConsensusJudgeEngine(
            adjudicator=adjudicator,
            qrels_dir=priv_harness.benchmark_dir / "qrels",
            redact_queries=True,
            immutable=True,
        )
    else:
        adjudicator = PoolAdjudicator(
            primary_judge=p_judge,
            secondary_judge=s_judge if args.judgment_mode == "consensus" else None,
            tie_breaker_judge=t_judge if args.judgment_mode == "consensus" else None,
        )
        engine = ConsensusJudgeEngine(adjudicator=adjudicator)

    exec_runner = EvaluationRunner(in_process=True)
    base_params, cand_params = _comparison_params(args)

    gain_mode = GainMode(args.gain_mode)

    family_base_ndcg: Dict[str, List[float]] = defaultdict(list)
    family_cand_ndcg: Dict[str, List[float]] = defaultdict(list)
    slice_family_deltas: Dict[str, List[float]] = defaultdict(list)

    total_base_rec100 = []
    total_cand_rec100 = []
    hard_violations = 0
    disliked_violations = 0
    chronology_violations = 0
    missing_canonical_items = 0
    cand_completeness_scores = []
    empty_output_cases = 0
    exploratory_covs = []
    unresolved_items = 0
    exec_failures = 0
    unexpected_fallbacks = 0
    duplicate_detected = False

    per_query_rows = []

    catalog_metadata = load_catalog_metadata()

    for case in cases:
        # Retrieve candidates up to depth 100 so Known-Positive Recall@100 measures true retrieval depth
        k_fetch = max(args.k, 100)

        # Run baseline
        base_items, base_trace = exec_runner.run_case(
            case, params=base_params, k=k_fetch
        )
        if base_trace.errors:
            exec_failures += 1
        if base_trace.fallbacks:
            unexpected_fallbacks += 1

        # Run candidate
        cand_items, cand_trace = exec_runner.run_case(
            case, params=cand_params, k=k_fetch
        )
        if cand_trace.errors:
            exec_failures += 1
        if cand_trace.fallbacks:
            unexpected_fallbacks += 1

        if not cand_items:
            empty_output_cases += 1

        if check_for_duplicates(base_items, k=args.k) or check_for_duplicates(
            cand_items, k=args.k
        ):
            duplicate_detected = True

        # Check disliked violations in candidate recommendations
        if case.constraints and case.constraints.disliked_ids:
            disliked_set = set(case.constraints.disliked_ids)
            for it in cand_items[: args.k]:
                try:
                    cid = int(TypedId.parse(it).id)
                    if cid in disliked_set:
                        disliked_violations += 1
                except Exception:
                    pass

        # Check canonical sequence chronology inversions and franchise presence
        seq = getattr(case, "canonical_sequence", None)
        if seq:
            order_res = check_canonical_order(cand_items[: args.k], seq, k=args.k)
            invs = order_res.get("inversions", 0)
            missing_count = order_res.get("missing_prefix_count", 0)
            exact_prefix = order_res.get("exact_prefix_match", False)

            # Separate missing-item checks from inversion counts, and require exact prefix preservation
            if invs > 0:
                chronology_violations += invs
            if missing_count > 0:
                missing_canonical_items += missing_count
            if not exact_prefix and invs == 0 and missing_count == 0:
                chronology_violations += 1

        # Judge declared references independently of whether either top-K retrieved them.
        # Reference IDs nominate evidence to judge; their labels never supply grades.
        references = case.golden_set or case.golden_ids or []
        default_media = (
            case.constraints.media_type if case.constraints else None
        ) or "movie"
        reference_ids = [str(TypedId.parse(ref, default_media)) for ref in references]
        pool_ids = adjudicator.deduplicate_pool(
            [base_items[: args.k], cand_items[: args.k], reference_ids],
            max_pool_size=2 * args.k + len(reference_ids),
        )
        candidate_top_ids = {str(TypedId.parse(it)) for it in cand_items[: args.k]}
        pool_evidence = []
        for pid_str in pool_ids:
            tid = TypedId.parse(pid_str)
            match = next(
                (
                    it
                    for it in (base_items[: args.k] + cand_items[: args.k])
                    if str(TypedId.parse(it)) == pid_str
                ),
                None,
            )
            ev = pool_item_evidence(tid, match or {}, catalog_metadata)
            pool_evidence.append(ev)
            if pid_str in candidate_top_ids and case.constraints:
                valid, _ = check_deterministic_constraints(ev, case.constraints)
                if not valid:
                    hard_violations += 1

        # Calculate completeness with independently determined catalog eligibility
        eligible_catalog_count = getattr(case, "eligible_catalog_count", None)
        if eligible_catalog_count is None:
            if getattr(case, "expected_empty", False):
                eligible_catalog_count = 0
            else:
                eligible_catalog_count = args.k

        if cand_items:
            comp = calculate_completeness(
                cand_items,
                eligible_catalog_count=eligible_catalog_count,
                k=args.k,
            )
            cand_completeness_scores.append(comp)
        else:
            cand_completeness_scores.append(0.0)

        # Adjudicate pool
        qrels, records, _ = engine.label_pool(
            query=case.query,
            pool_evidence=pool_evidence,
            constraints=case.constraints,
            mode=args.judgment_mode,
        )

        for rec in records:
            if any(
                status not in {"success", "abstain"}
                for status in rec.execution_statuses
            ):
                exec_failures += 1
        # Both systems' top-K and the shared recall references require valid judgments,
        # including single-judge abstentions and insufficient evidence.
        case_unresolved = len(set(pool_ids) - qrels.keys())
        unresolved_items += case_unresolved

        # Compute nDCG@K on top-K
        b_ndcg = calculate_ndcg_at_k(
            base_items[: args.k], qrels, k=args.k, gain_mode=gain_mode
        )
        c_ndcg = calculate_ndcg_at_k(
            cand_items[: args.k], qrels, k=args.k, gain_mode=gain_mode
        )

        family_base_ndcg[case.family_id].append(b_ndcg)
        family_cand_ndcg[case.family_id].append(c_ndcg)

        # Compute Known-Positive Recall@100 on retrieved up to 100 items
        b_r100 = calculate_known_positive_recall_at_k(base_items[:100], qrels, k=100)
        c_r100 = calculate_known_positive_recall_at_k(cand_items[:100], qrels, k=100)
        total_base_rec100.append(b_r100)
        total_cand_rec100.append(c_r100)

        # Coverage on top-K
        for items in (base_items, cand_items):
            if items:
                exploratory_covs.append(
                    calculate_coverage(items[: args.k], qrels, k=args.k)
                )

        per_query_rows.append(
            {
                "case_id": case.case_id,
                "family_id": case.family_id,
                "query": case.query,
                "base_ndcg": b_ndcg,
                "cand_ndcg": c_ndcg,
                "delta_ndcg": c_ndcg - b_ndcg,
                "base_recall_100": b_r100,
                "cand_recall_100": c_r100,
                "unresolved_judgments": case_unresolved,
            }
        )

    # Average paraphrases within each family
    family_ndcg_deltas = {}
    for fam_id in family_base_ndcg:
        mean_b = float(np.mean(family_base_ndcg[fam_id]))
        mean_c = float(np.mean(family_cand_ndcg[fam_id]))
        family_ndcg_deltas[fam_id] = mean_c - mean_b

    # Slice deltas: group by unique family_id within each slice to prevent double-counting
    slice_family_map: Dict[str, Dict[str, float]] = defaultdict(dict)
    for c in cases:
        fam_d = family_ndcg_deltas.get(c.family_id, 0.0)
        for s_tag in c.slice_tags:
            slice_family_map[s_tag][c.family_id] = fam_d

    slice_family_deltas = {
        s_tag: list(f_map.values()) for s_tag, f_map in slice_family_map.items()
    }

    base_r100_mean = float(np.mean(total_base_rec100)) if total_base_rec100 else 0.0
    cand_r100_mean = float(np.mean(total_cand_rec100)) if total_cand_rec100 else 0.0

    # Evaluate all gates
    is_stat_promo = (args.split in ("regression", "full")) or bool(args.private_eval)
    gate_result = evaluate_comparison_gates(
        family_ndcg_deltas=family_ndcg_deltas,
        slice_family_deltas=slice_family_deltas,
        baseline_recall_100=base_r100_mean,
        candidate_recall_100=cand_r100_mean,
        hard_constraint_violations=hard_violations,
        disliked_violations=disliked_violations,
        exploratory_coverages=exploratory_covs,
        authoritative_unresolved_count=unresolved_items,
        execution_failures=exec_failures,
        unexpected_fallbacks=unexpected_fallbacks,
        duplicate_outputs_detected=duplicate_detected,
        is_statistical_promotion=is_stat_promo,
        completeness_scores=cand_completeness_scores,
        chronology_violations=chronology_violations,
        missing_canonical_items=missing_canonical_items,
        empty_output_cases=empty_output_cases,
    )

    # Print Report Banner
    status_name = gate_result.status.name
    banner_char = (
        "="
        if gate_result.passed
        else ("!" if gate_result.status == EvaluationStatus.FAIL else "*")
    )
    print("\n" + banner_char * 76)
    print(
        f"               COMPARISON DECISION: {status_name} (Exit {int(gate_result.status)})"
    )
    print(banner_char * 76)
    print(f"  {'Gate Check':<34} {'Status':<10} {'Details'}")
    print("  " + "-" * 72)
    for chk in gate_result.checks:
        chk_status = "PASSED" if chk.passed else "FAILED"
        print(f"  {chk.name:<34} {chk_status:<10} {chk.details}")
    if gate_result.reasons:
        print("\n  Gate Evaluation Reasons:")
        for r in gate_result.reasons:
            print(f"    - {r}")
    print(banner_char * 76 + "\n")

    if priv_harness is not None:
        export_data = priv_harness.submit_candidate(
            candidate_identity=f"candidate_{args.candidate}",
            gate_result=gate_result.to_dict(),
        )
        print("\n" + "=" * 76)
        print("          PRIVATE PROMOTION BENCHMARK (CONFIDENTIAL)")
        print("=" * 76)
        print(f"  Submission Index:       {export_data.get('submission_index')}")
        print(f"  Status:                 {export_data.get('status')}")
        print(f"  Passed:                 {export_data.get('passed')}")
        print(f"  Holdout Refresh Needed: {export_data.get('refresh_required')}")
        print("=" * 76 + "\n")
        return int(export_data.get("exit_code", int(gate_result.status)))

    # Save v2 report JSON
    report_dict = {
        "track": args.track,
        "split": args.split,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "gain_mode": args.gain_mode,
        "judgment_mode": args.judgment_mode,
        "judge_config": args.judge_config,
        "gate_result": gate_result.to_dict(),
        "per_query_results": per_query_rows,
    }
    args.v2_report.parent.mkdir(parents=True, exist_ok=True)
    with args.v2_report.open("w", encoding="utf-8") as fp:
        json.dump(report_dict, fp, indent=2)
    print(f"Evaluation Suite v2 report saved to {args.v2_report}\n")

    return int(gate_result.status)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if (
        args.v2
        or args.qualify_judges
        or args.hardware_inspect
        or args.latency_benchmark
        or args.personalization_test
        or args.private_eval
        or args.rescore_only
    ):
        code = run_evaluation_v2(args)
        sys.exit(code)

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
        report = run_ab_comparison(
            entries=entries,
            k=args.k,
            baseline_param=args.baseline,
            candidate_param=args.candidate,
            backend=args.backend,
            dsn=args.dsn,
            in_process=args.in_process,
            baseline_file=args.baseline_file,
        )
        if not report.get("quality_gate", {}).get("passed", False):
            sys.exit(1)
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
            allow_regression=args.accept_baseline,
        )
    print_summary(aggregated, args.k)


if __name__ == "__main__":
    main()
