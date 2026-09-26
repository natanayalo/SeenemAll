"""Unit tests for recommendation evaluation suite and regression gate."""

import math
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from evaluation.evaluate import (
    EvaluationEntry,
    calculate_average_precision,
    calculate_catalog_coverage,
    calculate_intent_alignment,
    calculate_intra_list_diversity,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    default_param_grid,
    generate_query_id,
    load_baseline_snapshot,
    load_id_based_entries,
    normalise_golden_ids,
    run_ab_comparison,
    run_regression_gate,
    save_baseline_snapshot,
)
from evaluation.resolver import fetch_embeddings_for_tmdb_ids


# --- Relevance Metrics Tests ---

def test_precision_at_k():
    rec = [10, 20, 30, 40, 50]
    golden = [20, 40, 99]

    assert calculate_precision_at_k(rec, golden, 5) == 2 / 5
    assert calculate_precision_at_k(rec, golden, 2) == 1 / 2
    assert calculate_precision_at_k([], golden, 5) == 0.0
    assert calculate_precision_at_k(rec, golden, 0) == 0.0


def test_recall_at_k():
    rec = [10, 20, 30, 40, 50]
    golden = [20, 40]

    assert calculate_recall_at_k(rec, golden, 5) == 1.0
    assert calculate_recall_at_k(rec, golden, 2) == 0.5
    assert calculate_recall_at_k(rec, [], 5) == 0.0
    assert calculate_recall_at_k([], golden, 5) == 0.0


def test_average_precision():
    rec = [10, 20, 30, 40]
    golden = [10, 30]
    # Rank 1: hit -> P@1 = 1/1
    # Rank 2: miss -> P@2 = 1/2
    # Rank 3: hit -> P@3 = 2/3
    # AP = (1.0 + 2/3) / 2 = 5/6
    assert math.isclose(calculate_average_precision(rec, golden), (1.0 + 2 / 3) / 2, rel_tol=1e-5)
    assert calculate_average_precision(rec, []) == 0.0
    assert calculate_average_precision([99, 98], golden) == 0.0


def test_ndcg_at_k():
    rec = [1, 2, 3, 4]
    golden = [1, 2]

    # Perfect ranking at top 2
    ndcg_perfect = calculate_ndcg_at_k([1, 2, 99, 98], golden, 4)
    assert math.isclose(ndcg_perfect, 1.0, rel_tol=1e-5)

    # Suboptimal ranking (hits at ranks 2 and 4)
    ndcg_subopt = calculate_ndcg_at_k([99, 1, 98, 2], golden, 4)
    assert 0.0 < ndcg_subopt < 1.0

    # No hits
    assert calculate_ndcg_at_k([99, 98, 97], golden, 3) == 0.0
    # Empty golden or k <= 0
    assert calculate_ndcg_at_k(rec, [], 4) == 0.0
    assert calculate_ndcg_at_k(rec, golden, 0) == 0.0


# --- Intra-List Diversity (ILD) Tests ---

def test_intra_list_diversity_identical_vectors():
    # Identical vectors should have distance 0.0 -> ILD = 0.0
    vecs = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    assert math.isclose(calculate_intra_list_diversity(vecs), 0.0, abs_tol=1e-6)


def test_intra_list_diversity_orthogonal_vectors():
    # Standard orthogonal basis in 3D: pairwise cosine sim = 0.0 -> dist = 1.0
    vecs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert math.isclose(calculate_intra_list_diversity(vecs), 1.0, rel_tol=1e-5)


def test_intra_list_diversity_edge_cases():
    assert calculate_intra_list_diversity([]) == 0.0
    assert calculate_intra_list_diversity([[1.0, 2.0]]) == 0.0
    assert calculate_intra_list_diversity(None) == 0.0


# --- Catalog Coverage and Gini Coefficient Tests ---

def test_catalog_coverage_uniform():
    recs = [1, 2, 3, 4, 5]
    stats = calculate_catalog_coverage(recs, total_catalog_size=10)

    assert stats["unique_count"] == 5.0
    assert stats["coverage_ratio"] == 0.5
    # All counts = 1 -> Gini = 0.0
    assert math.isclose(stats["gini_coefficient"], 0.0, abs_tol=1e-5)


def test_catalog_coverage_skewed():
    recs = [1] * 10 + [2, 3]
    stats = calculate_catalog_coverage(recs, total_catalog_size=100)

    assert stats["unique_count"] == 3.0
    assert stats["coverage_ratio"] == 0.03
    assert stats["gini_coefficient"] > 0.0


def test_catalog_coverage_empty():
    stats = calculate_catalog_coverage([])
    assert stats["unique_count"] == 0.0
    assert stats["coverage_ratio"] == 0.0
    assert stats["gini_coefficient"] == 0.0


# --- Intent Alignment / Constraint Adherence Tests ---

def test_intent_alignment_full_match():
    constraints = {
        "media_type": "movie",
        "genres": ["Horror"],
        "max_runtime": 120,
        "language": "ko",
        "providers": ["Netflix"],
    }
    items = [
        {
            "media_type": "movie",
            "genres": [{"name": "Horror"}],
            "runtime": 110,
            "original_language": "ko",
            "watch_options": [{"service": "Netflix"}],
        },
        {
            "media_type": "movie",
            "genres": ["Horror", "Thriller"],
            "runtime": 95,
            "original_language": "ko",
            "watch_options": [{"service": "netflix"}],
        },
    ]

    score = calculate_intent_alignment(items, constraints)
    assert score == 1.0


def test_intent_alignment_partial_match():
    constraints = {
        "media_type": "movie",
        "max_runtime": 100,
    }
    items = [
        {"media_type": "movie", "runtime": 90},   # 2/2 satisfied = 1.0
        {"media_type": "movie", "runtime": 130},  # 1/2 satisfied = 0.5
    ]

    score = calculate_intent_alignment(items, constraints)
    assert math.isclose(score, 0.75, rel_tol=1e-5)


def test_intent_alignment_no_constraints():
    items = [{"title": "Movie"}]
    assert calculate_intent_alignment(items, None) is None
    assert calculate_intent_alignment([], {"media_type": "movie"}) is None


# --- A/B Benchmark & Regression Gate Tests ---

def test_regression_gate_pass():
    mock_aggregates = {
        ("pgvector", "default"): {
            "ndcg": 0.785,
            "ild": 0.512,
        }
    }
    with patch("evaluation.evaluate.evaluate_entries", return_value=([], [], mock_aggregates)):
        passed = run_regression_gate(
            entries=[],
            k=10,
            min_ndcg=0.72,
            min_ild=0.45,
            backend="pgvector",
            candidate_param="default",
        )
        assert passed is True


def test_regression_gate_fail_ndcg():
    mock_aggregates = {
        ("pgvector", "default"): {
            "ndcg": 0.680,  # Below 0.72 threshold
            "ild": 0.512,
        }
    }
    with patch("evaluation.evaluate.evaluate_entries", return_value=([], [], mock_aggregates)):
        passed = run_regression_gate(
            entries=[],
            k=10,
            min_ndcg=0.72,
            min_ild=0.45,
            backend="pgvector",
            candidate_param="default",
        )
        assert passed is False


def test_regression_gate_fail_ild():
    mock_aggregates = {
        ("pgvector", "default"): {
            "ndcg": 0.750,
            "ild": 0.380,  # Below 0.45 threshold
        }
    }
    with patch("evaluation.evaluate.evaluate_entries", return_value=([], [], mock_aggregates)):
        passed = run_regression_gate(
            entries=[],
            k=10,
            min_ndcg=0.72,
            min_ild=0.45,
            backend="pgvector",
            candidate_param="default",
        )
        assert passed is False


def test_ab_comparison():
    mock_aggregates = {
        ("pgvector", "ann_only"): {
            "ndcg": 0.600,
            "ild": 0.400,
            "map": 0.500,
            "precision": 0.400,
            "recall": 0.600,
            "unique_count": 50.0,
            "intent_alignment": 0.700,
        },
        ("pgvector", "default"): {
            "ndcg": 0.750,
            "ild": 0.500,
            "map": 0.650,
            "precision": 0.520,
            "recall": 0.750,
            "unique_count": 100.0,
            "intent_alignment": 0.900,
        },
    }
    with patch("evaluation.evaluate.evaluate_entries", return_value=([], [], mock_aggregates)):
        report = run_ab_comparison(
            entries=[],
            k=10,
            baseline_param="ann_only",
            candidate_param="default",
            backend="pgvector",
            report_path=None,
        )
        comp = report["comparisons"]
        assert math.isclose(comp["ndcg"]["delta"], 0.150, rel_tol=1e-5)
        assert math.isclose(comp["ndcg"]["lift_percent"], 25.0, rel_tol=1e-5)
        assert comp["unique_count"]["delta"] == 50.0


# --- Evaluation Set & Normalisation Tests ---

def test_normalise_golden_ids():
    assert normalise_golden_ids({"golden_set": [{"id": 10}, {"id": "20"}, {"id": "bad"}]}) == [10, 20]
    assert normalise_golden_ids({"golden_ids": [1, 2, "3", "bad"]}) == [1, 2, 3]

    with pytest.raises(KeyError):
        normalise_golden_ids({"invalid": []})


def test_evaluation_set_expansion(tmp_path):
    # Verify evaluation set loading with 51 queries
    from pathlib import Path
    set_path = Path("evaluation/evaluation_set.json")
    assert set_path.exists()
    entries = load_id_based_entries(set_path)
    assert len(entries) >= 50

    categories = {e.category for e in entries}
    assert "franchise" in categories
    assert "vibe" in categories
    assert "constraint" in categories
    assert "cold_start" in categories

    cold_starts = [e for e in entries if e.category == "cold_start"]
    assert len(cold_starts) >= 10
    assert any(e.user_id.startswith("u_cold_start") for e in cold_starts)


def test_generate_query_id():
    qid = generate_query_id("test query", 42)
    assert qid.startswith("q0042_")
    assert len(qid) == 14


def test_default_param_grid():
    grid = default_param_grid()
    assert "default" in grid
    assert "ann_only" in grid
    assert "classic_top_rated" in grid
    entry_raw = {"genre_override": "Sci-Fi"}
    assert grid["default"](entry_raw)["use_llm_intent"] is True
    assert grid["genre_override"](entry_raw)["genre_override"] == "Sci-Fi"


def test_save_and_load_baseline_snapshot(tmp_path):
    path = tmp_path / "test_baseline.json"
    agg = {
        ("pgvector", "ann_only"): {
            "ndcg": 0.1145,
            "ild": 0.6624,
            "precision": 0.0529,
            "recall": 0.1209,
            "map": 0.1014,
            "intent_alignment": 0.8079,
            "unique_count": 223,
        }
    }
    summary = [{"query": "Star Wars", "ndcg@10": 0.959}]

    save_baseline_snapshot(agg, summary, path, k=10, backend="pgvector", config="ann_only")
    assert path.exists()

    loaded = load_baseline_snapshot(path)
    assert loaded["backend"] == "pgvector"
    assert loaded["config"] == "ann_only"
    assert loaded["k"] == 10
    assert loaded["aggregates"]["ndcg"] == 0.1145
    assert len(loaded["per_query_scores"]) == 1


def test_ab_comparison_with_stored_baseline(tmp_path):
    b_path = tmp_path / "baseline.json"
    rep_path = tmp_path / "report.json"

    # Pre-save baseline snapshot
    agg = {
        ("pgvector", "ann_only"): {
            "ndcg": 0.50,
            "ild": 0.60,
            "precision": 0.20,
            "recall": 0.30,
            "map": 0.25,
            "intent_alignment": 0.70,
            "unique_count": 50,
        }
    }
    save_baseline_snapshot(agg, [], b_path, k=10, backend="pgvector", config="ann_only")

    # Run A/B compare with stored baseline
    mock_item = {"tmdb_id": 100, "media_type": "movie", "genres": ["Action"]}
    with patch("evaluation.evaluate.call_recommendation_api_items", return_value=[mock_item]):
        with patch("evaluation.evaluate.fetch_embeddings_for_tmdb_ids", return_value={100: [0.1] * 384}):
            entries = [
                EvaluationEntry(
                    query="action",
                    golden_ids=[100],
                    raw={},
                )
            ]
            report = run_ab_comparison(
                entries=entries,
                k=10,
                baseline_param="ann_only",
                candidate_param="default",
                backend="pgvector",
                report_path=rep_path,
                baseline_file=b_path,
            )

    assert "comparisons" in report
    assert report["baseline_param"] == "ann_only"
    assert rep_path.exists()

