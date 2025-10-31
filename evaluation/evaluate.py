"""Script for evaluating the recommendation system with different parameters."""

import json
import httpx
import csv
import math
from typing import Iterable, List, Dict, Any


def load_evaluation_set(path: str) -> List[Dict[str, Any]]:
    """Load the evaluation set from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def normalise_golden_ids(entry: Dict[str, Any]) -> List[int]:
    """Return the list of TMDB ids for a single evaluation entry."""
    if "golden_set" in entry:
        return [row["id"] for row in entry["golden_set"] if "id" in row]
    if "golden_ids" in entry:
        ids: Iterable[Any] = entry["golden_ids"]
        return [int(gid) for gid in ids]
    raise KeyError("Entry must include 'golden_set' or 'golden_ids'.")


def call_recommendation_api(query: str, params: Dict[str, Any]) -> List[int]:
    """Call the recommendation API and return a list of item IDs."""
    base_url = "http://localhost:8000/recommend"
    all_params = {"user_id": "u1", "query": query, **params}
    try:
        response = httpx.get(base_url, params=all_params, timeout=5.0)
        response.raise_for_status()
        data = response.json()
        return [item["tmdb_id"] for item in data.get("items", [])]
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        print(f"API request failed: {e}")
        return []


def calculate_precision_at_k(
    recommended_ids: List[int], golden_ids: List[int], k: int
) -> float:
    """Calculate Precision@k."""
    if not recommended_ids or k == 0:
        return 0.0
    return len(set(recommended_ids[:k]) & set(golden_ids)) / k


def calculate_recall_at_k(
    recommended_ids: List[int], golden_ids: List[int], k: int
) -> float:
    """Calculate Recall@k."""
    if not golden_ids:
        return 0.0
    return len(set(recommended_ids[:k]) & set(golden_ids)) / len(golden_ids)


def calculate_average_precision(
    recommended_ids: List[int], golden_ids: List[int]
) -> float:
    """Calculate Average Precision (AP)."""
    if not golden_ids:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for i, rec_id in enumerate(recommended_ids):
        if rec_id in golden_ids:
            hits += 1
            precision_sum += hits / (i + 1)

    return precision_sum / len(golden_ids)


def calculate_ndcg_at_k(
    recommended_ids: List[int], golden_ids: List[int], k: int
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (nDCG)@k."""
    dcg = 0.0
    for i, rec_id in enumerate(recommended_ids[:k]):
        if rec_id in golden_ids:
            dcg += 1 / math.log2(i + 2)

    idcg = 0.0
    for i in range(min(k, len(golden_ids))):
        idcg += 1 / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def main():
    """Run the evaluation."""
    evaluation_set = load_evaluation_set("evaluation/evaluation_set.json")

    param_grid = {
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
        "no_diversify": lambda entry: {"diversify": False, "use_llm_intent": False},
        "genre_override": lambda entry: (
            {"genre_override": entry.get("genre_override"), "use_llm_intent": False}
            if entry.get("genre_override")
            else None
        ),
    }

    k = 10
    results = []

    for params_name, params_fn in param_grid.items():
        print(f"--- Evaluating parameters: {params_name} ---")
        total_precision = 0.0
        total_recall = 0.0
        total_ap = 0.0
        total_ndcg = 0.0
        processed_entries = 0

        for item in evaluation_set:
            query = item["query"]
            try:
                golden_ids = normalise_golden_ids(item)
            except KeyError as exc:
                print(f"Skipping entry for query '{query}': {exc}")
                continue
            params = params_fn(item)
            if params is None:
                continue

            recommended_ids = call_recommendation_api(query, params)

            precision = calculate_precision_at_k(recommended_ids, golden_ids, k)
            recall = calculate_recall_at_k(recommended_ids, golden_ids, k)
            ap = calculate_average_precision(recommended_ids, golden_ids)
            ndcg = calculate_ndcg_at_k(recommended_ids, golden_ids, k)

            total_precision += precision
            total_recall += recall
            total_ap += ap
            total_ndcg += ndcg
            processed_entries += 1

            results.append(
                {
                    "params_name": params_name,
                    "query": query,
                    f"precision@{k}": precision,
                    f"recall@{k}": recall,
                    "average_precision": ap,
                    f"ndcg@{k}": ndcg,
                }
            )

            print(f'  Query: "{query}" ')
            print(f"    Precision@{k}: {precision:.4f}")
            print(f"    Recall@{k}: {recall:.4f}")
            print(f"    Average Precision: {ap:.4f}")
            print(f"    nDCG@{k}: {ndcg:.4f}")

        if processed_entries == 0:
            print("  No evaluation entries processed for this parameter set.\n")
            continue

        avg_precision = total_precision / processed_entries
        avg_recall = total_recall / processed_entries
        map_score = total_ap / processed_entries
        avg_ndcg = total_ndcg / processed_entries

        print(f"\n  Average Precision@{k}: {avg_precision:.4f}")
        print(f"  Average Recall@{k}: {avg_recall:.4f}")
        print(f"  Mean Average Precision (MAP): {map_score:.4f}")
        print(f"  Average nDCG@{k}: {avg_ndcg:.4f}")
        print("\n")

    # Save results to CSV
    if results:
        with open("evaluation/evaluation_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print("Evaluation results saved to evaluation/evaluation_results.csv")
    else:
        print("No evaluation results to save.")


if __name__ == "__main__":
    main()
