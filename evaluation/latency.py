"""Recommendation latency benchmark harness and judge throughput reporting."""

from __future__ import annotations

import time
import math
import os
from typing import Any, Dict, List, Sequence

import numpy as np

from evaluation.judge.base import LocalJudgeAdapter
from evaluation.models import JudgeInput
from evaluation.runner import EvaluationRunner


class LatencyHarness:
    """Warm-model, query-cache-cold latency benchmark harness.

    Enforces:
      - 5 warmup queries per system
      - 5 repetitions of ABBA BAAB per query with seeded query ordering (seed 42)
      - Exact-query cache bypassing
      - Median paired latency ratio <= 1.10
      - Candidate P95 <= 1.15 * Baseline P95
    """

    def __init__(
        self,
        runner: EvaluationRunner,
        seed: int = 42,
        warmup_count: int = 5,
        repetition_count: int = 5,
    ) -> None:
        if warmup_count < 0 or repetition_count < 1:
            raise ValueError(
                "Warmup count must be nonnegative and repetitions positive"
            )
        self.runner = runner
        self.seed = seed
        self.warmup_count = warmup_count
        self.repetition_count = repetition_count

    def warmup(
        self,
        queries: Sequence[str],
        baseline_params: Dict[str, Any],
        candidate_params: Dict[str, Any],
    ) -> List[str]:
        """Run warmups to prime models, connection pools, and metadata caches."""
        errors: List[str] = []
        warmup_queries = list(queries[: self.warmup_count])
        for q in warmup_queries:
            _, b_trace = self.runner.execute_query(
                q, params=baseline_params, bypass_cache=True
            )
            if b_trace.errors:
                errors.extend(b_trace.errors)
            _, c_trace = self.runner.execute_query(
                q, params=candidate_params, bypass_cache=True
            )
            if c_trace.errors:
                errors.extend(c_trace.errors)
        return errors

    def benchmark_paired_latency(
        self,
        queries: Sequence[str],
        baseline_params: Dict[str, Any],
        candidate_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute paired ABBA BAAB latency benchmark and verify gates."""
        if not queries:
            return {"error": "no_queries_provided", "passed": False}

        # 1. Warm-up
        total_errors: List[str] = []
        warmup_errors = self.warmup(queries, baseline_params, candidate_params)
        if warmup_errors:
            total_errors.extend(warmup_errors)

        # 2. Seeded ordering
        rng = np.random.default_rng(self.seed)
        shuffled_queries = list(queries)
        rng.shuffle(shuffled_queries)

        baseline_latencies: List[float] = []
        candidate_latencies: List[float] = []
        paired_ratios: List[float] = []

        total_inferences = 0
        total_cache_hits = 0
        # Sequence of ABBA BAAB for each repetition block
        # A = baseline, B = candidate
        pattern = ["A", "B", "B", "A", "B", "A", "A", "B"]

        def measure(query, parameters, label):
            nonlocal total_inferences, total_cache_hits
            items, trace = self.runner.execute_query(
                query, params=parameters, bypass_cache=True
            )
            elapsed = trace.timings_ms.get("total_latency_ms")
            if (
                not isinstance(elapsed, (float, int))
                or not math.isfinite(elapsed)
                or elapsed <= 0
            ):
                total_errors.append(
                    f"Missing or invalid request timing for {label}: {query}"
                )
                elapsed = 0.0
            total_errors.extend(trace.errors)
            total_errors.extend(
                f"Unexpected fallback: {fallback}" for fallback in trace.fallbacks
            )
            if not items:
                total_errors.append(f"Empty response for {label}: {query}")
            counts = trace.inference_counts
            if not isinstance(counts, dict):
                total_errors.append(
                    f"Malformed inference telemetry for {label}: {query}"
                )
                return elapsed
            inf_count = counts.get("actual_inferences_performed", 0)
            cache_hits = counts.get("cache_hits", 0)
            if any(
                not isinstance(v, int) or isinstance(v, bool) or v < 0
                for v in (inf_count, cache_hits)
            ):
                total_errors.append(
                    f"Malformed inference counters for {label}: {query}"
                )
                return elapsed
            total_inferences += inf_count
            total_cache_hits += cache_hits
            if cache_hits or trace.cache_hits:
                total_errors.append(f"Result-cache hit in cache-cold {label}: {query}")
            enabled = parameters.get("rerank")
            if enabled is None:
                enabled = os.getenv("RERANK_ENABLED", "1").strip().lower() not in {
                    "0",
                    "false",
                    "no",
                    "off",
                }
            provider = parameters.get("rerank_provider") or "cross_encoder"
            if provider not in {"cross_encoder", "small"}:
                provider = "cross_encoder"
            provider_stats = trace.inference_providers.get(provider, {})
            if enabled and (
                inf_count == 0 or provider_stats.get("successful_calls", 0) == 0
            ):
                total_errors.append(
                    f"Missing expected {provider} inference for {label}: {query}"
                )
            if not enabled and inf_count:
                total_errors.append(
                    f"Unexpected reranking inference for disabled {label}: {query}"
                )
            if any(
                stats.get("failures", 0) for stats in trace.inference_providers.values()
            ):
                total_errors.append(f"Reranker failure for {label}: {query}")
            return elapsed

        for q in shuffled_queries:
            q_base_times: List[float] = []
            q_cand_times: List[float] = []
            for _ in range(self.repetition_count):
                for step in pattern:
                    if step == "A":
                        q_base_times.append(measure(q, baseline_params, "baseline"))
                    else:
                        q_cand_times.append(measure(q, candidate_params, "candidate"))
            med_b = float(np.median(q_base_times))
            med_c = float(np.median(q_cand_times))
            baseline_latencies.extend(q_base_times)
            candidate_latencies.extend(q_cand_times)
            if med_b > 0:
                paired_ratios.append(med_c / med_b)

        base_p50 = float(np.percentile(baseline_latencies, 50))
        base_p95 = float(np.percentile(baseline_latencies, 95))
        cand_p50 = float(np.percentile(candidate_latencies, 50))
        cand_p95 = float(np.percentile(candidate_latencies, 95))

        median_paired_ratio = float(np.median(paired_ratios)) if paired_ratios else 1.0
        p95_ratio = (cand_p95 / base_p95) if base_p95 > 0 else 1.0

        median_ratio_pass = median_paired_ratio <= 1.10
        p95_pass = p95_ratio <= 1.15
        has_errors = len(total_errors) > 0
        overall_pass = median_ratio_pass and p95_pass and not has_errors

        return {
            "passed": overall_pass,
            "errors": total_errors,
            "error_count": len(total_errors),
            "actual_inferences_performed": total_inferences,
            "cache_hits": total_cache_hits,
            "median_paired_ratio": round(median_paired_ratio, 4),
            "median_paired_ratio_pass": median_ratio_pass,
            "threshold_median_ratio": 1.10,
            "p95_ratio": round(p95_ratio, 4),
            "p95_ratio_pass": p95_pass,
            "threshold_p95_ratio": 1.15,
            "baseline_p50_ms": round(base_p50, 2),
            "baseline_p95_ms": round(base_p95, 2),
            "candidate_p50_ms": round(cand_p50, 2),
            "candidate_p95_ms": round(cand_p95, 2),
            "total_measurements": len(baseline_latencies) + len(candidate_latencies),
        }


def benchmark_judge_throughput(
    judge: LocalJudgeAdapter,
    sample_inputs: Sequence[JudgeInput],
) -> Dict[str, Any]:
    """Measure local judge labeling throughput and cache speed.

    Reports:
      - First-pass labeling speed (items/sec)
      - Reusable cache / second-pass speed
      - Elapsed time
    """
    if not sample_inputs:
        return {"error": "no_inputs"}

    # 1. First-pass throughput
    t0 = time.perf_counter()
    for inp in sample_inputs:
        _ = judge.judge_pair(inp)
    first_pass_time = time.perf_counter() - t0
    first_pass_qps = (
        len(sample_inputs) / first_pass_time if first_pass_time > 0 else 0.0
    )

    # 2. Second-pass (exercises cache/reusable embeddings if supported)
    t1 = time.perf_counter()
    for inp in sample_inputs:
        _ = judge.judge_pair(inp)
    second_pass_time = time.perf_counter() - t1
    second_pass_qps = (
        len(sample_inputs) / second_pass_time if second_pass_time > 0 else 0.0
    )

    cache_stats = {}
    if hasattr(judge, "get_cache_stats"):
        cache_stats = judge.get_cache_stats()

    return {
        "model_name": judge.model_name,
        "sample_count": len(sample_inputs),
        "first_pass_elapsed_sec": round(first_pass_time, 3),
        "first_pass_items_per_sec": round(first_pass_qps, 2),
        "second_pass_elapsed_sec": round(second_pass_time, 3),
        "second_pass_items_per_sec": round(second_pass_qps, 2),
        "cache_stats": cache_stats,
    }
