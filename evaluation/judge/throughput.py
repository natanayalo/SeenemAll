"""Uncached worker benchmark; independent inputs retain the qualified judging contract."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

from evaluation.datasets import load_evaluation_cases
from evaluation.evidence import load_catalog_metadata, pool_item_evidence
from evaluation.judge.base import LocalJudgeAdapter
from evaluation.judge.ollama import discover_ollama_judges
from evaluation.judge.qualification import qualification_record_matches
from evaluation.models import JudgeInput, TypedId
from evaluation.preflight import quick_dev_cases


def benchmark_workers(
    judge: LocalJudgeAdapter,
    inputs: Sequence[JudgeInput],
    *,
    repeats: int = 2,
    worker_counts: Sequence[int] = (1, 2, 4),
) -> dict[str, Any]:
    """Measure fresh calls and compare decisions with repeated sequential controls."""
    if (
        not inputs
        or repeats < 2
        or 1 not in worker_counts
        or any(w not in range(1, 5) for w in worker_counts)
    ):
        raise ValueError(
            "Need inputs, >=2 repeats, and worker counts in 1..4 including one"
        )
    if not judge.is_available():
        raise ValueError("Pinned judge service unavailable")
    # Warm the model using a real input; no generated-text fallback or judgment cache.
    warmup = judge.judge_pair(inputs[0])
    if warmup.execution_status != "success":
        raise ValueError("Judge warmup failed")
    rows = []
    reference = []
    raw = []
    sequence = list(inputs) * repeats
    for workers in sorted(set(worker_counts)):
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outputs = list(pool.map(judge.judge_pair, sequence))
        elapsed = time.perf_counter() - started
        decisions = [
            (
                out.execution_status,
                out.evidence_sufficiency,
                out.grade if out.evidence_sufficiency else None,
            )
            for out in outputs
        ]
        failures = sum(out.execution_status != "success" for out in outputs)
        if workers == 1:
            reference = decisions
        repeatable = all(
            decisions[i] == decisions[i % len(inputs)] for i in range(len(sequence))
        )
        agreement = sum(a == b for a, b in zip(decisions, reference)) / len(sequence)
        rows.append(
            {
                "workers": workers,
                "requests": len(sequence),
                "elapsed_seconds": elapsed,
                "pairs_per_second": len(sequence) / max(elapsed, 1e-9),
                "execution_failures": failures,
                "decision_agreement": agreement,
                "repeatable": repeatable,
                "eligible": failures == 0 and repeatable and agreement == 1.0,
            }
        )
        raw.append({"workers": workers, "outputs": [out.to_dict() for out in outputs]})
        print(
            f"Workers {workers}: {elapsed:.2f}s, agreement {agreement:.1%}, failures {failures}",
            flush=True,
        )
    artifact_verified = judge.is_available()
    eligible = [row for row in rows if row["eligible"] and artifact_verified]
    selected = (
        min(eligible, key=lambda row: row["elapsed_seconds"]) if eligible else rows[0]
    )
    # Avoid recommending parallelism for noise-sized gains on a small diagnostic.
    recommended = (
        selected["workers"]
        if selected["elapsed_seconds"] <= rows[0]["elapsed_seconds"] * 0.9
        else 1
    )
    return {
        "schema": "seenemall.judge-throughput.v1",
        "diagnostic_only": True,
        "qualification_changed": False,
        "judgment_cache_used": False,
        "judge_fingerprint": judge.qualification_fingerprint(),
        "artifact_verified_after_run": artifact_verified,
        "distinct_pairs": len(inputs),
        "repeats": repeats,
        "native_requests": 1 + len(sequence) * len(rows),
        "recommended_workers": recommended if eligible else 1,
        "input_manifest_sha256": hashlib.sha256(
            json.dumps([inp.input_hash() for inp in inputs]).encode()
        ).hexdigest(),
        "input_manifest": [
            {
                "query": inp.query,
                "typed_id": str(inp.evidence.typed_id),
                "input_hash": inp.input_hash(),
                "evidence_hash": inp.evidence.content_hash(),
            }
            for inp in inputs
        ],
        "measurements": rows,
        "raw_outputs": raw,
        "limitations": [
            "Small fixed diagnostic; not a qualification or broad accuracy test",
            "Sequential arm runs first; thermal and scheduling effects may affect throughput",
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)
    judge = discover_ollama_judges()["bespoke-nimble-9b"]
    qualification = json.loads(
        Path("evaluation/.judge_qualification_ollama.json").read_text()
    )
    if not qualification_record_matches(
        judge, qualification["reports"].get(judge.model_name, {})
    ):
        raise ValueError("Benchmark requires the existing qualified v2.2 judge")
    catalog = load_catalog_metadata()
    inputs = []
    for case in quick_dev_cases(load_evaluation_cases(split="dev")):
        references = case.golden_set or case.golden_ids or []
        if not references:
            continue
        tid = TypedId.parse(
            references[0],
            (case.constraints.media_type if case.constraints else None) or "movie",
        )
        inputs.append(JudgeInput(case.query, pool_item_evidence(tid, {}, catalog)))
    result = benchmark_workers(judge, inputs, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - exercised through main in unit tests
    main()
