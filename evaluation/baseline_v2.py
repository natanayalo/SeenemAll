"""Versioned production references measured with the v2 evaluation contract."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.datasets import load_evaluation_cases
from evaluation.deterministic import (
    check_canonical_order,
    check_deterministic_constraints,
)
from evaluation.evidence import load_catalog_metadata, pool_item_evidence
from evaluation.judge.consensus import PoolAdjudicator
from evaluation.judge.ollama import discover_ollama_judges
from evaluation.judge.qualification import qualification_record_matches
from evaluation.latency import LatencyHarness
from evaluation.metrics import (
    calculate_average_precision,
    calculate_completeness,
    calculate_known_positive_recall_at_k,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    check_for_duplicates,
)
from evaluation.models import EvaluationStatus, GainMode, TypedId
from evaluation.runner import EvaluationRunner, IndexArtifactVerifier

SCHEMA = "seenemall.production-baseline.v2.1"


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def code_identity() -> dict[str, Any]:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD", "--", "api", "evaluation"])
    return {
        "revision": revision,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "production_source_sha256": fingerprint(
            {
                str(path.as_posix()): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path("api").rglob("*.py"))
            }
        ),
    }


def runtime_identity() -> dict[str, Any]:
    from api.core.cross_encoder import (
        DEFAULT_CROSS_ENCODER_MODEL,
        get_cross_encoder_device,
    )

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "execution": "in_process",
        "cross_encoder_model": os.getenv(
            "CROSS_ENCODER_MODEL", DEFAULT_CROSS_ENCODER_MODEL
        ),
        "cross_encoder_backend": os.getenv("CROSS_ENCODER_BACKEND", "openvino"),
        "cross_encoder_device": get_cross_encoder_device(),
        "note": "Locally executed production configuration; timings are specific to this host and model runtime",
    }


def warm_judgments(
    adjudicator: Any, query: str, evidence: list[Any], workers: int
) -> None:
    """Parallelize independent inference only; commit cache entries sequentially."""
    adjudicator.warm_pool(query, evidence, workers)


def index_identity(backend: str) -> dict[str, Any]:
    if backend != "elasticsearch":
        return {"checksum": None, "note": "No Elasticsearch artifact for this backend"}
    from api import config
    from api.core.elasticsearch_client import get_elasticsearch_client

    verifier = IndexArtifactVerifier()
    metadata = verifier.fetch_live_index_metadata(
        get_elasticsearch_client(), config.ELASTICSEARCH_ITEMS_INDEX
    )
    return {"checksum": verifier.compute_index_checksum(metadata), "metadata": metadata}


def load_reference(
    path: Path, cases: list[Any], judge: Any, args: Any
) -> dict[str, Any]:
    reference = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(reference)
    checksum = payload.pop("snapshot_sha256", None)
    if checksum != fingerprint(payload):
        raise ValueError("V2 baseline snapshot checksum mismatch")
    expected = {
        "schema": SCHEMA,
        "reference_valid": True,
        "dataset_sha256": fingerprint(
            [row["case"] for row in reference["per_query_results"]]
        ),
        "catalog_sha256": fingerprint(
            load_catalog_metadata()
            if getattr(args, "evidence_version", "v2.2") == "v2.2"
            else load_catalog_metadata(evidence_version=args.evidence_version)
        ),
        "judge_fingerprint": judge.qualification_fingerprint(),
        "backend": args.backend,
        "k": args.k,
        "gain_mode": args.gain_mode,
        "index_artifact_checksum": index_identity(args.backend)["checksum"],
    }
    for key, value in expected.items():
        if reference.get(key) != value:
            raise ValueError(f"V2 baseline contract mismatch: {key}")
    rows = {row["case"]["case_id"]: row for row in reference["per_query_results"]}
    for case in cases:
        if case.case_id not in rows or rows[case.case_id]["case"] != case.to_dict():
            raise ValueError("V2 baseline case coverage or dataset mismatch")
    return reference


def capture_baseline(args: Any) -> int:
    """Record current quality, including defects, without a promotion requirement.

    Failed executions cannot establish a usable reference. Successful abstentions
    remain explicit gaps: scores are provisional until the comparison coverage
    gate resolves them. Recording those gaps does not waive promotion checks.
    """
    from evaluation.evaluate import default_param_grid

    destination = args.save_v2_baseline
    if destination.exists():
        raise ValueError("V2 baselines are immutable; choose a new snapshot path")
    if (
        args.track != "product"
        or args.split != "full"
        or args.judge_config != "nimble"
        or args.judgment_mode != "single_judge"
        or args.allow_stub_judges_for_testing
        or args.private_eval
    ):
        raise ValueError(
            "Production v2 baselines require the full product split and qualified Nimble"
        )
    config = args.config or args.candidate
    grid = default_param_grid()
    if config not in grid:
        raise ValueError(f"Unknown production configuration: {config}")
    params = grid[config]({})
    if not params:
        raise ValueError("Production configuration must resolve explicit parameters")
    params["ann_backend_override"] = args.backend
    if not 1 <= args.baseline_judge_workers <= 4:
        raise ValueError("Baseline judging requires between one and four workers")
    cases = load_evaluation_cases(track=args.track, split=args.split)
    if not cases:
        raise ValueError("Full product dataset is missing or empty")
    evidence_version = getattr(args, "evidence_version", "v2.2")
    judge = (
        discover_ollama_judges()
        if evidence_version == "v2.2"
        else discover_ollama_judges(evidence_version)
    )["bespoke-nimble-9b"]
    qualification = json.loads(
        Path(
            "evaluation/.judge_qualification_ollama.json"
            if evidence_version == "v2.2"
            else "evaluation/.judge_qualification_ollama_v2.3.json"
        ).read_text()
    )
    if not judge.is_available() or not qualification_record_matches(
        judge, qualification["reports"].get(judge.model_name, {})
    ):
        raise ValueError(
            "Production baseline requires the available, qualified judge artifact"
        )
    catalog = (
        load_catalog_metadata()
        if evidence_version == "v2.2"
        else load_catalog_metadata(evidence_version=evidence_version)
    )
    index_artifact = index_identity(args.backend)
    build_identity = code_identity()
    started_at = datetime.now(timezone.utc).isoformat()
    adjudicator = PoolAdjudicator(primary_judge=judge)
    # EvaluationRunner currently executes via TestClient for both flag values.
    runner = EvaluationRunner(in_process=True)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    unresolved = 0
    for index, case in enumerate(cases, 1):
        print(f"Baseline query {index}/{len(cases)}: {case.query}", flush=True)
        items, trace = runner.run_case(case, params=params, k=max(100, args.k))
        ranked = [str(TypedId.parse(item)) for item in items]
        failures.extend(trace.errors + trace.fallbacks)
        if check_for_duplicates(items, k=args.k):
            failures.append(f"Duplicate output: {case.case_id}")
        references = case.golden_set or case.golden_ids or []
        media = (case.constraints.media_type if case.constraints else None) or "movie"
        pool = adjudicator.deduplicate_pool(
            [ranked[: args.k], [str(TypedId.parse(ref, media)) for ref in references]],
            max_pool_size=args.k + len(references),
        )
        results_by_id = {str(TypedId.parse(item)): item for item in items}
        records = []
        violations = []
        pool_evidence = [
            (
                pool_item_evidence(
                    TypedId.parse(identifier),
                    results_by_id.get(identifier, {}),
                    catalog,
                )
                if evidence_version == "v2.2"
                else pool_item_evidence(
                    TypedId.parse(identifier),
                    results_by_id.get(identifier, {}),
                    catalog,
                    evidence_version=evidence_version,
                )
            )
            for identifier in pool
        ]
        warm_judgments(
            adjudicator, case.query, pool_evidence, args.baseline_judge_workers
        )
        for identifier, evidence in zip(pool, pool_evidence):
            record = adjudicator.adjudicate_pair(
                case.query, evidence, case.constraints, mode="single_judge"
            )
            records.append(record.to_dict())
            if any(
                status not in {"success", "abstain"}
                for status in record.execution_statuses
            ):
                failures.append(f"Judge execution failed: {case.case_id}/{identifier}")
            if identifier in ranked[: args.k] and case.constraints:
                _, reasons = check_deterministic_constraints(evidence, case.constraints)
                violations.extend(reasons)
        qrels = {
            record["typed_id"]: float(record["grade"])
            for record in records
            if record["grade"] is not None
        }
        case_unresolved = len(set(pool) - qrels.keys())
        unresolved += case_unresolved
        metrics = {
            "ndcg_at_k": calculate_ndcg_at_k(
                ranked, qrels, args.k, GainMode(args.gain_mode)
            ),
            "precision_at_k": calculate_precision_at_k(ranked, qrels, args.k),
            "ap_at_k": calculate_average_precision(ranked[: args.k], qrels),
            "known_positive_recall_100": calculate_known_positive_recall_at_k(
                ranked, qrels, 100
            ),
            "completeness": calculate_completeness(
                items,
                (
                    case.eligible_catalog_count
                    if case.eligible_catalog_count is not None
                    else args.k
                ),
                args.k,
            ),
        }
        rows.append(
            {
                "case": case.to_dict(),
                "ranked_ids": ranked,
                "qrels": qrels,
                "judgments": records,
                "metrics": metrics,
                "unresolved_judgments": case_unresolved,
                "unexpected_empty_output": not items and not case.expected_empty,
                "constraint_violations": violations,
                "canonical_order": (
                    check_canonical_order(
                        items[: args.k], case.canonical_sequence, args.k
                    )
                    if case.canonical_sequence
                    else None
                ),
                "execution": {
                    "errors": trace.errors,
                    "fallbacks": trace.fallbacks,
                    "inference_counts": trace.inference_counts,
                    "inference_providers": trace.inference_providers,
                },
            }
        )
    families: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        families[row["case"]["family_id"]].append(row["metrics"])
    quality = {
        key: float(
            np.mean(
                [
                    np.mean([metrics[key] for metrics in members])
                    for members in families.values()
                ]
            )
        )
        for key in rows[0]["metrics"]
    }
    snapshot: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "purpose": "Current production reference, not an improvement or promotion claim",
        "config": config,
        "request_params": params,
        "backend": args.backend,
        "track": args.track,
        "split": args.split,
        "k": args.k,
        "gain_mode": args.gain_mode,
        "dataset_sha256": fingerprint([case.to_dict() for case in cases]),
        "catalog_sha256": fingerprint(catalog),
        "index_artifact_checksum": index_artifact["checksum"],
        "index_artifact": index_artifact,
        "judge_fingerprint": judge.qualification_fingerprint(),
        "judge_provenance": judge.get_provenance("").to_dict(),
        "judge_workers": args.baseline_judge_workers,
        "evidence_version": evidence_version,
        "code": build_identity,
        "runtime": runtime_identity(),
        "query_count": len(rows),
        "family_count": len(families),
        "quality": quality,
        "quality_aggregation": "mean within family, then mean across families; positives grade >=2",
        "execution_failures": failures,
        "unresolved_judgments": unresolved,
        "reference_valid": not failures,
        "judgments_complete": unresolved == 0,
        "quality_status": "complete" if unresolved == 0 else "inconclusive",
        "quality_defects": {
            "unexpected_empty_queries": sum(
                row["unexpected_empty_output"] for row in rows
            ),
        },
        "unjudged_policy": "Unknown grades count as zero only for provisional arithmetic; unresolved evidence prevents a conclusive comparison",
        "per_query_results": rows,
    }
    if index_identity(args.backend)["checksum"] != index_artifact["checksum"]:
        failures.append("Search index changed during production reference capture")
        snapshot["reference_valid"] = False
    if snapshot["reference_valid"]:
        # Ten queries, 400 timed requests; separate from the 101 quality queries.
        latency_queries = []
        for tag in (
            "franchise",
            "vibe",
            "constraint",
            "entity",
            "vibe",
            "constraint",
            "entity",
            "franchise",
            "vibe",
            "constraint",
        ):
            match = next(
                (
                    case.query
                    for case in cases
                    if tag in case.slice_tags and case.query not in latency_queries
                ),
                None,
            )
            if match is not None:
                latency_queries.append(match)
        snapshot["latency"] = LatencyHarness(runner).benchmark_paired_latency(
            latency_queries, params, params
        )
        snapshot["latency"]["queries"] = latency_queries
        snapshot["latency"][
            "policy"
        ] = "warm models; result caches bypassed; five ABBA/BAAB repetitions; same production config in both arms"
        snapshot["latency"][
            "runtime_note"
        ] = "Hardware and API runtime are deployment-specific; rerun both configurations together for future latency comparisons"
        snapshot["reference_valid"] = snapshot["latency"].get("error_count", 1) == 0
        if not snapshot["reference_valid"]:
            failures.extend(
                snapshot["latency"].get("errors", [])
                or ["Latency measurement contains execution failures"]
            )
    snapshot["snapshot_sha256"] = fingerprint(snapshot)
    args.v2_report.parent.mkdir(parents=True, exist_ok=True)
    args.v2_report.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not snapshot["reference_valid"]:
        print(
            f"Incomplete reference: {len(failures)} execution failures, {unresolved} unresolved judgments. Diagnostics: {args.v2_report}"
        )
        return int(
            EvaluationStatus.INVALID if failures else EvaluationStatus.INCONCLUSIVE
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        json.dump(snapshot, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(f"Production v2 baseline saved: {destination}; quality: {quality}")
    return 0
