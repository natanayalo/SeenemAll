"""Milestone 0: Local judge qualification harness, hardware inspection, and pilot."""

from __future__ import annotations

import logging
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from evaluation.evidence import build_item_evidence, load_catalog_metadata
from evaluation.judge.base import LocalJudgeAdapter
from evaluation.judge.ollama import discover_ollama_judges
from evaluation.judge.rubric import QUALIFICATION_PROTOCOL_VERSION
from evaluation.models import (
    DeterministicConstraint,
    ItemEvidence,
    JudgeInput,
    TestCase,
    TypedId,
)

logger = logging.getLogger(__name__)


def generate_judge_control_cases() -> List[TestCase]:
    """100 factual controls with represented evidence, half matches and half violations.

    Provider access and whole-list chronology have separate deterministic checks.
    """
    cases = []
    for category in ("year", "runtime", "media"):
        count = 40 if category != "media" else 20
        for index in range(count):
            expected = 0 if index < count // 2 else 2
            if category == "year":
                limit = 1990 + index
                query = f"movies released before {limit}"
                constraint = DeterministicConstraint(max_year=limit - 1)
            elif category == "runtime":
                limit = 80 + index
                query = f"short movies under {limit} minutes"
                constraint = DeterministicConstraint(max_runtime=limit - 1)
            else:
                media = "tv" if index % 2 else "movie"
                query = (
                    "science fiction television series"
                    if media == "tv"
                    else "feature movie"
                )
                constraint = DeterministicConstraint(media_type=media)
            cases.append(
                TestCase(
                    case_id=f"judge_{category}_{index}",
                    family_id=f"judge_{category}_{index}",
                    track="product",
                    split="dev",
                    task="constraint",
                    slice_tags=["factual_control", category],
                    query=query,
                    constraints=constraint,
                    is_factual_control=True,
                    expected_grade=expected,
                )
            )
    return cases


def inspect_local_hardware() -> Dict[str, Any]:
    """Inspect local hardware and available runtimes.

    Detects: CPU, Intel Arc 140v GPU, Intel NPU, RAM, PyTorch, OpenVINO, Ollama.
    """
    info: Dict[str, Any] = {
        "devices": ["CPU"],
        "gpu_available": False,
        "npu_available": False,
        "ram_gb": 32.0,  # Known host specification
        "openvino_devices": [],
        "pytorch_version": None,
        "transformers_version": None,
        "ollama_available": False,
    }

    # OpenVINO inspection
    try:
        import openvino as ov

        core = ov.Core()
        devices = list(core.available_devices)
        info["openvino_devices"] = devices
        if "GPU" in devices:
            info["gpu_available"] = True
            info["devices"].append("Intel Arc 140v GPU (16GB)")
        if "NPU" in devices:
            info["npu_available"] = True
            info["devices"].append("Intel NPU")
    except Exception as exc:
        info["openvino_error"] = str(exc)

    # PyTorch inspection
    try:
        import torch

        info["pytorch_version"] = torch.__version__
    except Exception:
        pass

    # Transformers inspection
    try:
        import transformers

        info["transformers_version"] = transformers.__version__
    except Exception:
        pass

    # Ollama inspection
    try:
        info["ollama_available"] = any(
            judge.is_available() for judge in discover_ollama_judges().values()
        )
    except Exception:
        info["ollama_available"] = False

    return info


def qualification_record_matches(judge: LocalJudgeAdapter, record: Any) -> bool:
    """Require current provenance and the measured gates, not a cached pass flag."""
    if not isinstance(record, dict) or record.get("qualified") is not True:
        return False
    if record.get("fingerprint") != judge.qualification_fingerprint():
        return False
    if record.get("qualification_protocol") != QUALIFICATION_PROTOCOL_VERSION:
        return False
    if record.get("scope") != "qualification":
        return False
    for key, minimum_count in (
        ("pilot_pairs", 400),
        ("pilot_distinct_pairs", 400),
        ("pilot_families", 20),
        ("repeat_tests", 100),
        ("control_tests", 100),
    ):
        value = record.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum_count
        ):
            return False
    for key, minimum in (
        ("repeatability_rate", 0.99),
        ("control_accuracy", 0.95),
    ):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not math.isfinite(value) or not minimum <= value <= 1.0:
            return False
    return all(
        isinstance(record.get(key), int)
        and not isinstance(record[key], bool)
        and record[key] == 0
        for key in ("execution_failures", "malformed_count")
    )


class JudgeQualificationRunner:
    """Milestone 0 qualification pilot evaluator."""

    def __init__(
        self, output_dir: Path = Path("evaluation/qualification_reports")
    ) -> None:
        self.output_dir = output_dir

    def build_pilot_items_for_family(
        self, family_id: str, count: int = 20, catalog_cases: Optional[List[Any]] = None
    ) -> List[ItemEvidence]:
        """Deterministically sample 20 items per family for the 400-pair pilot using real catalog cases."""
        import hashlib

        # If catalog cases not supplied, attempt loading from frozen product datasets
        cases = catalog_cases
        if not cases:
            try:
                from evaluation.datasets import load_evaluation_cases

                cases = load_evaluation_cases(
                    track="product", split="dev"
                ) + load_evaluation_cases(track="product", split="regression")
            except Exception:
                cases = []

        real_items: List[ItemEvidence] = []
        family_items: List[ItemEvidence] = []
        seen_ids = set()

        cat_meta = load_catalog_metadata()
        if cases:
            for case in cases:
                family = getattr(case, "family_id", None)
                goldens = (
                    getattr(case, "golden_set", None)
                    or getattr(case, "golden_ids", [])
                    or []
                )
                for golden in goldens:
                    try:
                        tid = TypedId.parse(golden)
                    except (ValueError, TypeError):
                        continue
                    metadata = dict(cat_meta.get(str(tid), {}))
                    # Benchmark labels are references, never catalog evidence.
                    # In particular a wrong ID/title pairing cannot rename a film.
                    if not metadata:
                        continue
                    evidence = build_item_evidence(tid, metadata)
                    if not evidence.title or not evidence.synopsis:
                        continue
                    key = str(tid)
                    if key in seen_ids:
                        continue
                    seen_ids.add(key)
                    if family == family_id:
                        family_items.append(evidence)
                    else:
                        real_items.append(evidence)

        # Combine family-specific items first, then sample deterministically from catalog
        combined = family_items + real_items
        if combined:
            seed_bytes = family_id.encode("utf-8")
            h_int = int(hashlib.sha256(seed_bytes).hexdigest()[:8], 16)
            start_idx = h_int % len(combined)
            selected = family_items[:count]
            remaining = [item for item in combined if item not in selected]
            selected.extend(
                remaining[(start_idx + i) % len(remaining)]
                for i in range(min(count - len(selected), len(remaining)))
            )
            return selected

        raise ValueError(
            f"No real catalog items found to build pilot items for family '{family_id}'. "
            "Qualification requires real catalog candidates; fictional generation is strictly prohibited."
        )

    def run_candidate_pilot(
        self,
        judge: LocalJudgeAdapter,
        query_families: List[str],
        control_cases: Optional[List[TestCase]] = None,
        catalog_cases: Optional[List[Any]] = None,
        *,
        items_per_family: int = 20,
        repeat_every: int = 4,
        run_option_order_diagnostic: bool = False,
    ) -> Dict[str, Any]:
        """Execute pilot benchmark on a single judge candidate."""
        controls = (
            control_cases
            if control_cases is not None
            else generate_judge_control_cases()
        )
        if not 1 <= items_per_family <= 20 or not 1 <= repeat_every <= 4:
            raise ValueError("Pilot size must be 1..20 items and repeat interval 1..4")
        report: Dict[str, Any] = {
            "model_name": judge.model_name,
            "runtime": judge.runtime,
            "quantization": judge.quantization,
            "total_judgments": 0,
            "repeatability_pass": False,
            "repeatability_rate": 0.0,
            "option_permutation_pass": False,
            "option_permutation_rate": 0.0,
            "option_permutation_role": "diagnostic",
            "permutation_execution_failures": 0,
            "control_accuracy_pass": False,
            "control_accuracy": 0.0,
            "malformed_count": 0,
            "abstention_count": 0,
            "execution_failures": 0,
            "mean_latency_ms": 0.0,
            "qualified": False,
            "fingerprint": judge.qualification_fingerprint(),
            "qualification_protocol": QUALIFICATION_PROTOCOL_VERSION,
            "scope": "diagnostic",
            "pilot_sufficient_count": 0,
            "pilot_evidence_coverage": 0.0,
            "repeat_sufficient_pairs": 0,
        }

        latencies: List[float] = []
        repeat_matches = 0
        total_repeat_tests = 0
        repeat_grade_matches = 0
        repeat_sufficiency_matches = 0
        permutation_matches = 0
        total_perm_tests = 0
        control_correct = 0

        # 1. Pilot query families (20 families x 20 items = 400 judgments)
        all_pilot_inputs: List[JudgeInput] = []
        for fam in query_families[:20]:
            family_queries = [
                c.query
                for c in (catalog_cases or [])
                if getattr(c, "family_id", None) == fam
                and isinstance(getattr(c, "query", None), str)
                and c.query.strip()
            ]
            if catalog_cases and not family_queries:
                raise ValueError(f"No real query found for pilot family '{fam}'")
            items = self.build_pilot_items_for_family(
                fam, count=items_per_family, catalog_cases=catalog_cases
            )
            for index, it in enumerate(items):
                query = (
                    family_queries[index % len(family_queries)]
                    if family_queries
                    else f"Recommendations for {fam}"
                )
                inp = JudgeInput(query=query, evidence=it)
                all_pilot_inputs.append(inp)

        report["pilot_pairs"] = len(all_pilot_inputs)
        report["pilot_distinct_pairs"] = len(
            {inp.input_hash() for inp in all_pilot_inputs}
        )
        report["pilot_families"] = len(set(query_families[:20]))

        if not judge.is_available():
            report["execution_failures"] = len(all_pilot_inputs) + len(controls[:100])
            report["repeatability_pct"] = 0.0
            report["control_accuracy_pct"] = 0.0
            report["permutation_agreement_pct"] = 0.0
            report["qualified"] = False
            return report

        def checked_judgment(inp):
            t0 = time.perf_counter()
            out = judge.judge_pair(inp)
            if out.execution_status == "failed":
                out = judge.judge_pair(inp)  # One bounded retry for transient failures.
            latencies.append((time.perf_counter() - t0) * 1000)
            report["total_judgments"] += 1
            if out.execution_status == "malformed":
                report["malformed_count"] += 1
            elif out.execution_status != "success":
                report["execution_failures"] += 1
            if out.execution_status == "abstain" or not out.evidence_sufficiency:
                report["abstention_count"] += 1
            return out

        for index, inp in enumerate(all_pilot_inputs):
            out = checked_judgment(inp)
            if out.execution_status == "success" and out.evidence_sufficiency:
                report["pilot_sufficient_count"] += 1
            if index == 0 and out.execution_status == "failed":
                # An unresolved runtime failure already makes qualification impossible.
                # Avoid hundreds of repeated requests to an incompatible local server.
                report["stopped_early"] = True
                report["runtime_error"] = out.raw_response
                return report
            if (index + 1) % repeat_every == 0:
                total_repeat_tests += 1
                repeat_out = checked_judgment(inp)
                if out.execution_status == repeat_out.execution_status == "success":
                    grade_match = repeat_out.grade == out.grade
                    sufficiency_match = (
                        repeat_out.evidence_sufficiency == out.evidence_sufficiency
                    )
                    repeat_grade_matches += int(grade_match)
                    repeat_sufficiency_matches += int(sufficiency_match)
                    repeat_matches += int(
                        sufficiency_match
                        and (grade_match or not out.evidence_sufficiency)
                    )
                    report["repeat_sufficient_pairs"] += int(
                        out.evidence_sufficiency and repeat_out.evidence_sufficiency
                    )
                if run_option_order_diagnostic:
                    total_perm_tests += 1
                    try:
                        _, _, is_cons = judge.test_option_order_permutation(inp)
                        if is_cons is True:
                            permutation_matches += 1
                    except Exception:
                        report["permutation_execution_failures"] += 1

        # 2. Automatically verifiable controls (100 controls)
        for idx, ctrl in enumerate(controls[:100]):
            expected = ctrl.expected_grade if ctrl.expected_grade is not None else 0
            ctrl_tid = TypedId("movie", 99000 + len(latencies))
            ctrl_media_type = "movie"
            ctrl_year = 2010
            ctrl_runtime = 95
            ctrl_genres = ["Science Fiction", "Drama"]
            ctrl_title = f"Control Item for {ctrl.case_id}"
            ctrl_synopsis = "A science-fiction drama following a difficult journey."

            if ctrl.constraints:
                if ctrl.constraints.max_year is not None:
                    if expected == 0:
                        # Violate max year constraint: released AFTER max_year
                        ctrl_year = ctrl.constraints.max_year + 5
                    else:
                        ctrl_year = max(1950, ctrl.constraints.max_year - 5)

                if ctrl.constraints.max_runtime is not None:
                    if expected == 0:
                        # Violate runtime constraint: much longer than allowed
                        ctrl_runtime = ctrl.constraints.max_runtime + 45
                    else:
                        ctrl_runtime = max(40, ctrl.constraints.max_runtime - 15)

                if ctrl.constraints.media_type:
                    if expected == 0:
                        ctrl_media_type = (
                            "movie" if ctrl.constraints.media_type == "tv" else "tv"
                        )
                        ctrl_tid = TypedId(ctrl_media_type, 99000 + len(latencies))
                    else:
                        ctrl_media_type = ctrl.constraints.media_type
                        ctrl_tid = TypedId(ctrl_media_type, 99000 + len(latencies))

                if ctrl.constraints.genres:
                    if expected == 0:
                        ctrl_genres = ["Cooking", "Gardening"]
                    else:
                        ctrl_genres = list(ctrl.constraints.genres)

            if expected == 0 and not ctrl.constraints:
                ctrl_title = "Cooking Pasta Guide"
                ctrl_synopsis = "A culinary instructional program about making noodles."
                ctrl_genres = ["Documentary"]
            elif expected >= 2 and not ctrl.constraints:
                ctrl_title = f"Exemplary match for {ctrl.query}"
                ctrl_synopsis = (
                    f"An acclaimed feature film directly addressing {ctrl.query}."
                )

            ctrl_ev = ItemEvidence(
                typed_id=ctrl_tid,
                title=ctrl_title,
                synopsis=ctrl_synopsis,
                genres=ctrl_genres,
                release_year=ctrl_year,
                runtime=ctrl_runtime,
                media_type=ctrl_media_type,
            )
            ctrl_inp = JudgeInput(query=ctrl.query, evidence=ctrl_ev)
            out = checked_judgment(ctrl_inp)
            if out.execution_status != "success" or not out.evidence_sufficiency:
                continue

            # Check correctness: if control has expected_grade, verify agreement or valid bounded grade
            if ctrl.expected_grade is not None:
                # Binary controls require an accepted positive (grade >=2).
                if ctrl.expected_grade == 0 and out.grade == 0:
                    control_correct += 1
                elif ctrl.expected_grade > 0 and out.grade >= 2:
                    control_correct += 1

        # Calculate metrics
        rep_rate = (
            repeat_matches / total_repeat_tests if total_repeat_tests > 0 else 0.0
        )
        # Optional robustness diagnostic. Untested or failed diagnostics never pass,
        # but cannot veto reliable execution of the frozen production prompt.
        if total_perm_tests > 0:
            perm_rate = permutation_matches / total_perm_tests
            perm_pass = perm_rate >= 0.95
        else:
            perm_rate = 0.0
            perm_pass = False

        ctrl_acc = control_correct / len(controls[:100]) if controls else 0.0

        report["repeatability_rate"] = rep_rate
        report["repeat_tests"] = total_repeat_tests
        report["permutation_tests"] = total_perm_tests
        report["control_tests"] = len(controls[:100])
        report["repeat_grade_agreement"] = (
            repeat_grade_matches / total_repeat_tests if total_repeat_tests else 0.0
        )
        report["repeat_sufficiency_agreement"] = (
            repeat_sufficiency_matches / total_repeat_tests
            if total_repeat_tests
            else 0.0
        )
        report["pilot_evidence_coverage"] = (
            report["pilot_sufficient_count"] / len(all_pilot_inputs)
            if all_pilot_inputs
            else 0.0
        )
        report["repeat_evidence_coverage"] = (
            report["repeat_sufficient_pairs"] / total_repeat_tests
            if total_repeat_tests
            else 0.0
        )
        report["repeatability_pass"] = rep_rate >= 0.99
        report["option_permutation_rate"] = perm_rate
        report["option_permutation_pass"] = perm_pass
        report["control_accuracy"] = ctrl_acc
        report["control_accuracy_pass"] = ctrl_acc >= 0.95
        report["mean_latency_ms"] = (
            round(float(sum(latencies) / len(latencies)), 2) if latencies else 0.0
        )

        # Qualification rule:
        # - No malformed outputs or silent truncation
        # - >= 95% accuracy on automatically verifiable controls
        # - >= 99% accepted-grade-or-unjudged agreement; raw grades remain diagnostic
        # Option permutation is reported separately and is not a qualification gate.
        # - No unresolved execution failures after 1 retry
        full_scope = (
            report["pilot_pairs"] >= 400
            and report["pilot_distinct_pairs"] >= 400
            and report["pilot_families"] >= 20
            and total_repeat_tests >= 100
            and len(controls[:100]) >= 100
        )
        report["scope"] = "qualification" if full_scope else "diagnostic"
        report["measured_gates_pass"] = (
            report["repeatability_pass"]
            and report["control_accuracy_pass"]
            and report["execution_failures"] == 0
            and report["malformed_count"] == 0
        )
        report["qualified"] = full_scope and report["measured_gates_pass"]

        return report

    def select_judge_panel(
        self,
        candidate_reports: Dict[str, Dict[str, Any]],
        candidates_map: Dict[str, LocalJudgeAdapter],
    ) -> Tuple[
        Optional[LocalJudgeAdapter],
        Optional[LocalJudgeAdapter],
        Optional[LocalJudgeAdapter],
        str,
    ]:
        """Select qualified Nimble; a failure never selects another model or a stub."""
        key = "bespoke-nimble-9b"
        judge = candidates_map.get(key)
        if judge is not None and qualification_record_matches(
            judge, candidate_reports.get(key)
        ):
            return judge, None, None, "single_judge"
        return None, None, None, "unqualified"
