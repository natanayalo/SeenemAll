"""Milestone 3: Private promotion benchmark harness and redaction."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict


from evaluation.models import EvaluationStatus


# Isolated directory outside development checkout (e.g. user home or local app data)
DEFAULT_PRIVATE_BENCHMARK_DIR = Path(
    os.environ.get(
        "PRIVATE_BENCHMARK_DIR", Path.home() / ".seenemall_private_benchmark"
    )
)


class PrivateBenchmarkHarness:
    """Private holdout evaluation outside development repository.

    Enforces:
      - Raw queries, rankings, and individual judgments stay confidential
      - Exports ONLY aggregated metrics, confidence intervals, and gate decisions
      - Rejects duplicate submission attempts
      - Flags holdout refresh after 10 distinct candidate submissions
    """

    def __init__(self, benchmark_dir: Path = DEFAULT_PRIVATE_BENCHMARK_DIR) -> None:
        self.benchmark_dir = benchmark_dir
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        self.submissions_file = self.benchmark_dir / "submission_ledger.json"
        self._ledger: Dict[str, Any] = self._load_ledger()

    def _load_ledger(self) -> Dict[str, Any]:
        if self.submissions_file.exists():
            try:
                with self.submissions_file.open("r", encoding="utf-8") as fp:
                    return json.load(fp)
            except Exception:
                pass
        return {"attempt_count": 0, "candidates": {}, "refresh_required": False}

    def _save_ledger(self) -> None:
        with self.submissions_file.open("w", encoding="utf-8") as fp:
            json.dump(self._ledger, fp, indent=2)

    def submit_candidate(
        self,
        candidate_identity: str,
        gate_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Record candidate evaluation submission with redaction of private contents."""
        cand_hash = hashlib.sha256(candidate_identity.encode("utf-8")).hexdigest()

        if cand_hash in self._ledger["candidates"]:
            return {
                "status": "REJECTED",
                "exit_code": int(EvaluationStatus.INVALID),
                "passed": False,
                "error": "Identical candidate submission rejected. Please advance candidate revision.",
                "candidate_hash": cand_hash,
            }

        self._ledger["attempt_count"] += 1
        self._ledger["candidates"][cand_hash] = {
            "submission_index": self._ledger["attempt_count"],
            "status": gate_result.get("status"),
            "passed": gate_result.get("passed", False),
        }

        if len(self._ledger["candidates"]) >= 10:
            self._ledger["refresh_required"] = True

        self._save_ledger()

        # Redacted public export (aggregates and decisions only)
        public_export = {
            "status": gate_result.get("status"),
            "exit_code": gate_result.get("exit_code"),
            "passed": gate_result.get("passed"),
            "submission_index": self._ledger["attempt_count"],
            "refresh_required": self._ledger["refresh_required"],
            "summary_checks": [
                {
                    "name": c.get("name"),
                    "passed": c.get("passed"),
                    "details": c.get("details"),
                }
                for c in gate_result.get("checks", [])
            ],
            "reasons": gate_result.get("reasons", []),
        }

        return public_export
