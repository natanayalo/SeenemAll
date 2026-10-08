"""Pooling, consensus adjudication, judgment caching, and frozen qrels management."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from evaluation.deterministic import apply_deterministic_override
from evaluation.judge.base import LocalJudgeAdapter
from evaluation.models import (
    DeterministicConstraint,
    ItemEvidence,
    JudgeInput,
    JudgmentRecord,
    TypedId,
)

DEFAULT_JUDGMENT_CACHE_PATH = Path("evaluation/.judgment_cache.json")
DEFAULT_QRELS_DIR = Path("evaluation/qrels")


class JudgmentCache:
    """Persistent cache for individual query-item judgments."""

    def __init__(self, path: Path = DEFAULT_JUDGMENT_CACHE_PATH) -> None:
        self.path = path
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.load()

    def _cache_key(self, judge_input: JudgeInput, judge: LocalJudgeAdapter) -> str:
        prov = judge.get_provenance(judge_input.evidence.content_hash())
        payload = {
            "input_hash": judge_input.input_hash(),
            "model_name": prov.model_name,
            "checkpoint": prov.checkpoint_revision,
            "quantization": prov.quantization,
            "prompt_hash": prov.prompt_rubric_hash,
            "qualification_fingerprint": judge.qualification_fingerprint(),
        }
        encoded = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def load(self) -> None:
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as fp:
                    self._cache = json.load(fp)
            except Exception:
                self._cache = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fp:
            json.dump(self._cache, fp, indent=2, ensure_ascii=False)

    def get(
        self, judge_input: JudgeInput, judge: LocalJudgeAdapter
    ) -> Optional[Dict[str, Any]]:
        key = self._cache_key(judge_input, judge)
        return self._cache.get(key)

    def set(
        self, judge_input: JudgeInput, judge: LocalJudgeAdapter, result: Dict[str, Any]
    ) -> None:
        key = self._cache_key(judge_input, judge)
        self._cache[key] = result
        self.save()


class PoolAdjudicator:
    """Pool deduplication and multi-judge consensus engine."""

    def __init__(
        self,
        primary_judge: LocalJudgeAdapter,
        secondary_judge: Optional[LocalJudgeAdapter] = None,
        tie_breaker_judge: Optional[LocalJudgeAdapter] = None,
        cache: Optional[JudgmentCache] = None,
    ) -> None:
        self.primary_judge = primary_judge
        self.secondary_judge = secondary_judge
        self.tie_breaker_judge = tie_breaker_judge
        self.cache = cache or JudgmentCache()

    def deduplicate_pool(
        self,
        candidate_lists: Sequence[Sequence[Any]],
        max_pool_size: int = 30,
    ) -> List[str]:
        """Pool and deduplicate top results up to max_pool_size (default: top 30)."""
        seen: Set[str] = set()
        pool: List[str] = []
        for lst in candidate_lists:
            for it in lst:
                try:
                    tid_str = str(TypedId.parse(it))
                except Exception:
                    tid_str = str(it)
                if tid_str not in seen:
                    seen.add(tid_str)
                    pool.append(tid_str)
                    if len(pool) >= max_pool_size:
                        return pool
        return pool

    def _judge_cached(
        self,
        judge: LocalJudgeAdapter,
        judge_input: JudgeInput,
    ) -> Tuple[int, Dict[int, float], bool, Any, str]:
        """Run judge or retrieve from persistent cache."""
        cached = self.cache.get(judge_input, judge)
        if cached:
            probs = {int(k): float(v) for k, v in cached["probabilities"].items()}
            return (
                int(cached["grade"]),
                probs,
                cached.get("execution_status") == "success"
                and bool(cached["evidence_sufficiency"]),
                judge.get_provenance(judge_input.evidence.content_hash()),
                cached.get("execution_status", "failed"),
            )

        output = judge.judge_pair(judge_input)
        self.cache.set(judge_input, judge, output.to_dict())
        return (
            output.grade,
            output.probabilities,
            output.execution_status == "success" and output.evidence_sufficiency,
            output.provenance,
            output.execution_status,
        )

    def adjudicate_pair(
        self,
        query: str,
        evidence: ItemEvidence,
        constraints: Optional[DeterministicConstraint] = None,
        mode: str = "consensus",  # "single_judge" or "consensus"
    ) -> JudgmentRecord:
        """Adjudicate a single query-item pair according to automated judgment contracts."""
        judge_input = JudgeInput(query=query, evidence=evidence)
        typed_id_str = str(evidence.typed_id)

        # 1. Exploratory Single-Judge Mode
        if mode == "single_judge" or self.secondary_judge is None:
            g1, p1, suff1, prov1, execution_status = self._judge_cached(
                self.primary_judge, judge_input
            )
            final_grade = g1 if suff1 else None
            status = "ACCEPTED" if suff1 else "INSUFFICIENT_EVIDENCE"
            if execution_status not in {"success", "abstain"}:
                status = "JUDGE_FAILED"

            # Apply deterministic constraint checks (authoritative override)
            override = False
            violations: List[str] = []
            if final_grade is not None and constraints:
                final_grade, override, violations = apply_deterministic_override(
                    grade=final_grade, item=evidence, constraints=constraints
                )

            return JudgmentRecord(
                query=query,
                typed_id=typed_id_str,
                grade=final_grade,
                status=status if not override else "ACCEPTED",
                provenances=[prov1],
                consensus_model_count=1,
                probabilities=p1,
                deterministic_override=override,
                violation_reasons=violations,
                execution_statuses=[execution_status],
            )

        # 2. Authoritative Multi-Judge Consensus Mode
        # Step 1: Primary and secondary independently label complete item
        g1, p1, suff1, prov1, status1 = self._judge_cached(
            self.primary_judge, judge_input
        )
        g2, p2, suff2, prov2, status2 = self._judge_cached(
            self.secondary_judge, judge_input
        )

        provenances = [prov1, prov2]
        execution_statuses = [status1, status2]

        # Step 2: Exact agreement when both find sufficient evidence
        if suff1 and suff2 and g1 == g2:
            accepted_grade = g1
            status = "ACCEPTED"
            # Average probabilities
            merged_probs = {g: (p1[g] + p2[g]) / 2.0 for g in range(4)}
        else:
            # Step 3: Disagreements or evidence conflicts go to 3rd qualifying model
            if self.tie_breaker_judge is not None:
                g3, p3, suff3, prov3, status3 = self._judge_cached(
                    self.tie_breaker_judge, judge_input
                )
                provenances.append(prov3)
                execution_statuses.append(status3)

                # Step 4: Accept a grade only when at least 2 distinct models assign that exact grade
                valid_votes = []
                if suff1:
                    valid_votes.append(g1)
                if suff2:
                    valid_votes.append(g2)
                if suff3:
                    valid_votes.append(g3)

                accepted_grade = None
                for candidate_grade in (0, 1, 2, 3):
                    if valid_votes.count(candidate_grade) >= 2:
                        accepted_grade = candidate_grade
                        break

                if accepted_grade is not None:
                    status = "ACCEPTED"
                    merged_probs = {g: (p1[g] + p2[g] + p3[g]) / 3.0 for g in range(4)}
                else:
                    # Step 5: Otherwise retain UNJUDGED
                    status = "UNJUDGED"
                    accepted_grade = None
                    merged_probs = {g: (p1[g] + p2[g] + p3[g]) / 3.0 for g in range(4)}
            else:
                # No tie-breaker available -> retain UNJUDGED
                status = "UNJUDGED"
                accepted_grade = None
                merged_probs = {g: (p1[g] + p2[g]) / 2.0 for g in range(4)}

        # Deterministic checks override subjective relevance
        override = False
        violations = []
        if accepted_grade is not None and constraints:
            accepted_grade, override, violations = apply_deterministic_override(
                grade=accepted_grade, item=evidence, constraints=constraints
            )

        return JudgmentRecord(
            query=query,
            typed_id=typed_id_str,
            grade=accepted_grade,
            status=status,
            provenances=provenances,
            consensus_model_count=len(provenances),
            probabilities=merged_probs,
            deterministic_override=override,
            violation_reasons=violations,
            execution_statuses=execution_statuses,
        )


class ConsensusJudgeEngine:
    """High-level management of qrels versions and pool labeling."""

    def __init__(
        self,
        adjudicator: Optional[PoolAdjudicator] = None,
        primary_judge: Optional[LocalJudgeAdapter] = None,
        secondary_judge: Optional[LocalJudgeAdapter] = None,
        tie_breaker_judge: Optional[LocalJudgeAdapter] = None,
        cache: Optional[JudgmentCache] = None,
        qrels_dir: Path = DEFAULT_QRELS_DIR,
        version: str = "v2.7",
        immutable: bool = True,
        redact_queries: bool = False,
    ) -> None:
        if adjudicator is None:
            if primary_judge is None:
                raise ValueError("Must provide either adjudicator or primary_judge")
            adjudicator = PoolAdjudicator(
                primary_judge=primary_judge,
                secondary_judge=secondary_judge,
                tie_breaker_judge=tie_breaker_judge,
                cache=cache,
            )
        self.adjudicator = adjudicator
        self.qrels_dir = qrels_dir
        self.version = version
        self.immutable = immutable
        self.redact_queries = redact_queries
        self.qrels_file = qrels_dir / f"qrels_{version}.json"
        self._qrels: Dict[str, Dict[str, float]] = {}  # {query_key: {typed_id: grade}}
        self.load_qrels()

    def _query_key(self, query: str) -> str:
        norm = query.strip().lower()
        if self.redact_queries:
            return (
                f"query_sha256_{hashlib.sha256(norm.encode('utf-8')).hexdigest()[:16]}"
            )
        return norm

    def load_qrels(self) -> None:
        if self.qrels_file.exists():
            try:
                with self.qrels_file.open("r", encoding="utf-8") as fp:
                    self._qrels = json.load(fp)
            except Exception:
                self._qrels = {}

    def save_qrels(self) -> None:
        if self.immutable and self.qrels_file.exists():
            # If marked immutable, write snapshot to a new versioned file rather than overwriting in place
            snapshot_file = (
                self.qrels_dir
                / f"qrels_{self.version}_{hashlib.sha256(str(time.time()).encode('utf-8')).hexdigest()[:8]}.json"
            )
            with snapshot_file.open("w", encoding="utf-8") as fp:
                json.dump(self._qrels, fp, indent=2, ensure_ascii=False)
                fp.write("\n")
            return

        self.qrels_dir.mkdir(parents=True, exist_ok=True)
        with self.qrels_file.open("w", encoding="utf-8") as fp:
            json.dump(self._qrels, fp, indent=2, ensure_ascii=False)
            fp.write("\n")

    def publish_version(self, new_version: str) -> Path:
        """Publish an immutable snapshot under a new explicit version."""
        target_file = self.qrels_dir / f"qrels_{new_version}.json"
        self.qrels_dir.mkdir(parents=True, exist_ok=True)
        with target_file.open("x", encoding="utf-8") as fp:
            json.dump(self._qrels, fp, indent=2, ensure_ascii=False)
            fp.write("\n")
        return target_file

    def get_query_qrels(self, query: str) -> Dict[str, float]:
        return self._qrels.get(self._query_key(query), {})

    def seed_query_qrels(self, query: str, qrels: Dict[str, float]) -> None:
        """Seed independently verified reference labels before judging a new pool."""
        self._qrels[self._query_key(query)] = dict(qrels)

    def label_pool(
        self,
        query: str,
        pool_evidence: List[ItemEvidence],
        constraints: Optional[DeterministicConstraint] = None,
        mode: str = "consensus",
    ) -> Tuple[Dict[str, float], List[JudgmentRecord], bool]:
        """Label all items in pool and return (qrels_dict, records, had_changes)."""
        q_key = self._query_key(query)
        query_qrels = self._qrels.setdefault(q_key, {})
        records: List[JudgmentRecord] = []
        had_changes = False

        for ev in pool_evidence:
            tid_str = str(ev.typed_id)
            rec = self.adjudicator.adjudicate_pair(
                query=query, evidence=ev, constraints=constraints, mode=mode
            )
            records.append(rec)

            if rec.grade is not None:
                old_val = query_qrels.get(tid_str)
                if old_val != float(rec.grade):
                    query_qrels[tid_str] = float(rec.grade)
                    had_changes = True
            else:
                # Issue 7 fix: When a new judgment is unresolved or fails, remove any stale grade!
                if tid_str in query_qrels:
                    del query_qrels[tid_str]
                    had_changes = True

        if had_changes:
            self.save_qrels()

        return dict(query_qrels), records, had_changes
