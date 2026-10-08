"""Milestone 2: Cold-start and synthetic personalization behavioral benchmark harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from evaluation.metrics import calculate_ndcg_at_k
from evaluation.models import GainMode, TypedId
from evaluation.runner import EvaluationRunner

DEFAULT_PERSONAS_PATH = Path("evaluation/fixtures/synthetic_personas.json")


def load_synthetic_personas(path: Optional[Path] = None) -> Dict[str, Any]:
    target = path or DEFAULT_PERSONAS_PATH
    if not target.exists():
        return {}
    with target.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def seed_persona_fixtures(db: Any, persona: Dict[str, Any], verify: bool = True) -> int:
    """Seed persona history and known negatives into database, computing user taste vectors."""
    from api.core.user_profile import upsert_user_vectors
    from api.db.models import Item, UserHistory

    user_id = str(persona.get("persona_id", "eval_user"))
    seed_history = persona.get("seed_history", [])
    known_negatives = persona.get("known_negatives", [])

    # Clean existing history for this user
    db.query(UserHistory).filter(UserHistory.user_id == user_id).delete()

    for item in seed_history:
        tmdb_id = int(item["tmdb_id"])
        db_item = db.query(Item).filter(Item.tmdb_id == tmdb_id).first()
        item_id = db_item.id if db_item else tmdb_id
        rating = float(item.get("rating", 4.0))
        uh = UserHistory(
            user_id=user_id,
            item_id=item_id,
            event_type="rated" if rating >= 3.0 else "disliked",
            weight=int(round(rating)),
        )
        db.add(uh)

    for neg in known_negatives:
        tmdb_id = int(neg["tmdb_id"])
        db_item = db.query(Item).filter(Item.tmdb_id == tmdb_id).first()
        item_id = db_item.id if db_item else tmdb_id
        uh = UserHistory(
            user_id=user_id,
            item_id=item_id,
            event_type="disliked",
            weight=1,
        )
        db.add(uh)

    db.commit()
    upsert_user_vectors(db, user_id)

    if verify and (seed_history or known_negatives):
        count = db.query(UserHistory).filter(UserHistory.user_id == user_id).count()
        if count == 0:
            raise RuntimeError(
                f"Seeding verification failed: 0 UserHistory records found for user '{user_id}'"
            )
        return count
    return len(seed_history) + len(known_negatives)


class PersonalizationHarness:
    """Evaluates synthetic personalization lift and cold-start behavioral gates."""

    def __init__(
        self,
        runner: EvaluationRunner,
        personas: Optional[Dict[str, Any]] = None,
        db_session_factory: Optional[Callable[[], Any]] = None,
        verify_seeding: bool = True,
    ) -> None:
        self.runner = runner
        self.personas = personas or load_synthetic_personas()
        if db_session_factory is None:
            try:
                from api.db.session import get_sessionmaker

                self.db_session_factory = get_sessionmaker()
            except Exception:
                self.db_session_factory = None
        else:
            self.db_session_factory = db_session_factory
        self.verify_seeding = verify_seeding
        self.seeding_verified = False
        self.seeding_error: Optional[str] = None

    def seed_all_personas(self) -> None:
        """Seed all active personas into the application database if session factory is provided."""
        if not self.db_session_factory:
            self.seeding_error = (
                "No database session factory available for persona seeding"
            )
            if self.verify_seeding:
                raise RuntimeError(self.seeding_error)
            return

        db = self.db_session_factory()
        try:
            total_seeded = 0
            for p in self.personas.values():
                if p.get("seed_history") or p.get("known_negatives"):
                    cnt = seed_persona_fixtures(db, p, verify=self.verify_seeding)
                    total_seeded += cnt
            self.seeding_verified = total_seeded > 0
        except Exception as exc:
            self.seeding_error = str(exc)
            if self.verify_seeding:
                raise RuntimeError(
                    f"Personalization seeding verification failed: {exc}"
                ) from exc
        finally:
            if hasattr(db, "close"):
                db.close()

    def evaluate_persona(
        self,
        persona_key: str,
        k: int = 10,
        baseline_params: Optional[Dict[str, Any]] = None,
        candidate_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single synthetic persona comparing personalized candidate vs masked cold-start baseline."""
        persona = self.personas.get(persona_key)
        if not persona:
            return {
                "error": f"persona '{persona_key}' not found",
                "execution_failed": True,
            }

        user_id = persona.get("persona_id", persona_key)
        hidden_targets = persona.get("hidden_targets", [])
        known_negatives = persona.get("known_negatives", [])

        # Build target qrels for the hidden target items
        qrels: Dict[str, float] = {}
        for ht in hidden_targets:
            tid = f"movie:{ht['tmdb_id']}"
            qrels[tid] = float(ht.get("expected_grade", 3))

        disliked_ids = {int(kn["tmdb_id"]) for kn in known_negatives}

        # 1. Run personalized candidate
        cand_p = dict(candidate_params or {})
        cand_items, cand_trace = self.runner.execute_query(
            query="recommendations",
            user_id=user_id,
            params=cand_p,
            limit=k,
            bypass_cache=True,
        )

        # 2. Run masked cold-start baseline on the SAME user
        # Preserves the user's seen/disliked exclusions while masking preference vectors & taste clusters
        masked_p = dict(baseline_params or {})
        masked_p["mask_preferences"] = True
        masked_items, masked_trace = self.runner.execute_query(
            query="recommendations",
            user_id=user_id,
            params=masked_p,
            limit=k,
            bypass_cache=True,
        )

        # 3. Separately evaluate standard personalized baseline
        pers_base_p = dict(baseline_params or {})
        pers_base_items, pers_base_trace = self.runner.execute_query(
            query="recommendations",
            user_id=user_id,
            params=pers_base_p,
            limit=k,
            bypass_cache=True,
        )

        # Inspect traces for execution errors: errors must invalidate measurement
        all_errors = (
            list(cand_trace.errors)
            + list(masked_trace.errors)
            + list(pers_base_trace.errors)
        )
        has_execution_error = len(all_errors) > 0
        is_empty_output = (len(cand_items) == 0) or (len(masked_items) == 0)

        cand_ndcg = calculate_ndcg_at_k(
            cand_items, qrels, k=k, gain_mode=GainMode.GRADED_EXPONENTIAL
        )
        masked_ndcg = calculate_ndcg_at_k(
            masked_items, qrels, k=k, gain_mode=GainMode.GRADED_EXPONENTIAL
        )
        pers_base_ndcg = calculate_ndcg_at_k(
            pers_base_items, qrels, k=k, gain_mode=GainMode.GRADED_EXPONENTIAL
        )

        # Total personalization-system lift = candidate nDCG - masked baseline nDCG
        total_personalization_lift = cand_ndcg - masked_ndcg
        # Relative ranking upgrade lift = candidate nDCG - personalized baseline nDCG
        relative_candidate_lift = cand_ndcg - pers_base_ndcg

        # Check for known-disliked results in candidate top-K
        cand_ids = []
        for it in cand_items[:k]:
            try:
                cand_ids.append(int(TypedId.parse(it).id))
            except Exception:
                pass
        disliked_hits = [cid for cid in cand_ids if cid in disliked_ids]

        return {
            "persona_key": persona_key,
            "user_id": user_id,
            "candidate_ndcg": round(cand_ndcg, 4),
            "masked_baseline_ndcg": round(masked_ndcg, 4),
            "personalized_baseline_ndcg": round(pers_base_ndcg, 4),
            "total_personalization_system_lift": round(total_personalization_lift, 4),
            "relative_candidate_lift": round(relative_candidate_lift, 4),
            "disliked_violations_count": len(disliked_hits),
            "disliked_violation_ids": disliked_hits,
            "returned_count": len(cand_items),
            "execution_failed": has_execution_error or is_empty_output,
            "errors": all_errors,
        }

    def run_personalization_benchmark(
        self,
        k: int = 10,
        baseline_params: Optional[Dict[str, Any]] = None,
        candidate_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute behavioral gates across all non-empty synthetic personas.

        Gates:
          - No execution errors or empty response payloads
          - Mean personalization lift >= 0
          - Zero known-disliked results
          - Every persona's candidate-versus-baseline nDCG decline <= 0.03
        """
        self.seed_all_personas()

        active_personas = [
            k_name
            for k_name, p in self.personas.items()
            if p.get("seed_history") and p.get("hidden_targets")
        ]

        if not active_personas:
            return {"error": "no_active_personas", "passed": False}

        results: List[Dict[str, Any]] = []
        for p_key in active_personas:
            res = self.evaluate_persona(
                p_key,
                k=k,
                baseline_params=baseline_params,
                candidate_params=candidate_params,
            )
            results.append(res)

        # Check for execution errors, failed requests, or unverified seeding
        seeding_unverified = (self.verify_seeding and not self.seeding_verified) or (
            self.seeding_error is not None
        )
        any_failed = (
            any(r.get("execution_failed") for r in results) or seeding_unverified
        )
        lifts = [r["total_personalization_system_lift"] for r in results]
        mean_lift = float(np.mean(lifts)) if lifts else 0.0
        total_disliked_viol = sum(r["disliked_violations_count"] for r in results)

        # Check max decline across individual personas relative to personalized baseline:
        # Candidate ranking must not regress against the personalized baseline by more than 0.03.
        max_decline = (
            max(
                (r["personalized_baseline_ndcg"] - r["candidate_ndcg"]) for r in results
            )
            if results
            else 0.0
        )

        mean_lift_pass = (mean_lift >= 0.0) and not any_failed
        disliked_pass = (total_disliked_viol == 0) and not any_failed
        decline_pass = (max_decline <= 0.03) and not any_failed
        overall_pass = (
            mean_lift_pass and disliked_pass and decline_pass and not any_failed
        )

        return {
            "passed": overall_pass,
            "mean_personalization_lift": round(mean_lift, 4),
            "mean_lift_pass": mean_lift_pass,
            "total_disliked_violations": total_disliked_viol,
            "disliked_pass": disliked_pass,
            "max_persona_ndcg_decline": round(max_decline, 4),
            "decline_pass": decline_pass,
            "execution_failed": any_failed,
            "persona_results": results,
        }
