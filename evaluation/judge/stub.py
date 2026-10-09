"""Deterministic test stub judge adapter for offline unit/integration testing."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from evaluation.judge.base import LocalJudgeAdapter
from evaluation.models import JudgeInput


class StubJudgeAdapter(LocalJudgeAdapter):
    """Deterministic, configurable judge adapter requiring no external runtimes or downloads."""

    def __init__(
        self,
        model_name: str = "stub-judge",
        checkpoint_revision: str = "test-v1",
        tokenizer_revision: str = "test-v1",
        quantization: str = "none",
        fixed_grade: Optional[int] = None,
        fixed_probabilities: Optional[Dict[int, float]] = None,
        fixed_sufficiency: bool = True,
        grade_lookup: Optional[Dict[Tuple[str, str], int]] = None,
        simulate_error: bool = False,
        error_message: str = "Simulated stub error",
        auto_qualify: bool = False,
        grade: Optional[int] = None,
        sufficient: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            checkpoint_revision=checkpoint_revision,
            tokenizer_revision=tokenizer_revision,
            quantization=quantization,
            runtime="stub",
        )
        self.fixed_grade = grade if grade is not None else fixed_grade
        self.fixed_probabilities = fixed_probabilities
        self.fixed_sufficiency = (
            sufficient if sufficient is not None else fixed_sufficiency
        )
        self.grade_lookup = grade_lookup or {}
        self.simulate_error = simulate_error
        self.error_message = error_message
        self.auto_qualify = auto_qualify
        self.call_count: int = 0

    def is_available(self) -> bool:
        return True

    def test_option_order_permutation(
        self, judge_input: JudgeInput
    ) -> Tuple[Dict[int, float], Dict[int, float], bool]:
        """Stub permutation check returns consistent probabilities."""
        probs, suff, _ = self._run_inference("", judge_input)
        return probs, probs, True

    def _run_inference(
        self,
        prompt: str,
        judge_input: JudgeInput,
    ) -> Tuple[Dict[int, float], bool, str]:
        self.call_count += 1
        if self.simulate_error:
            raise RuntimeError(self.error_message)

        # Control query recognition when auto_qualify is enabled
        if self.auto_qualify:
            q_lower = judge_input.query.lower()
            match = re.search(r"released before (\d+)", q_lower)
            if match:
                year = judge_input.evidence.release_year
                g = 2 if year is not None and year < int(match.group(1)) else 0
                return (
                    {i: float(i == g) for i in range(4)},
                    year is not None,
                    "stub_year_control",
                )
            match = re.search(r"short movies under (\d+)", q_lower)
            if match:
                runtime = judge_input.evidence.runtime
                g = 2 if runtime is not None and runtime < int(match.group(1)) else 0
                return (
                    {i: float(i == g) for i in range(4)},
                    runtime is not None,
                    "stub_runtime_control",
                )
            if "television series" in q_lower or "feature movie" in q_lower:
                expected_media = "tv" if "television series" in q_lower else "movie"
                g = 2 if judge_input.evidence.media_type == expected_media else 0
                return {i: float(i == g) for i in range(4)}, True, "stub_media_control"
            if "in order" in q_lower:
                return {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0}, True, "stub_control_three"

        key = (
            judge_input.query.strip().lower(),
            str(judge_input.evidence.typed_id).lower(),
        )
        if key in self.grade_lookup:
            g = self.grade_lookup[key]
            probs = {i: (1.0 if i == g else 0.0) for i in range(4)}
            return probs, self.fixed_sufficiency, "stub_lookup"

        if self.fixed_probabilities is not None:
            return (
                dict(self.fixed_probabilities),
                self.fixed_sufficiency,
                "stub_fixed_probs",
            )

        if self.fixed_grade is not None:
            g = self.fixed_grade
            probs = {i: (1.0 if i == g else 0.0) for i in range(4)}
            return probs, self.fixed_sufficiency, "stub_fixed_grade"

        # Deterministic default based on query/evidence string hash
        combined = f"{judge_input.query}::{judge_input.evidence.typed_id}"
        val = sum(ord(c) for c in combined) % 4
        probs = {i: (1.0 if i == val else 0.0) for i in range(4)}
        return probs, self.fixed_sufficiency, "stub_hash_default"
