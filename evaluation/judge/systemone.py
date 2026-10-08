"""Published System One score contract, including semantic permutation tests."""

import json
import math
import urllib.request
from typing import Dict, Tuple

from evaluation.judge.base import LocalJudgeAdapter
from evaluation.judge.rubric import (
    GRADE_CRITERIA,
    RELEVANCE_INSTRUCTIONS,
    SUFFICIENCY_INSTRUCTIONS,
    normalize_probabilities,
)
from evaluation.models import JudgeInput, select_grade_from_probabilities
from evaluation.query_interpretation import interpret_query


class SystemOneJudgeAdapter(LocalJudgeAdapter):
    endpoint_url: str
    service_model: str
    timeout_seconds: float
    api_key: str | None = None

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": "SeenemAllEval"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _score(
        self, judge_input: JudgeInput, order: Tuple[int, ...] = (0, 1, 2, 3)
    ) -> Tuple[Dict[int, float], bool, str]:
        if not self.is_available():
            raise RuntimeError(f"Judge service unavailable: {self.endpoint_url}")
        # Keep state free of an independently ordered rubric: only criteria vary.
        payload = {
            "model": self.service_model,
            "state": {
                "query": interpret_query(judge_input.query),
                "evidence": judge_input.evidence.to_evidence_text(),
                "persona_context": judge_input.persona_context,
            },
            "questions": {
                "relevance": {
                    "type": "score",
                    "instructions": RELEVANCE_INSTRUCTIONS,
                    "criteria": [GRADE_CRITERIA[g] for g in order],
                },
                "evidence_sufficient": {
                    "type": "noul",
                    "instructions": SUFFICIENCY_INSTRUCTIONS,
                },
            },
        }
        request = urllib.request.Request(
            self.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict) or result.get("truncated"):
            raise ValueError("Malformed or truncated System One response")
        answers = result.get("answers")
        if not isinstance(answers, dict):
            raise ValueError("Missing named System One answers")
        answer = answers.get("relevance")
        if not isinstance(answer, dict) or answer.get("truncated"):
            raise ValueError("Missing or truncated relevance answer")
        probs = normalize_probabilities(answer.get("probabilities"))
        suff_answer = answers.get("evidence_sufficient")
        if not isinstance(suff_answer, dict) or suff_answer.get("truncated"):
            raise ValueError("Missing evidence sufficiency answer")
        sufficient = suff_answer.get("noul")
        if isinstance(sufficient, bool) or not isinstance(sufficient, (float, int)):
            raise ValueError("Invalid evidence sufficiency probability")
        if not math.isfinite(sufficient) or not 0 <= sufficient <= 1:
            raise ValueError("Invalid evidence sufficiency probability")
        return (
            {order[i]: p for i, p in probs.items()},
            sufficient >= 0.5,
            json.dumps(result),
        )

    def _run_inference(self, prompt: str, judge_input: JudgeInput):
        return self._score(judge_input)

    def test_option_order_permutation(self, judge_input: JudgeInput):
        forward = self.judge_pair(judge_input)
        if forward.execution_status != "success":
            raise RuntimeError("Forward permutation judgment failed")
        reverse, sufficient, _ = self._score(judge_input, (3, 2, 1, 0))
        consistent = forward.evidence_sufficiency == sufficient and (
            not sufficient or forward.grade == select_grade_from_probabilities(reverse)
        )
        return forward.probabilities, reverse, consistent
