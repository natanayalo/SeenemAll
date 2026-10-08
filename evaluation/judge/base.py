"""Base provider-neutral local judge adapter interface and prompt formatting."""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
from typing import Dict, Tuple
import json
from evaluation.query_interpretation import INTERPRETATION_VERSION, interpret_query

from evaluation.judge.rubric import (
    ADAPTER_CONTRACT_VERSION,
    GRADE_CRITERIA,
    QUALIFICATION_PROTOCOL_VERSION,
    RELEVANCE_INSTRUCTIONS,
    SUFFICIENCY_INSTRUCTIONS,
    normalize_probabilities,
)

from evaluation.models import (
    EVIDENCE_CONTRACT_VERSION,
    JudgeInput,
    JudgeOutput,
    JudgeProvenance,
    RubricGrade,
    select_grade_from_probabilities,
)

STANDARD_RUBRIC_TEXT = (
    "Evaluation Rubric:\n"
    + "\n".join(f"- Grade {g}: {text}" for g, text in enumerate(GRADE_CRITERIA))
    + "\n"
    + RELEVANCE_INSTRUCTIONS
    + "\n"
    + SUFFICIENCY_INSTRUCTIONS
    + "\n"
)


# A conservative character guard, separate from the model's token context.
MAX_INPUT_CHARS = 4096


class LocalJudgeAdapter(ABC):
    """Abstract base class for provider-neutral local judge adapters."""

    def __init__(
        self,
        model_name: str,
        checkpoint_revision: str = "v1.0",
        tokenizer_revision: str = "v1.0",
        quantization: str = "int8",
        runtime: str = "local",
    ) -> None:
        self.model_name = model_name
        self.checkpoint_revision = checkpoint_revision
        self.tokenizer_revision = tokenizer_revision
        self.quantization = quantization
        self.runtime = runtime

    @abstractmethod
    def is_available(self) -> bool:
        """Check if local inference runtime and model artifact are ready."""
        raise NotImplementedError

    @abstractmethod
    def _run_inference(
        self,
        prompt: str,
        judge_input: JudgeInput,
    ) -> Tuple[Dict[int, float], bool, str]:
        """Execute local model inference.

        Returns:
            (probabilities_dict, evidence_sufficiency_bool, raw_response_str)
        """
        raise NotImplementedError

    def build_prompt(self, judge_input: JudgeInput) -> str:
        """Construct deterministic prompt presentation for local models."""
        evidence_text = judge_input.evidence.to_evidence_text()
        prompt = (
            f"You are a strict, objective movie recommendation judge.\n"
            f"Evaluate whether the candidate item satisfies the user's search query.\n\n"
            f"{STANDARD_RUBRIC_TEXT}\n\n"
            f"Query: {interpret_query(judge_input.query)}\n\n"
            f"Item Evidence:\n{evidence_text}\n\n"
            f"Respond with a structured assessment containing:\n"
            f"1. grade_probabilities: probabilities for grades [0, 1, 2, 3] summing to 1.0\n"
            f"2. evidence_sufficient: true or false\n"
        )
        return prompt

    def get_prompt_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "contract": ADAPTER_CONTRACT_VERSION,
                    "evidence_contract": EVIDENCE_CONTRACT_VERSION,
                    "qualification_protocol": QUALIFICATION_PROTOCOL_VERSION,
                    "rubric": STANDARD_RUBRIC_TEXT,
                    "query_interpretation": INTERPRETATION_VERSION,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    def qualification_fingerprint(self) -> str:
        provenance = self.get_provenance("").to_dict()
        provenance["service_model"] = getattr(self, "service_model", None)
        return hashlib.sha256(
            json.dumps(provenance, sort_keys=True).encode()
        ).hexdigest()

    def get_provenance(self, evidence_hash: str) -> JudgeProvenance:
        return JudgeProvenance(
            model_name=self.model_name,
            checkpoint_revision=self.checkpoint_revision,
            tokenizer_revision=self.tokenizer_revision,
            quantization=self.quantization,
            runtime=self.runtime,
            prompt_rubric_hash=self.get_prompt_hash(),
            evidence_hash=evidence_hash,
        )

    def judge_pair(self, judge_input: JudgeInput) -> JudgeOutput:
        """Judge a single query-item pair with input validation and tie breaking."""
        evidence_hash = judge_input.evidence.content_hash()
        provenance = self.get_provenance(evidence_hash)

        prompt = self.build_prompt(judge_input)

        # Reject over-limit inputs before inference; never permit silent truncation
        if len(prompt) > MAX_INPUT_CHARS:
            return JudgeOutput(
                grade=int(RubricGrade.IRRELEVANT),
                probabilities={0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                evidence_sufficiency=False,
                execution_status="over_limit",
                provenance=provenance,
                raw_response=f"Input length {len(prompt)} exceeds maximum {MAX_INPUT_CHARS} characters.",
            )

        try:
            probs, sufficient, raw_resp = self._run_inference(prompt, judge_input)
            normalized_probs = normalize_probabilities(probs)
            if not isinstance(sufficient, bool):
                raise ValueError("Evidence sufficiency must be boolean")

            # Select discrete grade using highest probability, ties broken to lower grade
            selected_grade = select_grade_from_probabilities(normalized_probs)

            return JudgeOutput(
                grade=selected_grade,
                probabilities=normalized_probs,
                evidence_sufficiency=sufficient,
                execution_status="success",
                provenance=provenance,
                raw_response=raw_resp,
            )
        except ValueError as exc:
            return JudgeOutput(
                grade=int(RubricGrade.IRRELEVANT),
                probabilities={0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                evidence_sufficiency=False,
                execution_status="malformed",
                provenance=provenance,
                raw_response=f"Malformed judge response: {exc}",
            )
        except Exception as exc:
            return JudgeOutput(
                grade=int(RubricGrade.IRRELEVANT),
                probabilities={0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                evidence_sufficiency=False,
                execution_status="failed",
                provenance=provenance,
                raw_response=f"Execution error: {exc}",
            )

    def test_option_order_permutation(
        self,
        judge_input: JudgeInput,
    ) -> Tuple[Dict[int, float], Dict[int, float], bool]:
        """Evaluate option-order permutation sensitivity [0, 1, 2, 3] vs [3, 2, 1, 0].

        Returns:
            (forward_probs, reversed_probs, is_semantically_consistent)
        """
        raise NotImplementedError("Adapter has no genuine permutation test")
