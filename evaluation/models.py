"""Core data models, schemas, and contracts for Seen'emAll Evaluation Suite v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
import hashlib
import json
from typing import Any, Dict, List, Optional

EVIDENCE_CONTRACT_VERSION = "v2.2"


class EvaluationStatus(IntEnum):
    """Exit status codes for evaluation execution and comparison gates."""

    PASS = 0
    FAIL = 1
    INVALID = 2
    INCONCLUSIVE = 3

    @classmethod
    def from_legacy(cls, value: str) -> EvaluationStatus:
        """Map legacy report strings to EvaluationStatus."""
        val = str(value).strip().upper()
        if val in ("PASS", "PASSED", "SUCCESS"):
            return cls.PASS
        if val in ("FAIL", "FAILED"):
            return cls.FAIL
        if val in ("INVALID", "MALFORMED"):
            return cls.INVALID
        # Map legacy NEEDS_ADJUDICATION and unknown statuses to INCONCLUSIVE
        return cls.INCONCLUSIVE


# Backwards compatibility alias
NEEDS_ADJUDICATION = EvaluationStatus.INCONCLUSIVE


class GainMode(str, Enum):
    """Gain formulation for Discounted Cumulative Gain (DCG/nDCG)."""

    GRADED_EXPONENTIAL = "graded_exponential"  # 2^relevance - 1 (grades 0-3)
    CONTINUOUS_IDENTITY = "continuous_identity"  # relevance (finite in [0, 1])


class RubricGrade(IntEnum):
    """Standardized discrete rubric grades 0 to 3."""

    IRRELEVANT = 0  # Incompatible or irrelevant to the request
    WEAK_MATCH = 1  # Partial or weak match
    CLEAR_MATCH = 2  # Clear match
    STRONG_MATCH = 3  # Especially strong match across main requirements

    @classmethod
    def is_positive(cls, grade: Optional[int]) -> bool:
        """Grades >= 2 are considered positives for Precision, MAP, and Recall."""
        return grade is not None and grade >= int(cls.CLEAR_MATCH)


def select_grade_from_probabilities(probs: Dict[int, float]) -> int:
    """Select highest-probability rubric level; exact ties break toward lower grade."""
    if not probs:
        return int(RubricGrade.IRRELEVANT)

    best_grade = 0
    best_prob = -1.0
    # Iterating in ascending order (0, 1, 2, 3) ensures that strict inequality
    # breaks exact ties toward the lower grade.
    for g in sorted(probs.keys()):
        p = probs[g]
        if p > best_prob:
            best_prob = p
            best_grade = g
    return best_grade


@dataclass(frozen=True)
class TypedId:
    """Strongly typed item identifier e.g. 'movie:1893' or 'tv:1234'."""

    media_type: str
    id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "media_type", str(self.media_type).strip().lower())
        object.__setattr__(self, "id", int(self.id))

    def __str__(self) -> str:
        return f"{self.media_type}:{self.id}"

    @classmethod
    def parse(cls, identifier: Any, default_media_type: str = "movie") -> TypedId:
        """Parse from int, string ('movie:1893' or '1893'), or dictionary."""
        if isinstance(identifier, TypedId):
            return identifier
        if isinstance(identifier, dict):
            media_type = (
                identifier.get("media_type")
                or identifier.get("type")
                or default_media_type
            )
            tid = identifier.get("tmdb_id") or identifier.get("id")
            if tid is None:
                raise ValueError(f"Dictionary missing ID field: {identifier}")
            return cls(str(media_type), int(tid))
        if isinstance(identifier, int):
            return cls(default_media_type, identifier)
        if isinstance(identifier, str):
            identifier = identifier.strip()
            if ":" in identifier:
                prefix, raw_id = identifier.split(":", 1)
                return cls(prefix, int(raw_id))
            return cls(default_media_type, int(identifier))
        raise TypeError(
            f"Cannot parse TypedId from {type(identifier).__name__}: {identifier!r}"
        )

    @classmethod
    def from_item(
        cls, item: Dict[str, Any], default_media_type: str = "movie"
    ) -> TypedId:
        return cls.parse(item, default_media_type=default_media_type)


@dataclass
class ItemEvidence:
    """Catalog item semantic evidence presented to judge models.

    Hides producing system, retrieval scores, rank, popularity metrics,
    existing ground-truth labels, and recommendation explanations.
    """

    typed_id: TypedId
    title: str
    synopsis: Optional[str] = None
    genres: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    media_type: str = ""
    cast: List[str] = field(default_factory=list)
    crew: List[str] = field(default_factory=list)
    release_year: Optional[int] = None
    runtime: Optional[int] = None
    original_language: Optional[str] = None
    missing_fields: List[str] = field(default_factory=list)
    preference_context: Optional[str] = None
    directors: List[str] = field(default_factory=list)
    maturity_rating: Optional[str] = None
    collection_id: Optional[int] = None
    collection_name: Optional[str] = None

    def __post_init__(self) -> None:
        self.media_type = (self.media_type or self.typed_id.media_type).strip().lower()
        if self.media_type != self.typed_id.media_type:
            raise ValueError("Evidence media type must match its typed ID")
        missing = []
        if not self.synopsis or self.synopsis.strip() in ("", "[Not Provided]"):
            missing.append("synopsis")
        if not self.genres:
            missing.append("genres")
        if not self.keywords:
            missing.append("keywords")
        if not self.cast:
            missing.append("cast")
        if not self.crew:
            missing.append("crew")
        if not self.directors:
            missing.append("directors")
        if not self.maturity_rating:
            missing.append("maturity_rating")
        if not self.collection_name:
            missing.append("collection")
        if self.release_year is None:
            missing.append("release_year")
        if self.runtime is None:
            missing.append("runtime")
        if not self.original_language:
            missing.append("original_language")
        self.missing_fields = sorted(list(set(self.missing_fields + missing)))

    def to_evidence_text(self, max_chars: Optional[int] = None) -> str:
        """Deterministic, formatted representation of catalog evidence."""
        lines = [
            f"Title: {self.title or '[Not Provided]'}",
            f"Media Type: {self.media_type or '[Not Provided]'}",
            f"Release Year: {self.release_year if self.release_year is not None else '[Not Provided]'}",
            f"Runtime (minutes): {self.runtime if self.runtime is not None else '[Not Provided]'}",
            f"Language: {self.original_language or '[Not Provided]'}",
            f"Genres: {', '.join(self.genres) if self.genres else '[Not Provided]'}",
            f"Keywords: {', '.join(self.keywords) if self.keywords else '[Not Provided]'}",
            f"Cast: {', '.join(self.cast) if self.cast else '[Not Provided]'}",
            f"Directors: {', '.join(self.directors) if self.directors else '[Not Provided]'}",
            f"Other Crew: {', '.join(self.crew) if self.crew else '[Not Provided]'}",
            f"Maturity Rating: {self.maturity_rating or '[Not Provided]'}",
            f"Collection: {self.collection_name or '[Not Provided]'}",
        ]

        synopsis_val = self.synopsis.strip() if self.synopsis else "[Not Provided]"

        extra_lines = []
        if self.missing_fields:
            extra_lines.append(
                f"Explicitly Missing Metadata: {', '.join(self.missing_fields)}"
            )
        if self.preference_context:
            extra_lines.append(
                f"Synthetic Persona Preference Context: {self.preference_context}"
            )

        other_text = "\n".join(lines + extra_lines)
        synopsis_line_prefix = "\nSynopsis: "

        if max_chars is not None:
            overhead = len(other_text) + len(synopsis_line_prefix)
            if overhead + len(synopsis_val) > max_chars:
                suffix = "... [TRUNCATED]"
                avail = max(0, max_chars - overhead - len(suffix))
                synopsis_val = synopsis_val[:avail].rstrip() + suffix

        full_text = other_text + synopsis_line_prefix + synopsis_val
        return full_text

    def content_hash(self) -> str:
        payload = {
            "typed_id": str(self.typed_id),
            "title": self.title,
            "synopsis": self.synopsis,
            "genres": self.genres,
            "keywords": self.keywords,
            "media_type": self.media_type,
            "cast": self.cast,
            "crew": self.crew,
            "release_year": self.release_year,
            "runtime": self.runtime,
            "original_language": self.original_language,
            "missing_fields": self.missing_fields,
            "preference_context": self.preference_context,
            "directors": self.directors,
            "maturity_rating": self.maturity_rating,
            "collection_id": self.collection_id,
            "collection_name": self.collection_name,
            "evidence_contract": EVIDENCE_CONTRACT_VERSION,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class JudgeInput:
    """Full input payload presented to a local judge."""

    query: str
    evidence: ItemEvidence
    rubric_version: str = "v2.7"
    persona_context: Optional[Dict[str, Any]] = None

    def input_hash(self) -> str:
        payload = {
            "query": self.query.strip().lower(),
            "evidence_hash": self.evidence.content_hash(),
            "rubric_version": self.rubric_version,
            "persona_context": self.persona_context,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class JudgeProvenance:
    """Full model and environment provenance for automated judgments."""

    model_name: str
    checkpoint_revision: str
    tokenizer_revision: str
    projection_head_revision: Optional[str] = None
    quantization: str = "fp16"
    runtime: str = "local"
    prompt_rubric_hash: str = ""
    evidence_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeOutput:
    """Structured response from a local judge model."""

    grade: int  # 0, 1, 2, 3
    probabilities: Dict[int, float]  # Normalized probabilities for grades 0-3
    evidence_sufficiency: bool  # True if evidence supports a definitive judgment
    execution_status: str  # "success", "failed", "abstain", "over_limit"
    provenance: JudgeProvenance
    expected_score: float = 0.0  # Sum(grade * p), preserved for diagnostics only
    latency_ms: float = 0.0
    raw_response: Optional[str] = None

    def __post_init__(self) -> None:
        # Validate grade bounds
        self.grade = max(0, min(3, int(self.grade)))
        # Compute expected score if not pre-filled
        if self.probabilities and not self.expected_score:
            self.expected_score = float(
                sum(g * p for g, p in self.probabilities.items())
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grade": self.grade,
            "probabilities": self.probabilities,
            "evidence_sufficiency": self.evidence_sufficiency,
            "execution_status": self.execution_status,
            "expected_score": round(self.expected_score, 4),
            "latency_ms": round(self.latency_ms, 2),
            "provenance": self.provenance.to_dict(),
        }


@dataclass
class JudgmentRecord:
    """Authoritative or pooled judgment record with consensus state."""

    query: str
    typed_id: str
    grade: Optional[int]  # None if UNJUDGED
    status: str  # "ACCEPTED", "UNJUDGED", "CONFLICT", "INSUFFICIENT_EVIDENCE"
    provenances: List[JudgeProvenance] = field(default_factory=list)
    consensus_model_count: int = 1
    probabilities: Optional[Dict[int, float]] = None
    deterministic_override: bool = False
    violation_reasons: List[str] = field(default_factory=list)

    def is_positive(self) -> bool:
        return RubricGrade.is_positive(self.grade)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "typed_id": self.typed_id,
            "grade": self.grade,
            "status": self.status,
            "consensus_model_count": self.consensus_model_count,
            "deterministic_override": self.deterministic_override,
            "violation_reasons": self.violation_reasons,
            "probabilities": self.probabilities,
            "provenances": [p.to_dict() for p in self.provenances],
        }


@dataclass
class DeterministicConstraint:
    """Explicit machine-verifiable constraints for a query."""

    media_type: Optional[str] = None  # "movie" or "tv"
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    min_runtime: Optional[int] = None
    max_runtime: Optional[int] = None
    language: Optional[str] = None
    providers: Optional[List[str]] = None
    genres: Optional[List[str]] = None
    require_all_genres: bool = False
    seen_ids: Optional[List[int]] = None
    disliked_ids: Optional[List[int]] = None
    canonical_sequence_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TestCase:
    """Unified test case for Seen'emAll Evaluation Suite v2."""

    __test__ = False

    case_id: str
    family_id: str
    track: str  # "product", "cold_start", "personalization", "anchor"
    split: str  # "dev", "regression", "full"
    task: str  # "search", "franchise", "constraint", "onboarding"
    slice_tags: List[str]  # ["vibe"], ["franchise"], ["constraint"], etc.
    query: str
    user_id: str = "u1"
    persona_context: Optional[Dict[str, Any]] = None
    constraints: Optional[DeterministicConstraint] = None
    golden_ids: Optional[List[int]] = None
    canonical_sequence: Optional[List[str]] = None
    expected_empty: bool = False
    is_factual_control: bool = False
    expected_grade: Optional[int] = None
    eligible_catalog_count: Optional[int] = None
    golden_set: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "case_id": self.case_id,
            "family_id": self.family_id,
            "track": self.track,
            "split": self.split,
            "task": self.task,
            "slice_tags": self.slice_tags,
            "query": self.query,
            "user_id": self.user_id,
            "expected_empty": self.expected_empty,
            "eligible_catalog_count": self.eligible_catalog_count,
        }
        if self.persona_context:
            res["persona_context"] = self.persona_context
        if self.constraints:
            res["constraints"] = self.constraints.to_dict()
        if self.golden_ids:
            res["golden_ids"] = self.golden_ids
        if self.golden_set:
            res["golden_set"] = self.golden_set
        if self.canonical_sequence:
            res["canonical_sequence"] = self.canonical_sequence
        if self.is_factual_control:
            res["is_factual_control"] = True
            res["expected_grade"] = self.expected_grade
        return res


@dataclass
class SharedManifest:
    """Shared immutable benchmark inputs."""

    dataset_hash: str
    split: str
    qrels_hash: str
    judge_config_hash: str
    catalog_hash: str
    il_availability_hash: str
    il_snapshot_date: str
    persona_fixtures_hash: str
    business_clock: str
    metric_definitions: Dict[str, Any]
    gain_mode: GainMode
    gate_policy: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["gain_mode"] = self.gain_mode.value
        return data


@dataclass
class PerSystemSpec:
    """Per-system specification for reproducible comparison."""

    system_name: str
    code_build_identity: str
    resolved_feature_flags: Dict[str, Any]
    model_tokenizer_revisions: Dict[str, str]
    embeddings_spec: Dict[str, Any]
    search_index_config: Dict[str, Any]
    search_index_artifact_checksum: str
    runtime: str
    device: str
    cache_policy: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GateCheckResult:
    """Individual gate check outcome."""

    name: str
    passed: bool
    observed: Any
    threshold: Any
    details: str = ""


@dataclass
class ComparisonGateResult:
    """Comprehensive comparison gate result."""

    status: EvaluationStatus
    passed: bool
    checks: List[GateCheckResult] = field(default_factory=list)
    family_count: int = 0
    slice_counts: Dict[str, int] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.name,
            "exit_code": int(self.status),
            "passed": self.passed,
            "family_count": self.family_count,
            "slice_counts": self.slice_counts,
            "reasons": self.reasons,
            "checks": [asdict(c) for c in self.checks],
        }
