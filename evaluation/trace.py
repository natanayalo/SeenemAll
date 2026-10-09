"""Evaluation execution tracing and stage capture."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from evaluation.models import TypedId


@dataclass
class EvaluationTrace:
    """Detailed stage trace for an individual recommendation execution."""

    query: str
    user_id: str
    effective_context: Dict[str, Any] = field(default_factory=dict)
    parsed_intent: Dict[str, Any] = field(default_factory=dict)
    retrieval_sources: Dict[str, List[str]] = field(default_factory=dict)
    prefilter_outcome: Dict[str, Any] = field(default_factory=dict)
    merged_candidates: List[str] = field(default_factory=list)
    scored_order: List[str] = field(default_factory=list)
    diversified_order: List[str] = field(default_factory=list)
    reranked_order: List[str] = field(default_factory=list)
    final_order: List[str] = field(default_factory=list)
    depths: Dict[str, int] = field(default_factory=dict)
    exclusions_applied: List[str] = field(default_factory=list)
    cache_hits: int = 0
    inference_counts: Dict[str, int] = field(default_factory=dict)
    inference_providers: Dict[str, Dict[str, int]] = field(default_factory=dict)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    fallbacks: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TraceCollector:
    """Collector and serializer for evaluation traces."""

    def __init__(self) -> None:
        self.traces: List[EvaluationTrace] = []

    def start_trace(self, query: str, user_id: str = "u1") -> EvaluationTrace:
        trace = EvaluationTrace(query=query, user_id=user_id)
        self.traces.append(trace)
        return trace

    def record_stage(
        self,
        trace: EvaluationTrace,
        stage_name: str,
        items: List[Any],
        timing_ms: float = 0.0,
    ) -> None:
        typed_ids = [str(TypedId.parse(it)) for it in items]
        if stage_name == "retrieval_merged":
            trace.merged_candidates = typed_ids
            trace.depths["retrieval"] = len(typed_ids)
        elif stage_name == "scored":
            trace.scored_order = typed_ids
            trace.depths["scored"] = len(typed_ids)
        elif stage_name == "diversified":
            trace.diversified_order = typed_ids
            trace.depths["diversified"] = len(typed_ids)
        elif stage_name == "reranked":
            trace.reranked_order = typed_ids
            trace.depths["reranked"] = len(typed_ids)
        elif stage_name == "final":
            trace.final_order = typed_ids
            trace.depths["final"] = len(typed_ids)
        if timing_ms > 0:
            trace.timings_ms[stage_name] = round(timing_ms, 2)
