from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from fastapi import Query

from api.core.elasticsearch_search import SearchFilters
from api.core.intent_parser import Intent
from api.core.legacy_intent_parser import IntentFilters


@dataclass
class RecommendParams:
    user_id: str = Query(..., description="Seen'emAll user_id (e.g., 'u1')")
    limit: int = Query(20, ge=1, le=100)
    query: str | None = Query(
        None,
        description="Optional natural-language intent (e.g. 'light sci-fi < 2h')",
    )
    diversify: bool = Query(True, description="Whether to diversify recommendations.")
    profile: str | None = Query(None, description="Optional profile identifier")
    use_llm_intent: bool = Query(
        True,
        description="Enable the LLM intent parser (set to false for manual overrides).",
    )
    ann_description_override: str | None = Query(
        None,
        description="Manual ANN description override to blend into the rewrite vector.",
    )
    rewrite_override: str | None = Query(
        None,
        description="Manual rewrite text override (skips rewrite_query when provided).",
    )
    ann_weight_override: float | None = Query(
        None,
        ge=0.0,
        description="Override weight for the ANN description component.",
    )
    rewrite_weight_override: float | None = Query(
        None,
        ge=0.0,
        description="Override weight for the rewrite text component.",
    )
    ann_backend_override: str | None = Query(
        None,
        description="Force ANN backend ('elasticsearch' or 'pgvector') for this request.",
    )
    genre_override: str | None = Query(
        None,
        description="Comma-separated manual genres to enforce (e.g., 'Drama, Sci-Fi').",
    )
    classic_top_rated: bool | None = Query(
        None,
        description=(
            "Prioritize top-rated catalog titles over trending suggestions. "
            "When omitted, a heuristic may enable this automatically for "
            "queries like 'best classic movies'."
        ),
    )
    mixer_ann_weight: float | None = Query(
        None,
        ge=0.0,
        description="Override hybrid ANN weight (default from HYBRID_ANN_WEIGHT).",
    )
    mixer_collab_weight: float | None = Query(
        None,
        ge=0.0,
        description="Override collaborative weight (default MIXER_COLLAB_WEIGHT).",
    )
    mixer_trending_weight: float | None = Query(
        None,
        ge=0.0,
        description="Override trending weight (default HYBRID_TRENDING_WEIGHT).",
    )
    mixer_popularity_weight: float | None = Query(
        None,
        ge=0.0,
        description="Override popularity weight (default HYBRID_POPULARITY_WEIGHT).",
    )
    mixer_vote_weight: float | None = Query(
        None,
        ge=0.0,
        description="Override vote-count weight (default HYBRID_VOTE_WEIGHT).",
    )
    mixer_novelty_weight: float | None = Query(
        None,
        ge=0.0,
        description="Override novelty weight (default MIXER_NOVELTY_WEIGHT).",
    )
    strict_filters: bool = Query(
        False,
        description="Require items to satisfy every inferred genre (AND semantics).",
    )
    debug: bool = Query(False, description="Include debug diagnostics in response.")

    def __post_init__(self):
        from fastapi.params import Param

        for field_name in self.__dataclass_fields__:
            val = getattr(self, field_name)
            if isinstance(val, Param):
                setattr(self, field_name, None if val.default is ... else val.default)


@dataclass
class ComputeResult:
    items: List[Dict[str, Any]]
    debug_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrefilterDecision:
    allowed_ids: List[int] | None
    boost_ids: List[int]
    enforce_genres: bool
    keyword_boosted: bool = False


@dataclass
class UserContext:
    canonical_id: str
    user_id: str
    profile: str | None
    long_v: np.ndarray | None
    short_v: np.ndarray | None
    exclude_set: Set[int]
    profile_meta: Dict[str, Any]
    cold_start: bool
    provider_alias_map: Dict[str, Set[str]]
    top_query_keywords: Set[str]
    preferred_services: Set[str] = field(default_factory=set)


@dataclass
class QueryUnderstanding:
    query: str | None
    llm_intent: Intent
    intent_filters: IntentFilters
    structured_search_filters: SearchFilters | None
    es_text_query: str | None
    rewrite_vec: np.ndarray | None
    prefer_top_rated: bool
    custom_genres: List[str]
    has_people_filters: bool
    candidate_limit: int
    backend_override_normalized: str | None = None
    preferred_services: Set[str] = field(default_factory=set)
    prefilter_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidatePool:
    ids: List[int]
    merged_scores: Dict[int, Dict[str, float]]
    prefilter: PrefilterDecision
    items_with_data: Dict[int, Tuple[Any, np.ndarray, Any]]
    boost_ids: List[int]
    enforce_genres: bool
    structured_search_filters: SearchFilters | None


@dataclass
class ScoredCandidates:
    ordered: List[Dict[str, Any]]
    serendipity_context: List[Dict[str, Any]]
    skipped_intent: int = 0
    skipped_people: int = 0
