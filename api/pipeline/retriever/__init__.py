from __future__ import annotations

from api.pipeline.retriever.ann import ANNRetriever
from api.pipeline.retriever.base import BaseRetriever
from api.pipeline.retriever.cold_start import (
    ColdStartRetriever,
    cold_start_candidates,
)
from api.pipeline.retriever.collaborative import (
    CollaborativeGraphRetriever,
    collaborative_candidates,
)
from api.pipeline.retriever.fusion import retrieve_candidates
from api.pipeline.retriever.prefilter import (
    filter_excluded_candidate_ids,
    genre_contains_clause,
    keyword_contains_clause,
    ordered_unique,
    people_only_candidate_ids,
    prefilter_allowed_ids,
    relax_filters_for_people,
    run_prefilter_query,
)
from api.pipeline.retriever.trending import (
    TrendingPriorRetriever,
    trending_prior_candidates,
)

# Aliases for backwards compatibility with tests and controller re-exports
_prefilter_allowed_ids = prefilter_allowed_ids
_run_prefilter_query = run_prefilter_query
_genre_contains_clause = genre_contains_clause
_keyword_contains_clause = keyword_contains_clause
_ordered_unique = ordered_unique
_cold_start_candidates = cold_start_candidates
_collaborative_candidates = collaborative_candidates
_trending_prior_candidates = trending_prior_candidates
_people_only_candidate_ids = people_only_candidate_ids
_filter_excluded_candidate_ids = filter_excluded_candidate_ids
_relax_filters_for_people = relax_filters_for_people

__all__ = [
    "ANNRetriever",
    "BaseRetriever",
    "ColdStartRetriever",
    "CollaborativeGraphRetriever",
    "TrendingPriorRetriever",
    "cold_start_candidates",
    "collaborative_candidates",
    "filter_excluded_candidate_ids",
    "genre_contains_clause",
    "keyword_contains_clause",
    "ordered_unique",
    "people_only_candidate_ids",
    "prefilter_allowed_ids",
    "relax_filters_for_people",
    "retrieve_candidates",
    "run_prefilter_query",
    "trending_prior_candidates",
    "_prefilter_allowed_ids",
    "_run_prefilter_query",
    "_genre_contains_clause",
    "_keyword_contains_clause",
    "_ordered_unique",
    "_cold_start_candidates",
    "_collaborative_candidates",
    "_trending_prior_candidates",
    "_people_only_candidate_ids",
    "_filter_excluded_candidate_ids",
    "_relax_filters_for_people",
]
