from __future__ import annotations

from api.pipeline.intent.builder import _SUPPORTED_ANN_BACKENDS, resolve_query_intent
from api.pipeline.intent.matcher import (
    _has_people_filters,
    matches_keywords,
    merge_query_filter_hints,
    strict_required_genres,
)
from api.pipeline.intent.parser import (
    intent_filters_from_llm,
    merge_maturity_rating,
    merge_with_legacy_filters,
    parse_llm_intent,
)
from api.pipeline.intent.rewrite import (
    append_weighted_text,
    build_rewrite_vector,
    float_from_env,
)

# Aliases for backwards compatibility with tests and controller re-exports
_float_from_env = float_from_env
_matches_keywords = matches_keywords
_append_weighted_text = append_weighted_text
_build_rewrite_vector = build_rewrite_vector
_parse_llm_intent = parse_llm_intent
_intent_filters_from_llm = intent_filters_from_llm
_merge_with_legacy_filters = merge_with_legacy_filters
_merge_maturity_rating = merge_maturity_rating
_strict_required_genres = strict_required_genres
_merge_query_filter_hints = merge_query_filter_hints

__all__ = [
    "_SUPPORTED_ANN_BACKENDS",
    "append_weighted_text",
    "build_rewrite_vector",
    "float_from_env",
    "intent_filters_from_llm",
    "matches_keywords",
    "merge_maturity_rating",
    "merge_query_filter_hints",
    "merge_with_legacy_filters",
    "parse_llm_intent",
    "resolve_query_intent",
    "strict_required_genres",
    "_has_people_filters",
    "_float_from_env",
    "_matches_keywords",
    "_append_weighted_text",
    "_build_rewrite_vector",
    "_parse_llm_intent",
    "_intent_filters_from_llm",
    "_merge_with_legacy_filters",
    "_merge_maturity_rating",
    "_strict_required_genres",
    "_merge_query_filter_hints",
]
