from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Sequence, Set

from api.core.business_rules import apply_business_rules
from api.core.elasticsearch_search import SearchFilters
from api.core.legacy_intent_parser import item_matches_intent
from api.db.models import Item
from api.pipeline.hooks import get_hook
from api.pipeline.intent import _has_people_filters, float_from_env
from api.pipeline.models import (
    CandidatePool,
    QueryUnderstanding,
    RecommendParams,
    ScoredCandidates,
    UserContext,
)

logger = logging.getLogger(__name__)

_HYBRID_ANN_WEIGHT = float_from_env("HYBRID_ANN_WEIGHT", 0.5)
_HYBRID_POPULARITY_WEIGHT = float_from_env("HYBRID_POPULARITY_WEIGHT", 0.25)
_HYBRID_TRENDING_WEIGHT = float_from_env("HYBRID_TRENDING_WEIGHT", 0.4)
_HYBRID_MIN_ANN_WEIGHT = 0.05
_MIXER_COLLAB_WEIGHT = float_from_env("MIXER_COLLAB_WEIGHT", 0.3)
_MIXER_TRENDING_WEIGHT = float_from_env("MIXER_TRENDING_WEIGHT", 0.2)
_MIXER_NOVELTY_WEIGHT = float_from_env("MIXER_NOVELTY_WEIGHT", 0.1)
_HYBRID_VOTE_WEIGHT = float_from_env("HYBRID_VOTE_WEIGHT", 0.2)


def item_matches_people_filters(item: Item, filters: SearchFilters) -> bool:
    def _extract_names(payload: Any) -> Set[str]:
        names: Set[str] = set()
        if not payload:
            return names
        if isinstance(payload, list):
            for entry in payload:
                if isinstance(entry, str):
                    names.add(entry.strip().lower())
                elif isinstance(entry, Mapping):
                    name = entry.get("name")
                    if isinstance(name, str):
                        names.add(name.strip().lower())
        return names

    cast_names = _extract_names(getattr(item, "cast", None))
    if filters.cast:
        if not any(name.strip().lower() in cast_names for name in filters.cast):
            return False

    director_names = _extract_names(getattr(item, "directors", None))
    if filters.directors:
        if not any(
            name.strip().lower() in director_names for name in filters.directors
        ):
            return False

    producer_names = _extract_names(getattr(item, "producers", None))
    if filters.producers:
        if not any(
            name.strip().lower() in producer_names for name in filters.producers
        ):
            return False

    writer_names = _extract_names(getattr(item, "writers", None))
    if filters.writers:
        if not any(name.strip().lower() in writer_names for name in filters.writers):
            return False

    return True


def prioritize_boosted_items(
    items: Sequence[Dict[str, Any]], boost_ids: Sequence[int]
) -> List[Dict[str, Any]]:
    if not items or not boost_ids:
        return list(items)

    lookup: Dict[int, Dict[str, Any]] = {}
    ordered: List[Dict[str, Any]] = []
    seen: set[int] = set()

    for item in items:
        ident = item.get("id")
        if isinstance(ident, int):
            lookup[ident] = item

    for candidate in boost_ids:
        boosted_item = lookup.get(candidate)
        if boosted_item is None:
            continue
        ident = boosted_item.get("id")
        if isinstance(ident, int) and ident not in seen:
            ordered.append(boosted_item)
            seen.add(ident)

    for item in items:
        ident = item.get("id")
        if isinstance(ident, int) and ident in seen:
            continue
        ordered.append(item)
        if isinstance(ident, int):
            seen.add(ident)

    return ordered


def apply_mixer_scores(
    candidates: List[Dict[str, Any]],
    *,
    ann_weight_override: float | None = None,
    collab_weight_override: float | None = None,
    trending_weight_override: float | None = None,
    popularity_weight_override: float | None = None,
    vote_weight_override: float | None = None,
    novelty_weight_override: float | None = None,
) -> None:
    if not candidates:
        return

    max_popularity = max(
        (float(item.get("popularity") or 0.0) for item in candidates), default=0.0
    )
    max_vote_count = max(
        (float(item.get("vote_count") or 0.0) for item in candidates), default=0.0
    )

    base_ann_weight = (
        _HYBRID_ANN_WEIGHT if ann_weight_override is None else ann_weight_override
    )
    ann_weight = max(_HYBRID_MIN_ANN_WEIGHT, base_ann_weight)
    collab_weight = (
        _MIXER_COLLAB_WEIGHT
        if collab_weight_override is None
        else collab_weight_override
    )
    trending_weight = (
        _MIXER_TRENDING_WEIGHT
        if trending_weight_override is None
        else trending_weight_override
    )
    pop_weight = (
        _HYBRID_POPULARITY_WEIGHT
        if popularity_weight_override is None
        else popularity_weight_override
    )
    vote_weight = (
        _HYBRID_VOTE_WEIGHT if vote_weight_override is None else vote_weight_override
    )
    novelty_weight = (
        _MIXER_NOVELTY_WEIGHT
        if novelty_weight_override is None
        else novelty_weight_override
    )

    for item in candidates:
        ann_rank = float(item.get("ann_rank", item.get("original_rank", 0)))
        ann_score = 1.0 / (1.0 + ann_rank)

        source_scores = item.get("source_scores") or {}
        collab_score = float(source_scores.get("collab") or 0.0)
        trending_source = float(source_scores.get("trending") or 0.0)

        pop_score = (
            float(item.get("popularity") or 0.0) / max_popularity
            if max_popularity > 0
            else 0.0
        )
        vote_bonus = (
            float(item.get("vote_count") or 0.0) / max_vote_count
            if max_vote_count > 0
            else 0.0
        )

        novelty_score = 1.0 - min(1.0, pop_score)
        novelty_score = (novelty_score + (1.0 - vote_bonus)) / 2.0

        retrieval_score = (
            ann_weight * ann_score
            + collab_weight * collab_score
            + trending_weight * trending_source
            + pop_weight * pop_score
            + vote_weight * vote_bonus
            + novelty_weight * novelty_score
        )
        item["retrieval_score"] = retrieval_score

    candidates.sort(
        key=lambda item: (
            -(item.get("retrieval_score") or 0.0),
            item.get("ann_rank", item.get("original_rank", 0)),
        )
    )
    for idx, item in enumerate(candidates):
        item["original_rank"] = idx


def score_candidates(
    pool: CandidatePool,
    intent: QueryUnderstanding,
    params: RecommendParams,
    context: UserContext,
) -> ScoredCandidates:
    if not pool.ids:
        return ScoredCandidates(ordered=[], serendipity_context=[])

    ordered: List[Dict[str, Any]] = []
    fallback_candidates: List[Dict[str, Any]] = []
    max_candidates = min(intent.candidate_limit, max(params.limit * 2, 25))
    skipped_intent = 0
    skipped_people = 0
    rank_counter = 0

    people_match_fn = get_hook(
        "_item_matches_people_filters", item_matches_people_filters
    )

    for iid in pool.ids:
        it, vec, watch_options = pool.items_with_data.get(iid, (None, None, None))
        if it is None or vec is None or len(vec) == 0:
            continue
        if not item_matches_intent(
            it, intent.intent_filters, enforce_genres=pool.enforce_genres
        ):
            skipped_intent += 1
            continue
        if pool.structured_search_filters and _has_people_filters(
            pool.structured_search_filters
        ):
            if not people_match_fn(it, pool.structured_search_filters):
                skipped_people += 1
                continue

        sources = pool.merged_scores.get(iid, {})

        cleaned_options = []
        if watch_options and watch_options[0] is not None:
            cleaned_options = [
                {"service": opt["service"], "url": opt["url"]}
                for opt in watch_options
                if opt["url"] is not None
            ]

        provider_allowed = True
        filtered_options = cleaned_options
        if intent.preferred_services:
            matching_options = [
                opt
                for opt in cleaned_options
                if isinstance(opt, dict)
                and ((service := str(opt.get("service") or "").strip().lower()))
                and service in intent.preferred_services
            ]
            if not matching_options:
                provider_allowed = False
            else:
                filtered_options = matching_options

        candidate_payload = {
            "id": it.id,
            "tmdb_id": it.tmdb_id,
            "media_type": it.media_type,
            "title": it.title,
            "overview": it.overview,
            "poster_url": it.poster_url,
            "runtime": it.runtime,
            "original_language": it.original_language,
            "genres": it.genres,
            "release_year": it.release_year,
            "collection_id": it.collection_id,
            "collection_name": it.collection_name,
            "maturity_rating": getattr(it, "maturity_rating", None),
            "watch_options": filtered_options,
            "watch_url": (filtered_options[0]["url"] if filtered_options else None),
            "original_rank": rank_counter,
            "ann_rank": rank_counter,
            "vector": vec,
            "popularity": getattr(it, "popularity", None),
            "vote_average": getattr(it, "vote_average", None),
            "vote_count": getattr(it, "vote_count", None),
            "popular_rank": getattr(it, "popular_rank", None),
            "trending_rank": getattr(it, "trending_rank", None),
            "top_rated_rank": getattr(it, "top_rated_rank", None),
            "retrieval_score": None,
            "source_scores": sources,
        }

        if provider_allowed:
            ordered.append(candidate_payload)
        else:
            candidate_payload["watch_options"] = cleaned_options
            candidate_payload["watch_url"] = (
                cleaned_options[0]["url"] if cleaned_options else None
            )
            fallback_candidates.append(candidate_payload)

        rank_counter += 1
        if provider_allowed and len(ordered) >= max_candidates:
            break

    if (
        intent.preferred_services
        and len(ordered) < params.limit
        and fallback_candidates
    ):
        deficit = params.limit - len(ordered)
        ordered.extend(fallback_candidates[:deficit])

    if not ordered:
        return ScoredCandidates(
            ordered=[],
            serendipity_context=[],
            skipped_intent=skipped_intent,
            skipped_people=skipped_people,
        )

    ann_weight_override = params.mixer_ann_weight
    collab_weight_override = params.mixer_collab_weight
    trending_weight_override = params.mixer_trending_weight
    popularity_weight_override = params.mixer_popularity_weight
    vote_weight_override = params.mixer_vote_weight
    novelty_weight_override = params.mixer_novelty_weight

    if intent.query:
        if trending_weight_override is None:
            trending_weight_override = 0.0
        if popularity_weight_override is None:
            popularity_weight_override = 0.0
        if vote_weight_override is None:
            vote_weight_override = 0.0
        if novelty_weight_override is None:
            novelty_weight_override = 0.0

    mixer_fn = get_hook("_apply_mixer_scores", apply_mixer_scores)
    mixer_fn(
        ordered,
        ann_weight_override=ann_weight_override,
        collab_weight_override=collab_weight_override,
        trending_weight_override=trending_weight_override,
        popularity_weight_override=popularity_weight_override,
        vote_weight_override=vote_weight_override,
        novelty_weight_override=novelty_weight_override,
    )

    rules_fn = get_hook("apply_business_rules", apply_business_rules)
    ordered = rules_fn(ordered, intent=intent.intent_filters)
    if not ordered:
        return ScoredCandidates(
            ordered=[],
            serendipity_context=[],
            skipped_intent=skipped_intent,
            skipped_people=skipped_people,
        )

    serendipity_context = list(ordered)

    return ScoredCandidates(
        ordered=ordered,
        serendipity_context=serendipity_context,
        skipped_intent=skipped_intent,
        skipped_people=skipped_people,
    )


# Aliases for backwards compatibility with tests
_item_matches_people_filters = item_matches_people_filters
_prioritize_boosted_items = prioritize_boosted_items
_apply_mixer_scores = apply_mixer_scores
