from __future__ import annotations

import base64
import json
import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Sequence, Set

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from sqlalchemy import select, func, or_, cast
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import SQLAlchemyError

from api.db.session import get_db
from api.db.models import Item, ItemEmbedding, Availability, UserHistory
from api.config import COUNTRY_DEFAULT
from api.core.user_utils import load_user_state, canonical_profile_id
from api.core.candidate_gen import ann_candidates
from api.core import llm_parser
from api.core.intent_parser import Intent
from api.core.legacy_intent_parser import (
    parse_intent as legacy_parse_intent,
    item_matches_intent,
    IntentFilters,
)
from api.core.reranker import rerank_with_explanations, diversify_with_mmr
from api.core.business_rules import apply_business_rules
from api.core.llm_parser import rewrite_query, linked_media_types
from api.core.embeddings import encode_texts
from api.core.user_profile import NEGATIVE_EVENT_TYPES, _event_weight
from api.core.maturity import rating_level
from api.core.rewrite import Rewrite

router = APIRouter(prefix="/recommend", tags=["recommend"])
logger = logging.getLogger(__name__)

_STREAMING_PROVIDER_ALIASES: Dict[str, Set[str]] = {
    "netflix": {"netflix", "nfx"},
    "disney_plus": {"disney_plus", "disney", "dnp"},
    "prime_video": {"prime_video", "primevideo", "amazon", "amz", "amp"},
    "hulu": {"hulu", "hlu"},
    "max": {"max", "hbomax", "hbo", "hbm"},
    "apple_tv_plus": {"apple_tv_plus", "appletvplus", "apple", "atp"},
    "paramount_plus": {"paramount_plus", "paramountplus", "prm", "pmnt", "paramount"},
}


def _normalize_streaming_services(
    providers: Sequence[str] | None,
) -> Set[str]:
    normalized: Set[str] = set()
    if not providers:
        return normalized

    for provider in providers:
        if not isinstance(provider, str):
            continue
        key = provider.strip().lower()
        if not key:
            continue
        matched = False
        for canonical, variants in _STREAMING_PROVIDER_ALIASES.items():
            all_aliases = variants.union({canonical})
            if key in all_aliases:
                normalized.update(all_aliases)
                matched = True
                break
        if not matched:
            normalized.add(key)
    return normalized


def _float_from_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value


_HYBRID_ANN_WEIGHT = _float_from_env("HYBRID_ANN_WEIGHT", 0.5)
_HYBRID_POPULARITY_WEIGHT = _float_from_env("HYBRID_POPULARITY_WEIGHT", 0.25)
_HYBRID_TRENDING_WEIGHT = _float_from_env("HYBRID_TRENDING_WEIGHT", 0.4)
_HYBRID_MIN_ANN_WEIGHT = 0.05
_REWRITE_BLEND_ALPHA = 0.5
_REWRITE_BLEND_ALPHA_QUERY = _float_from_env("REWRITE_BLEND_ALPHA_QUERY", 0.2)
_COLLAB_HISTORY_LIMIT_MULTIPLIER = 4
_MIXER_COLLAB_WEIGHT = _float_from_env("MIXER_COLLAB_WEIGHT", 0.3)
_MIXER_TRENDING_WEIGHT = _float_from_env("MIXER_TRENDING_WEIGHT", 0.2)
_MIXER_NOVELTY_WEIGHT = _float_from_env("MIXER_NOVELTY_WEIGHT", 0.1)
_HYBRID_VOTE_WEIGHT = _float_from_env("HYBRID_VOTE_WEIGHT", 0.2)

_SERENDIPITY_RATIO_RAW = _float_from_env("SERENDIPITY_RATIO", 0.15)
if _SERENDIPITY_RATIO_RAW <= 0.0:
    _SERENDIPITY_RATIO = 0.0
else:
    _SERENDIPITY_RATIO = max(0.1, min(0.2, _SERENDIPITY_RATIO_RAW))

_ANN_DESCRIPTION_WEIGHT = _float_from_env("ANN_DESCRIPTION_WEIGHT", 1.2)
_REWRITE_TEXT_WEIGHT = _float_from_env("REWRITE_TEXT_WEIGHT", 1.0)


def _append_weighted_text(
    text: str | None,
    weight_override: float | None,
    default_weight: float,
    texts: List[str],
    weights: List[float],
) -> float:
    normalized = (text or "").strip()
    if not normalized:
        return 0.0
    weight = default_weight if weight_override is None else weight_override
    weight = max(0.0, weight)
    if weight <= 0.0:
        return 0.0
    texts.append(normalized)
    weights.append(weight)
    return weight


@dataclass
class PrefilterDecision:
    allowed_ids: List[int] | None
    boost_ids: List[int]
    enforce_genres: bool


def _build_rewrite_vector(
    rewrite_text: str | None,
    ann_description: str | None,
    ann_weight_override: float | None = None,
    rewrite_weight_override: float | None = None,
) -> np.ndarray | None:
    texts: List[str] = []
    weights: List[float] = []

    description_weight = _append_weighted_text(
        ann_description,
        ann_weight_override,
        _ANN_DESCRIPTION_WEIGHT,
        texts,
        weights,
    )
    rewrite_weight = _append_weighted_text(
        rewrite_text,
        rewrite_weight_override,
        _REWRITE_TEXT_WEIGHT,
        texts,
        weights,
    )

    if not texts:
        return None

    vectors = encode_texts(texts)
    if not isinstance(vectors, np.ndarray) or vectors.size == 0:
        return None

    combined = np.zeros(vectors.shape[1], dtype=np.float32)
    total_weight = 0.0
    for vec, weight in zip(vectors, weights):
        if weight <= 0.0:
            continue
        combined += weight * vec
        total_weight += weight

    if total_weight <= 0.0:
        return None

    norm = float(np.linalg.norm(combined))
    if norm == 0.0 or not np.isfinite(norm):
        return None
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Rewrite vector sources | description=%s rewrite=%s desc_weight=%.3f rewrite_weight=%.3f total_weight=%.3f",
            bool(description_weight),
            bool(rewrite_weight),
            description_weight,
            rewrite_weight,
            total_weight,
        )
    return combined / norm


def _parse_llm_intent(
    query: str | None,
    user_context: Dict[str, Any],
    linked_entities: Dict[str, Any] | None = None,
) -> Intent:
    if not query:
        return llm_parser.default_intent()
    try:
        return llm_parser.parse_intent(query, user_context, linked_entities)
    except Exception:
        logger.exception("LLM intent parser failed; falling back to default intent.")
        return llm_parser.default_intent()


def _intent_filters_from_llm(query: str | None, llm_intent: Intent) -> IntentFilters:
    genres = list(llm_intent.include_genres or [])
    filters = IntentFilters(
        raw_query=query or "",
        genres=genres,
        moods=[],
        media_types=[],
        min_runtime=llm_intent.runtime_minutes_min,
        max_runtime=llm_intent.runtime_minutes_max,
        maturity_rating_max=llm_intent.maturity_rating_max,
    )
    return filters


def _merge_with_legacy_filters(
    primary: IntentFilters, fallback: IntentFilters | None
) -> IntentFilters:
    if not fallback:
        return primary

    # Merge genres with order preservation
    seen_genres = {genre for genre in primary.genres}
    for genre in fallback.genres:
        if genre and genre not in seen_genres:
            primary.genres.append(genre)
            seen_genres.add(genre)

    if not primary.moods and fallback.moods:
        primary.moods = list(fallback.moods)

    if not primary.media_types and fallback.media_types:
        primary.media_types = list(fallback.media_types)

    if primary.min_runtime is None and fallback.min_runtime is not None:
        primary.min_runtime = fallback.min_runtime

    if primary.max_runtime is None and fallback.max_runtime is not None:
        primary.max_runtime = fallback.max_runtime

    _merge_maturity_rating(primary, fallback)

    return primary


def _merge_maturity_rating(primary: IntentFilters, fallback: IntentFilters) -> None:
    """Merge maturity caps, keeping the stricter (lower) rating."""
    fallback_cap = getattr(fallback, "maturity_rating_max", None)
    if not fallback_cap:
        return

    if not primary.maturity_rating_max:
        primary.maturity_rating_max = fallback_cap
        return

    fallback_level = rating_level(fallback_cap)
    primary_level = rating_level(primary.maturity_rating_max)
    if fallback_level is None:
        return
    if primary_level is None or fallback_level < primary_level:
        primary.maturity_rating_max = fallback_cap


def _apply_franchise_cap(
    candidates: List[Dict[str, Any]], cap: int = 2
) -> List[Dict[str, Any]]:
    if not candidates or cap <= 0:
        return candidates

    franchise_counts: Dict[int, int] = {}
    filtered_candidates: List[Dict[str, Any]] = []

    for item in candidates:
        collection_id = item.get("collection_id")
        if collection_id is None:
            filtered_candidates.append(item)
            continue

        count = franchise_counts.get(collection_id, 0)
        if count < cap:
            filtered_candidates.append(item)
            franchise_counts[collection_id] = count + 1

    return filtered_candidates


def _is_long_tail(item: Dict[str, Any], limit: int) -> bool:
    if limit <= 0:
        return False
    original_rank = int(item.get("original_rank", 0) or 0)
    return original_rank >= limit


def _serendipity_target(limit: int) -> int:
    """
    Determine how many serendipity slots to allocate.

    We skip serendipity when the limit is very small (<=2) because swapping
    long-tail items into a list that short tends to degrade perceived quality.
    """
    if _SERENDIPITY_RATIO <= 0.0 or limit <= 2:
        return 0
    target = round(limit * _SERENDIPITY_RATIO)
    target = max(1, target)
    return min(limit, target)


def _apply_serendipity_slot(
    current: List[Dict[str, Any]],
    candidate_pool: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    if not current or limit <= 0 or _SERENDIPITY_RATIO <= 0.0:
        return current

    top_count = min(limit, len(current))
    target = _serendipity_target(limit)
    if target == 0:
        return current

    top_section = list(current[:top_count])
    existing_long_tail = [item for item in top_section if _is_long_tail(item, limit)]
    if len(existing_long_tail) >= target:
        return current

    short_tail_candidates = [
        idx for idx, item in enumerate(top_section) if not _is_long_tail(item, limit)
    ]
    if not short_tail_candidates:
        return current

    top_ids = {item.get("id") for item in top_section if item.get("id") is not None}

    replacement_pool: List[Dict[str, Any]] = []
    seen_pool: set[int] = set()
    for item in candidate_pool:
        ident = item.get("id")
        if ident is None or ident in top_ids or ident in seen_pool:
            continue
        if _is_long_tail(item, limit):
            replacement_pool.append(item)
            seen_pool.add(ident)

    if not replacement_pool:
        return current

    needed = min(target - len(existing_long_tail), len(replacement_pool))
    if needed <= 0:
        return current

    short_tail_candidates = short_tail_candidates[-needed:]
    replacements = replacement_pool[:needed]

    new_top = top_section
    for idx, replacement in zip(short_tail_candidates, replacements):
        new_top[idx] = replacement

    deduped: List[Dict[str, Any]] = []
    seen_ids: set[int] = set()
    for item in new_top + list(current):
        ident = item.get("id")
        if ident is None or ident not in seen_ids:
            if ident is not None:
                seen_ids.add(ident)
            deduped.append(item)

    return deduped


@router.get("")
async def recommend(
    request: Request,
    user_id: str = Query(..., description="Seen'emAll user_id (e.g., 'u1')"),
    limit: int = Query(20, ge=1, le=100),
    query: str | None = Query(
        None,
        description="Optional natural-language intent (e.g. 'light sci-fi < 2h')",
    ),
    cursor: str | None = Query(
        None,
        description="Opaque cursor returned by a previous request for pagination.",
    ),
    diversify: bool = Query(True, description="Whether to diversify recommendations."),
    profile: str | None = Query(None, description="Optional profile identifier"),
    use_llm_intent: bool = Query(
        True,
        description="Enable the LLM intent parser (set to false for manual overrides).",
    ),
    ann_description_override: str | None = Query(
        None,
        description="Manual ANN description override to blend into the rewrite vector.",
    ),
    rewrite_override: str | None = Query(
        None,
        description="Manual rewrite text override (skips rewrite_query when provided).",
    ),
    ann_weight_override: float | None = Query(
        None,
        ge=0.0,
        description="Override weight for the ANN description component.",
    ),
    rewrite_weight_override: float | None = Query(
        None,
        ge=0.0,
        description="Override weight for the rewrite text component.",
    ),
    genre_override: str | None = Query(
        None,
        description="Comma-separated manual genres to enforce (e.g., 'Drama, Sci-Fi').",
    ),
    mixer_ann_weight: float | None = Query(
        None,
        ge=0.0,
        description="Override hybrid ANN weight (default from HYBRID_ANN_WEIGHT).",
    ),
    mixer_collab_weight: float | None = Query(
        None,
        ge=0.0,
        description="Override collaborative weight (default MIXER_COLLAB_WEIGHT).",
    ),
    mixer_trending_weight: float | None = Query(
        None,
        ge=0.0,
        description="Override trending weight (default HYBRID_TRENDING_WEIGHT).",
    ),
    mixer_popularity_weight: float | None = Query(
        None,
        ge=0.0,
        description="Override popularity weight (default HYBRID_POPULARITY_WEIGHT).",
    ),
    mixer_vote_weight: float | None = Query(
        None,
        ge=0.0,
        description="Override vote-count weight (default HYBRID_VOTE_WEIGHT).",
    ),
    mixer_novelty_weight: float | None = Query(
        None,
        ge=0.0,
        description="Override novelty weight (default MIXER_NOVELTY_WEIGHT).",
    ),
    db: Session = Depends(get_db),
):
    canonical_id = canonical_profile_id(user_id, profile)
    long_v, short_v, exclude, profile_meta = load_user_state(db, canonical_id)
    cold_start = short_v is None

    linked_entities = None
    if query:
        entity_linker = getattr(request.app.state, "entity_linker", None)
        if entity_linker:
            linked_entities = await entity_linker.link_entities(query)

    llm_user_context = {"user_id": canonical_id, "profile_id": profile}
    if use_llm_intent:
        llm_intent = _parse_llm_intent(query, llm_user_context, linked_entities)
    else:
        llm_intent = Intent()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "LLM intent parser disabled for user %s; using manual/default intent.",
                canonical_id,
            )
    if logger.isEnabledFor(logging.DEBUG):
        intent_snapshot = {
            key: value
            for key, value in llm_intent.model_dump(exclude_none=True).items()
            if key in {"include_genres", "exclude_genres", "maturity_rating_max"}
        }
        logger.debug("LLM intent parsed for user %s: %s", canonical_id, intent_snapshot)
        if linked_entities:
            entity_counts = {
                key: len(value) if isinstance(value, list) else 0
                for key, value in linked_entities.items()
            }
            logger.debug("Linked entity counts: %s", entity_counts)
    intent = _intent_filters_from_llm(query, llm_intent)
    if genre_override:
        custom_genres = [g.strip() for g in genre_override.split(",") if g.strip()]
        if custom_genres:
            intent.genres = custom_genres
            llm_intent.include_genres = custom_genres
    preferred_services = _normalize_streaming_services(llm_intent.streaming_providers)
    llm_media_types = list(intent.media_types or [])
    legacy_filters = legacy_parse_intent(query) if query else None
    if legacy_filters:
        intent = _merge_with_legacy_filters(intent, legacy_filters)
    entity_media_types = linked_media_types(linked_entities)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Media type signals for user %s | llm=%s legacy=%s linked=%s -> merged=%s",
            canonical_id,
            llm_media_types,
            legacy_filters.media_types if legacy_filters else None,
            entity_media_types,
            intent.media_types,
        )
        if preferred_services:
            logger.debug(
                "Requested streaming providers for user %s: %s",
                canonical_id,
                sorted(preferred_services),
            )
    if ann_description_override:
        llm_intent.ann_description = ann_description_override.strip()
    candidate_limit = min(500, max(limit, limit * 3))
    if intent.has_filters():
        candidate_limit = min(500, max(candidate_limit, limit * 5))

    prefilter = _prefilter_allowed_ids(
        db, intent, candidate_limit, preferred_services=preferred_services
    )
    allowlist = prefilter.allowed_ids
    boost_ids = prefilter.boost_ids or []
    # Carry the stricter genre requirement forward so downstream filtering
    # (ann_candidates + item_matches_intent) stays aligned with the prefilter.
    enforce_genres = prefilter.enforce_genres
    rewrite_result = None
    rewrite_vec: np.ndarray | None = None
    manual_rewrite_text = (rewrite_override or "").strip() if rewrite_override else ""
    if manual_rewrite_text:
        manual_rewrite_text = " ".join(manual_rewrite_text.split()[:8])
        rewrite_result = Rewrite(rewritten_text=manual_rewrite_text)
    elif query:
        rewrite_result = rewrite_query(query or "", llm_intent)
    rewrite_vec = _build_rewrite_vector(
        getattr(rewrite_result, "rewritten_text", None),
        getattr(llm_intent, "ann_description", None),
        ann_weight_override,
        rewrite_weight_override,
    )

    if cold_start:
        rewrite_used = False
        if rewrite_vec is not None:
            vec_norm = float(np.linalg.norm(rewrite_vec))
            if vec_norm > 0 and np.isfinite(vec_norm):
                ann_ids = ann_candidates(
                    db,
                    rewrite_vec,
                    exclude,
                    limit=candidate_limit,
                    allowed_ids=allowlist,
                )
                if ann_ids:
                    logger.info(
                        "Using rewrite ANN candidates for cold-start user %s",
                        canonical_id,
                    )
                    logger.debug(
                        "Rewrite ANN retrieval | candidate_count=%d allowlist_size=%s",
                        len(ann_ids),
                        len(allowlist) if allowlist is not None else None,
                    )
                    ids = ann_ids
                    rewrite_used = True

        if not rewrite_used:
            logger.info("Using cold-start candidates for user %s", canonical_id)
            ids = _cold_start_candidates(db, intent, candidate_limit, allowlist)
            logger.debug(
                "Cold-start retrieval | candidate_count=%d allowlist_size=%s enforce_genres=%s",
                len(ids),
                len(allowlist) if allowlist is not None else None,
                enforce_genres,
            )
    else:
        logger.info("Using ANN candidates for user %s", canonical_id)
        assert short_v is not None
        if rewrite_vec is not None:
            alpha = _REWRITE_BLEND_ALPHA_QUERY if query else _REWRITE_BLEND_ALPHA
            alpha = max(0.0, min(1.0, alpha))
            q_vec = (alpha * short_v) + ((1 - alpha) * rewrite_vec)
            q_vec = q_vec / np.linalg.norm(q_vec)
        else:
            q_vec = short_v

        ids = ann_candidates(
            db, q_vec, exclude, limit=candidate_limit, allowed_ids=allowlist
        )
        rewrite_applied = bool(rewrite_vec is not None)
        logger.debug(
            "ANN retrieval | candidate_count=%d allowlist_size=%s enforce_genres=%s rewrite=%s",
            len(ids),
            len(allowlist) if allowlist is not None else None,
            enforce_genres,
            rewrite_applied,
        )

    collab_results = _collaborative_candidates(
        db,
        profile_meta.get("neighbors"),
        exclude,
        candidate_limit,
        allowed_ids=allowlist,
    )
    logger.debug(
        "Collaborative retrieval | neighbor_count=%d candidate_count=%d allowlist_size=%s",
        len(profile_meta.get("neighbors") or []),
        len(collab_results),
        len(allowlist) if allowlist is not None else None,
    )
    collab_scores = {iid: score for iid, score in collab_results}
    merged_scores: Dict[int, Dict[str, float]] = {}
    if not cold_start:
        ann_scores = {iid: 1.0 / (1.0 + idx) for idx, iid in enumerate(ids)}
        for iid, score in ann_scores.items():
            merged_scores.setdefault(iid, {})["ann"] = score
    else:
        ann_scores = {}

    if collab_scores:
        max_collab = max(collab_scores.values()) or 1.0
        for iid, score in collab_scores.items():
            merged_scores.setdefault(iid, {})["collab"] = score / max_collab

    trending_results: List[Tuple[int, float]] = []
    if not query:
        if cold_start or ann_scores or collab_scores or boost_ids:
            trending_results = _trending_prior_candidates(
                db,
                intent,
                exclude,
                candidate_limit,
                allowed_ids=allowlist,
            )
            logger.debug(
                "Trending prior retrieval | candidate_count=%d allowlist_size=%s",
                len(trending_results),
                len(allowlist) if allowlist is not None else None,
            )
            trending_scores = {iid: score for iid, score in trending_results}
            if trending_scores:
                max_trending = max(trending_scores.values()) or 1.0
                for iid, score in trending_scores.items():
                    merged_scores.setdefault(iid, {})["trending"] = score / max_trending

    collab_ids = [iid for iid, _ in collab_results]
    trending_ids = [iid for iid, _ in trending_results]

    if collab_ids or trending_ids:
        merged: List[int] = []
        seen: set[int] = set()
        sequences = [collab_ids, ids, trending_ids]
        for seq in sequences:
            for iid in seq:
                if iid in seen:
                    continue
                merged.append(iid)
                seen.add(iid)
                if len(merged) >= candidate_limit:
                    break
            if len(merged) >= candidate_limit:
                break
        ids = merged
    # Limit applied after boost reordering

    if boost_ids:
        exclude_set = set(exclude or [])
        priority: List[int] = []
        seen_priority: set[int] = set()
        for candidate in boost_ids:
            if candidate in exclude_set or candidate in seen_priority:
                continue
            priority.append(candidate)
            seen_priority.add(candidate)

        if priority:
            priority = [cand for cand in priority if cand in ids]
            if priority:
                combined: List[int] = list(priority)
                seen_all = set(priority)
                for candidate in ids:
                    if candidate in seen_all:
                        continue
                    combined.append(candidate)
                    seen_all.add(candidate)

                ids = combined
                for idx, candidate in enumerate(priority):
                    score = 1.0 / (1.0 + idx)
                    scores = merged_scores.setdefault(candidate, {})
                    current = scores.get("ann")
                    if current is None or score > current:
                        scores["ann"] = score
                logger.debug(
                    "Boost reordering applied | boost_count=%d merged_length=%d",
                    len(priority),
                    len(ids),
                )

    ids = ids[:candidate_limit]

    negative_items = set(profile_meta.get("negative_items") or [])
    if negative_items:
        ids = [i for i in ids if i not in negative_items]
    if not ids:
        return _empty_response()

    # fetch metadata and streaming links, preserve ANN order
    items_with_data = {
        row.id: (row, vec, watch_options)
        for row, vec, watch_options in db.execute(
            select(
                Item,
                ItemEmbedding.vector,
                func.json_agg(
                    func.json_build_object(
                        "service",
                        Availability.service,
                        "url",
                        func.coalesce(Availability.web_url, Availability.deeplink),
                    )
                ).label("watch_options"),
            )
            .join(ItemEmbedding, Item.id == ItemEmbedding.item_id)
            .outerjoin(
                Availability,
                (Item.id == Availability.item_id)
                & (Availability.country == COUNTRY_DEFAULT),
            )
            .where(Item.id.in_(ids))
            .group_by(Item.id, ItemEmbedding.vector)
        ).all()
    }
    ordered: List[Dict[str, Any]] = []
    fallback_candidates: List[Dict[str, Any]] = []
    max_candidates = min(candidate_limit, max(limit * 2, 25))
    skipped_intent = 0
    rank_counter = 0
    for iid in ids:
        it, vec, watch_options = items_with_data.get(iid, (None, None, None))
        if it is None or vec is None or len(vec) == 0:
            continue
        if not item_matches_intent(it, intent, enforce_genres=enforce_genres):
            skipped_intent += 1
            continue

        sources = merged_scores.get(iid, {})

        # Clean up watch_options - remove null entries and handle None case
        cleaned_options = []
        if (
            watch_options and watch_options[0] is not None
        ):  # PostgreSQL returns [null] when no matches
            cleaned_options = [
                {"service": opt["service"], "url": opt["url"]}
                for opt in watch_options
                if opt["url"] is not None
            ]
        provider_allowed = True
        filtered_options = cleaned_options
        if preferred_services:
            matching_options = [
                opt
                for opt in cleaned_options
                if isinstance(opt, dict)
                and ((service := str(opt.get("service") or "").strip().lower()))
                and service in preferred_services
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

    if preferred_services and len(ordered) < limit and fallback_candidates:
        deficit = limit - len(ordered)
        ordered.extend(fallback_candidates[:deficit])

    if not ordered:
        return _empty_response()
    if skipped_intent and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Post-filter dropped %d candidates due to intent match (enforce_genres=%s)",
            skipped_intent,
            enforce_genres,
        )

    if diversify:
        ordered = _apply_franchise_cap(ordered)
    ann_weight_override = mixer_ann_weight
    collab_weight_override = mixer_collab_weight
    trending_weight_override = mixer_trending_weight
    popularity_weight_override = mixer_popularity_weight
    vote_weight_override = mixer_vote_weight
    novelty_weight_override = mixer_novelty_weight

    if query:
        if trending_weight_override is None:
            trending_weight_override = 0.0
        if popularity_weight_override is None:
            popularity_weight_override = 0.0
        if vote_weight_override is None:
            vote_weight_override = 0.0
        if novelty_weight_override is None:
            novelty_weight_override = 0.0

    _apply_mixer_scores(
        ordered,
        ann_weight_override=ann_weight_override,
        collab_weight_override=collab_weight_override,
        trending_weight_override=trending_weight_override,
        popularity_weight_override=popularity_weight_override,
        vote_weight_override=vote_weight_override,
        novelty_weight_override=novelty_weight_override,
    )
    ordered = apply_business_rules(ordered, intent=intent)
    if not ordered:
        return _empty_response()

    serendipity_context = list(ordered)

    if diversify:
        ordered = diversify_with_mmr(ordered, limit=limit)

    ordered = _apply_serendipity_slot(ordered, serendipity_context, limit)

    reranked = rerank_with_explanations(
        ordered,
        intent=intent,
        query=query,
        user={
            "user_id": canonical_id,
            "base_user_id": user_id,
            "profile": profile,
            "genre_prefs": profile_meta.get("genre_prefs"),
            "neighbors": profile_meta.get("neighbors"),
            "negative_items": profile_meta.get("negative_items"),
        },
    )

    start_index = _decode_cursor(cursor)
    if start_index < 0:
        raise HTTPException(status_code=400, detail="Invalid cursor")

    page = reranked[start_index : start_index + limit]
    response: List[Dict[str, Any]] = []
    for entry in page:
        cleaned = dict(entry)
        cleaned.pop("original_rank", None)
        cleaned.pop("vector", None)
        cleaned.pop("ann_rank", None)
        cleaned.pop("retrieval_score", None)
        cleaned.pop("source_scores", None)
        response.append(cleaned)

    next_cursor = None
    if start_index + limit < len(reranked):
        next_cursor = _encode_cursor(start_index + limit)

    payload: Dict[str, Any] = {"items": response}
    if next_cursor:
        payload["next_cursor"] = next_cursor
    return payload


def _apply_mixer_scores(
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


def _collaborative_candidates(
    db: Session,
    neighbors: List[Dict[str, Any]] | None,
    exclude_ids: List[int],
    limit: int,
    allowed_ids: List[int] | None,
) -> List[Tuple[int, float]]:
    if not neighbors:
        return []

    if allowed_ids is not None and len(allowed_ids) == 0:
        return []

    neighbor_weights: Dict[str, float] = {}
    for entry in neighbors:
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get("user_id")
        if not raw_id:
            continue
        uid = str(raw_id).strip()
        if not uid:
            continue
        weight = float(entry.get("weight", 0.0) or 0.0)
        if weight <= 0:
            continue
        neighbor_weights[uid] = weight

    if not neighbor_weights:
        return []

    exclude_set = set(exclude_ids or [])
    allowed_set = set(allowed_ids) if allowed_ids is not None else None
    history_limit = max(limit * _COLLAB_HISTORY_LIMIT_MULTIPLIER, 200)

    stmt = (
        select(
            UserHistory.item_id,
            UserHistory.user_id,
            UserHistory.weight,
            UserHistory.event_type,
            UserHistory.ts,
        )
        .where(UserHistory.user_id.in_(list(neighbor_weights)))
        .order_by(UserHistory.ts.desc())
        .limit(history_limit)
    )

    try:
        rows = db.execute(stmt).all()
    except SQLAlchemyError:
        logger.debug("Collaborative recall query failed.", exc_info=True)
        return []

    if not rows:
        return []

    aggregated: Dict[int, tuple[float, float]] = {}
    for item_id, uid, base_weight, event_type, ts in rows:
        if item_id is None or uid is None:
            continue
        uid_str = str(uid)
        neighbor_weight = neighbor_weights.get(uid_str)
        if neighbor_weight is None:
            continue
        if item_id in exclude_set:
            continue
        if allowed_set is not None and item_id not in allowed_set:
            continue
        normalized_event = (event_type or "").lower()
        if normalized_event in NEGATIVE_EVENT_TYPES:
            continue
        event_score = _event_weight(normalized_event, base_weight)
        if event_score <= 0:
            continue
        score = neighbor_weight * event_score
        ts_value = 0.0
        if ts is not None:
            ts_value = float(ts.timestamp()) if hasattr(ts, "timestamp") else 0.0
        if item_id in aggregated:
            current_score, current_ts = aggregated[item_id]
            aggregated[item_id] = (
                current_score + score,
                max(current_ts, ts_value),
            )
        else:
            aggregated[item_id] = (score, ts_value)

    if not aggregated:
        return []

    ranked = sorted(
        aggregated.items(),
        key=lambda kv: (-kv[1][0], -kv[1][1], kv[0]),
    )
    top = ranked[:limit]
    if not top:
        return []
    max_score = max(score for _, (score, _) in top)
    if max_score <= 0:
        max_score = 1.0
    scaled: List[Tuple[int, float]] = []
    for item_id, (score, _) in top:
        normalized = score / max_score if max_score > 0 else 0.0
        scaled.append((item_id, normalized))
    return scaled


def _trending_prior_candidates(
    db: Session,
    intent: IntentFilters | None,
    exclude_ids: List[int],
    limit: int,
    allowed_ids: List[int] | None,
) -> List[Tuple[int, float]]:
    if limit <= 0:
        return []

    stmt = select(
        Item.id,
        Item.trending_rank,
        Item.popular_rank,
        Item.popularity,
    )

    filters = []

    if allowed_ids is not None:
        if not allowed_ids:
            return []
        filters.append(Item.id.in_(allowed_ids))

    if exclude_ids:
        filters.append(~Item.id.in_(exclude_ids))

    if intent and intent.media_types:
        filters.append(Item.media_type.in_(intent.media_types))

    if intent:
        genres = intent.effective_genres()
        if genres:
            genre_filters = [
                _genre_contains_clause(db, genre) for genre in genres if genre
            ]
            if genre_filters:
                filters.append(or_(*genre_filters))

    if filters:
        stmt = stmt.where(*filters)

    stmt = stmt.order_by(
        Item.trending_rank.asc().nullslast(),
        Item.popular_rank.asc().nullslast(),
        Item.popularity.desc().nullslast(),
        Item.id.asc(),
    ).limit(limit)

    rows = db.execute(stmt).all()
    if not rows:
        return []

    scored: List[Tuple[int, float, float]] = []
    for item_id, trending_rank, popular_rank, popularity in rows:
        if item_id is None:
            continue

        if not isinstance(trending_rank, (int, float)):
            trending_rank = None
        if not isinstance(popular_rank, (int, float)):
            popular_rank = None
        if not isinstance(popularity, (int, float)):
            popularity = None

        rank_score = 0.0
        if trending_rank and trending_rank > 0:
            rank_score += 1.0 / (1.0 + float(trending_rank))
        if popular_rank and popular_rank > 0:
            rank_score += 0.5 / (1.0 + float(popular_rank))
        pop_score = float(popularity) if popularity is not None else 0.0
        scored.append((int(item_id), rank_score, pop_score))

    max_rank = max((entry[1] for entry in scored), default=0.0)
    max_pop = max((entry[2] for entry in scored), default=0.0)
    if max_rank <= 0:
        max_rank = 1.0
    if max_pop <= 0:
        max_pop = 1.0

    results: List[Tuple[int, float]] = []
    for item_id, rank_score, pop_score in scored:
        normalized_rank = rank_score / max_rank if max_rank > 0 else 0.0
        normalized_pop = pop_score / max_pop if max_pop > 0 else 0.0
        combined = 0.7 * normalized_rank + 0.3 * normalized_pop
        results.append((item_id, combined))
    return results


def _prefilter_allowed_ids(
    db: Session,
    intent: IntentFilters | None,
    limit: int,
    preferred_services: Set[str] | None = None,
) -> PrefilterDecision:
    if intent is None or not intent.has_filters():
        # Nothing to prefilter: leave the allowlist unset, but keep genre enforcement
        # enabled so downstream checks still respect catalog intent defaults.
        return PrefilterDecision(None, [], True)

    threshold = max(10, limit // 2)
    fetch_limit = max(limit * 5, limit)

    strict_ids = _run_prefilter_query(
        db,
        intent,
        fetch_limit=fetch_limit,
        include_genres=True,
        required_services=preferred_services,
    )
    logger.debug(
        "Prefilter strict run | threshold=%d strict_count=%d media_types=%s genres=%s",
        threshold,
        len(strict_ids),
        intent.media_types,
        intent.effective_genres(),
    )
    boost_cap = max(5, min(limit, 15))

    if len(strict_ids) >= threshold:
        logger.debug("Prefilter returning strict allowlist.")
        return PrefilterDecision(strict_ids, strict_ids[:boost_cap], True)

    relaxed_ids = _run_prefilter_query(
        db,
        intent,
        fetch_limit=fetch_limit,
        include_genres=False,
        required_services=preferred_services,
    )

    if len(relaxed_ids) >= threshold:
        # Genre matches were sparse, treat them as boosts but keep other filters strict.
        logger.debug(
            "Prefilter relaxed pass selected | relaxed_count=%d strict_count=%d",
            len(relaxed_ids),
            len(strict_ids),
        )
        return PrefilterDecision(relaxed_ids, strict_ids, False)

    # Fall back to ANN-first retrieval. Preserve genre matches as soft boosts.
    logger.debug(
        "Prefilter falling back to ANN-first | relaxed_count=%d strict_count=%d",
        len(relaxed_ids),
        len(strict_ids),
    )
    return PrefilterDecision(None, strict_ids[:boost_cap], False)


def _run_prefilter_query(
    db: Session,
    intent: IntentFilters,
    *,
    fetch_limit: int,
    include_genres: bool,
    required_services: Set[str] | None = None,
) -> List[int]:
    stmt = select(Item.id)

    if required_services:
        stmt = (
            stmt.join(
                Availability,
                (Availability.item_id == Item.id)
                & (Availability.country == COUNTRY_DEFAULT),
            )
            .where(Availability.service.in_(required_services))
            .distinct()
        )

    media_types = intent.media_types or []
    if media_types:
        stmt = stmt.where(Item.media_type.in_(media_types))

    if include_genres:
        genres = intent.effective_genres()
        if genres:
            filters = [_genre_contains_clause(db, genre) for genre in genres if genre]
            if filters:
                stmt = stmt.where(or_(*filters))

    rows = db.execute(stmt.limit(fetch_limit)).scalars().all()
    return _ordered_unique(rows)


def _ordered_unique(values: List[int]) -> List[int]:
    seen: set[int] = set()
    ordered: List[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _genre_contains_clause(db: Session, genre: str):
    bind = getattr(db, "bind", None)
    if bind is None:
        get_bind = getattr(db, "get_bind", None)
        if callable(get_bind):
            try:
                bind = get_bind()
            except Exception:
                bind = None

    dialect_name: Optional[str] = None
    if bind is not None:
        dialect = getattr(bind, "dialect", None)
        if dialect is not None:
            dialect_name = getattr(dialect, "name", None)

    if dialect_name == "postgresql":
        return cast(Item.genres, JSONB).contains([{"name": genre}])

    # Fallback for other dialects used in tests (e.g., SQLite memory stubs)
    return Item.genres.contains([{"name": genre}])


def _cold_start_candidates(
    db: Session,
    intent: IntentFilters,
    limit: int,
    allowlist: List[int] | None,
) -> List[int]:
    stmt = select(Item.id).join(ItemEmbedding, ItemEmbedding.item_id == Item.id)

    if allowlist is not None:
        if not allowlist:
            return []
        stmt = stmt.where(Item.id.in_(allowlist))
    else:
        if intent.media_types:
            stmt = stmt.where(Item.media_type.in_(intent.media_types))
        genres = intent.effective_genres()
        if genres:
            genre_filters = [
                _genre_contains_clause(db, genre) for genre in genres if genre
            ]
            if genre_filters:
                stmt = stmt.where(or_(*genre_filters))

    stmt = stmt.order_by(
        Item.popular_rank.asc().nullslast(),
        Item.trending_rank.asc().nullslast(),
        Item.popularity.desc().nullslast(),
        Item.id.asc(),
    ).limit(limit)

    rows = db.execute(stmt).scalars().all()
    seen: set[int] = set()
    ordered: List[int] = []
    for value in rows:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _encode_cursor(rank: int) -> str:
    payload = json.dumps({"rank": rank}).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("utf-8").rstrip("=")


def _decode_cursor(cursor: str | None) -> int:
    if not cursor:
        return 0
    padding = "=" * (-len(cursor) % 4)
    try:
        decoded = base64.urlsafe_b64decode(cursor + padding)
        payload = json.loads(decoded.decode("utf-8"))
        rank = int(payload["rank"])
        if rank < 0:
            raise ValueError
        return rank
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=400, detail="Invalid cursor") from exc


def _empty_response() -> Dict[str, Any]:
    return {"items": []}
