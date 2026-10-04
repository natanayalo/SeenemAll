from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from fastapi import HTTPException, Request
from sqlalchemy import cast, select, String
from sqlalchemy.orm import Session

from api.config import COUNTRY_DEFAULT
from api.db.models import Availability, Item
from api.core.elasticsearch_search import SearchFilters
from api.core.filter_matcher import get_filter_matcher, get_query_filters
from api.core.legacy_intent_parser import parse_intent as legacy_parse_intent
from api.core.llm_parser import default_intent, linked_media_types
from api.core.metrics import METRICS
from api.core.query_formulation import select_retrieval_query
from api.pipeline.context import normalize_streaming_services
from api.pipeline.hooks import get_hook
from api.pipeline.intent.matcher import (
    _has_people_filters,
    matches_keywords,
    merge_query_filter_hints,
    strict_required_genres,
)
from api.pipeline.intent.parser import (
    intent_filters_from_llm,
    merge_with_legacy_filters,
    parse_llm_intent,
)
from api.pipeline.intent.rewrite import build_query_vector
from api.pipeline.models import QueryUnderstanding, RecommendParams, UserContext

logger = logging.getLogger("api.routes.recommend")

_SUPPORTED_ANN_BACKENDS = {"elasticsearch", "pgvector"}


async def resolve_query_intent(
    request: Request,
    params: RecommendParams,
    user_context: UserContext,
    db: Session,
) -> QueryUnderstanding:
    query = params.query
    canonical_id = user_context.canonical_id
    backend_override_normalized: str | None = None
    if params.ann_backend_override:
        candidate_backend = params.ann_backend_override.strip().lower()
        if candidate_backend and candidate_backend not in _SUPPORTED_ANN_BACKENDS:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported ann_backend_override. "
                    "Valid options: 'elasticsearch', 'pgvector'."
                ),
            )
        backend_override_normalized = candidate_backend or None

    linked_entities = None
    if query:
        entity_linker = getattr(request.app.state, "entity_linker", None)
        if entity_linker:
            linked_entities = await entity_linker.link_entities(query)

    llm_user_context = {"user_id": canonical_id, "profile_id": params.profile}
    parser_fn = get_hook("_parse_llm_intent", parse_llm_intent)
    if params.use_llm_intent:
        llm_intent = parser_fn(query, llm_user_context, linked_entities)
    else:
        llm_intent = default_intent()
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

    intent = intent_filters_from_llm(query, llm_intent)
    custom_genres: List[str] = []
    if params.genre_override:
        custom_genres = [
            g.strip() for g in params.genre_override.split(",") if g.strip()
        ]
        if custom_genres:
            intent.genres = custom_genres
            llm_intent.include_genres = custom_genres

    preferred_services = normalize_streaming_services(
        llm_intent.streaming_providers, user_context.provider_alias_map
    )
    user_context.preferred_services = preferred_services

    legacy_parser_fn = get_hook("legacy_parse_intent", legacy_parse_intent)
    legacy_filters = legacy_parser_fn(query) if query else None
    if legacy_filters:
        merge_legacy_fn = get_hook(
            "_merge_with_legacy_filters", merge_with_legacy_filters
        )
        intent = merge_legacy_fn(intent, legacy_filters)

    matcher_fn = get_hook("get_query_filters", get_query_filters)
    query_filter_result = matcher_fn(query)
    merge_query_filter_hints(intent, query_filter_result, db)

    prefer_top_rated = (
        bool(params.classic_top_rated)
        if params.classic_top_rated is not None
        else False
    )
    heuristic_applied = False
    has_top_keyword = matches_keywords(query, user_context.top_query_keywords)
    fallback_keywords_used = bool(getattr(intent, "_fallback_keywords_used", False))
    if (
        params.classic_top_rated is None
        and has_top_keyword
        and not fallback_keywords_used
    ):
        prefer_top_rated = True
        heuristic_applied = True

    if prefer_top_rated:
        if params.diversify:
            params.diversify = False
        if params.mixer_ann_weight is None:
            params.mixer_ann_weight = 0.2
        if params.mixer_collab_weight is None:
            params.mixer_collab_weight = 0.2
        if params.mixer_trending_weight is None:
            params.mixer_trending_weight = 0.0
        if params.mixer_popularity_weight is None:
            params.mixer_popularity_weight = 0.0
        if params.mixer_vote_weight is None:
            params.mixer_vote_weight = 1.2
        if params.mixer_novelty_weight is None:
            params.mixer_novelty_weight = 0.0
        if heuristic_applied and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Auto-enabled classic_top_rated for query '%s' due to keyword match.",
                query,
            )

    llm_media_types = list(intent.media_types or [])

    prefilter_kwargs: Dict[str, Any] = {}
    if params.strict_filters:
        prefilter_kwargs["require_all_genres"] = True
        strict_required = strict_required_genres(
            db, custom_genres, legacy_filters, intent
        )
        if strict_required:
            intent.required_genres = strict_required

    providers_list = sorted(preferred_services) if preferred_services else []

    def _unique_sequence(values: Sequence[str]) -> List[str]:
        seen: set[str] = set()
        ordered: List[str] = []
        for value in values:
            if not value:
                continue
            if value not in seen:
                ordered.append(value)
                seen.add(value)
        return ordered

    merged_reference_titles = _unique_sequence(
        list(query_filter_result.reference_titles or ())
        + list(llm_intent.reference_titles or [])
        + list(llm_intent.franchises or [])
    )
    if merged_reference_titles:
        for t in merged_reference_titles:
            if t not in intent.reference_titles:
                intent.reference_titles.append(t)
        from dataclasses import is_dataclass, replace

        if is_dataclass(query_filter_result) and not isinstance(
            query_filter_result, type
        ):
            query_filter_result = replace(
                query_filter_result, reference_titles=tuple(merged_reference_titles)
            )
        else:
            try:
                setattr(
                    query_filter_result,
                    "reference_titles",
                    tuple(merged_reference_titles),
                )
            except Exception:
                pass

    genre_filters = intent.required_genres or intent.effective_genres()
    if not genre_filters:
        genre_filters = list(query_filter_result.genres)

    from api.core.fast_intent_parser import FRANCHISE_GENRES

    for f in llm_intent.franchises or []:
        f_clean = f.lower().strip()
        for franchise_key, mapped_genres in FRANCHISE_GENRES.items():
            if franchise_key in f_clean or f_clean in franchise_key:
                for g in mapped_genres:
                    if g not in genre_filters:
                        genre_filters.append(g)

    genre_filters = _unique_sequence(genre_filters)

    media_type_filters = intent.media_types or list(query_filter_result.media_types)
    media_type_filters = _unique_sequence(media_type_filters)

    keywords_refined = bool(getattr(intent, "_query_keywords_merged", False))
    keyword_filters = list(intent.keywords or [])
    if not keyword_filters and not keywords_refined:
        keyword_filters = list(query_filter_result.keywords)
    for kw in getattr(llm_intent, "keywords", None) or []:
        if kw not in keyword_filters:
            keyword_filters.append(kw)
    keyword_filters = _unique_sequence(keyword_filters)

    cast_filters = list(query_filter_result.cast)
    director_filters = list(query_filter_result.directors)
    producer_filters = list(query_filter_result.producers)
    writer_filters = list(query_filter_result.writers)

    matcher_getter = get_hook("get_filter_matcher", get_filter_matcher)
    matcher = matcher_getter()

    def _add_person(name: str, target_role: str) -> None:
        if not name or not name.strip():
            return
        resolved_name, roles = matcher.resolve_person(name)
        canonical = resolved_name or name.strip()
        role_to_list = {
            "cast": cast_filters,
            "directors": director_filters,
            "producers": producer_filters,
            "writers": writer_filters,
        }
        assigned_role = target_role
        if roles and target_role not in roles:
            for r in ("directors", "cast", "producers", "writers"):
                if r in roles:
                    assigned_role = r
                    break
        target = role_to_list.get(assigned_role, cast_filters)
        if canonical not in target:
            target.append(canonical)

    for actor in llm_intent.include_actors or []:
        _add_person(actor, "cast")
    for director in llm_intent.include_directors or []:
        _add_person(director, "directors")
    for producer in llm_intent.include_producers or []:
        _add_person(producer, "producers")
    for writer in llm_intent.include_writers or []:
        _add_person(writer, "writers")

    if llm_intent.include_people:
        for person in llm_intent.include_people:
            if person:
                _add_person(person, "cast")

    ref_names_lower = {t.lower().strip() for t in merged_reference_titles if t}
    for ref_t in merged_reference_titles:
        canonical_ref, _ = matcher.resolve_person(ref_t)
        if canonical_ref:
            ref_names_lower.add(canonical_ref.lower().strip())

    matched_coll_ids: List[int] = []
    for cid, cname in getattr(query_filter_result, "matched_collections", ()):
        if cid not in matched_coll_ids:
            matched_coll_ids.append(cid)
        ref_names_lower.add(cname.lower().strip())
        base = re.sub(
            r"\s+(?:collection|trilogy|saga|series|movies|films)$", "", cname.lower()
        ).strip()
        if base:
            ref_names_lower.add(base)
            if base.startswith("the "):
                ref_names_lower.add(base[4:].strip())

    for f in llm_intent.franchises or []:
        ref_names_lower.add(f.lower().strip())
        resolved_list = matcher.resolve_collections(f)
        if not resolved_list:
            resolved_single = matcher.resolve_collection(f)
            if resolved_single:
                resolved_list = [resolved_single]
        for cid, cname in resolved_list:
            if cid not in matched_coll_ids:
                matched_coll_ids.append(cid)
            ref_names_lower.add(cname.lower().strip())

    query_lower = (query or "").lower()

    if "dark knight" in query_lower:
        dark_knight_ids = [
            cid
            for cid, cname in getattr(query_filter_result, "matched_collections", ())
            if "dark knight" in cname.lower()
        ]
        if not dark_knight_ids:
            dk_res = matcher.resolve_collection("dark knight")
            if dk_res:
                dark_knight_ids = [dk_res[0]]
        if dark_knight_ids:
            matched_coll_ids = [
                cid for cid in matched_coll_ids if cid in dark_knight_ids
            ]

    collection_item_ids: List[int] = []
    if matched_coll_ids:
        coll_stmt = select(Item.id, Item.release_year).where(
            Item.collection_id.in_(matched_coll_ids)
        )
        if llm_intent.year_min:
            coll_stmt = coll_stmt.where(Item.release_year >= llm_intent.year_min)
        if llm_intent.year_max:
            coll_stmt = coll_stmt.where(Item.release_year <= llm_intent.year_max)
        if llm_intent.exclude_genres:
            for ex in llm_intent.exclude_genres:
                coll_stmt = coll_stmt.where(~cast(Item.genres, String).ilike(f"%{ex}%"))
        if "movie" in media_type_filters or "movie" in (intent.media_types or []):
            coll_stmt = coll_stmt.where(Item.media_type == "movie")
        if providers_list:
            coll_stmt = coll_stmt.where(
                Item.id.in_(
                    select(Availability.item_id).where(
                        Availability.country == COUNTRY_DEFAULT,
                        Availability.service.in_(providers_list),
                    )
                )
            )
        coll_stmt = coll_stmt.order_by(Item.release_year.asc().nulls_last())
        coll_rows = db.execute(coll_stmt).all()
        collection_item_ids = [getattr(r, "id", r[0]) for r in coll_rows]

    # MCU franchise handling
    is_mcu_query = any(
        c in query_lower for c in ("marvel cinematic universe", "mcu")
    ) or (
        "marvel" in query_lower
        and any(w in query_lower for w in ("phase", "universe", "cinematic"))
    )
    if is_mcu_query:
        mcu_stmt = select(Item.id, Item.collection_id).where(
            cast(Item.keywords, String).ilike("%marvel cinematic universe%")
        )
        if llm_intent.year_min:
            mcu_stmt = mcu_stmt.where(Item.release_year >= llm_intent.year_min)
        if llm_intent.year_max:
            mcu_stmt = mcu_stmt.where(Item.release_year <= llm_intent.year_max)
        if "movie" in media_type_filters or "movie" in (intent.media_types or []):
            mcu_stmt = mcu_stmt.where(Item.media_type == "movie")
        if providers_list:
            mcu_stmt = mcu_stmt.where(
                Item.id.in_(
                    select(Availability.item_id).where(
                        Availability.country == COUNTRY_DEFAULT,
                        Availability.service.in_(providers_list),
                    )
                )
            )
        mcu_stmt = mcu_stmt.order_by(Item.release_year.asc().nulls_last())
        mcu_rows = db.execute(mcu_stmt).all()
        for r in mcu_rows:
            r_id = getattr(r, "id", r[0])
            r_coll_id = (
                getattr(r, "collection_id", r[1])
                if (
                    hasattr(r, "collection_id")
                    or (isinstance(r, (tuple, list)) and len(r) > 1)
                )
                else None
            )
            if r_id not in collection_item_ids:
                collection_item_ids.append(r_id)
            if r_coll_id and r_coll_id not in matched_coll_ids:
                matched_coll_ids.append(r_coll_id)

    chronological_cues = (
        "chronological",
        "in order",
        "release order",
        "timeline",
        "order",
        "trilogy",
        "saga",
        "series",
        "phase",
        "era",
    )
    is_chronological = (bool(matched_coll_ids) or bool(collection_item_ids)) and any(
        cue in query_lower for cue in chronological_cues
    )

    if ref_names_lower:
        cast_filters = [
            c for c in cast_filters if c.lower().strip() not in ref_names_lower
        ]
        director_filters = [
            d for d in director_filters if d.lower().strip() not in ref_names_lower
        ]
        producer_filters = [
            p for p in producer_filters if p.lower().strip() not in ref_names_lower
        ]
        writer_filters = [
            w for w in writer_filters if w.lower().strip() not in ref_names_lower
        ]

    cast_filters = _unique_sequence(cast_filters)
    director_filters = _unique_sequence(director_filters)
    producer_filters = _unique_sequence(producer_filters)
    writer_filters = _unique_sequence(writer_filters)

    if matched_coll_ids or collection_item_ids:
        has_tv_explicit = any(
            w in query_lower
            for w in ("tv ", "tv show", "television", "miniseries", "sitcom")
        )
        if not has_tv_explicit:
            if "tv" in media_type_filters:
                media_type_filters = [m for m in media_type_filters if m != "tv"]
            if intent.media_types and "tv" in intent.media_types:
                intent.media_types = [m for m in intent.media_types if m != "tv"]

    language_filters = _unique_sequence(
        list(query_filter_result.languages) + list(llm_intent.languages or [])
    )

    structured_search_filters: SearchFilters | None = None
    if (
        providers_list
        or any(
            len(seq)
            for seq in (
                language_filters,
                keyword_filters,
                genre_filters,
                media_type_filters,
                cast_filters,
                director_filters,
                producer_filters,
                writer_filters,
            )
        )
        or any(
            value is not None
            for value in (
                llm_intent.year_min,
                llm_intent.year_max,
                llm_intent.runtime_minutes_min,
                llm_intent.runtime_minutes_max,
                intent.min_runtime,
                intent.max_runtime,
            )
        )
    ):
        structured_search_filters = SearchFilters(
            providers=tuple(providers_list),
            languages=tuple(language_filters),
            keywords=tuple(keyword_filters),
            genres=tuple(genre_filters),
            media_types=tuple(media_type_filters),
            cast=tuple(cast_filters),
            directors=tuple(director_filters),
            producers=tuple(producer_filters),
            writers=tuple(writer_filters),
            release_year_gte=llm_intent.year_min,
            release_year_lte=llm_intent.year_max,
            runtime_gte=(
                llm_intent.runtime_minutes_min
                if llm_intent.runtime_minutes_min is not None
                else intent.min_runtime
            ),
            runtime_lte=(
                llm_intent.runtime_minutes_max
                if llm_intent.runtime_minutes_max is not None
                else intent.max_runtime
            ),
            strict_genres=bool(params.strict_filters),
            is_vibe=bool(getattr(intent, "is_vibe", False)),
        )

    retrieval_query_text, query_formulation = select_retrieval_query(
        query, query_filter_result
    )
    es_text_query: Optional[str] = retrieval_query_text or None
    METRICS.counter(f"recommend.query_formulation.{query_formulation}").inc()

    has_people = _has_people_filters(structured_search_filters)
    entity_media_types = linked_media_types(linked_entities)
    if logger.isEnabledFor(logging.DEBUG):
        if structured_search_filters:
            logger.debug(
                "Search filters | media=%s genres=%s keywords=%s languages=%s people=%s",
                structured_search_filters.media_types,
                structured_search_filters.genres,
                structured_search_filters.keywords,
                structured_search_filters.languages,
                {
                    "cast": structured_search_filters.cast,
                    "directors": structured_search_filters.directors,
                    "producers": structured_search_filters.producers,
                    "writers": structured_search_filters.writers,
                },
            )
        logger.debug("ES text query: %s", es_text_query)
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

    candidate_limit = min(500, max(params.limit, params.limit * 3))
    if intent.has_filters() or getattr(llm_intent, "is_vibe", False):
        candidate_limit = min(500, max(candidate_limit, params.limit * 10, 100))

    vector_builder_fn = get_hook("_build_query_vector", build_query_vector)
    query_vec = vector_builder_fn(retrieval_query_text)

    return QueryUnderstanding(
        query=query,
        llm_intent=llm_intent,
        intent_filters=intent,
        structured_search_filters=structured_search_filters,
        es_text_query=es_text_query,
        query_vec=query_vec,
        prefer_top_rated=prefer_top_rated,
        custom_genres=custom_genres,
        has_people_filters=has_people,
        candidate_limit=candidate_limit,
        backend_override_normalized=backend_override_normalized,
        preferred_services=preferred_services,
        prefilter_kwargs=prefilter_kwargs,
        matched_collection_ids=matched_coll_ids,
        collection_item_ids=collection_item_ids,
        is_chronological_requested=is_chronological,
    )
