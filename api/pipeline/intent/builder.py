from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from fastapi import HTTPException, Request
from sqlalchemy.orm import Session

from api.core.elasticsearch_search import SearchFilters
from api.core.filter_matcher import get_query_filters
from api.core.legacy_intent_parser import parse_intent as legacy_parse_intent
from api.core.llm_parser import default_intent, linked_media_types, rewrite_query
from api.core.rewrite import Rewrite
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
from api.pipeline.intent.rewrite import build_rewrite_vector
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
        if params.ann_weight_override is None:
            params.ann_weight_override = 0.2
        if heuristic_applied and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Auto-enabled classic_top_rated for query '%s' due to keyword match.",
                query,
            )

    llm_media_types = list(intent.media_types or [])
    if hasattr(intent, "genre_keywords") and intent.genre_keywords:
        genre_keyword_text = " ".join(intent.genre_keywords)
        if llm_intent.ann_description:
            llm_intent.ann_description = (
                f"{llm_intent.ann_description.rstrip('.')} {genre_keyword_text}"
            )
        else:
            llm_intent.ann_description = genre_keyword_text

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

    genre_filters = intent.required_genres or intent.effective_genres()
    if not genre_filters:
        genre_filters = list(query_filter_result.genres)
    genre_filters = _unique_sequence(genre_filters)

    media_type_filters = intent.media_types or list(query_filter_result.media_types)
    media_type_filters = _unique_sequence(media_type_filters)

    keywords_refined = bool(getattr(intent, "_query_keywords_merged", False))
    keyword_filters = list(intent.keywords or [])
    if not keyword_filters and not keywords_refined:
        keyword_filters = list(query_filter_result.keywords)
    keyword_filters = _unique_sequence(keyword_filters)

    structured_search_filters: SearchFilters | None = None
    if providers_list or any(
        len(seq)
        for seq in (
            query_filter_result.languages,
            keyword_filters,
            genre_filters,
            media_type_filters,
            query_filter_result.cast,
            query_filter_result.directors,
            query_filter_result.producers,
            query_filter_result.writers,
        )
    ):
        structured_search_filters = SearchFilters(
            providers=tuple(providers_list),
            languages=query_filter_result.languages,
            keywords=tuple(keyword_filters),
            genres=tuple(genre_filters),
            media_types=tuple(media_type_filters),
            cast=query_filter_result.cast,
            directors=query_filter_result.directors,
            producers=query_filter_result.producers,
            writers=query_filter_result.writers,
        )

    es_text_query: Optional[str] = (
        query_filter_result.residual_text or ""
    ).strip() or None
    if not es_text_query and keyword_filters:
        keyword_blob = " ".join(keyword_filters).strip()
        if keyword_blob:
            es_text_query = keyword_blob
    if not es_text_query and query:
        es_text_query = query.strip() or None
    titles_source = intent.reference_titles or query_filter_result.reference_titles
    if titles_source:
        titles_blob = " ".join(titles_source)
        if titles_blob:
            es_text_query = (
                f"{es_text_query} {titles_blob}".strip()
                if es_text_query
                else titles_blob
            )
    if query and not es_text_query:
        es_text_query = query

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

    if params.ann_description_override:
        llm_intent.ann_description = params.ann_description_override.strip()
    include_people = list(dict.fromkeys(llm_intent.include_people or []))
    if include_people:
        people_phrase = ", ".join(include_people)
        if llm_intent.ann_description:
            if people_phrase not in llm_intent.ann_description:
                llm_intent.ann_description = (
                    f"{llm_intent.ann_description.rstrip('.')}. "
                    f"Featuring {people_phrase}."
                )
        else:
            llm_intent.ann_description = f"Featuring {people_phrase}."

    candidate_limit = min(500, max(params.limit, params.limit * 3))
    if intent.has_filters():
        candidate_limit = min(500, max(candidate_limit, params.limit * 5))

    rewrite_result = None
    manual_rewrite_text = (
        (params.rewrite_override or "").strip() if params.rewrite_override else ""
    )
    if manual_rewrite_text:
        manual_rewrite_text = " ".join(manual_rewrite_text.split()[:8])
        rewrite_result = Rewrite(
            rewritten_text=manual_rewrite_text,
            facet_allow=None,
            facet_block=None,
        )
    elif query:
        rewriter_fn = get_hook("rewrite_query", rewrite_query)
        rewrite_result = rewriter_fn(query or "", llm_intent)

    vector_builder_fn = get_hook("_build_rewrite_vector", build_rewrite_vector)
    rewrite_vec = vector_builder_fn(
        getattr(rewrite_result, "rewritten_text", None),
        getattr(llm_intent, "ann_description", None),
        params.ann_weight_override,
        params.rewrite_weight_override,
        query_filter_result.reference_titles,
    )

    return QueryUnderstanding(
        query=query,
        llm_intent=llm_intent,
        intent_filters=intent,
        structured_search_filters=structured_search_filters,
        es_text_query=es_text_query,
        rewrite_vec=rewrite_vec,
        prefer_top_rated=prefer_top_rated,
        custom_genres=custom_genres,
        has_people_filters=has_people,
        candidate_limit=candidate_limit,
        backend_override_normalized=backend_override_normalized,
        preferred_services=preferred_services,
        prefilter_kwargs=prefilter_kwargs,
    )
