from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
)

from elasticsearch.exceptions import TransportError

from api import config
from api.core.elasticsearch_client import get_elasticsearch_client


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchFilters:
    include_item_ids: Sequence[str] = ()
    genres: Sequence[str] = ()
    media_types: Sequence[str] = ()
    providers: Sequence[str] = ()
    maturity: Sequence[str] = ()
    languages: Sequence[str] = ()
    keywords: Sequence[str] = ()
    cast: Sequence[str] = ()
    directors: Sequence[str] = ()
    producers: Sequence[str] = ()
    writers: Sequence[str] = ()
    release_year_gte: Optional[int] = None
    release_year_lte: Optional[int] = None
    runtime_gte: Optional[int] = None
    runtime_lte: Optional[int] = None
    exclude_item_ids: Sequence[str] = ()


class ElasticsearchSearchError(RuntimeError):
    """Raised when Elasticsearch search requests fail."""


def _terms_filter(field: str, values: Iterable[str]) -> Optional[Dict[str, Any]]:
    terms = [v for v in values if v]
    if not terms:
        return None
    return {"terms": {field: terms}}


def _range_filter(
    field: str,
    *,
    gte: Optional[int] = None,
    lte: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    payload: Dict[str, Any] = {}
    if gte is not None:
        payload["gte"] = int(gte)
    if lte is not None:
        payload["lte"] = int(lte)
    if not payload:
        return None
    return {"range": {field: payload}}


def _build_filter_clauses(
    filters: Optional[SearchFilters],
) -> Dict[str, List[Mapping[str, Any]]]:
    bool_filters: Dict[str, List[Mapping[str, Any]]] = {}
    if not filters:
        return bool_filters

    clauses: List[Mapping[str, Any]] = []
    must_not: List[Mapping[str, Any]] = []
    keyword_should: List[Mapping[str, Any]] = []

    for field, values in (
        ("item_id", filters.include_item_ids),
        ("genres", filters.genres),
        ("media_type", filters.media_types),
        ("streaming_providers", filters.providers),
        ("maturity", filters.maturity),
        ("spoken_languages", filters.languages),
        ("cast", filters.cast),
        ("directors", filters.directors),
        ("producers", filters.producers),
        ("writers", filters.writers),
    ):
        term_clause = _terms_filter(field, values)
        if term_clause:
            clauses.append(term_clause)

    keyword_clause = _terms_filter("keywords", filters.keywords)
    if keyword_clause:
        keyword_should.append(keyword_clause)

    range_clause = _range_filter(
        "release_year",
        gte=filters.release_year_gte,
        lte=filters.release_year_lte,
    )
    if range_clause:
        clauses.append(range_clause)

    runtime_clause = _range_filter(
        "runtime",
        gte=filters.runtime_gte,
        lte=filters.runtime_lte,
    )
    if runtime_clause:
        clauses.append(runtime_clause)

    exclude_clause = _terms_filter("item_id", filters.exclude_item_ids)
    if exclude_clause:
        must_not.append(exclude_clause)

    if clauses:
        bool_filters["filter"] = clauses
    if must_not:
        bool_filters["must_not"] = must_not
    if keyword_should:
        bool_filters["keyword_should"] = keyword_should
    return bool_filters


def _extract_item_id(hit: Mapping[str, Any]) -> Optional[str]:
    source = hit.get("_source") or {}
    item_id = source.get("item_id")
    if item_id:
        return str(item_id)
    raw_id = hit.get("_id")
    return str(raw_id) if raw_id is not None else None


def _fuse_hits_rrf(
    hit_lists: Sequence[Sequence[Mapping[str, Any]]],
    *,
    max_size: int,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Combine multiple ranked hit lists using Reciprocal Rank Fusion.
    """
    fused: Dict[str, Dict[str, Any]] = {}
    for hits in hit_lists:
        for rank, hit in enumerate(hits, start=1):
            item_id = _extract_item_id(hit)
            if not item_id:
                continue
            entry = fused.setdefault(
                item_id,
                {
                    "_id": hit.get("_id"),
                    "_source": hit.get("_source") or {},
                    "score": 0.0,
                },
            )
            entry["score"] += 1.0 / (rrf_k + rank)
            # Preserve a representative _source for downstream consumers.
            if not entry["_source"] and hit.get("_source"):
                entry["_source"] = hit["_source"]

    sorted_hits = sorted(fused.values(), key=lambda data: data["score"], reverse=True)
    limited_hits: List[Dict[str, Any]] = []
    for data in sorted_hits:
        limited_hits.append(
            {
                "_id": data.get("_id"),
                "_source": data.get("_source", {}),
                "_score": data["score"],
            }
        )
        if len(limited_hits) >= max_size:
            break
    return limited_hits


def knn_search(
    query_vector: Sequence[float],
    *,
    k: Optional[int] = None,
    num_candidates: Optional[int] = None,
    text_query: Optional[str] = None,
    filters: Optional[SearchFilters] = None,
    source_includes: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Execute a kNN search against the items index and return hits.
    """
    client = get_elasticsearch_client()
    effective_k = k or config.ELASTICSEARCH_KNN_K
    effective_candidates = num_candidates or config.ELASTICSEARCH_KNN_NUM_CANDIDATES
    search_size = max(effective_k, effective_candidates)

    bool_filters = _build_filter_clauses(filters)

    knn_query: MutableMapping[str, Any] = {
        "field": "embedding",
        "query_vector": list(query_vector),
        "k": effective_k,
        "num_candidates": max(effective_candidates, effective_k),
    }
    if bool_filters:
        knn_filter_bool: Dict[str, List[Mapping[str, Any]]] = {}
        if "filter" in bool_filters:
            knn_filter_bool["filter"] = list(bool_filters["filter"])
        if "must_not" in bool_filters:
            knn_filter_bool["must_not"] = list(bool_filters["must_not"])
        if knn_filter_bool:
            knn_query["filter"] = {"bool": knn_filter_bool}

    knn_body: MutableMapping[str, Any] = {
        "size": effective_k,
        "knn": knn_query,
    }

    search_kwargs: Dict[str, Any] = {
        "index": config.ELASTICSEARCH_ITEMS_INDEX,
        "body": knn_body,
    }
    if source_includes is not None:
        search_kwargs["_source"] = {"includes": list(source_includes)}

    try:
        response = client.search(**search_kwargs)
    except TransportError as exc:
        raise ElasticsearchSearchError(f"Elasticsearch search failed: {exc}") from exc

    hits = response.get("hits", {}).get("hits", [])
    text_hits: List[Mapping[str, Any]] = []
    if logger.isEnabledFor(logging.DEBUG):
        sample_ids = [
            hit.get("_source", {}).get("item_id") or hit.get("_id") for hit in hits[:5]
        ]
        logger.debug(
            "ES kNN search completed | filters=%s hit_count=%d sample_ids=%s",
            bool_filters,
            len(hits),
            sample_ids,
        )

    if text_query:
        text_bool: Dict[str, List[Mapping[str, Any]]] = {
            key: list(value) for key, value in bool_filters.items()
        }
        keyword_should = text_bool.pop("keyword_should", [])
        if keyword_should:
            text_bool.setdefault("should", []).extend(keyword_should)
        text_bool.setdefault("must", []).append(
            {
                "multi_match": {
                    "query": text_query,
                    "fields": [
                        "title^3",
                        "overview",
                        "keywords^2",
                        "cast",
                        "directors",
                        "producers",
                        "writers",
                    ],
                }
            }
        )
        text_body: Dict[str, Any] = {
            "size": search_size,
            "query": {"bool": text_bool},
        }
        text_kwargs: Dict[str, Any] = {
            "index": config.ELASTICSEARCH_ITEMS_INDEX,
            "body": text_body,
        }
        if source_includes is not None:
            text_kwargs["_source"] = {"includes": list(source_includes)}
        try:
            text_response = client.search(**text_kwargs)
            text_hits = text_response.get("hits", {}).get("hits", [])
        except TransportError as exc:
            logger.warning(
                "Elasticsearch text query failed; continuing with kNN results: %s",
                exc,
            )
        else:
            if logger.isEnabledFor(logging.DEBUG):
                sample_text_ids = [
                    hit.get("_source", {}).get("item_id") or hit.get("_id")
                    for hit in text_hits[:5]
                ]
                logger.debug(
                    "ES text search completed | query=%s hit_count=%d sample_ids=%s",
                    text_query,
                    len(text_hits),
                    sample_text_ids,
                )

    if text_hits:
        hits = _fuse_hits_rrf([hits, text_hits], max_size=effective_k)

    results: List[Dict[str, Any]] = []
    for hit in hits:
        source = hit.get("_source") or {}
        results.append(
            {
                "item_id": source.get("item_id") or hit.get("_id"),
                "score": hit.get("_score"),
                "source": source,
            }
        )
    return results
