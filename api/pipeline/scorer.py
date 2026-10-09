from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Mapping, Sequence, Set

import numpy as np

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
_MIXER_INTENT_WEIGHT = float_from_env("MIXER_INTENT_WEIGHT", 0.35)
_BAYESIAN_MIN_VOTES = float_from_env("BAYESIAN_MIN_VOTES", 50.0)
_BAYESIAN_PRIOR_MEAN = float_from_env("BAYESIAN_PRIOR_MEAN", 0.65)


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

    role_names_map = {
        "cast": _extract_names(getattr(item, "cast", None)),
        "directors": _extract_names(getattr(item, "directors", None)),
        "producers": _extract_names(getattr(item, "producers", None)),
        "writers": _extract_names(getattr(item, "writers", None)),
    }

    person_to_roles: Dict[str, Set[str]] = {}
    for role, person_list in (
        ("cast", filters.cast or ()),
        ("directors", filters.directors or ()),
        ("producers", filters.producers or ()),
        ("writers", filters.writers or ()),
    ):
        for p in person_list:
            if p and p.strip():
                person_to_roles.setdefault(p.strip().lower(), set()).add(role)

    if not person_to_roles:
        return True

    for person_name, allowed_roles in person_to_roles.items():
        matched = False
        for role in allowed_roles:
            if person_name in role_names_map[role]:
                matched = True
                break
        if not matched:
            return False

    return True


def item_matches_language_filters(item: Item, languages: Sequence[str]) -> bool:
    if not languages:
        return True
    from api.pipeline.retriever.prefilter import LANGUAGE_TO_ISO

    item_lang = (getattr(item, "original_language", None) or "").strip().lower()
    for lang in languages:
        if not lang:
            continue
        lang_clean = lang.strip().lower()
        iso = LANGUAGE_TO_ISO.get(lang_clean, lang_clean)
        if item_lang == iso or item_lang == lang_clean:
            return True
    return False


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


def compute_bayesian_vote_score(
    vote_average: float | None,
    vote_count: int | float | None,
    min_votes: float = _BAYESIAN_MIN_VOTES,
    prior_mean: float = _BAYESIAN_PRIOR_MEAN,
) -> float:
    """Calculate damped Bayesian average rating in [0.0, 1.0]."""
    if vote_average is None or vote_count is None:
        return float(prior_mean)
    try:
        v = float(vote_count)
        if v <= 0.0:
            return float(prior_mean)
        r = float(vote_average) / 10.0
        r = max(0.0, min(1.0, r))
        damped = (v / (v + min_votes)) * r + (min_votes / (v + min_votes)) * prior_mean
        return float(max(0.0, min(1.0, damped)))
    except (TypeError, ValueError):
        return float(prior_mean)


def compute_popularity_percentiles(items: Sequence[Dict[str, Any]]) -> List[float]:
    """Calculate fractional percentile ranking in [0.0, 1.0] for candidate popularities.

    Mitigates severe power-law / Pareto blockbuster skew across candidate distributions.
    """
    n = len(items)
    if n == 0:
        return []
    if n == 1:
        return [0.5]

    raw_pops = [float(item.get("popularity") or 0.0) for item in items]
    min_pop, max_pop = min(raw_pops), max(raw_pops)
    if min_pop == max_pop:
        return [0.5] * n

    sorted_indices = sorted(range(n), key=lambda idx: raw_pops[idx])
    percentiles = [0.0] * n
    i = 0
    while i < n:
        j = i
        val = raw_pops[sorted_indices[i]]
        while j < n and raw_pops[sorted_indices[j]] == val:
            j += 1
        avg_rank = (i + (j - 1)) / 2.0
        pct = avg_rank / float(n - 1)
        for k in range(i, j):
            percentiles[sorted_indices[k]] = float(max(0.0, min(1.0, pct)))
        i = j
    return percentiles


def _extract_cluster_vector(cluster: Any) -> np.ndarray | None:
    raw = (
        cluster.centroid
        if hasattr(cluster, "centroid")
        else (cluster.get("centroid") if isinstance(cluster, Mapping) else None)
    )
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float32)
    norm_val = float(np.linalg.norm(arr))
    return arr / norm_val if norm_val > 0.0 else arr


def compute_semantic_affinity(
    item_vec: Any,
    *,
    query_vec: np.ndarray | None = None,
    taste_clusters: Sequence[Any] | None = None,
    user_vec: np.ndarray | None = None,
    ann_rank: float = 0.0,
) -> float:
    """Calculate continuous semantic affinity in [0.0, 1.0] fusing vector cosine similarity and retrieval rank."""
    s_rank = 1.0 / (1.0 + float(ann_rank))

    if item_vec is None:
        return float(s_rank)

    try:
        v = np.asarray(item_vec, dtype=np.float32)
        v_norm = float(np.linalg.norm(v))
        if v_norm <= 0.0 or not np.isfinite(v_norm):
            return float(s_rank)
        v_unit = v / v_norm
    except Exception:
        return float(s_rank)

    sim_q: float | None = None
    if query_vec is not None:
        try:
            q = np.asarray(query_vec, dtype=np.float32)
            q_norm = float(np.linalg.norm(q))
            if q_norm > 0.0 and np.isfinite(q_norm):
                cos_q = float(np.dot(v_unit, q / q_norm))
                # Map continuous MiniLM cosine [-0.1, 0.8] to [0.0, 1.0]
                sim_q = max(0.0, min(1.0, (cos_q + 0.1) / 0.9))
        except Exception:
            sim_q = None

    sim_cluster: float | None = None
    if taste_clusters:
        cluster_sims: List[float] = []
        for cl in taste_clusters:
            c_vec = _extract_cluster_vector(cl)
            if c_vec is not None and len(c_vec) == len(v_unit):
                cos_c = float(np.dot(v_unit, c_vec))
                cluster_sims.append(max(0.0, min(1.0, (cos_c + 0.1) / 0.9)))
        if cluster_sims:
            sim_cluster = max(cluster_sims)

    sim_user: float | None = None
    if user_vec is not None and sim_cluster is None:
        try:
            u = np.asarray(user_vec, dtype=np.float32)
            u_norm = float(np.linalg.norm(u))
            if u_norm > 0.0 and np.isfinite(u_norm):
                cos_u = float(np.dot(v_unit, u / u_norm))
                sim_user = max(0.0, min(1.0, (cos_u + 0.1) / 0.9))
        except Exception:
            sim_user = None

    if sim_q is not None:
        if sim_cluster is not None:
            vec_sim = 0.65 * sim_q + 0.35 * sim_cluster
        else:
            vec_sim = sim_q
        f_ann = 0.7 * vec_sim + 0.3 * s_rank
    elif sim_cluster is not None:
        f_ann = 0.7 * sim_cluster + 0.3 * s_rank
    elif sim_user is not None:
        f_ann = 0.7 * sim_user + 0.3 * s_rank
    else:
        f_ann = s_rank

    return float(max(0.0, min(1.0, f_ann)))


def compute_intent_overlap(
    item: Dict[str, Any],
    intent_filters: Any = None,
    top_keywords: Set[str] | None = None,
) -> float:
    """Calculate structured intent overlap in [0.0, 1.0] against target genres and keywords."""
    if intent_filters is None and not top_keywords:
        return 1.0

    target_genres = getattr(intent_filters, "genres", None) or ()
    has_target_genres = bool(target_genres)

    genre_score = 1.0
    if has_target_genres:
        target_set = {str(g).strip().lower() for g in target_genres if g}
        raw_genres = item.get("genres") or []
        cand_genres: Set[str] = set()
        if isinstance(raw_genres, list):
            for g in raw_genres:
                if isinstance(g, Mapping):
                    name = g.get("name")
                    if isinstance(name, str):
                        cand_genres.add(name.strip().lower())
                elif isinstance(g, str):
                    cand_genres.add(g.strip().lower())
        elif isinstance(raw_genres, str):
            cand_genres.add(raw_genres.strip().lower())

        if target_set:
            matched = len(cand_genres & target_set)
            genre_score = matched / float(len(target_set))

    target_kws = getattr(intent_filters, "keywords", None) or set()
    all_kws: Set[str] = set()
    for kw in target_kws:
        if kw:
            all_kws.add(str(kw).strip().lower())
    for kw in top_keywords or ():
        if kw:
            all_kws.add(str(kw).strip().lower())

    kw_score = 1.0
    has_target_kws = bool(all_kws)
    if has_target_kws:
        cand_kws: Set[str] = set()
        raw_kws = item.get("keywords") or []
        if isinstance(raw_kws, list):
            for k in raw_kws:
                if isinstance(k, Mapping):
                    name = k.get("name")
                    if isinstance(name, str):
                        cand_kws.add(name.strip().lower())
                elif isinstance(k, str):
                    cand_kws.add(k.strip().lower())
        overview = str(item.get("overview") or "").lower()
        title = str(item.get("title") or "").lower()

        hits = 0
        for kw in all_kws:
            if kw in cand_kws or kw in overview or kw in title:
                hits += 1
        kw_score = hits / float(len(all_kws)) if all_kws else 1.0

    if has_target_genres and has_target_kws:
        total = 0.65 * genre_score + 0.35 * kw_score
    elif has_target_genres:
        total = genre_score
    elif has_target_kws:
        total = kw_score
    else:
        total = 1.0

    return float(max(0.0, min(1.0, total)))


def apply_mixer_scores(
    candidates: List[Dict[str, Any]],
    *,
    ann_weight_override: float | None = None,
    collab_weight_override: float | None = None,
    trending_weight_override: float | None = None,
    popularity_weight_override: float | None = None,
    vote_weight_override: float | None = None,
    novelty_weight_override: float | None = None,
    intent_weight_override: float | None = None,
    query_vector: np.ndarray | None = None,
    taste_clusters: Sequence[Any] | None = None,
    user_vector: np.ndarray | None = None,
    intent_filters: Any = None,
    top_keywords: Set[str] | None = None,
    is_query: bool = False,
    **extra_kwargs: Any,
) -> None:
    """Score candidates using normalized candidate features and calibrated logistic scaling predicting P(relevant)."""
    if not candidates:
        return

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
    intent_weight = (
        _MIXER_INTENT_WEIGHT
        if intent_weight_override is None
        else intent_weight_override
    )
    if not is_query and intent_weight_override is None:
        intent_weight = 0.0

    pop_percentiles = compute_popularity_percentiles(candidates)
    total_weight = (
        ann_weight
        + collab_weight
        + trending_weight
        + pop_weight
        + vote_weight
        + novelty_weight
        + intent_weight
    )

    for idx, item in enumerate(candidates):
        ann_rank = float(item.get("ann_rank", item.get("original_rank", idx)))
        source_scores = item.get("source_scores") or {}

        # 1. Semantic affinity
        f_ann = compute_semantic_affinity(
            item.get("vector"),
            query_vec=query_vector,
            taste_clusters=taste_clusters,
            user_vec=user_vector,
            ann_rank=ann_rank,
        )

        # 2. Collaborative overlap
        f_collab = max(0.0, min(1.0, float(source_scores.get("collab") or 0.0)))

        # 3. Trending signal
        trending_src = max(0.0, min(1.0, float(source_scores.get("trending") or 0.0)))
        t_rank = item.get("trending_rank")
        if t_rank is not None and float(t_rank) > 0.0:
            rank_trending = 1.0 / (1.0 + math.log(1.0 + float(t_rank)))
            f_trending = max(trending_src, rank_trending)
        else:
            f_trending = trending_src

        # 4. Percentile popularity
        f_pop = pop_percentiles[idx] if idx < len(pop_percentiles) else 0.5

        # 5. Bayesian vote quality
        f_vote = compute_bayesian_vote_score(
            item.get("vote_average"),
            item.get("vote_count"),
        )

        # 6. Novelty
        f_novelty = max(0.0, min(1.0, 1.0 - f_pop))

        # 7. Intent overlap
        f_intent = compute_intent_overlap(
            item,
            intent_filters=intent_filters,
            top_keywords=top_keywords,
        )

        linear_score = (
            ann_weight * f_ann
            + collab_weight * f_collab
            + trending_weight * f_trending
            + pop_weight * f_pop
            + vote_weight * f_vote
            + novelty_weight * f_novelty
            + intent_weight * f_intent
        )

        norm_score = (linear_score / total_weight) if total_weight > 0.0 else f_ann

        rel_year = item.get("release_year")
        try:
            rel_year_int = int(rel_year) if rel_year is not None else 0
        except (ValueError, TypeError):
            rel_year_int = 0

        v_count = item.get("vote_count")
        try:
            v_count_int = int(v_count) if v_count is not None else 0
        except (ValueError, TypeError):
            v_count_int = 0

        if (
            getattr(intent_filters, "is_vibe", False)
            and rel_year_int >= 2025
            and v_count_int < 500
        ):
            norm_score *= 0.5

        # Calibrated logistic scaling: P(relevant) = σ(β * (S - S0))
        # Maps [0.0, 1.0] smoothly into a calibrated probability in (0, 1)
        z = 6.0 * (norm_score - 0.5)
        p_relevant = 1.0 / (1.0 + math.exp(-z))

        item["retrieval_score"] = float(p_relevant)
        item["score"] = float(p_relevant)
        item["features"] = {
            "semantic_affinity": round(f_ann, 4),
            "collab_overlap": round(f_collab, 4),
            "trending_score": round(f_trending, 4),
            "popularity_percentile": round(f_pop, 4),
            "bayesian_vote": round(f_vote, 4),
            "novelty_score": round(f_novelty, 4),
            "intent_overlap": round(f_intent, 4),
            "weighted_normalized": round(norm_score, 4),
            "p_relevant": round(p_relevant, 4),
        }

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
    max_candidates = min(intent.candidate_limit, max(params.limit * 5, 50))
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
        if pool.structured_search_filters and pool.structured_search_filters.languages:
            if not item_matches_language_filters(
                it, pool.structured_search_filters.languages
            ):
                skipped_intent += 1
                continue

        sources = pool.merged_scores.get(iid, {})

        cleaned_options = []
        if watch_options and watch_options[0] is not None:
            cleaned_options = [
                {
                    "service": opt["service"],
                    "url": opt["url"],
                    "offer_type": opt.get("offer_type"),
                }
                for opt in watch_options
                if opt.get("url") is not None
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
            "directors": getattr(it, "directors", None),
            "cast": getattr(it, "cast", None),
            "keywords": getattr(it, "keywords", None),
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
    collab_weight_override = (
        0.0
        if getattr(params, "mask_preferences", False)
        else params.mixer_collab_weight
    )
    trending_weight_override = params.mixer_trending_weight
    popularity_weight_override = params.mixer_popularity_weight
    vote_weight_override = params.mixer_vote_weight
    novelty_weight_override = params.mixer_novelty_weight
    intent_weight_override = getattr(params, "mixer_intent_weight", None)
    is_vibe = getattr(intent.llm_intent, "is_vibe", False) or getattr(
        intent.intent_filters, "is_vibe", False
    )

    if intent.query:
        if trending_weight_override is None:
            trending_weight_override = 0.0
        if popularity_weight_override is None:
            popularity_weight_override = 0.0
        if vote_weight_override is None:
            vote_weight_override = 0.5 if is_vibe else 0.0
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
        intent_weight_override=intent_weight_override,
        query_vector=intent.query_vec,
        taste_clusters=context.active_taste_clusters,
        user_vector=context.short_v,
        intent_filters=intent.intent_filters,
        top_keywords=context.top_query_keywords,
        is_query=bool(intent.query),
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
