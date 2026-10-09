from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

from api.core.reranker import diversify_with_mmr
from api.pipeline.hooks import get_hook
from api.pipeline.intent import float_from_env
from api.pipeline.scorer import prioritize_boosted_items

logger = logging.getLogger(__name__)

_SERENDIPITY_RATIO_RAW = float_from_env("SERENDIPITY_RATIO", 0.15)
if _SERENDIPITY_RATIO_RAW <= 0.0:
    _SERENDIPITY_RATIO = 0.0
else:
    _SERENDIPITY_RATIO = max(0.1, min(0.2, _SERENDIPITY_RATIO_RAW))


def apply_franchise_cap(
    candidates: List[Dict[str, Any]],
    cap: int = 2,
    exempt_collection_ids: Set[int] | None = None,
) -> List[Dict[str, Any]]:
    if not candidates or cap <= 0:
        return candidates

    exempt = exempt_collection_ids or set()
    franchise_counts: Dict[int, int] = {}
    filtered_candidates: List[Dict[str, Any]] = []

    for item in candidates:
        collection_id = item.get("collection_id")
        if collection_id is None:
            filtered_candidates.append(item)
            continue

        if collection_id in exempt:
            filtered_candidates.append(item)
            continue

        count = franchise_counts.get(collection_id, 0)
        if count < cap:
            filtered_candidates.append(item)
            franchise_counts[collection_id] = count + 1

    return filtered_candidates


def is_long_tail(item: Dict[str, Any], limit: int) -> bool:
    if limit <= 0:
        return False
    original_rank = int(item.get("original_rank", 0) or 0)
    return original_rank >= limit


def serendipity_target(limit: int) -> int:
    """
    Determine how many serendipity slots to allocate.

    We skip serendipity when the limit is very small (<=2) because swapping
    long-tail items into a list that short tends to degrade perceived quality.
    """
    ratio = get_hook("_SERENDIPITY_RATIO", _SERENDIPITY_RATIO)
    if ratio <= 0.0 or limit <= 2:
        return 0
    target = round(limit * ratio)
    target = max(1, target)
    return min(limit, target)


def apply_serendipity_slot(
    current: List[Dict[str, Any]],
    candidate_pool: List[Dict[str, Any]],
    limit: int,
    cap: int = 2,
    exempt_collection_ids: Set[int] | None = None,
    enforce_franchise_cap: bool = True,
    protected_ids: Set[int] | None = None,
    is_chronological: bool = False,
) -> List[Dict[str, Any]]:
    ratio = get_hook("_SERENDIPITY_RATIO", _SERENDIPITY_RATIO)
    if not current or not candidate_pool or limit <= 0 or ratio <= 0.0:
        return current

    top_count = min(limit, len(current))
    target_fn = get_hook("_serendipity_target", serendipity_target)
    target = target_fn(limit)
    if target == 0:
        return current

    is_lt_fn = get_hook("_is_long_tail", is_long_tail)
    top_section = list(current[:top_count])
    existing_long_tail = [item for item in top_section if is_lt_fn(item, limit)]
    if len(existing_long_tail) >= target:
        return current

    protected = set(protected_ids or ())
    short_tail_candidates = [
        idx
        for idx, item in enumerate(top_section)
        if not is_lt_fn(item, limit) and item.get("id") not in protected
    ]
    if not short_tail_candidates:
        return current

    exempt = exempt_collection_ids or set()
    franchise_counts: Dict[int, int] = {}
    if enforce_franchise_cap and cap > 0:
        for item in top_section:
            cid = item.get("collection_id")
            if cid is not None and cid not in exempt:
                franchise_counts[cid] = franchise_counts.get(cid, 0) + 1

    def _preserves_chronology(
        section: List[Dict[str, Any]], idx: int, cand: Dict[str, Any]
    ) -> bool:
        cand_year = cand.get("release_year")
        if cand_year is None:
            return False
        if idx > 0:
            prev_year = section[idx - 1].get("release_year")
            if prev_year is not None and cand_year < prev_year:
                return False
        if idx < len(section) - 1:
            next_year = section[idx + 1].get("release_year")
            if next_year is not None and cand_year > next_year:
                return False
        return True

    top_ids = {item.get("id") for item in top_section if item.get("id") is not None}

    replacement_pool: List[Dict[str, Any]] = []
    seen_pool: set[int] = set()
    for item in candidate_pool:
        ident = item.get("id")
        if ident is None or ident in top_ids or ident in seen_pool:
            continue
        if enforce_franchise_cap and cap > 0:
            cid = item.get("collection_id")
            if (
                cid is not None
                and cid not in exempt
                and franchise_counts.get(cid, 0) >= cap
            ):
                continue
        if is_lt_fn(item, limit):
            replacement_pool.append(item)
            seen_pool.add(ident)

    if not replacement_pool:
        return current

    needed = min(target - len(existing_long_tail), len(replacement_pool))
    if needed <= 0:
        return current

    new_top = list(top_section)
    used_replacement_indices: set[int] = set()

    for idx in reversed(short_tail_candidates):
        if needed <= 0:
            break
        old_item = new_top[idx]
        old_cid = old_item.get("collection_id")

        for r_idx, replacement in enumerate(replacement_pool):
            if r_idx in used_replacement_indices:
                continue

            new_cid = replacement.get("collection_id")
            if (
                enforce_franchise_cap
                and cap > 0
                and new_cid is not None
                and new_cid not in exempt
            ):
                cur_count = franchise_counts.get(new_cid, 0)
                net_count = cur_count + (0 if old_cid == new_cid else 1)
                if net_count > cap:
                    continue

            if is_chronological and not _preserves_chronology(
                new_top, idx, replacement
            ):
                continue

            # Accepted replacement
            new_top[idx] = replacement
            used_replacement_indices.add(r_idx)
            if enforce_franchise_cap and cap > 0:
                if old_cid is not None and old_cid not in exempt:
                    franchise_counts[old_cid] = max(
                        0, franchise_counts.get(old_cid, 0) - 1
                    )
                if new_cid is not None and new_cid not in exempt:
                    franchise_counts[new_cid] = franchise_counts.get(new_cid, 0) + 1
            needed -= 1
            break

    deduped: List[Dict[str, Any]] = []
    seen_ids: set[int] = set()
    for item in new_top + list(current):
        ident = item.get("id")
        if ident is None or ident not in seen_ids:
            if ident is not None:
                seen_ids.add(ident)
            deduped.append(item)
    if enforce_franchise_cap and cap > 0:
        return apply_franchise_cap(deduped, cap=cap, exempt_collection_ids=exempt)
    return deduped


def apply_diversity_policies(
    ordered: List[Dict[str, Any]],
    serendipity_context: List[Dict[str, Any]],
    limit: int,
    diversify: bool,
    boost_ids: List[int],
    exempt_collection_ids: Set[int] | None = None,
    serendipity: bool = True,
    presentation_limit: int | None = None,
    mmr: bool = True,
    franchise_cap: bool = True,
) -> List[Dict[str, Any]]:
    pres_limit = presentation_limit if presentation_limit is not None else limit
    if diversify and franchise_cap:
        cap_fn = get_hook("_apply_franchise_cap", apply_franchise_cap)
        try:
            ordered = cap_fn(ordered, exempt_collection_ids=exempt_collection_ids)
        except TypeError:
            ordered = cap_fn(ordered)

    boost_set = set(boost_ids or ())
    if boost_set:
        boost_lookup = {it["id"]: it for it in ordered if it.get("id") in boost_set}
        ordered_boosted = [
            boost_lookup[bid] for bid in (boost_ids or ()) if bid in boost_lookup
        ]
        remaining = [it for it in ordered if it.get("id") not in boost_set]
    else:
        ordered_boosted = []
        remaining = ordered

    if diversify and mmr:
        mmr_fn = get_hook("diversify_with_mmr", diversify_with_mmr)
        if boost_set:
            slots_needed = max(0, limit - len(ordered_boosted))
            remaining = (
                mmr_fn(remaining, limit=slots_needed) if slots_needed > 0 else []
            )
            ordered = ordered_boosted + remaining
        else:
            ordered = mmr_fn(ordered, limit=limit)

    if serendipity:
        serendipity_fn = get_hook("_apply_serendipity_slot", apply_serendipity_slot)
        try:
            ordered = serendipity_fn(
                ordered,
                serendipity_context,
                pres_limit,
                cap=2,
                exempt_collection_ids=exempt_collection_ids,
                enforce_franchise_cap=bool(diversify and franchise_cap),
            )
        except TypeError:
            ordered = serendipity_fn(ordered, serendipity_context, pres_limit)

    if boost_ids:
        boost_fn = get_hook("_prioritize_boosted_items", prioritize_boosted_items)
        ordered = boost_fn(ordered, boost_ids)

    return ordered


# Aliases for backwards compatibility with tests
_apply_franchise_cap = apply_franchise_cap
_is_long_tail = is_long_tail
_serendipity_target = serendipity_target
_apply_serendipity_slot = apply_serendipity_slot
