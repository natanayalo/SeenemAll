from __future__ import annotations

import logging
from typing import Any, Dict, List

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
) -> List[Dict[str, Any]]:
    ratio = get_hook("_SERENDIPITY_RATIO", _SERENDIPITY_RATIO)
    if not current or limit <= 0 or ratio <= 0.0:
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

    short_tail_candidates = [
        idx for idx, item in enumerate(top_section) if not is_lt_fn(item, limit)
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
        if is_lt_fn(item, limit):
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


def apply_diversity_policies(
    ordered: List[Dict[str, Any]],
    serendipity_context: List[Dict[str, Any]],
    limit: int,
    diversify: bool,
    boost_ids: List[int],
) -> List[Dict[str, Any]]:
    if diversify:
        cap_fn = get_hook("_apply_franchise_cap", apply_franchise_cap)
        ordered = cap_fn(ordered)

    if diversify:
        mmr_fn = get_hook("diversify_with_mmr", diversify_with_mmr)
        ordered = mmr_fn(ordered, limit=limit)

    serendipity_fn = get_hook("_apply_serendipity_slot", apply_serendipity_slot)
    ordered = serendipity_fn(ordered, serendipity_context, limit)

    if boost_ids:
        boost_fn = get_hook("_prioritize_boosted_items", prioritize_boosted_items)
        ordered = boost_fn(ordered, boost_ids)

    return ordered


# Aliases for backwards compatibility with tests
_apply_franchise_cap = apply_franchise_cap
_is_long_tail = is_long_tail
_serendipity_target = serendipity_target
_apply_serendipity_slot = apply_serendipity_slot
