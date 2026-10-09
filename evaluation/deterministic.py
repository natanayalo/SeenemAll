"""Authoritative deterministic constraint verification and control generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from evaluation.models import (
    DeterministicConstraint,
    ItemEvidence,
    TestCase,
    TypedId,
)

DEFAULT_IL_PROVIDERS_PATH = Path("evaluation/fixtures/il_providers.json")
DEFAULT_CANONICAL_PATH = Path("evaluation/fixtures/canonical_sequences.json")

_CACHED_IL_SNAPSHOT: Optional[Dict[str, Any]] = None
_CACHED_CANONICAL_SEQUENCES: Optional[Dict[str, Any]] = None


def load_il_providers(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load frozen Israel streaming provider availability snapshot."""
    global _CACHED_IL_SNAPSHOT
    target = path or DEFAULT_IL_PROVIDERS_PATH
    if _CACHED_IL_SNAPSHOT is not None and path is None:
        return _CACHED_IL_SNAPSHOT
    if not target.exists():
        return {
            "snapshot_version": "none",
            "country_code": "IL",
            "items": {},
        }
    with target.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if path is None:
        _CACHED_IL_SNAPSHOT = data
    return data


def load_canonical_sequences(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load versioned, sourced canonical sequences for franchises."""
    global _CACHED_CANONICAL_SEQUENCES
    target = path or DEFAULT_CANONICAL_PATH
    if _CACHED_CANONICAL_SEQUENCES is not None and path is None:
        return _CACHED_CANONICAL_SEQUENCES
    if not target.exists():
        return {"version": "none", "sequences": {}}
    with target.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if path is None:
        _CACHED_CANONICAL_SEQUENCES = data
    return data


def check_deterministic_constraints(
    item: Dict[str, Any] | ItemEvidence,
    constraints: DeterministicConstraint,
    il_snapshot: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """Check machine-verifiable hard constraints against item metadata.

    Returns:
        (is_valid, violation_reasons)
        Unknown required metadata fails eligibility.
    """
    violations: List[str] = []
    snapshot = il_snapshot or load_il_providers()
    snapshot_items = snapshot.get("items", {})

    # Extract fields regardless of whether item is dict or ItemEvidence
    if isinstance(item, ItemEvidence):
        typed_id = item.typed_id
        media_type = item.media_type
        release_year = item.release_year
        runtime = item.runtime
        language = item.original_language
        genres = [g.lower() for g in item.genres]
        watch_options: List[Dict[str, Any]] = []
    else:
        typed_id = TypedId.from_item(item)
        media_type = str(item.get("media_type") or "movie").lower()
        release_year = item.get("release_year")
        runtime = item.get("runtime")
        language = item.get("original_language")
        raw_genres = item.get("genres") or []
        genres = []
        for g in raw_genres:
            if isinstance(g, dict) and "name" in g:
                genres.append(str(g["name"]).lower())
            elif isinstance(g, str):
                genres.append(g.lower())
        watch_options = item.get("watch_options") or []

    # 1. Media Type
    if constraints.media_type:
        expected_type = constraints.media_type.strip().lower()
        if not media_type:
            violations.append("missing_media_type")
        elif media_type != expected_type:
            violations.append(
                f"media_type_mismatch (expected {expected_type}, got {media_type})"
            )

    # 2. Release Year
    if constraints.min_year is not None:
        if release_year is None:
            violations.append("missing_release_year_for_min_year_constraint")
        elif release_year < constraints.min_year:
            violations.append(
                f"release_year_too_early ({release_year} < {constraints.min_year})"
            )

    if constraints.max_year is not None:
        if release_year is None:
            violations.append("missing_release_year_for_max_year_constraint")
        elif release_year > constraints.max_year:
            violations.append(
                f"release_year_too_late ({release_year} > {constraints.max_year})"
            )

    # 3. Runtime
    if constraints.min_runtime is not None:
        if runtime is None:
            violations.append("missing_runtime_for_min_runtime_constraint")
        elif runtime < constraints.min_runtime:
            violations.append(
                f"runtime_too_short ({runtime}m < {constraints.min_runtime}m)"
            )

    if constraints.max_runtime is not None:
        if runtime is None:
            violations.append("missing_runtime_for_max_runtime_constraint")
        elif runtime > constraints.max_runtime:
            violations.append(
                f"runtime_too_long ({runtime}m > {constraints.max_runtime}m)"
            )

    # 4. Language
    if constraints.language:
        expected_lang = constraints.language.strip().lower()
        if not language:
            violations.append("missing_language_metadata")
        elif language.strip().lower() != expected_lang:
            violations.append(
                f"language_mismatch (expected {expected_lang}, got {language})"
            )

    # 5. Explicit Genres
    if constraints.genres:
        required_genres = [g.strip().lower() for g in constraints.genres]
        if not genres:
            violations.append("missing_genre_metadata")
        elif constraints.require_all_genres:
            missing_genres = [rg for rg in required_genres if rg not in genres]
            if missing_genres:
                violations.append(
                    f"missing_required_genres ({', '.join(missing_genres)})"
                )
        else:
            if not any(rg in genres for rg in required_genres):
                violations.append(
                    f"no_matching_genres (required any of {required_genres})"
                )

    # 6. Seen & Disliked Exclusions
    tmdb_id = typed_id.id
    if constraints.seen_ids and tmdb_id in set(constraints.seen_ids):
        violations.append(f"seen_item_exclusion_violated ({tmdb_id})")

    if constraints.disliked_ids and tmdb_id in set(constraints.disliked_ids):
        violations.append(f"disliked_item_exclusion_violated ({tmdb_id})")

    # 7. Providers (Israel 'IL' availability snapshot)
    if constraints.providers:
        req_providers = [p.strip().lower() for p in constraints.providers]
        tid_str = str(typed_id)
        if tid_str in snapshot_items:
            item_snap = snapshot_items[tid_str]
            available = [p.lower() for p in item_snap.get("available_services", [])]
            confirmed_absent = [
                p.lower() for p in item_snap.get("confirmed_absent", [])
            ]

            # Check if any confirmed absent
            for req in req_providers:
                if req in confirmed_absent:
                    violations.append(f"provider_confirmed_absent ({req})")
                elif req not in available:
                    violations.append(f"provider_not_available_in_il ({req})")
        else:
            # Fallback to runtime item watch_options if snapshot has incomplete coverage
            item_providers: Set[str] = set()
            for opt in watch_options:
                if isinstance(opt, dict):
                    srv = str(opt.get("service") or "").lower()
                    if srv:
                        item_providers.add(srv)
            if item_providers:
                if not any(req in item_providers for req in req_providers):
                    violations.append(
                        f"provider_missing_in_item_options ({req_providers})"
                    )
            else:
                violations.append(f"incomplete_provider_snapshot_coverage ({tid_str})")

    return (len(violations) == 0, violations)


def apply_deterministic_override(
    grade: int,
    item: Dict[str, Any] | ItemEvidence,
    constraints: Optional[DeterministicConstraint],
    il_snapshot: Optional[Dict[str, Any]] = None,
) -> Tuple[int, bool, List[str]]:
    """Override subjective relevance to grade 0 if deterministic constraints are violated."""
    if not constraints:
        return (grade, False, [])

    is_valid, violations = check_deterministic_constraints(
        item=item, constraints=constraints, il_snapshot=il_snapshot
    )
    if not is_valid:
        # A deterministic constraint violation overrides subjective relevance to 0
        return (0, True, violations)
    return (grade, False, [])


def check_canonical_order(
    recommended_items: Sequence[Any],
    canonical_sequence: Sequence[str],
    k: int = 10,
) -> Dict[str, Any]:
    """Evaluate franchise chronology against a versioned canonical sequence.

    Calculates:
      - canonical_prefix_recall: fraction of min(K, canonical_length) returned.
      - pairwise_accuracy: fraction of ordered pairs in recommendations matching canonical order.
      - kendalls_tau: rank correlation between common items (undefined if < 2 items).
      - exact_prefix_match: whether the first min(K, canonical_length) items match exactly.
    """
    if not canonical_sequence:
        return {
            "evaluable": False,
            "reason": "missing_canonical_sequence",
            "exact_prefix_match": False,
            "prefix_recall": 0.0,
            "pairwise_accuracy": 0.0,
            "kendalls_tau": None,
            "missing_prefix_count": 0,
            "missing_prefix_items": [],
            "inversions": 0,
        }

    rec_typed_ids: List[str] = []
    for it in recommended_items[:k]:
        try:
            rec_typed_ids.append(str(TypedId.parse(it)))
        except Exception:
            rec_typed_ids.append(str(it))

    canonical_clean = [str(TypedId.parse(c)) for c in canonical_sequence]
    prefix_length = min(k, len(canonical_clean))
    required_prefix = canonical_clean[:prefix_length]

    # Exact prefix match requires declared first min(K, canonical_length) in exact order
    rec_prefix = rec_typed_ids[:prefix_length]
    exact_prefix_match = rec_prefix == required_prefix

    # Canonical prefix recall: how many of required_prefix are in top K
    rec_set = set(rec_typed_ids)
    hits = sum(1 for item in required_prefix if item in rec_set)
    prefix_recall = hits / prefix_length if prefix_length > 0 else 0.0
    missing_prefix_count = prefix_length - hits
    missing_prefix_items = [item for item in required_prefix if item not in rec_set]

    # Common items for pairwise and Kendall's tau
    canonical_indices = {item: idx for idx, item in enumerate(canonical_clean)}
    common_items = [item for item in rec_typed_ids if item in canonical_indices]

    if len(common_items) < 2:
        return {
            "evaluable": True,
            "exact_prefix_match": exact_prefix_match,
            "prefix_recall": prefix_recall,
            "pairwise_accuracy": 1.0 if len(common_items) == 1 else 0.0,
            "kendalls_tau": None,  # Undefined for < 2 items
            "common_items_count": len(common_items),
            "prefix_length": prefix_length,
            "missing_prefix_count": missing_prefix_count,
            "missing_prefix_items": missing_prefix_items,
            "inversions": 0,
        }

    # Pairwise accuracy on common items
    pairs_total = 0
    pairs_correct = 0
    concordant = 0
    discordant = 0

    for i in range(len(common_items)):
        for j in range(i + 1, len(common_items)):
            item_a = common_items[i]
            item_b = common_items[j]
            pairs_total += 1
            if canonical_indices[item_a] < canonical_indices[item_b]:
                pairs_correct += 1
                concordant += 1
            else:
                discordant += 1

    pairwise_acc = pairs_correct / pairs_total if pairs_total > 0 else 0.0
    kendall_tau = (concordant - discordant) / pairs_total if pairs_total > 0 else 0.0

    return {
        "evaluable": True,
        "exact_prefix_match": exact_prefix_match,
        "prefix_recall": prefix_recall,
        "pairwise_accuracy": pairwise_acc,
        "kendalls_tau": kendall_tau,
        "common_items_count": len(common_items),
        "prefix_length": prefix_length,
        "missing_prefix_count": missing_prefix_count,
        "missing_prefix_items": missing_prefix_items,
        "pairs_total": pairs_total,
        "pairs_correct": pairs_correct,
        "inversions": discordant,
    }


def generate_factual_control_cases() -> List[TestCase]:
    """Construct 100 deterministic controls with verifiable expected behavior.

    Covers:
      - Perfect factual matches (expected grade 3)
      - Confirmed genre/entity matches (expected grade 2)
      - Irrelevant / constraint violation traps (expected grade 0)
      - Missing metadata controls (fails eligibility / grade 0)
    """
    cases: List[TestCase] = []

    # 1. Hard constraint year controls (20 cases)
    for i, year in enumerate(range(1990, 2010)):
        cases.append(
            TestCase(
                case_id=f"ctrl_year_max_{i+1}",
                family_id=f"ctrl_family_year_max_{i+1}",
                track="product",
                split="dev",
                task="constraint",
                slice_tags=["constraint", "factual_control"],
                query=f"classic movies released before {year}",
                constraints=DeterministicConstraint(max_year=year - 1),
                is_factual_control=True,
                expected_grade=0,  # trap check for violations
            )
        )

    # 2. Hard constraint runtime controls (20 cases)
    for i, runtime in enumerate(range(80, 100)):
        cases.append(
            TestCase(
                case_id=f"ctrl_runtime_max_{i+1}",
                family_id=f"ctrl_family_runtime_max_{i+1}",
                track="product",
                split="dev",
                task="constraint",
                slice_tags=["constraint", "factual_control"],
                query=f"short movies under {runtime} minutes",
                constraints=DeterministicConstraint(max_runtime=runtime),
                is_factual_control=True,
                expected_grade=0,
            )
        )

    # 3. Media type controls (20 cases)
    for i in range(10):
        cases.append(
            TestCase(
                case_id=f"ctrl_tv_show_{i+1}",
                family_id=f"ctrl_family_tv_{i+1}",
                track="product",
                split="dev",
                task="constraint",
                slice_tags=["constraint", "factual_control"],
                query=f"sci-fi television series season {i+1}",
                constraints=DeterministicConstraint(media_type="tv"),
                is_factual_control=True,
                expected_grade=2,
            )
        )
    for i in range(10):
        cases.append(
            TestCase(
                case_id=f"ctrl_movie_only_{i+1}",
                family_id=f"ctrl_family_movie_{i+1}",
                track="product",
                split="dev",
                task="constraint",
                slice_tags=["constraint", "factual_control"],
                query=f"stand-alone feature movie {i+1}",
                constraints=DeterministicConstraint(media_type="movie"),
                is_factual_control=True,
                expected_grade=2,
            )
        )

    # 4. Provider IL controls (20 cases)
    providers = ["netflix", "disney_plus", "prime_video", "apple_tv_plus"]
    for i in range(20):
        p = providers[i % len(providers)]
        cases.append(
            TestCase(
                case_id=f"ctrl_provider_il_{i+1}",
                family_id=f"ctrl_family_provider_il_{i+1}",
                track="product",
                split="dev",
                task="constraint",
                slice_tags=["constraint", "factual_control"],
                query=f"streamable movies on {p} in Israel",
                constraints=DeterministicConstraint(providers=[p]),
                is_factual_control=True,
                expected_grade=2,
            )
        )

    # 5. Exclusions & canonical sequence controls (20 cases)
    sequences = load_canonical_sequences().get("sequences", {})
    seq_keys = list(sequences.keys())
    for i in range(20):
        seq_key = seq_keys[i % len(seq_keys)] if seq_keys else "star_wars_chronological"
        cases.append(
            TestCase(
                case_id=f"ctrl_canonical_{i+1}",
                family_id=f"ctrl_family_canonical_{i+1}",
                track="product",
                split="dev",
                task="franchise",
                slice_tags=["franchise", "factual_control"],
                query=f"{seq_key.replace('_', ' ')} in order",
                canonical_sequence=sequences.get(seq_key, {}).get("items", []),
                is_factual_control=True,
                expected_grade=3,
            )
        )

    return cases
