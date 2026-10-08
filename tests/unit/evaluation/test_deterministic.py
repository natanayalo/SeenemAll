"""Unit tests for authoritative deterministic checks, overrides, and canonical ordering."""

from evaluation.deterministic import (
    apply_deterministic_override,
    check_canonical_order,
    check_deterministic_constraints,
    generate_factual_control_cases,
)
from evaluation.models import (
    DeterministicConstraint,
)


def test_deterministic_media_type_and_year():
    item = {
        "media_type": "movie",
        "release_year": 1995,
        "runtime": 120,
        "genres": ["Action", "Sci-Fi"],
        "tmdb_id": 100,
    }
    # Passing constraints
    c_pass = DeterministicConstraint(media_type="movie", min_year=1990, max_year=2000)
    is_valid, violations = check_deterministic_constraints(item, c_pass)
    assert is_valid is True
    assert len(violations) == 0

    # Failing media type
    c_fail_type = DeterministicConstraint(media_type="tv")
    is_valid, violations = check_deterministic_constraints(item, c_fail_type)
    assert is_valid is False
    assert any("media_type_mismatch" in v for v in violations)

    # Failing year
    c_fail_year = DeterministicConstraint(min_year=2000)
    is_valid, violations = check_deterministic_constraints(item, c_fail_year)
    assert is_valid is False
    assert any("release_year_too_early" in v for v in violations)


def test_deterministic_runtime_and_language():
    item = {
        "media_type": "movie",
        "runtime": 85,
        "original_language": "fr",
        "genres": ["Comedy"],
        "tmdb_id": 200,
    }
    c_pass = DeterministicConstraint(max_runtime=90, language="fr")
    is_valid, violations = check_deterministic_constraints(item, c_pass)
    assert is_valid is True

    c_fail_runtime = DeterministicConstraint(max_runtime=80)
    is_valid, violations = check_deterministic_constraints(item, c_fail_runtime)
    assert is_valid is False
    assert any("runtime_too_long" in v for v in violations)

    c_fail_lang = DeterministicConstraint(language="en")
    is_valid, violations = check_deterministic_constraints(item, c_fail_lang)
    assert is_valid is False
    assert any("language_mismatch" in v for v in violations)


def test_deterministic_genres_strict_and_or():
    item = {
        "media_type": "movie",
        "genres": ["Action", "Adventure"],
        "tmdb_id": 300,
    }
    # OR semantics (default)
    c_or = DeterministicConstraint(
        genres=["Adventure", "Horror"], require_all_genres=False
    )
    is_valid, violations = check_deterministic_constraints(item, c_or)
    assert is_valid is True

    # AND semantics (strict)
    c_and = DeterministicConstraint(
        genres=["Adventure", "Horror"], require_all_genres=True
    )
    is_valid, violations = check_deterministic_constraints(item, c_and)
    assert is_valid is False
    assert any("missing_required_genres" in v for v in violations)


def test_deterministic_seen_and_disliked_exclusions():
    item = {"tmdb_id": 400, "media_type": "movie"}
    c_seen = DeterministicConstraint(seen_ids=[400, 401])
    is_valid, violations = check_deterministic_constraints(item, c_seen)
    assert is_valid is False
    assert any("seen_item_exclusion_violated" in v for v in violations)

    c_disliked = DeterministicConstraint(disliked_ids=[400])
    is_valid, violations = check_deterministic_constraints(item, c_disliked)
    assert is_valid is False
    assert any("disliked_item_exclusion_violated" in v for v in violations)


def test_deterministic_il_provider_snapshot():
    # Star Wars: Episode IV (movie:11) is available on disney_plus and confirmed_absent on netflix
    item_star_wars = {"tmdb_id": 11, "media_type": "movie"}

    # Available on Disney Plus
    c_disney = DeterministicConstraint(providers=["disney_plus"])
    is_valid, _ = check_deterministic_constraints(item_star_wars, c_disney)
    assert is_valid is True

    # Confirmed absent on Netflix
    c_netflix = DeterministicConstraint(providers=["netflix"])
    is_valid, violations = check_deterministic_constraints(item_star_wars, c_netflix)
    assert is_valid is False
    assert any("provider_confirmed_absent" in v for v in violations)


def test_deterministic_override():
    item = {"tmdb_id": 500, "media_type": "movie", "runtime": 150}
    c_fail = DeterministicConstraint(max_runtime=100)

    # High subjective grade (3) overridden to 0 due to hard constraint violation
    final_grade, override, violations = apply_deterministic_override(
        grade=3, item=item, constraints=c_fail
    )
    assert final_grade == 0
    assert override is True
    assert len(violations) > 0

    # Passing constraint keeps original grade
    c_pass = DeterministicConstraint(max_runtime=200)
    final_grade, override, violations = apply_deterministic_override(
        grade=3, item=item, constraints=c_pass
    )
    assert final_grade == 3
    assert override is False
    assert len(violations) == 0


def test_canonical_order_evaluation():
    canonical = ["movie:1", "movie:2", "movie:3", "movie:4"]

    # Perfect exact prefix match
    res_perfect = check_canonical_order(
        ["movie:1", "movie:2", "movie:3", "movie:4"], canonical, k=4
    )
    assert res_perfect["evaluable"] is True
    assert res_perfect["exact_prefix_match"] is True
    assert res_perfect["prefix_recall"] == 1.0
    assert res_perfect["pairwise_accuracy"] == 1.0
    assert res_perfect["kendalls_tau"] == 1.0

    # Inverted order
    res_inv = check_canonical_order(
        ["movie:4", "movie:3", "movie:2", "movie:1"], canonical, k=4
    )
    assert res_inv["exact_prefix_match"] is False
    assert res_inv["prefix_recall"] == 1.0
    assert res_inv["pairwise_accuracy"] == 0.0
    assert res_inv["kendalls_tau"] == -1.0

    # Incomplete prefix
    res_partial = check_canonical_order(
        ["movie:1", "movie:99", "movie:2"], canonical, k=3
    )
    assert res_partial["exact_prefix_match"] is False
    assert res_partial["prefix_recall"] == 2 / 3

    # Undefined Kendall's tau for fewer than 2 items
    res_single = check_canonical_order(["movie:1"], canonical, k=3)
    assert res_single["kendalls_tau"] is None


def test_factual_control_generation():
    controls = generate_factual_control_cases()
    assert len(controls) == 100
    for c in controls:
        assert c.is_factual_control is True
        assert c.expected_grade is not None


def test_missing_canonical_title_fails_even_without_inversion():
    from evaluation.deterministic import check_canonical_order
    from evaluation.metrics import evaluate_comparison_gates
    from evaluation.models import EvaluationStatus

    chronology = check_canonical_order(
        ["movie:1", "movie:3"], ["movie:1", "movie:2", "movie:3"], k=3
    )
    assert chronology["inversions"] == 0
    assert chronology["missing_prefix_items"] == ["movie:2"]
    result = evaluate_comparison_gates(
        family_ndcg_deltas={str(i): 0.02 for i in range(50)},
        slice_family_deltas={
            tag: [0.02] * 12 for tag in ["vibe", "franchise", "constraint", "entity"]
        },
        baseline_recall_100=0.8,
        candidate_recall_100=0.81,
        hard_constraint_violations=0,
        disliked_violations=0,
        exploratory_coverages=[1.0] * 50,
        authoritative_unresolved_count=0,
        execution_failures=0,
        unexpected_fallbacks=0,
        duplicate_outputs_detected=False,
        is_statistical_promotion=True,
        chronology_violations=chronology["inversions"],
        missing_canonical_items=chronology["missing_prefix_count"],
    )
    assert result.status == EvaluationStatus.FAIL
    assert any("missing items" in reason for reason in result.reasons)
