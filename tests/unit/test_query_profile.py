from api.core.query_profile import build_query_profile


def test_build_query_profile_captures_global_dimensions_for_optimistic_scifi_tv() -> (
    None
):
    profile = build_query_profile(
        "hopeful space adventure tv series with a starship crew"
    )

    assert profile.signals.optimistic_scifi_tv is True
    assert "science_fiction" in profile.semantic_domains
    assert "optimistic" in profile.tone
    assert "optimistic_scifi_tv" in profile.semantic_facets


def test_build_query_profile_generalizes_cerebral_temporal_thriller_paraphrase() -> (
    None
):
    profile = build_query_profile(
        "cerebral time-loop thriller with paradoxes and memory games"
    )

    assert profile.signals.temporal_thriller is True
    assert profile.signals.cerebral_temporal_thriller is True
    assert "thriller" in profile.semantic_domains
    assert "cerebral" in profile.tone
    assert profile.hard_constraints.year_min == 2000


def test_build_query_profile_tracks_family_adventure_audience_and_constraints() -> None:
    profile = build_query_profile("kids profile bilingual fantasy adventure shows")

    assert profile.signals.kids_profile is True
    assert profile.signals.multilingual_family_adventure is True
    assert "kids_profile" in profile.audience
    assert "family_adventure" in profile.semantic_domains
    assert profile.hard_constraints.maturity_rating_max == "PG-13"


def test_build_query_profile_captures_crime_signal() -> None:
    profile = build_query_profile("money heist crime drama")
    assert profile.signals.crime_query_signal is True
