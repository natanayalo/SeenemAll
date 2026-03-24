from api.routes.recommend import (
    _normalize_merged_intent,
    _apply_explicit_query_overrides,
)
from api.core.legacy_intent_parser import IntentFilters
from api.core.intent_parser import Intent


def test_light_drama_preserves_mood_and_no_wrong_genres():
    # Setup intent with 'light' mood
    intent = IntentFilters(
        raw_query="light drama movies",
        genres=["Drama"],
        moods=["light"],
        media_types=["movie"],
    )
    query = "light drama movies"
    # LLM also says light
    llm_intent = Intent(moods=["light"], include_genres=["Drama"])

    normalized = _normalize_merged_intent(intent, query, llm_intent=llm_intent)

    assert "light" not in normalized.moods
    # Should NOT have Comedy or Family because it's not a kid/audience signal query
    assert "Comedy" not in normalized.genres
    assert "Family" not in normalized.genres
    assert "Drama" in normalized.genres


def test_light_family_injects_genres():
    intent = IntentFilters(
        raw_query="light family adventure",
        genres=["Adventure"],
        moods=["light"],
        media_types=["movie"],
    )
    query = "light family adventure"
    llm_intent = Intent(moods=["light"], include_genres=["Adventure"])

    normalized = _normalize_merged_intent(intent, query, llm_intent=llm_intent)

    assert "light" not in normalized.moods
    # SHOULD have Comedy and Family because 'family' is an audience signal
    assert "Comedy" in normalized.genres
    assert "Family" in normalized.genres


def test_animation_provenance_trusts_llm():
    # Legacy parser might inject Animation for 'light', but if LLM explicitly wants it, keep it.
    intent = IntentFilters(
        raw_query="animated comedy",
        genres=["Animation", "Comedy"],
        media_types=["movie"],
    )
    query = "animated comedy"
    llm_intent = Intent(include_genres=["Animation", "Comedy"])

    normalized = _normalize_merged_intent(intent, query, llm_intent=llm_intent)
    assert "Animation" in normalized.genres


def test_animation_stripped_if_not_in_llm_or_query():
    # Legacy parser injected Animation but user didn't ask for it
    intent = IntentFilters(
        raw_query="spooky thriller",
        genres=["Animation", "Horror"],
        media_types=["movie"],
    )
    query = "spooky thriller"
    llm_intent = Intent(include_genres=["Horror"])

    normalized = _normalize_merged_intent(intent, query, llm_intent=llm_intent)
    assert "Animation" not in normalized.genres
    assert "Horror" in normalized.genres


def test_maturity_explicit_llm_is_respected():
    intent = IntentFilters(
        raw_query="safe movies for 10 year olds",
        maturity_rating_max="PG",
        media_types=["movie"],
    )
    query = "safe movies for 10 year olds"
    # LLM explicitly chose PG
    llm_intent = Intent(maturity_rating_max="PG")

    normalized = _apply_explicit_query_overrides(intent, query, llm_intent=llm_intent)
    # Should NOT be reset to None because it was explicit in LLM
    assert normalized.maturity_rating_max == "PG"


def test_maturity_reset_if_not_explicit():
    intent = IntentFilters(
        raw_query="action movies", maturity_rating_max="G", media_types=["movie"]
    )
    query = "action movies"  # Nothing about kids
    llm_intent = Intent()  # No explicit cap

    normalized = _apply_explicit_query_overrides(intent, query, llm_intent=llm_intent)
    # G is too restrictive and not explicitly asked for
    assert normalized.maturity_rating_max is None
