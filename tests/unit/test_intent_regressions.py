import pytest
from api.core.intent_parser import Intent
from api.core.legacy_intent_parser import IntentFilters
from api.routes.recommend import (
    _normalize_merged_intent,
    _apply_explicit_query_overrides,
)

def test_normalize_merged_intent_respects_llm_animation():
    # Scenario: Query is "anime movies", LLM correctly identifies Animation genre.
    # Legacy 'light' might also be present if the query was "light anime".
    intent = IntentFilters(
        raw_query="anime movies",
        genres=["Animation", "Action"],
        moods=["light"],
        media_types=["movie"]
    )
    llm_intent = Intent(include_genres=["Animation", "Action"], moods=["light"])
    
    normalized = _normalize_merged_intent(intent, "anime movies", llm_intent=llm_intent)
    
    # "Animation" should be preserved because it was in llm_intent.include_genres
    assert "Animation" in normalized.genres
    # "light" tag is always stripped as it's a mood, and Comedy/Family are added
    assert "light" not in normalized.moods
    assert "Comedy" in normalized.genres
    assert "Family" in normalized.genres

def test_normalize_merged_intent_strips_legacy_animation():
    # Scenario: Query is "feel-good movie", legacy parser might add Animation via 'light'.
    # LLM doesn't want Animation.
    intent = IntentFilters(
        raw_query="feel-good movie",
        genres=["Animation", "Comedy"],
        moods=["light"],
        media_types=["movie"]
    )
    llm_intent = Intent(include_genres=["Comedy"], moods=["light"])
    
    normalized = _normalize_merged_intent(intent, "feel-good movie", llm_intent=llm_intent)
    
    # "Animation" should be stripped because it was NOT in llm_intent.include_genres 
    # and "feel-good movie" doesn't match _ANIMATION_AUDIENCE_PATTERN
    assert "Animation" not in normalized.genres
    assert "Comedy" in normalized.genres

def test_apply_explicit_query_overrides_respects_llm_maturity():
    # Scenario: Query is "space adventure", LLM identifies it's for kids and sets PG.
    # Without this fix, the heuristic would strip the PG cap for being "too restrictive".
    intent = IntentFilters(
        raw_query="space adventure",
        media_types=["movie"],
        maturity_rating_max="PG"
    )
    llm_intent = Intent(maturity_rating_max="PG")
    
    overridden = _apply_explicit_query_overrides(intent, "space adventure", llm_intent=llm_intent)
    
    # PG should be preserved because llm_intent specifically requested it
    assert overridden.maturity_rating_max == "PG"

def test_apply_explicit_query_overrides_strips_unintended_restrictive_cap():
    # Scenario: Query is "psychological thriller", LLM hallucinations or legacy quirk 
    # might set a G/PG cap even if not requested.
    intent = IntentFilters(
        raw_query="psychological thriller",
        media_types=["movie"],
        maturity_rating_max="G"
    )
    llm_intent = Intent() # No maturity cap here
    
    overridden = _apply_explicit_query_overrides(intent, "psychological thriller", llm_intent=llm_intent)
    
    # G cap should be stripped because it wasn't explicit and no audience signal exists
    assert overridden.maturity_rating_max is None

def test_apply_explicit_query_overrides_family_signal():
    # Scenario: Query is "family movies", maturity should be at least PG-13 or more restrictive
    intent = IntentFilters(
        raw_query="family movies",
        media_types=["movie"],
        maturity_rating_max=None
    )
    
    overridden = _apply_explicit_query_overrides(intent, "family movies")
    
    assert overridden.maturity_rating_max == "PG-13"
