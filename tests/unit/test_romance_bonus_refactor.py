"""
Unit tests for Task 4: Romance bonus refactor to use intent signals.

Tests validate that the romantic tone bonus function now uses LLM-extracted
mood signals instead of keyword matching, making it robust to paraphrasing.
"""

import pytest
from unittest.mock import Mock

from api.core.query_profile import IntentSignals
from api.routes.recommend import (
    _constraint_bonus_for_romance_tone,
    ConstraintBonusContext,
)


class TestRomanceBonusRefactor:
    """Tests for the refactored romance bonus function using intent signals."""

    def _make_context(self, **kwargs) -> ConstraintBonusContext:
        """Helper to create ConstraintBonusContext with defaults."""
        defaults = {
            "haystack": "",
            "genre_names": set(),
            "original_language": "en",
            "media_type": "movie",
            "runtime": None,
            "release_year": 2020,
            "maturity_rating": "PG-13",
            "vote_count": 100.0,
        }
        defaults.update(kwargs)
        return ConstraintBonusContext(**defaults)

    def _make_signals(self, **overrides) -> Mock:
        """Helper to create a signals object."""
        signals = Mock()
        signals.date_night = False
        signals.feel_good_comedy = False
        signals.rom_com = False
        signals.adult_feelgood_movieish = False

        for key, value in overrides.items():
            setattr(signals, key, value)

        return signals

    def _make_intent_signals(self, **overrides) -> IntentSignals:
        """Helper to create IntentSignals."""
        defaults = {
            "mood": None,
            "intent_type": None,
            "semantic_facets": (),
            "prestige_indicator": False,
        }
        defaults.update(overrides)
        return IntentSignals(**defaults)

    def test_romantic_mood_with_date_night(self):
        """Date night + romantic mood should apply romance bonuses."""
        intent_signals = self._make_intent_signals(mood="romantic")
        signals = self._make_signals(date_night=True)
        ctx = self._make_context(
            genre_names={"romance", "comedy"}, haystack="love fall in love"
        )

        bonus = _constraint_bonus_for_romance_tone(intent_signals, signals, ctx)

        # Should apply bonuses for date_night + romantic mood
        assert (
            bonus > 0.0
        ), f"Expected positive bonus for romantic date night, got {bonus}"

    def test_uplifting_mood_with_date_night(self):
        """Date night + uplifting mood should also apply romance bonuses."""
        intent_signals = self._make_intent_signals(mood="uplifting")
        signals = self._make_signals(date_night=True)
        ctx = self._make_context(genre_names={"comedy"}, haystack="feel-good")

        bonus = _constraint_bonus_for_romance_tone(intent_signals, signals, ctx)

        # Should apply bonuses for uplifting mood (paraphrase of romantic)
        assert (
            bonus > 0.0
        ), f"Expected positive bonus for uplifting date night, got {bonus}"

    def test_heartwarming_mood_with_date_night(self):
        """Date night + heartwarming mood should also apply romance bonuses."""
        intent_signals = self._make_intent_signals(mood="heartwarming")
        signals = self._make_signals(date_night=True)
        ctx = self._make_context(
            genre_names={"drama", "comedy"}, haystack="warm family"
        )

        bonus = _constraint_bonus_for_romance_tone(intent_signals, signals, ctx)

        # Should apply bonuses for heartwarming mood (another paraphrase)
        assert (
            bonus > 0.0
        ), f"Expected positive bonus for heartwarming date night, got {bonus}"

    def test_dark_mood_no_romance_bonus(self):
        """Dark mood should NOT apply romance bonuses even with date_night."""
        intent_signals = self._make_intent_signals(mood="dark")
        signals = self._make_signals(date_night=True)
        ctx = self._make_context(genre_names={"romance", "thriller"})

        bonus = _constraint_bonus_for_romance_tone(intent_signals, signals, ctx)

        # Dark mood should not trigger romance bonuses
        assert bonus == 0.0, f"Expected no bonus for dark mood, got {bonus}"

    def test_feel_good_mood_standalone(self):
        """Uplifting mood alone should trigger feel-good bonuses."""
        intent_signals = self._make_intent_signals(mood="uplifting")
        signals = self._make_signals(feel_good_comedy=False)
        ctx = self._make_context(genre_names={"comedy"}, haystack="uplifting happy")

        bonus = _constraint_bonus_for_romance_tone(intent_signals, signals, ctx)

        # Feel-good mood should trigger bonuses even without date_night
        assert bonus > 0.0, f"Expected positive bonus for feel-good mood, got {bonus}"

    def test_romantic_facet_rom_com(self):
        """Romantic semantic facet should trigger rom-com bonuses."""
        intent_signals = self._make_intent_signals(
            mood=None, semantic_facets=("romantic", "love", "couple")
        )
        signals = self._make_signals(rom_com=False)
        ctx = self._make_context(
            genre_names={"romance", "comedy"}, haystack="rom com romantic comedy"
        )

        bonus = _constraint_bonus_for_romance_tone(intent_signals, signals, ctx)

        # Romantic facets should trigger rom-com bonuses
        assert bonus > 0.0, f"Expected positive bonus for romantic facets, got {bonus}"

    def test_paraphrase_robustness_comparison(self):
        """
        Validate paraphrase robustness:
        "feel-good" vs "uplifting" should give similar bonuses.
        """
        signals = self._make_signals(date_night=True)
        ctx = self._make_context(
            genre_names={"comedy", "romance"}, haystack="feel-good uplifting"
        )

        # Test with "romantic" mood
        bonus_romantic = _constraint_bonus_for_romance_tone(
            self._make_intent_signals(mood="romantic"), signals, ctx
        )

        # Test with "uplifting" mood (paraphrase)
        bonus_uplifting = _constraint_bonus_for_romance_tone(
            self._make_intent_signals(mood="uplifting"), signals, ctx
        )

        # Both should give similar bonuses (at least both positive)
        assert (
            bonus_romantic > 0.0 and bonus_uplifting > 0.0
        ), f"Both moods should trigger bonuses: romantic={bonus_romantic}, uplifting={bonus_uplifting}"

    def test_no_bonus_without_mood_or_facets(self):
        """No mood or romantic facets should give no bonus."""
        intent_signals = self._make_intent_signals(mood="action", semantic_facets=())
        signals = self._make_signals()
        ctx = self._make_context(genre_names={"action", "thriller"})

        bonus = _constraint_bonus_for_romance_tone(intent_signals, signals, ctx)

        # No bonuses for action mood
        assert bonus == 0.0, f"Expected no bonus for action mood, got {bonus}"

    def test_keyword_debuff_still_applies(self):
        """
        Keyword debuffs (horror, action penalties) should still apply.
        The refactor maintains the penalty logic.
        """
        intent_signals = self._make_intent_signals(mood="romantic")
        signals = self._make_signals(date_night=True)
        ctx = self._make_context(genre_names={"romance", "horror"}, haystack="scary")

        bonus = _constraint_bonus_for_romance_tone(intent_signals, signals, ctx)

        # Horror genre should apply penalty to romance
        # The penalty (-0.28) should be applied in the logic
        # So final bonus should be less than full romance bonus
        assert bonus < 0.5, f"Expected penalty for horror with romance, got {bonus}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
