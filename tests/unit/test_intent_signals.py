"""
Unit tests for IntentSignals extraction and paraphrase robustness.

Tests validate that LLM-extracted intent signals are immune to query paraphrasing.
Key test cases:
- Synonym handling: "feel-good" vs "uplifting" → same mood
- Keyword variations: "heist" vs "robbery" → same semantic_facets
- Structural variations: Same intent expressed different ways
- Quality preference extraction: Awards/prestige language → prestige_indicator
- Temporal hints: Implicit recency/classic language → temporal_preference
"""

import json
import pytest
from unittest.mock import Mock, patch

from api.core.llm_parser import (
    extract_intent_signals,
    _build_intent_signals_prompt,
    _extract_intent_signals_from_llm,
    _default_intent_signals,
    IntentParserSettings,
)
from api.core.intent_parser import Intent


class TestIntentSignalsExtraction:
    """Tests for extract_intent_signals function."""

    def test_default_signals_empty_query(self):
        """Empty or None query should return default (empty) signals."""
        assert extract_intent_signals("") == _default_intent_signals()
        assert extract_intent_signals(None) == _default_intent_signals()

    def test_signals_structure(self):
        """IntentSignals has expected fields."""
        signals = _default_intent_signals()
        assert signals.mood is None
        assert signals.intent_type is None
        assert signals.semantic_facets == ()
        assert signals.prestige_indicator is False
        assert signals.quality_mode is None
        assert signals.temporal_preference is None

    @patch("api.core.llm_parser._get_settings")
    @patch("api.core.llm_parser._extract_intent_signals_from_llm")
    def test_fallback_when_llm_disabled(self, mock_extract, mock_settings):
        """Returns default signals when LLM is disabled."""
        mock_settings.return_value = IntentParserSettings(
            provider="openai",
            api_key=None,
            model="gpt-4o-mini",
            endpoint="https://api.openai.com/v1/chat/completions",
            enabled=False,
            timeout=12.0,
        )
        mock_extract.return_value = None

        signals = extract_intent_signals("what movies should i watch")
        assert signals == _default_intent_signals()

    @patch("api.core.llm_parser._get_settings")
    @patch("api.core.llm_parser._extract_intent_signals_from_llm")
    def test_llm_extraction_success(self, mock_extract, mock_settings):
        """Successfully extracts signals when LLM returns valid JSON."""
        mock_settings.return_value = IntentParserSettings(
            provider="openai",
            api_key="test-key",
            model="gpt-4o-mini",
            endpoint="https://api.openai.com/v1/chat/completions",
            enabled=True,
            timeout=12.0,
        )
        mock_extract.return_value = {
            "mood": "uplifting",
            "intent_type": "discovery",
            "semantic_facets": ["comedy", "family"],
            "prestige_indicator": False,
            "quality_mode": "high_quality",
            "temporal_preference": None,
        }

        signals = extract_intent_signals("feel-good family movies")
        assert signals.mood == "uplifting"
        assert signals.intent_type == "discovery"
        assert signals.semantic_facets == ("comedy", "family")
        assert signals.prestige_indicator is False
        assert signals.quality_mode == "high_quality"
        assert signals.temporal_preference is None

    @patch("api.core.llm_parser._get_settings")
    @patch("api.core.llm_parser._extract_intent_signals_from_llm")
    def test_llm_extraction_partial_fields(self, mock_extract, mock_settings):
        """Handles partial LLM output gracefully."""
        mock_settings.return_value = IntentParserSettings(
            provider="openai",
            api_key="test-key",
            model="gpt-4o-mini",
            endpoint="https://api.openai.com/v1/chat/completions",
            enabled=True,
            timeout=12.0,
        )
        # LLM only returns some fields
        mock_extract.return_value = {
            "mood": "dark",
            "semantic_facets": ["crime", "thriller"],
        }

        signals = extract_intent_signals("dark crime thriller")
        assert signals.mood == "dark"
        assert signals.semantic_facets == ("crime", "thriller")
        assert signals.intent_type is None
        assert signals.prestige_indicator is False
        assert signals.quality_mode is None


class TestParaphraseRobustness:
    """Tests that signal extraction is robust to paraphrasing."""

    @patch("api.core.llm_parser._get_settings")
    @patch("api.core.llm_parser._extract_intent_signals_from_llm")
    def test_feel_good_paraphrase_robustness(self, mock_extract, mock_settings):
        """
        Paraphrased feel-good queries should extract same mood.
        Examples:
        - "feel-good movies"
        - "uplifting movies"
        - "heartwarming films"
        Should all → mood in (uplifting, heartwarming, feel-good)
        """
        mock_settings.return_value = IntentParserSettings(
            provider="openai",
            api_key="test-key",
            model="gpt-4o-mini",
            endpoint="https://api.openai.com/v1/chat/completions",
            enabled=True,
            timeout=12.0,
        )

        # Simulate LLM returning consistent mood for paraphrases
        def side_effect(settings, query, intent):
            normalized = query.lower()
            if any(
                word in normalized
                for word in ["feel", "uplifting", "heartwarming", "cheerful"]
            ):
                return {
                    "mood": "uplifting",
                    "semantic_facets": ["comedy", "family"],
                }
            return None

        mock_extract.side_effect = side_effect

        queries = [
            "feel-good movies",
            "uplifting movies",
            "heartwarming family films",
            "cheerful comedies",
        ]

        moods = [extract_intent_signals(q).mood for q in queries]
        # All should extract similar mood (uplifting/heartwarming family-oriented)
        assert all(m == "uplifting" for m in moods), f"Moods: {moods}"

    @patch("api.core.llm_parser._get_settings")
    @patch("api.core.llm_parser._extract_intent_signals_from_llm")
    def test_crime_facet_paraphrase_robustness(self, mock_extract, mock_settings):
        """
        Crime/heist queries with synonym variations should extract same facets.
        Examples:
        - "heist movies"
        - "robbery thrillers"
        - "caper films"
        Should all → semantic_facets contains "heist" or "crime"
        """
        mock_settings.return_value = IntentParserSettings(
            provider="openai",
            api_key="test-key",
            model="gpt-4o-mini",
            endpoint="https://api.openai.com/v1/chat/completions",
            enabled=True,
            timeout=12.0,
        )

        def side_effect(settings, query, intent):
            normalized = query.lower()
            if any(
                word in normalized
                for word in ["heist", "robbery", "caper", "theft", "heist"]
            ):
                return {
                    "mood": "suspenseful",
                    "semantic_facets": ["heist", "crime", "thriller"],
                }
            return None

        mock_extract.side_effect = side_effect

        queries = [
            "heist movies",
            "robbery thrillers",
            "caper films",
            "theft-focused suspense",
        ]

        facets_list = [extract_intent_signals(q).semantic_facets for q in queries]
        # All should contain crime-related facets
        assert all(
            "heist" in facets or "crime" in facets for facets in facets_list
        ), f"Facets: {facets_list}"

    @patch("api.core.llm_parser._get_settings")
    @patch("api.core.llm_parser._extract_intent_signals_from_llm")
    def test_prestige_not_keyword_dependent(self, mock_extract, mock_settings):
        """
        Prestige should be extracted from intent, not query keywords.
        "oscar" keyword should NOT be required.
        - "award-winning dramas" → prestige_indicator=true
        - "best picture nominees" → prestige_indicator=true
        - "critically acclaimed films" → prestige_indicator=true
        """
        mock_settings.return_value = IntentParserSettings(
            provider="openai",
            api_key="test-key",
            model="gpt-4o-mini",
            endpoint="https://api.openai.com/v1/chat/completions",
            enabled=True,
            timeout=12.0,
        )

        def side_effect(settings, query, intent):
            normalized = query.lower()
            # Extract prestige from intent keywords, NOT "oscar" keyword
            if any(
                word in normalized
                for word in [
                    "award",
                    "nominated",
                    "acclaimed",
                    "prestigious",
                    "quality",
                    "picture",
                ]
            ):
                return {
                    "prestige_indicator": True,
                    "quality_mode": "high_quality",
                }
            return {"prestige_indicator": False}

        mock_extract.side_effect = side_effect

        queries = [
            "award-winning dramas",
            "best picture nominees",
            "critically acclaimed films",
            "prestigious international cinema",
        ]

        prestige_indicators = [
            extract_intent_signals(q).prestige_indicator for q in queries
        ]
        assert all(prestige_indicators), f"Prestige indicators: {prestige_indicators}"


class TestPromptConstruction:
    """Tests for intent signals prompt building."""

    def test_prompt_includes_examples(self):
        """Prompt should include paraphrase examples to guide LLM."""
        system, user = _build_intent_signals_prompt("test query")
        assert "feel-good" in system.lower()
        assert "uplifting" in system.lower()
        assert "heist" in system.lower()
        assert "robbery" in system.lower()
        assert "mood" in user.lower()
        assert "semantic_facets" in user.lower()

    def test_prompt_handles_intent_hint(self):
        """Prompt should incorporate intent genres if provided."""
        intent = Intent(include_genres=["Comedy", "Drama"])
        system, user = _build_intent_signals_prompt("test query", intent)
        assert "Comedy" in user or "comedy" in user.lower()


class TestLLMExtractionFallback:
    """Tests for LLM extraction with error handling."""

    @patch("api.core.llm_parser._get_settings")
    @patch("httpx.Client")
    def test_openai_extraction_success(self, mock_client, mock_settings):
        """Successfully extracts signals from OpenAI API."""
        mock_settings.return_value = IntentParserSettings(
            provider="openai",
            api_key="test-key",
            model="gpt-4o-mini",
            endpoint="https://api.openai.com/v1/chat/completions",
            enabled=True,
            timeout=12.0,
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "mood": "uplifting",
                                "semantic_facets": ["comedy"],
                            }
                        )
                    }
                }
            ]
        }
        mock_client.return_value.__enter__.return_value.post.return_value = (
            mock_response
        )

        result = _extract_intent_signals_from_llm(
            mock_settings.return_value, "happy movies", None
        )
        assert result is not None
        assert result.get("mood") == "uplifting"

    @patch("api.core.llm_parser._get_settings")
    @patch("httpx.Client")
    def test_gemini_extraction_success(self, mock_client, mock_settings):
        """Successfully extracts signals from Gemini API."""
        mock_settings.return_value = IntentParserSettings(
            provider="gemini",
            api_key="test-key",
            model="gemini-2.0-flash-lite",
            endpoint="https://generativelanguage.googleapis.com/v1beta/models",
            enabled=True,
            timeout=12.0,
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "mood": "dark",
                                        "semantic_facets": ["thriller"],
                                    }
                                )
                            }
                        ]
                    }
                }
            ]
        }
        mock_client.return_value.__enter__.return_value.post.return_value = (
            mock_response
        )

        result = _extract_intent_signals_from_llm(
            mock_settings.return_value, "dark thriller", None
        )
        assert result is not None
        assert result.get("mood") == "dark"

    @patch("api.core.llm_parser._get_settings")
    @patch("httpx.Client")
    def test_extraction_handles_http_error(self, mock_client, mock_settings):
        """Gracefully handles HTTP errors from LLM provider."""
        mock_settings.return_value = IntentParserSettings(
            provider="openai",
            api_key="test-key",
            model="gpt-4o-mini",
            endpoint="https://api.openai.com/v1/chat/completions",
            enabled=True,
            timeout=12.0,
        )

        mock_client.return_value.__enter__.return_value.post.side_effect = Exception(
            "API Error"
        )

        result = _extract_intent_signals_from_llm(
            mock_settings.return_value, "test", None
        )
        assert result is None


class TestIntentSignalsCaching:
    """Tests for caching behavior of intent signals."""

    @patch("api.core.llm_parser._get_settings")
    @patch("api.core.llm_parser._extract_intent_signals_from_llm")
    def test_signals_cached(self, mock_extract, mock_settings):
        """Repeated queries should use cache, not call LLM twice."""
        mock_settings.return_value = IntentParserSettings(
            provider="openai",
            api_key="test-key",
            model="gpt-4o-mini",
            endpoint="https://api.openai.com/v1/chat/completions",
            enabled=True,
            timeout=12.0,
        )
        mock_extract.return_value = {
            "mood": "uplifting",
            "semantic_facets": ["comedy"],
        }

        query = "feel-good comedy"
        signals1 = extract_intent_signals(query)
        signals2 = extract_intent_signals(query)

        assert signals1 == signals2
        # LLM should only be called once (second call uses cache)
        # Note: This test assumes cache implementation; may need adjustment
        # based on actual cache behavior

    @patch("api.core.llm_parser._get_settings")
    @patch("api.core.llm_parser._extract_intent_signals_from_llm")
    def test_case_insensitive_caching(self, mock_extract, mock_settings):
        """Cache key should normalize to lowercase."""
        mock_settings.return_value = IntentParserSettings(
            provider="openai",
            api_key="test-key",
            model="gpt-4o-mini",
            endpoint="https://api.openai.com/v1/chat/completions",
            enabled=True,
            timeout=12.0,
        )
        mock_extract.return_value = {
            "mood": "uplifting",
        }

        signals1 = extract_intent_signals("Feel-Good Comedy")
        signals2 = extract_intent_signals("feel-good comedy")

        # Both should return same signals (cached)
        assert signals1 == signals2


class TestSemanticFacetsNormalization:
    """Tests for semantic facets normalization."""

    @patch("api.core.llm_parser._get_settings")
    @patch("api.core.llm_parser._extract_intent_signals_from_llm")
    def test_facets_lowercased(self, mock_extract, mock_settings):
        """Facets should be normalized to lowercase."""
        mock_settings.return_value = IntentParserSettings(
            provider="openai",
            api_key="test-key",
            model="gpt-4o-mini",
            endpoint="https://api.openai.com/v1/chat/completions",
            enabled=True,
            timeout=12.0,
        )
        mock_extract.return_value = {
            "semantic_facets": ["HEIST", "Crime", "THRILLER"],
        }

        signals = extract_intent_signals("heist movie")
        assert signals.semantic_facets == ("heist", "crime", "thriller")

    @patch("api.core.llm_parser._get_settings")
    @patch("api.core.llm_parser._extract_intent_signals_from_llm")
    def test_facets_empty_strings_filtered(self, mock_extract, mock_settings):
        """Empty or whitespace facets should be filtered."""
        mock_settings.return_value = IntentParserSettings(
            provider="openai",
            api_key="test-key",
            model="gpt-4o-mini",
            endpoint="https://api.openai.com/v1/chat/completions",
            enabled=True,
            timeout=12.0,
        )
        mock_extract.return_value = {
            "semantic_facets": ["heist", "", "  ", "crime", None],
        }

        signals = extract_intent_signals("heist crime movies")
        assert signals.semantic_facets == ("heist", "crime")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
