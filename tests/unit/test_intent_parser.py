import pytest
from api.core.intent_parser import Intent
from api.core.llm_parser import parse_intent, rewrite_query, DEFAULT_INTENT
from api.core.rewrite import Rewrite


@pytest.mark.asyncio
async def test_parse_intent_fixtures():
    """
    Tests the parse_intent function with a set of predefined fixtures.
    """
    test_cases = [
        (
            "light sci-fi <2h",
            Intent(include_genres=["sci-fi"], runtime_minutes_max=120),
        ),
        (
            "no gore",
            Intent(exclude_genres=["horror"]),
        ),
        (
            "movies from the 90s",
            Intent(year_min=1990, year_max=1999),
        ),
        (
            "something in french",
            Intent(languages=["fr"]),
        ),
        (
            "bad query",
            DEFAULT_INTENT,
        ),
    ]

    for query, expected_intent in test_cases:
        # The user_context is not used in the mock implementation, so we can pass an empty dict.
        intent = await parse_intent(query, {})
        assert intent == expected_intent, f"Query: '{query}' failed"


@pytest.mark.asyncio
async def test_rewrite_query():
    """
    Tests the rewrite_query function with a set of predefined fixtures.
    """
    test_cases = [
        (
            Intent(include_genres=["sci-fi"], runtime_minutes_max=120),
            Rewrite(rewritten_text="sci-fi movies"),
        ),
        (
            Intent(exclude_genres=["horror"]),
            Rewrite(rewritten_text=""),
        ),
    ]

    for intent, expected_rewrite in test_cases:
        rewrite = await rewrite_query("some query", intent)
        assert rewrite == expected_rewrite
