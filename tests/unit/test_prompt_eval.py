from __future__ import annotations
import pytest
from api.core.prompt_eval import (
    load_prompt_template,
    evaluate_reranker_output,
    evaluate_intent_parser_output,
)


def test_load_prompt_template():
    template = load_prompt_template("reranker")
    assert "system_prompt" in template
    assert "examples" in template
    assert isinstance(template["examples"], list)
    assert len(template["examples"]) > 0


def test_evaluate_reranker_output_perfect_match():
    output = {
        "items": [
            {
                "id": 2,
                "score": 0.95,
                "explanation": "Perfect match: A thoughtful sci-fi under 100min with an intimate story.",
            },
            {
                "id": 1,
                "score": 0.85,
                "explanation": "Fits your request: Cerebral sci-fi just under 2 hours.",
            },
        ]
    }
    expected = {
        "items": [
            {
                "id": 2,
                "score": 0.95,
                "explanation": "Different explanation but same ranking",
            },
            {
                "id": 1,
                "score": 0.85,
                "explanation": "Different explanation but same ranking",
            },
        ]
    }
    intent = {
        "media_types": ["movie"],
        "genres": ["Science Fiction"],
        "max_runtime": 120,
    }
    query = "light sci-fi under 2 hours"

    score, errors = evaluate_reranker_output(output, expected, intent, query)

    # Scores:
    # 1. Perfect order match = 1.0
    # 2. Good explanation length = 1.0
    # 3. Critical term relevance: {"hours", "movie", "sci-fi", "2", "under"}
    #    First exp: "sci-fi", "under" = 0.4
    #    Second exp: "hours", "sci-fi", "2", "under" = 0.8
    #    Average relevance = (0.4 + 0.8) / 2 = 0.6
    # Final score = (1.0 + 1.0 + 0.6) / 3 ≈ 0.84
    assert 0.8 <= score <= 0.85
    assert not errors


def test_evaluate_reranker_output_wrong_order():
    output = {
        "items": [
            {"id": 1, "score": 0.9, "explanation": "Good match"},
            {"id": 2, "score": 0.8, "explanation": "Also good"},
        ]
    }
    expected = {
        "items": [
            {"id": 2, "score": 0.9, "explanation": "Should be first"},
            {"id": 1, "score": 0.8, "explanation": "Should be second"},
        ]
    }
    intent = {"media_types": ["movie"]}

    score, errors = evaluate_reranker_output(output, expected, intent, None)
    assert score < 0.5
    assert any("ranking" in err.lower() for err in errors)


def test_evaluate_reranker_output_long_and_irrelevant_explanations():
    long_explanation = " ".join(["word"] * 60)
    output = {"items": [{"id": 5, "explanation": long_explanation, "score": 0.2}]}
    expected = {"items": [{"id": 5}]}
    intent = {"media_types": ["movie"], "genres": ["Drama"]}
    query = "under 90 minutes"

    score, errors = evaluate_reranker_output(output, expected, intent, query)
    assert score < 0.5
    assert any("too long" in err.lower() for err in errors)
    assert any("low relevance" in err.lower() for err in errors)


def test_evaluate_intent_parser_output_perfect_match():
    output = {
        "media_types": ["movie"],
        "genres": ["Science Fiction"],
        "max_runtime": 120,
    }
    expected = {
        "media_types": ["movie"],
        "genres": ["Science Fiction"],
        "max_runtime": 120,
    }
    query = "sci-fi movies under 2 hours"

    score, errors = evaluate_intent_parser_output(output, expected, query)
    assert score == 1.0
    assert not errors


def test_evaluate_intent_parser_output_partial_match():
    output = {
        "media_types": ["movie"],
        "genres": ["Science Fiction", "Action"],  # Extra genre
        "max_runtime": 120,
    }
    expected = {
        "media_types": ["movie"],
        "genres": ["Science Fiction"],
        "max_runtime": 120,
    }
    query = "sci-fi movies under 2 hours"

    score, errors = evaluate_intent_parser_output(output, expected, query)

    # Scores:
    # - Media type match = 1.0
    # - Genre similarity = 1/2 (Jaccard) = 0.5
    # - Runtime match = 1.0
    # Average = (1.0 + 0.5 + 1.0) / 3 ≈ 0.833
    assert 0.8 <= score <= 0.9
    assert len(errors) > 0  # Should note the genre mismatch
    assert any("genre" in err.lower() for err in errors)


def test_load_template_missing_file():
    with pytest.raises(FileNotFoundError):
        load_prompt_template("nonexistent")


def test_evaluate_intent_parser_output_numeric_mismatch():
    output = {
        "media_types": ["movie"],
        "genres": ["Drama"],
        "max_runtime": 150,
        "min_runtime": 80,
    }
    expected = {
        "media_types": ["movie"],
        "genres": ["Drama"],
        "max_runtime": 120,
        "min_runtime": 120,
    }

    score, errors = evaluate_intent_parser_output(
        output, expected, "dramas under 2 hours"
    )
    assert score < 1.0
    assert any("max_runtime" in err or "min_runtime" in err for err in errors)
