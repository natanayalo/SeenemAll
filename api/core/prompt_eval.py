import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


def load_prompt_template(name: str) -> Dict[str, Any]:
    """Load a prompt template and its examples from JSON file."""
    template_path = Path(__file__).parent.parent / "prompts" / f"{name}.json"
    with open(template_path) as f:
        return json.load(f)


def evaluate_reranker_output(
    output: Dict[str, Any],
    expected: Dict[str, Any],
    intent: Dict[str, Any],
    query: str | None,
) -> Tuple[float, List[str]]:
    """
    Evaluate reranker output against expected results.
    Returns (score, list of error messages).
    """
    errors: List[str] = []
    metrics: List[float] = []

    # Check ranking order
    output_ids = [item["id"] for item in output.get("items", [])]
    expected_ids = [item["id"] for item in expected.get("items", [])]

    if output_ids != expected_ids:
        errors.append(f"Ranking mismatch. Expected: {expected_ids}, Got: {output_ids}")
        metrics.append(0.0)
    else:
        metrics.append(1.0)

    # Check explanations
    intent_terms: Set[str] = set()
    if query:
        intent_terms.update(query.lower().split())
    if intent.get("genres"):
        intent_terms.update(g.lower() for g in intent["genres"])
    if intent.get("media_types"):
        intent_terms.update(mt.lower() for mt in intent["media_types"])

    for item in output.get("items", []):
        exp = item.get("explanation", "").lower()
        if len(exp.split()) > 45:
            errors.append(
                f"Explanation too long for item {item['id']}: {len(exp.split())} words"
            )
            metrics.append(0.0)
        else:
            metrics.append(1.0)

        term_matches = sum(1 for term in intent_terms if term in exp)
        relevance = term_matches / len(intent_terms) if intent_terms else 1.0
        metrics.append(relevance)
        if relevance < 0.5:
            errors.append(f"Low relevance explanation for item {item['id']}: {exp}")

    return sum(metrics) / len(metrics) if metrics else 0.0, errors


def evaluate_intent_parser_output(
    output: Dict[str, Any],
    expected: Dict[str, Any],
    query: str,
) -> Tuple[float, List[str]]:
    """
    Evaluate intent parser output against expected results.
    Returns (score, list of error messages).
    """
    errors: List[str] = []
    metrics: List[float] = []

    # Check media types
    if output.get("media_types") != expected.get("media_types"):
        errors.append(
            f"Media type mismatch. Expected: {expected.get('media_types')}, "
            f"Got: {output.get('media_types')}"
        )
        metrics.append(0.0)
    else:
        metrics.append(1.0)

    # Check genres
    expected_genres = set(expected.get("genres", []))
    output_genres = set(output.get("genres", []))
    if output_genres != expected_genres:
        errors.append(
            f"Genre mismatch. Expected: {expected_genres}, Got: {output_genres}"
        )
        metrics.append(
            len(output_genres & expected_genres) / len(expected_genres)
            if expected_genres
            else 0.0
        )
    else:
        metrics.append(1.0)

    # Check runtime constraints
    for key in ["min_runtime", "max_runtime"]:
        if output.get(key) != expected.get(key):
            errors.append(
                f"Runtime constraint mismatch for {key}. "
                f"Expected: {expected.get(key)}, Got: {output.get(key)}"
            )
            metrics.append(0.0)
        else:
            metrics.append(1.0)

    # Check year constraints
    for key in ["min_year", "max_year"]:
        if output.get(key) != expected.get(key):
            errors.append(
                f"Year constraint mismatch for {key}. "
                f"Expected: {expected.get(key)}, Got: {output.get(key)}"
            )
            metrics.append(0.0)
        else:
            metrics.append(1.0)

    return sum(metrics) / len(metrics) if metrics else 0.0, errors
