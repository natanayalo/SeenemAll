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
    # Extract critical terms (genres, media types, runtime hints)
    intent_terms: Set[str] = set()
    if intent.get("genres"):
        # Normalize sci-fi/science fiction to one term
        genres = set(g.lower() for g in intent["genres"])
        if "science fiction" in genres:
            intent_terms.add("sci-fi")
        else:
            intent_terms.update(genres)
    if intent.get("media_types"):
        intent_terms.update(mt.lower() for mt in intent["media_types"])
    if query:
        # Only add numeric/temporal terms from query
        critical_terms = {
            "under",
            "over",
            "min",
            "max",
            "hours",
            "minutes",
            "year",
            "decade",
        }
        for term in query.lower().split():
            if term.isdigit() or term in critical_terms or term == "sci-fi":
                intent_terms.add(term)

    for item in output.get("items", []):
        exp = item.get("explanation", "").lower()
        if len(exp.split()) > 45:
            errors.append(
                f"Explanation too long for item {item['id']}: {len(exp.split())} words"
            )
            metrics.append(0.0)
        else:
            metrics.append(1.0)

        # Debug term matches
        matching_terms = [term for term in intent_terms if term in exp]
        term_matches = len(matching_terms)
        relevance = term_matches / len(intent_terms) if intent_terms else 1.0
        metrics.append(relevance)

        logger.info(f"Intent terms: {intent_terms}")
        logger.info(f"Matching terms: {matching_terms}")
        logger.info(f"Relevance: {relevance}")

        if relevance < 0.3:
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

    # Check genres with partial credit for overlapping genres
    expected_genres = set(expected.get("genres", []))
    output_genres = set(output.get("genres", []))

    if expected_genres or output_genres:
        # Calculate Jaccard similarity for genres
        intersection = len(expected_genres & output_genres)
        union = len(expected_genres | output_genres)
        genre_score = intersection / union if union > 0 else 0.0
        if genre_score < 1.0:
            errors.append(
                f"Genre partial match. Expected: {expected_genres}, Got: {output_genres}"
            )
        metrics.append(genre_score)
    else:
        # Both empty is a match
        metrics.append(1.0)

    # Check numeric fields with tolerance
    numeric_fields = ["min_runtime", "max_runtime", "min_year", "max_year"]
    for field in numeric_fields:
        exp_val = expected.get(field)
        out_val = output.get(field)
        if exp_val is not None and out_val is not None:
            # Allow 5% tolerance for numeric values
            tolerance = 0.05
            diff = abs(exp_val - out_val) / max(exp_val, out_val)
            if diff > tolerance:
                errors.append(f"{field} mismatch. Expected: {exp_val}, Got: {out_val}")
                metrics.append(0.0)
            else:
                metrics.append(1.0)

    # Average all metrics for final score
    return sum(metrics) / len(metrics) if metrics else 0.0, errors
