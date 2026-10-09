"""Versioned grade semantics shared by every judge adapter."""

import math
from typing import Any, Dict

ADAPTER_CONTRACT_VERSION = "v2.7"
QUALIFICATION_PROTOCOL_VERSION = "v2.8"
RELEVANCE_INSTRUCTIONS = (
    "Use only supplied evidence to assess this candidate, not model memory, quality or popularity. "
    "Identify the main intent and mandatory facts. One known mandatory violation makes the item "
    "incompatible despite other matches or unknowns. Media Type movie means film; tv means series. "
    "Use Release Year for release decades, not story dates; Runtime is in minutes (60 per hour). "
    "Language is not country of production. Missing credits do not prove a person's absence. "
    "For franchises assess membership; list order and completeness are checked separately. "
    "Mood/style needs synopsis or keyword support: generic genre overlap alone is irrelevant. "
    "Partial support needs a concrete connection to the main intent; clear support needs the main "
    "intent, not just a secondary qualifier. Fully evidenced literal requests are complete matches. "
    "Decide the meaning first, then select its criterion irrespective of position."
)
SUFFICIENCY_INSTRUCTIONS = (
    "Can evidence support acceptance OR rejection? Apply in order, independently of option order: "
    "(1) YES for a known mandatory mismatch. Media Type tv versus a feature-movie request, or "
    "movie versus a TV-series request, means YES even with missing synopsis or cast. Wrong release "
    "year/runtime also permits rejection despite unknown availability. (2) YES if represented "
    "facts or synopsis support a relevance grade or establish irrelevance. YES does not mean "
    "relevant. (3) Otherwise NO when missing/conflicting/corrupt critical facts prevent a decision. "
    "Ignore missing optional facts. Required streaming access needs provider, country and time; "
    "unknown means NO unless rejection is already supported. Do not fill missing facts from memory."
)
GRADE_CRITERIA = (
    "Incompatible with an explicit required factual condition, or semantically irrelevant to the request.",
    "Weak or partial match: concrete evidence supports part of the main intent, but not a clear match; generic genre overlap alone is insufficient.",
    "Clear match: the main requested intent is supported, with only partial support for secondary subjective qualifiers.",
    "Complete match: explicit evidence satisfies all stated requirements; additional popularity or artistic merit is unnecessary.",
)


def normalize_probabilities(raw: Any) -> Dict[int, float]:
    """Reject malformed distributions rather than manufacture a grade."""
    if isinstance(raw, dict):
        if set(raw) == {"0", "1", "2", "3"}:
            values = [raw[str(i)] for i in range(4)]
        elif set(raw) == {0, 1, 2, 3}:
            values = [raw[i] for i in range(4)]
        else:
            raise ValueError("Expected exactly four grade probabilities (0 through 3)")
    elif isinstance(raw, (list, tuple)) and len(raw) == 4:
        values = list(raw)
    else:
        raise ValueError("Expected exactly four grade probabilities")
    try:
        if any(isinstance(v, bool) for v in values):
            raise ValueError("Boolean probabilities are invalid")
        numbers = [float(v) for v in values]
    except (TypeError, OverflowError) as exc:
        raise ValueError("Non-numeric probability") from exc
    if any(not math.isfinite(v) or v < 0 for v in numbers):
        raise ValueError("Probabilities must be finite and nonnegative")
    # Scaling first avoids overflow when individually finite values sum to infinity.
    maximum = max(numbers)
    if maximum <= 0:
        raise ValueError("Probabilities must have a positive total")
    scaled = [v / maximum for v in numbers]
    total = math.fsum(scaled)
    return {g: value / total for g, value in enumerate(scaled)}
