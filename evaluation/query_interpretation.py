"""Conservative normalization of explicit film release and duration conditions."""

import re

INTERPRETATION_VERSION = "v1"
_DECADE = re.compile(
    r"\b((?:movies|films|dramas)\s+)from\s+(?:the\s+)?((?:19|20)\d0)s\b",
    re.IGNORECASE,
)
_DURATION = re.compile(
    r"\b(under|over|longer than|shorter than|at least|at most)\s+"
    r"(\d+(?:\.\d+)?|one|two|three)\s+hours?\b",
    re.IGNORECASE,
)
_NUMBERS = {"one": 1, "two": 2, "three": 3}


def interpret_query(query: str) -> str:
    """Preserve intent; never turn story dates or elapsed plot time into filters.

    Ambiguous narrative or relational wording is deliberately left untouched.
    Original text remains in JudgeInput and the run's immutable audit inputs.
    """
    canonical = _DECADE.sub(
        lambda match: f"{match[1]}released between {match[2]} and {int(match[2]) + 9}",
        query,
    )
    if re.search(r"\b(story|plot|unfolds|set in|takes place)\b", query, re.I):
        return canonical

    def duration(match: re.Match) -> str:
        number = match[2].lower()
        hours = _NUMBERS[number] if number in _NUMBERS else float(number)
        operator = {"over": "longer than", "under": "shorter than"}.get(
            match[1].lower(), match[1].lower()
        )
        return f"with runtime {operator} {hours * 60:g} minutes"

    return _DURATION.sub(duration, canonical)
