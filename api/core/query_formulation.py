"""Production query text selection for lexical and dense retrieval."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Tuple

logger = logging.getLogger(__name__)

RAW_FORMULATION = "raw"
STRATEGY_F_FORMULATION = "strategy_f"


def select_retrieval_query(
    query: str | None,
    query_filters: Any = None,
    *,
    strategy: str | None = None,
) -> Tuple[str, str]:
    """Return the text used by both embedding and BM25, plus its selected strategy.

    Raw user text is the production default. Strategy F is available only through
    the explicit ``RETRIEVAL_QUERY_FORMULATION=strategy_f`` experiment flag.
    Structured filters are parsed separately from the original query.
    """
    raw_query = (query or "").strip()
    selected = (
        (
            strategy
            if strategy is not None
            else os.getenv("RETRIEVAL_QUERY_FORMULATION", RAW_FORMULATION)
        )
        .strip()
        .lower()
    )

    if selected == STRATEGY_F_FORMULATION:
        return _strip_hard_filters(raw_query, query_filters), STRATEGY_F_FORMULATION
    if selected != RAW_FORMULATION:
        logger.warning(
            "Unsupported RETRIEVAL_QUERY_FORMULATION '%s'; using raw query text.",
            selected,
        )
    return raw_query, RAW_FORMULATION


def _strip_hard_filters(query: str, query_filters: Any) -> str:
    """Experimental Strategy F: remove hard constraints from retrieval text."""
    text = re.sub(r"^cold_start:\s*", "", query, flags=re.IGNORECASE)
    patterns = (
        r"\b(?:on\s+)?(?:netflix|prime(?:\s+video)?|amazon(?:\s+prime)?|disney\s*(?:\+|plus)?|apple\s*tv\s*(?:\+)?|hbo(?:\s+max)?|max|peacock|paramount\s*(?:\+)?|hulu)\b",
        r"\b(?:under|less\s+than|over|more\s+than|at\s+least|around|approx(?:\.)?)\s+\d+(?:\s*(?:hours?|hrs?|h|minutes?|mins?|m))?\b|\b\d+\s*(?:hours?|hrs?|h|minutes?|mins?|m)\b",
        r"\b(?:from\s+the|in\s+the|from|in|during\s+the)?\s*(?:19\d0s?|20\d0s?|\b\d0s)\b",
        r"\b(?:between|from|after|before)?\s*(?:19\d\d|20\d\d)(?:\s*(?:and|to|-)\s*(?:19\d\d|20\d\d))?\b",
        r"\b(?:r-rated|pg-13|pg|g|nc-17|tv-ma)\b",
        r"\b(?:movies?|films?|tv\s+series|tv\s+shows?|shows?|series|miniseries)\b",
    )
    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text = re.sub(r"[^\w\s\-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.-")

    entities: list[str] = []
    if query_filters is not None:
        entities.extend(getattr(query_filters, "reference_titles", ()) or ())
        entities.extend(getattr(query_filters, "cast", ()) or ())
        entities.extend(getattr(query_filters, "directors", ()) or ())
    for entity in entities:
        if (
            isinstance(entity, str)
            and entity.strip()
            and entity.lower() not in text.lower()
        ):
            text = f"{text} {entity.strip()}"
    return text.strip() or query
