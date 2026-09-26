from __future__ import annotations

import logging
import os
from typing import List, Sequence

import numpy as np

from api.core.embeddings import encode_texts
from api.pipeline.hooks import get_hook

logger = logging.getLogger("api.routes.recommend")


def float_from_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


_ANN_DESCRIPTION_WEIGHT = float_from_env("ANN_DESCRIPTION_WEIGHT", 1.2)
_REWRITE_TEXT_WEIGHT = float_from_env("REWRITE_TEXT_WEIGHT", 1.0)
_REFERENCE_TITLE_WEIGHT = float_from_env("REFERENCE_TITLE_WEIGHT", 0.8)


def append_weighted_text(
    text: str | None,
    weight_override: float | None,
    default_weight: float,
    texts: List[str],
    weights: List[float],
) -> float:
    normalized = (text or "").strip()
    if not normalized:
        return 0.0
    weight = default_weight if weight_override is None else weight_override
    weight = max(0.0, weight)
    if weight <= 0.0:
        return 0.0
    texts.append(normalized)
    weights.append(weight)
    return weight


def build_rewrite_vector(
    rewrite_text: str | None,
    ann_description: str | None,
    ann_weight_override: float | None = None,
    rewrite_weight_override: float | None = None,
    reference_titles: Sequence[str] | None = None,
) -> np.ndarray | None:
    texts: List[str] = []
    weights: List[float] = []

    description_weight = append_weighted_text(
        ann_description,
        ann_weight_override,
        _ANN_DESCRIPTION_WEIGHT,
        texts,
        weights,
    )
    rewrite_weight = append_weighted_text(
        rewrite_text,
        rewrite_weight_override,
        _REWRITE_TEXT_WEIGHT,
        texts,
        weights,
    )
    reference_weight = 0.0
    if reference_titles:
        for title in reference_titles:
            reference_weight += append_weighted_text(
                title,
                None,
                _REFERENCE_TITLE_WEIGHT,
                texts,
                weights,
            )

    if not texts:
        return None

    encoder = get_hook("encode_texts", encode_texts)
    vectors = encoder(texts)
    if not isinstance(vectors, np.ndarray) or vectors.size == 0:
        return None

    combined = np.zeros(vectors.shape[1], dtype=np.float32)
    total_weight = 0.0
    for vec, weight in zip(vectors, weights):
        if weight <= 0.0:
            continue
        combined += weight * vec
        total_weight += weight

    if total_weight <= 0.0:
        return None

    norm = float(np.linalg.norm(combined))
    if norm == 0.0 or not np.isfinite(norm):
        return None
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Rewrite vector sources | description=%s rewrite=%s references=%d desc_weight=%.3f rewrite_weight=%.3f reference_weight=%.3f total_weight=%.3f",
            bool(description_weight),
            bool(rewrite_weight),
            len(reference_titles or ()),
            description_weight,
            rewrite_weight,
            reference_weight,
            total_weight,
        )
    return combined / norm
