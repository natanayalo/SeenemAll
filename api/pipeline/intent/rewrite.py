from __future__ import annotations

import os

import numpy as np

from api.core.embeddings import encode_texts
from api.core.metrics import timer
from api.pipeline.hooks import get_hook


def float_from_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def build_query_vector(query_text: str | None) -> np.ndarray | None:
    """Embed one retrieval query without generated descriptions or rewrites."""
    normalized = (query_text or "").strip()
    if not normalized:
        return None

    encoder = get_hook("encode_texts", encode_texts)
    with timer("recommend.query_embedding_latency_ms"):
        vectors = encoder([normalized])
    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2 or not vectors.size:
        return None

    vector = np.asarray(vectors[0], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not np.isfinite(norm):
        return None
    return vector / norm
