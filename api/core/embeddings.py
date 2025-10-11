from __future__ import annotations

import os
from typing import Iterable

import numpy as np
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer  # pragma: no cover
else:
    SentenceTransformer = Any  # type: ignore

DEFAULT_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        import torch  # local import to avoid initialization overhead when unused
        from sentence_transformers import SentenceTransformer as _SentenceTransformer

        # device auto: cuda if available, else cpu
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _SentenceTransformer(DEFAULT_MODEL, device=device)
    return _model


def encode_texts(texts: Iterable[str]) -> np.ndarray:
    """
    Returns float32 numpy array shape (N, 384) for MiniLM, L2-normalized row-wise.
    """
    # Convert generator to list once to avoid double iteration
    texts = list(texts)
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    model = get_model()
    emb = model.encode(
        texts,
        batch_size=int(os.getenv("EMBED_BATCH", "64")),
        normalize_embeddings=True,  # L2-normalize for cosine
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return emb.astype("float32")
