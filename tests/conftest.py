from __future__ import annotations

import sys
import types

import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_torch_stub() -> None:
    sys.modules.pop("torch", None)
    torch_stub = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    torch_stub.cuda = _Cuda()  # type: ignore[attr-defined]
    sys.modules["torch"] = torch_stub


def _ensure_sentence_transformers_stub() -> None:
    sys.modules.pop("sentence_transformers", None)
    module = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        def __init__(self, model_name: str, device: str | None = None):
            self.model_name = model_name
            self.device = device

        def encode(
            self,
            texts,
            batch_size: int = 32,
            normalize_embeddings: bool = False,
            convert_to_numpy: bool = True,
            show_progress_bar: bool = False,
        ):
            texts = list(texts)
            size = len(texts)
            vectors = np.zeros((size, 384), dtype="float32")
            return vectors

    module.SentenceTransformer = SentenceTransformer  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = module


_ensure_torch_stub()
_ensure_sentence_transformers_stub()
