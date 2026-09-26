from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np

os.environ["API_AUTH_ENABLED"] = "0"

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
    torch_stub.Tensor = type(  # type: ignore[attr-defined]
        "Tensor", (), {}
    )  # Scipy checks for torch.Tensor at import time

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    torch_stub.no_grad = _NoGrad  # type: ignore[attr-defined]

    # Also add it to the sys.modules so imports like `from torch import cuda` don't fail immediately
    cuda_stub = types.ModuleType("torch.cuda")
    cuda_stub.is_available = _Cuda.is_available  # type: ignore[attr-defined]
    sys.modules["torch.cuda"] = cuda_stub

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

    class CrossEncoder:
        def __init__(self, model_name: str, device: str | None = None):
            self.model_name = model_name
            self.device = device

        def predict(self, pairs, **kwargs):
            return np.zeros(len(pairs), dtype="float32")

    module.SentenceTransformer = SentenceTransformer  # type: ignore[attr-defined]
    module.CrossEncoder = CrossEncoder  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = module


_ensure_torch_stub()
_ensure_sentence_transformers_stub()
