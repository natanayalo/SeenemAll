from __future__ import annotations

import numpy as np

from api.core import embeddings


def test_encode_texts_returns_zero_vector_for_empty_input():
    result = embeddings.encode_texts([])
    assert result.shape == (0, 384)
    assert result.dtype == np.float32


def test_encode_texts_uses_model_encode(monkeypatch):
    monkeypatch.setattr(embeddings, "_model", None, raising=False)

    class StubModel:
        def encode(
            self,
            texts,
            batch_size,
            normalize_embeddings,
            convert_to_numpy,
            show_progress_bar,
        ):
            assert normalize_embeddings is True
            assert convert_to_numpy is True
            assert show_progress_bar is True
            return np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")

    monkeypatch.setattr(embeddings, "get_model", lambda: StubModel())

    result = embeddings.encode_texts(["foo", "bar"])

    assert result.shape == (2, 2)
    assert result.dtype == np.float32


def test_get_model_initialises_once(monkeypatch):
    # reset cached singleton
    monkeypatch.setattr(embeddings, "_model", None, raising=False)

    import types
    import sys

    created = {}

    class DummySentenceTransformer:
        def __init__(self, model_name, device):
            created["model_name"] = model_name
            created["device"] = device

    dummy_module = types.SimpleNamespace(SentenceTransformer=DummySentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", dummy_module)

    class DummyCuda:
        @staticmethod
        def is_available():
            return True

    dummy_torch = types.SimpleNamespace(cuda=DummyCuda())
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)

    first = embeddings.get_model()
    second = embeddings.get_model()

    assert first is second
    assert created["model_name"] == embeddings.DEFAULT_MODEL
    assert created["device"] == "cuda"
