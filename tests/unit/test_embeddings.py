from __future__ import annotations

import numpy as np

import pytest

from api.core import embeddings


@pytest.fixture(autouse=True)
def reset_embeddings_state():
    embeddings.reset_embedding_cache_for_tests()
    yield
    embeddings.reset_embedding_cache_for_tests()


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
    embeddings.reset_embedding_cache_for_tests()
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    monkeypatch.delenv("DEVICE", raising=False)
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)

    import types
    import sys

    created = {}

    class DummySentenceTransformer:
        def __init__(self, model_name, device):
            created["model_name"] = model_name
            created["device"] = device

        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 384), dtype=np.float32)

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


def test_get_embedding_device_resolution(monkeypatch):
    # 1. Direct env override
    monkeypatch.setenv("EMBEDDING_DEVICE", "NPU")
    assert embeddings.get_embedding_device() == "NPU"

    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    monkeypatch.setenv("DEVICE", "GPU")
    assert embeddings.get_embedding_device() == "GPU"

    # 2. OpenVINO auto-discovery
    monkeypatch.delenv("DEVICE", raising=False)
    monkeypatch.setenv("EMBEDDING_BACKEND", "openvino")

    from unittest.mock import MagicMock

    mock_core = MagicMock()
    mock_core.return_value.available_devices = ["CPU", "GPU", "NPU"]
    monkeypatch.setattr("openvino.Core", mock_core, raising=False)

    assert embeddings.get_embedding_device() == "GPU"

    mock_core.return_value.available_devices = ["CPU", "NPU"]
    assert embeddings.get_embedding_device() == "NPU"

    mock_core.return_value.available_devices = ["CPU"]
    assert embeddings.get_embedding_device() == "CPU"


def test_openvino_embedding_model_encode():
    from types import SimpleNamespace

    class FakeOVModel:
        def __call__(self, **kwargs):
            bs = len(kwargs["input_ids"])
            seq_len = 10
            hidden = np.ones((bs, seq_len, 384), dtype=np.float32)
            return SimpleNamespace(last_hidden_state=hidden)

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            bs = len(texts)
            return {
                "input_ids": np.zeros((bs, 10), dtype=np.int64),
                "attention_mask": np.ones((bs, 10), dtype=np.int64),
            }

    model = embeddings.OpenVINOEmbeddingModel.__new__(embeddings.OpenVINOEmbeddingModel)
    model.model = FakeOVModel()
    model.tokenizer = FakeTokenizer()
    model.device = "GPU"

    vectors = model.encode(["test 1", "test 2"], batch_size=2)
    assert vectors.shape == (2, 384)
    # Check L2 normalization (norm should be 1.0)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0)

    # Empty texts
    empty = model.encode([])
    assert empty.shape == (0, 384)
