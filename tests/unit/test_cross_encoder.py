from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from api.core import cross_encoder


@pytest.fixture(autouse=True)
def _clean_cross_encoder():
    cross_encoder.reset_cross_encoder_cache_for_tests()
    old_device = os.environ.pop("CROSS_ENCODER_DEVICE", None)
    old_model = os.environ.pop("CROSS_ENCODER_MODEL", None)
    yield
    cross_encoder.reset_cross_encoder_cache_for_tests()
    if old_device is not None:
        os.environ["CROSS_ENCODER_DEVICE"] = old_device
    if old_model is not None:
        os.environ["CROSS_ENCODER_MODEL"] = old_model


def test_get_cross_encoder_device_env_override(monkeypatch):
    monkeypatch.setenv("CROSS_ENCODER_DEVICE", "xpu")
    assert cross_encoder.get_cross_encoder_device() == "xpu"

    monkeypatch.setenv("CROSS_ENCODER_DEVICE", "cuda:0")
    assert cross_encoder.get_cross_encoder_device() == "cuda:0"


def test_get_cross_encoder_device_hardware_detection(monkeypatch):
    import torch

    monkeypatch.delenv("CROSS_ENCODER_DEVICE", raising=False)
    monkeypatch.delenv("CROSS_ENCODER_BACKEND", raising=False)

    # Test CUDA detection
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert cross_encoder.get_cross_encoder_device() == "cuda"

    # Test XPU detection
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch, "xpu", SimpleNamespace(is_available=lambda: True), raising=False
    )
    assert cross_encoder.get_cross_encoder_device() in {"cpu", "cuda", "xpu", "mps"}


def test_get_cross_encoder_model_caching(monkeypatch):
    mock_model = MagicMock()
    mock_load = MagicMock(return_value=mock_model)
    monkeypatch.setattr(cross_encoder, "_load_cross_encoder", mock_load)

    m1 = cross_encoder.get_cross_encoder_model("test-model")
    m2 = cross_encoder.get_cross_encoder_model("test-model")

    assert m1 is m2
    assert mock_load.call_count == 1

    cross_encoder.reset_cross_encoder_cache_for_tests()
    _ = cross_encoder.get_cross_encoder_model("test-model")
    assert mock_load.call_count == 2


def test_sigmoid():
    assert np.isclose(cross_encoder.sigmoid(0.0), 0.5)
    assert np.isclose(cross_encoder.sigmoid(100.0), 1.0)
    assert np.isclose(cross_encoder.sigmoid(-100.0), 0.0)

    arr = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    sig = cross_encoder.sigmoid(arr)
    assert len(sig) == 3
    assert sig[0] < sig[1] < sig[2]


def test_extract_names():
    assert cross_encoder._extract_names(None) == []
    assert cross_encoder._extract_names([]) == []
    assert cross_encoder._extract_names("Christopher Nolan") == ["Christopher Nolan"]
    assert cross_encoder._extract_names({"name": "Denis Villeneuve"}) == [
        "Denis Villeneuve"
    ]

    items = [
        {"name": "Leonardo DiCaprio"},
        "Joseph Gordon-Levitt",
        {"name": "Elliot Page"},
        {"name": "Leonardo DiCaprio"},  # duplicate
    ]
    extracted = cross_encoder._extract_names(items)
    assert extracted == ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Elliot Page"]


def test_build_candidate_document():
    full_item = {
        "id": 101,
        "title": "Inception",
        "release_year": 2010,
        "media_type": "movie",
        "genres": [{"name": "Action"}, {"name": "Sci-Fi"}],
        "directors": [{"name": "Christopher Nolan"}],
        "cast": [{"name": "Leonardo DiCaprio"}, {"name": "Joseph Gordon-Levitt"}],
        "overview": "A thief who steals corporate secrets through dream-sharing technology.",
    }
    doc = cross_encoder.build_candidate_document(full_item)
    assert "Inception (2010) [movie]" in doc
    assert "Genres: Action, Sci-Fi" in doc
    assert "Directed by Christopher Nolan" in doc
    assert "Starring Leonardo DiCaprio" in doc
    assert "dream-sharing technology" in doc

    # Sparse item fallback
    sparse = {"id": 999}
    assert cross_encoder.build_candidate_document(sparse) == "Item 999"


def test_format_query_text():
    assert (
        cross_encoder.format_query_text("space battle", ["Sci-Fi", "Action"])
        == "space battle | Genres: Sci-Fi, Action"
    )
    assert cross_encoder.format_query_text("space battle", None) == "space battle"
    assert cross_encoder.format_query_text("", ["Drama"]) == "Genres: Drama"
    assert cross_encoder.format_query_text(None, None) == ""


def test_score_query_candidates_empty_inputs():
    assert cross_encoder.score_query_candidates("test", []) == []

    # Empty query falls back to candidate retrieval score or rank prior
    candidates = [
        {"id": 1, "retrieval_score": 0.8},
        {"id": 2, "score": 0.6},
        {"id": 3},
    ]
    scores = cross_encoder.score_query_candidates("", candidates)
    assert len(scores) == 3
    assert scores[0] == (1, 0.8, 0.8)
    assert scores[1] == (2, 0.6, 0.6)
    assert scores[2][0] == 3


def test_score_query_candidates_scoring_and_blending(monkeypatch):
    candidates = [
        {
            "id": 1,
            "title": "Star Wars: A New Hope",
            "overview": "Luke Skywalker joins forces with a Jedi Knight.",
            "retrieval_score": 0.6,
            "original_rank": 0,
        },
        {
            "id": 2,
            "title": "Baking Bread",
            "overview": "How to bake sourdough in a Dutch oven.",
            "retrieval_score": 0.9,
            "original_rank": 1,
        },
    ]

    mock_model = MagicMock()
    # Candidate 1 gets high raw logit (2.0), Candidate 2 gets low logit (-2.0)
    mock_model.predict.return_value = np.array([2.0, -2.0], dtype=np.float32)
    monkeypatch.setattr(
        cross_encoder, "get_cross_encoder_model", lambda name=None: mock_model
    )

    scored = cross_encoder.score_query_candidates(
        query="star wars jedi space battle",
        candidates=candidates,
        alpha=0.7,
    )

    assert len(scored) == 2
    # Candidate 1 should win despite lower retrieval score because Cross-Encoder score is much higher
    assert scored[0][0] == 1
    assert scored[1][0] == 2

    # Verify scores are bounded [0, 1]
    assert 0.0 <= scored[0][1] <= 1.0
    assert 0.0 <= scored[1][1] <= 1.0


def test_score_query_candidates_handles_inference_exception(monkeypatch):
    candidates = [{"id": 10, "retrieval_score": 0.75, "title": "Test"}]

    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("GPU out of memory")
    monkeypatch.setattr(
        cross_encoder, "get_cross_encoder_model", lambda name=None: mock_model
    )

    scored = cross_encoder.score_query_candidates("query", candidates)
    assert len(scored) == 1
    assert scored[0][0] == 10
    assert scored[0][1] == 0.75  # safely falls back to retrieval score


def test_get_cross_encoder_device_openvino(monkeypatch):
    monkeypatch.setenv("CROSS_ENCODER_BACKEND", "openvino")
    monkeypatch.delenv("CROSS_ENCODER_DEVICE", raising=False)

    mock_core = MagicMock()
    mock_core.return_value.available_devices = ["CPU", "GPU", "NPU"]
    monkeypatch.setattr("openvino.Core", mock_core, raising=False)

    dev = cross_encoder.get_cross_encoder_device()
    assert dev == "GPU"

    mock_core.return_value.available_devices = ["CPU", "NPU"]
    dev_npu = cross_encoder.get_cross_encoder_device()
    assert dev_npu == "NPU"

    mock_core.return_value.available_devices = ["CPU"]
    dev_cpu = cross_encoder.get_cross_encoder_device()
    assert dev_cpu == "CPU"


def test_load_cross_encoder_openvino_and_fallback(monkeypatch):
    monkeypatch.setenv("CROSS_ENCODER_BACKEND", "openvino")
    mock_ov_init = MagicMock(return_value=None)
    monkeypatch.setattr(cross_encoder.OpenVINOCrossEncoder, "__init__", mock_ov_init)

    loaded = cross_encoder._load_cross_encoder("test-model", "GPU")
    assert isinstance(loaded, cross_encoder.OpenVINOCrossEncoder)

    # Fallback to PyTorch on exception
    monkeypatch.setattr(
        cross_encoder.OpenVINOCrossEncoder,
        "__init__",
        MagicMock(side_effect=RuntimeError("OpenVINO init failed")),
    )
    import sentence_transformers

    mock_st_ce = MagicMock()
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", mock_st_ce)

    fallback_loaded = cross_encoder._load_cross_encoder("test-model", "GPU")
    assert fallback_loaded is not None
    assert mock_st_ce.called


def test_openvino_cross_encoder_predict():
    class FakeOVModel:
        def __call__(self, **kwargs):
            bs = len(kwargs["input_ids"])
            return SimpleNamespace(logits=np.ones((bs, 1), dtype=np.float32) * 1.5)

    class FakeTokenizer:
        def __call__(self, pairs, **kwargs):
            return {"input_ids": np.zeros((len(pairs), 10), dtype=np.int64)}

    ov_ce = cross_encoder.OpenVINOCrossEncoder.__new__(
        cross_encoder.OpenVINOCrossEncoder
    )
    ov_ce.model = FakeOVModel()
    ov_ce.tokenizer = FakeTokenizer()
    ov_ce.device = "GPU"

    pairs = [("q", f"doc {i}") for i in range(5)]
    preds = ov_ce.predict(pairs, batch_size=2)
    assert len(preds) == 5
    assert np.allclose(preds, 1.5)

    # Empty pairs
    empty_preds = ov_ce.predict([])
    assert len(empty_preds) == 0
