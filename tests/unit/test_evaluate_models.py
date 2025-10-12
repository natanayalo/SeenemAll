"""Tests for model evaluation script."""

import pytest
from scripts.evaluate_models import ModelConfig, evaluate_model


@pytest.fixture
def test_items():
    return [
        {
            "id": 1,
            "title": "Test Movie 1",
            "overview": "A test movie about testing.",
            "genres": [{"id": 28, "name": "Action"}],
            "release_year": 2025,
            "media_type": "movie",
            "runtime": 120,
        },
        {
            "id": 2,
            "title": "Test Movie 2",
            "overview": "Another test movie about similar things.",
            "genres": [{"id": 28, "name": "Action"}],
            "release_year": 2025,
            "media_type": "movie",
            "runtime": 120,
        },
    ]


@pytest.fixture
def test_pairs():
    return [
        (1, 2, 0.8),  # Similar items
        (1, 1, 1.0),  # Self-similarity
    ]  # Expect high similarity for both pairs


def test_model_evaluation(test_items, test_pairs):
    model_config = ModelConfig("test", "all-MiniLM-L6-v2", 384, "Test model")

    results = evaluate_model(model_config, test_items, test_pairs)

    assert "model" in results
    assert "mse" in results
    assert "mae" in results
    assert "encode_time" in results
    assert "num_items" in results

    assert results["model"] == "test"
    assert results["num_items"] == 2
    assert 0 <= float(results["mse"]) <= 1.0
    assert 0 <= float(results["mae"]) <= 1.0
    assert float(results["encode_time"]) > 0


def test_invalid_model(test_items, test_pairs):
    with pytest.raises(Exception):
        evaluate_model(
            ModelConfig("invalid", "nonexistent-model", 384, "Invalid"),
            test_items,
            test_pairs,
        )
