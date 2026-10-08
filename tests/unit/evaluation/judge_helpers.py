"""Pinned Nimble test metadata; no model or external service is required."""

from evaluation.judge.ollama import OllamaJudgeAdapter


def make_nimble(url="http://127.0.0.1:11434"):
    return OllamaJudgeAdapter(
        "bespoke-nimble-9b",
        "nimble:latest",
        {"digest": "test-digest", "details": {"quantization_level": "Q8_0"}},
        "test-version",
        url,
    )
