"""Evaluation unit tests own their caches, qrels and database substitutes."""

import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def isolated_catalog_source(tmp_path, monkeypatch):
    # A frozen local fixture is sufficient for unit tests even after chdir;
    # generated live snapshots and real DB reads belong to integration checks.
    from evaluation import evidence

    monkeypatch.setattr(evidence, "CATALOG_EVIDENCE_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(
        evidence,
        "LEGACY_CATALOG_PATH",
        Path(__file__).resolve().parents[3]
        / "evaluation/fixtures/catalog_metadata.json",
    )


@pytest.fixture(autouse=True)
def isolated_judgments(tmp_path, monkeypatch):
    from evaluation.judge import consensus

    monkeypatch.setattr(
        consensus.JudgmentCache.__init__, "__defaults__", (tmp_path / "judgments.json",)
    )
    defaults = list(consensus.ConsensusJudgeEngine.__init__.__defaults__)
    defaults[5] = tmp_path / "qrels"
    monkeypatch.setattr(
        consensus.ConsensusJudgeEngine.__init__, "__defaults__", tuple(defaults)
    )


@pytest.fixture(autouse=True)
def isolated_runtime_cache(monkeypatch):
    monkeypatch.setattr("api.core.llm_parser._persistent_intent_store", lambda: None)
