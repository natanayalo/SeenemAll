"""Nimble judging, qualification and qrels for Seen'emAll Evaluation Suite v2."""

from evaluation.judge.base import LocalJudgeAdapter
from evaluation.judge.consensus import ConsensusJudgeEngine, PoolAdjudicator
from evaluation.judge.ollama import OllamaJudgeAdapter, discover_ollama_judges
from evaluation.judge.qualification import JudgeQualificationRunner
from evaluation.judge.stub import StubJudgeAdapter

__all__ = [
    "LocalJudgeAdapter",
    "OllamaJudgeAdapter",
    "discover_ollama_judges",
    "StubJudgeAdapter",
    "ConsensusJudgeEngine",
    "PoolAdjudicator",
    "JudgeQualificationRunner",
]
