"""The deterministic judge remains available for isolated contract tests."""

from evaluation.judge.stub import StubJudgeAdapter
from evaluation.models import ItemEvidence, JudgeInput, TypedId


def test_over_limit_input_rejection():
    judge = StubJudgeAdapter()
    evidence = ItemEvidence(TypedId("movie", 1), "Test Movie", "A " * 3000)
    out = judge.judge_pair(JudgeInput("Query " * 200, evidence))
    assert out.execution_status == "over_limit"
    assert out.grade == 0 and out.evidence_sufficiency is False


def test_stub_judge_custom_lookup_and_error():
    evidence = ItemEvidence(TypedId("movie", 99), "Target")
    judge = StubJudgeAdapter(grade_lookup={("my query", "movie:99"): 3})
    assert judge.judge_pair(JudgeInput("my query", evidence)).grade == 3
    failed = StubJudgeAdapter(simulate_error=True).judge_pair(JudgeInput("q", evidence))
    assert failed.execution_status == "failed" and failed.grade == 0
