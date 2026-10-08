"""The deterministic judge remains available for isolated contract tests."""

from evaluation.judge.stub import StubJudgeAdapter
from evaluation.judge.consensus import JudgmentCache, PoolAdjudicator
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


def test_cached_resource_rejection_is_retried_when_input_fits(tmp_path):
    judge = StubJudgeAdapter(fixed_grade=2)
    evidence = ItemEvidence(TypedId("movie", 1), "A film", synopsis="x")
    inp = JudgeInput("A complete query", evidence)
    evidence.synopsis += "x" * (4011 - len(judge.build_prompt(inp)))
    assert len(judge.build_prompt(inp)) == 4011
    cache = JudgmentCache(tmp_path / "cache.json")
    cache.set(
        inp,
        judge,
        {
            "grade": 0,
            "probabilities": {0: 1.0},
            "evidence_sufficiency": False,
            "execution_status": "over_limit",
        },
    )
    record = PoolAdjudicator(judge, cache=cache).adjudicate_pair(
        inp.query, evidence, mode="single_judge"
    )
    assert record.grade == 2 and record.execution_statuses == ["success"]
    assert cache.get(inp, judge)["execution_status"] == "success"


def test_over_budget_cached_input_still_fails(tmp_path):
    judge = StubJudgeAdapter(fixed_grade=2)
    evidence = ItemEvidence(TypedId("movie", 1), "A film", synopsis="x")
    inp = JudgeInput("A complete query", evidence)
    evidence.synopsis += "x" * (4097 - len(judge.build_prompt(inp)))
    assert len(judge.build_prompt(inp)) == 4097
    cache = JudgmentCache(tmp_path / "cache.json")
    cache.set(inp, judge, judge.judge_pair(inp).to_dict())
    record = PoolAdjudicator(judge, cache=cache).adjudicate_pair(
        inp.query, evidence, mode="single_judge"
    )
    assert record.grade is None and record.status == "JUDGE_FAILED"
    assert record.execution_statuses == ["over_limit"]
