"""Comparison regressions exercised through the real adjudicator and gate."""

import json
from unittest.mock import MagicMock

import pytest

from evaluation import evaluate
from evaluation.datasets import load_evaluation_cases
from evaluation.judge.consensus import JudgmentCache, PoolAdjudicator
from evaluation.judge.stub import StubJudgeAdapter
from evaluation.models import (
    DeterministicConstraint,
    EvaluationStatus,
    ItemEvidence,
    TestCase,
    TypedId,
)
from evaluation.trace import EvaluationTrace


def film(identifier, runtime=80, media="movie"):
    return {
        "tmdb_id": identifier,
        "media_type": media,
        "title": f"Catalog title {identifier}",
        "overview": "A detailed account of a journey.",
        "genres": ["Drama"],
        "runtime": runtime,
    }


class SelectiveJudge(StubJudgeAdapter):
    def __init__(self, name, affected_ids=(), failure=None, reference_grade=2):
        super().__init__(name, fixed_grade=2)
        self.affected_ids = set(affected_ids)
        self.failure = failure
        self.reference_grade = reference_grade
        self.inputs = []

    def _run_inference(self, prompt, judge_input):
        self.inputs.append(judge_input)
        identifier = judge_input.evidence.typed_id.id
        if identifier in self.affected_ids:
            if self.failure == "failed":
                raise RuntimeError("Judge service unavailable")
            if self.failure == "malformed":
                raise ValueError("Missing probabilities")
            if self.failure == "insufficient":
                return {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0}, False, "insufficient"
        if identifier == 999:
            grade = self.reference_grade
            return {g: float(g == grade) for g in range(4)}, True, "reference"
        return super()._run_inference(prompt, judge_input)


@pytest.fixture
def compare(tmp_path, monkeypatch):
    def run(
        base,
        candidate,
        *,
        constraints=None,
        references=None,
        failure=None,
        affected_ids=(),
        mode="single_judge",
        split="dev",
        reference_grade=2,
    ):
        family_count = 50 if split == "regression" else 1
        cases = [
            TestCase(
                case_id=f"case-{i}",
                family_id=f"family-{i}",
                track="product",
                split=split,
                task="search",
                slice_tags=["vibe", "entity", "franchise", "constraint"],
                query=f"journey {i}",
                constraints=constraints,
                golden_set=references,
            )
            for i in range(family_count)
        ]
        runner = MagicMock()
        runner.run_case.side_effect = lambda case, params, k: (
            candidate if params["rerank"] else base,
            EvaluationTrace(case.query, case.user_id),
        )
        judges = []

        def make_judge(name, **kwargs):
            judge = SelectiveJudge(name, affected_ids, failure, reference_grade)
            judges.append(judge)
            return judge

        media = (constraints.media_type if constraints else None) or "movie"
        catalog = {
            str(TypedId.parse(item)): item
            for item in base + candidate + [film(999, media=media)]
        }
        monkeypatch.setattr(evaluate, "load_evaluation_cases", lambda **kw: cases)
        monkeypatch.setattr(evaluate, "load_catalog_metadata", lambda: catalog)
        monkeypatch.setattr(evaluate, "EvaluationRunner", lambda **kw: runner)
        monkeypatch.setattr(evaluate, "discover_ollama_judges", lambda: {})
        monkeypatch.setattr(evaluate, "StubJudgeAdapter", make_judge)
        # These comparisons exercise adjudication and gates; persistence has its
        # own disk round-trip regression below. Avoid rewriting 1,000 labels per run.
        monkeypatch.setattr(JudgmentCache, "save", lambda self: None)
        report_path = tmp_path / "report.json"
        args = evaluate.parse_args(
            [
                "--v2",
                "--split",
                split,
                "--judgment-mode",
                mode,
                "--judge-config",
                "stub",
                "--allow-stub-judges-for-testing",
                "--backend",
                "pgvector",
                "--v2-report",
                str(report_path),
            ]
        )
        code = evaluate.run_evaluation_v2(args)
        report = json.loads(report_path.read_text())
        return code, report, runner, judges

    return run


@pytest.mark.parametrize("mode", ["single_judge", "consensus"])
@pytest.mark.parametrize(
    "failure, expected",
    [
        ("failed", EvaluationStatus.INVALID),
        ("malformed", EvaluationStatus.INVALID),
        ("insufficient", EvaluationStatus.INCONCLUSIVE),
    ],
)
def test_failed_baseline_judgments_never_pass_promotion(
    compare, mode, failure, expected
):
    code, report, _, _ = compare(
        [film(i) for i in range(1, 11)],
        [film(i) for i in range(101, 111)],
        failure=failure,
        affected_ids=range(1, 11),
        mode=mode,
        split="regression",
    )
    assert code == expected
    assert not report["gate_result"]["passed"]
    assert report["per_query_results"][0]["unresolved_judgments"] == 10


@pytest.mark.parametrize("failure", ["failed", "malformed", "insufficient"])
def test_cached_judgments_preserve_execution_status(tmp_path, failure):
    judge = SelectiveJudge("test", [1], failure)
    cache_path = tmp_path / "cache.json"
    evidence = ItemEvidence(TypedId("movie", 1), "Film", "Synopsis")
    first = PoolAdjudicator(judge, cache=JudgmentCache(cache_path)).adjudicate_pair(
        "q", evidence
    )
    calls = len(judge.inputs)
    cached = PoolAdjudicator(judge, cache=JudgmentCache(cache_path)).adjudicate_pair(
        "q", evidence
    )
    assert len(judge.inputs) == calls
    assert first.grade is cached.grade is None
    status = "success" if failure == "insufficient" else failure
    assert first.execution_statuses == cached.execution_statuses == [status]
    assert cached.to_dict()["execution_statuses"] == [status]


def test_baseline_constraint_violations_do_not_penalize_compliant_candidate(compare):
    code, report, _, _ = compare(
        [film(i, 120) for i in range(1, 11)],
        [film(i) for i in range(101, 111)],
        constraints=DeterministicConstraint(max_runtime=90),
    )
    assert code == EvaluationStatus.PASS
    gate = next(
        c
        for c in report["gate_result"]["checks"]
        if c["name"] == "zero_constraint_violations_gate"
    )
    assert gate["observed"]["hard_constraints"] == 0


@pytest.mark.parametrize("failure", [None, "insufficient"])
def test_candidate_constraints_checked_even_without_accepted_grade(compare, failure):
    code, report, _, _ = compare(
        [film(i) for i in range(1, 11)],
        [film(101, 120)] + [film(i) for i in range(102, 111)],
        constraints=DeterministicConstraint(max_runtime=90),
        affected_ids=[101],
        failure=failure,
    )
    assert code != EvaluationStatus.PASS
    gate = next(
        c
        for c in report["gate_result"]["checks"]
        if c["name"] == "zero_constraint_violations_gate"
    )
    assert gate["observed"]["hard_constraints"] == 1


@pytest.mark.parametrize("media", ["movie", "tv"])
def test_recall_detects_known_positive_removed_at_rank_50(compare, media):
    shared = [film(i, media=media) for i in range(1, 100)]
    baseline = shared[:49] + [film(999, media=media)] + shared[49:]
    candidate = shared + [film(100, media=media)]
    code, report, _, _ = compare(
        baseline,
        candidate,
        references=[f"{media}:999"],
        constraints=DeterministicConstraint(media_type=media),
    )
    assert code == EvaluationStatus.FAIL
    row = report["per_query_results"][0]
    assert row["base_recall_100"] == 1
    assert row["cand_recall_100"] == pytest.approx(10 / 11)
    assert row["delta_ndcg"] == 0


def test_reference_grades_and_titles_are_judged_from_catalog(compare):
    items = [film(i) for i in range(1, 11)]
    code, report, _, judges = compare(
        items,
        items,
        references=[{"id": 999, "title": "Wrong annotation", "grade": 3}],
        reference_grade=0,
    )
    assert code == EvaluationStatus.PASS
    evidence = next(
        inp.evidence for inp in judges[0].inputs if inp.evidence.typed_id.id == 999
    )
    assert evidence.title == "Catalog title 999"
    assert report["per_query_results"][0]["cand_recall_100"] == 1


def test_unresolved_recall_reference_is_inconclusive(compare):
    items = [film(i) for i in range(1, 11)]
    code, report, _, _ = compare(
        items,
        items,
        references=["movie:999"],
        affected_ids=[999],
        failure="insufficient",
    )
    assert code == EvaluationStatus.INCONCLUSIVE
    assert report["per_query_results"][0]["unresolved_judgments"] == 1


def test_selected_backend_reaches_both_systems(compare):
    items = [film(i) for i in range(1, 11)]
    code, _, runner, _ = compare(items, items)
    assert code == EvaluationStatus.PASS
    assert runner.run_case.call_count == 2
    for call in runner.run_case.call_args_list:
        assert call.kwargs["params"]["ann_backend_override"] == "pgvector"


def test_full_product_split_contains_both_v2_splits_and_constraints():
    dev = load_evaluation_cases(split="dev")
    regression = load_evaluation_cases(split="regression")
    full = load_evaluation_cases(split="full")
    assert [case.to_dict() for case in full] == [
        case.to_dict() for case in dev + regression
    ]
    assert {case.split for case in full} == {"dev", "regression"}
    assert any(case.constraints is not None for case in full)


def test_full_split_never_replaces_missing_v2_file_with_legacy(tmp_path):
    (tmp_path / "product_dev.json").write_text('[{"query": "dev"}]')
    assert load_evaluation_cases(split="full", base_dir=tmp_path) == []


def test_dataset_preserves_exclusions_for_deterministic_checks(tmp_path):
    from evaluation.deterministic import check_deterministic_constraints

    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "query": "exclude history",
                    "constraints": {
                        "seen_ids": [1],
                        "disliked_ids": [2],
                        "canonical_sequence_id": "sequence",
                    },
                }
            ]
        )
    )
    case = load_evaluation_cases(path=path)[0]
    assert case.constraints.seen_ids == [1]
    assert case.constraints.disliked_ids == [2]
    assert case.constraints.canonical_sequence_id == "sequence"
    for identifier in (1, 2):
        valid, reasons = check_deterministic_constraints(
            film(identifier), case.constraints
        )
        assert not valid and reasons
