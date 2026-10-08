import pytest

from evaluation.judge.base import LocalJudgeAdapter
from tests.unit.evaluation.judge_helpers import make_nimble
from evaluation.judge.qualification import (
    JudgeQualificationRunner,
    generate_judge_control_cases,
    qualification_record_matches,
)
from evaluation.judge.stub import StubJudgeAdapter
from evaluation.models import ItemEvidence, TestCase, TypedId


def valid_record(judge):
    return dict(
        qualified=True,
        fingerprint=judge.qualification_fingerprint(),
        repeatability_rate=1.0,
        option_permutation_rate=1.0,
        control_accuracy=1.0,
        execution_failures=0,
        malformed_count=0,
        qualification_protocol="v2.8",
        scope="qualification",
        pilot_pairs=400,
        pilot_distinct_pairs=400,
        pilot_families=20,
        repeat_tests=100,
        permutation_tests=100,
        control_tests=100,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("fingerprint", None),
        ("qualification_protocol", "v2.1"),
        ("qualification_protocol", "v2.3"),
        ("qualification_protocol", "v2.4"),
        ("scope", "diagnostic"),
        ("pilot_pairs", 80),
        ("pilot_distinct_pairs", 399),
        ("qualification_protocol", "v2.5"),
        ("qualification_protocol", "v2.6"),
        ("pilot_families", 19),
        ("repeat_tests", False),
        ("control_tests", 99),
        ("qualified", False),
        ("repeatability_rate", 0.989999),
        ("qualification_protocol", "v2.7"),
        ("control_accuracy", 0.949999),
        ("control_accuracy", float("nan")),
        ("control_accuracy", True),
        ("control_accuracy", "1"),
        ("execution_failures", 1),
        ("malformed_count", 1),
        ("malformed_count", False),
    ],
)
def test_stale_or_invalid_records_cannot_authorize(field, value):
    judge = make_nimble()
    record = valid_record(judge)
    assert qualification_record_matches(judge, record)
    record[field] = value
    assert not qualification_record_matches(judge, record)
    assert not qualification_record_matches(judge, None)


def test_changed_provenance_requires_requalification():
    judge = make_nimble()
    record = valid_record(judge)
    judge.checkpoint_revision = "changed"
    assert not qualification_record_matches(judge, record)
    judge = make_nimble()
    judge.service_model += "-changed"
    assert not qualification_record_matches(judge, record)


def test_changed_evidence_contract_requires_fresh_qualification(monkeypatch):
    judge = make_nimble()
    record = valid_record(judge)
    monkeypatch.setattr("evaluation.judge.base.EVIDENCE_CONTRACT_VERSION", "v2.1")
    assert not qualification_record_matches(judge, record)


def test_changed_qualification_protocol_changes_cache_and_fingerprint(
    monkeypatch, tmp_path
):
    from evaluation.judge.consensus import JudgmentCache
    from evaluation.models import JudgeInput

    judge = make_nimble()
    inp = JudgeInput("film", ItemEvidence(TypedId("movie", 1), "Film", "Plot"))
    record = valid_record(judge)
    cache = JudgmentCache(tmp_path / "cache.json")
    key = cache._cache_key(inp, judge)
    monkeypatch.setattr("evaluation.judge.base.QUALIFICATION_PROTOCOL_VERSION", "old")
    assert cache._cache_key(inp, judge) != key
    assert not qualification_record_matches(judge, record)


@pytest.mark.parametrize(
    "mode",
    [
        "stable_abstention",
        "abstention_grade_flip",
        "sufficiency_flip",
        "grade_flip",
        "failed_repeat",
    ],
)
def test_repeat_decisions_and_coverage_are_separate(monkeypatch, mode):
    from evaluation.models import JudgeOutput

    runner = JudgeQualificationRunner()
    evidence = ItemEvidence(TypedId("movie", 1), "Film", "Plot")
    monkeypatch.setattr(
        runner, "build_pilot_items_for_family", lambda *a, **k: [evidence] * 4
    )
    judge = StubJudgeAdapter(auto_qualify=True)
    original = judge.judge_pair
    calls = 0

    def run(inp):
        nonlocal calls
        calls += 1
        if inp.query == "Recommendations for family":
            return JudgeOutput(
                grade=(
                    1
                    if mode in ("grade_flip", "abstention_grade_flip") and calls == 5
                    else 0
                ),
                probabilities={0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                evidence_sufficiency=mode == "grade_flip"
                or (mode == "sufficiency_flip" and calls == 5),
                execution_status=(
                    "failed"
                    if mode == "failed_repeat" and calls in (5, 6)
                    else "success"
                ),
                provenance=judge.get_provenance(evidence.content_hash()),
            )
        return original(inp)

    monkeypatch.setattr(judge, "judge_pair", run)
    report = runner.run_candidate_pilot(judge, ["family"])
    assert report["repeatability_rate"] == (
        1 if mode in ("stable_abstention", "abstention_grade_flip") else 0
    )
    assert report["pilot_evidence_coverage"] == (1 if mode == "grade_flip" else 0)
    assert report["repeat_evidence_coverage"] == (1 if mode == "grade_flip" else 0)
    assert report["repeat_tests"] == 1
    assert report["scope"] == "diagnostic"
    assert not report["qualified"]
    assert report["control_accuracy"] == 1


@pytest.mark.parametrize("items,interval", [(0, 4), (21, 4), (4, 0), (4, 5)])
def test_invalid_diagnostic_sampling_rejected(items, interval):
    with pytest.raises(ValueError, match="Pilot size"):
        JudgeQualificationRunner().run_candidate_pilot(
            StubJudgeAdapter(), ["fam"], items_per_family=items, repeat_every=interval
        )


def test_passing_small_pilot_cannot_authorize_evaluation(monkeypatch):
    runner = JudgeQualificationRunner()
    evidence = ItemEvidence(TypedId("movie", 1), "Film", "Plot")
    monkeypatch.setattr(
        runner, "build_pilot_items_for_family", lambda *a, **k: [evidence] * 4
    )
    judge = StubJudgeAdapter(auto_qualify=True)
    report = runner.run_candidate_pilot(judge, ["family"], items_per_family=4)
    assert report["measured_gates_pass"]
    assert report["scope"] == "diagnostic"
    assert not report["qualified"]
    report["qualified"] = True
    assert not qualification_record_matches(judge, report)


def test_all_abstentions_cannot_qualify_through_repeatability(monkeypatch):
    runner = JudgeQualificationRunner()
    monkeypatch.setattr(
        runner,
        "build_pilot_items_for_family",
        lambda *a, **k: [
            ItemEvidence(TypedId("movie", i), "Film", "Plot") for i in range(20)
        ],
    )
    judge = StubJudgeAdapter(auto_qualify=True, fixed_sufficiency=False)
    # Every stage abstains, including controls; identical abstentions do not become accuracy.
    original = judge.judge_pair

    def abstain(inp):
        out = original(inp)
        out.evidence_sufficiency = False
        return out

    monkeypatch.setattr(judge, "judge_pair", abstain)
    report = runner.run_candidate_pilot(judge, [f"fam-{i}" for i in range(20)])
    assert report["scope"] == "qualification"
    assert report["repeatability_rate"] == 1
    assert report["pilot_evidence_coverage"] == report["control_accuracy"] == 0
    assert not report["qualified"]


def test_control_set_has_represented_conditions_and_both_outcomes():
    controls = generate_judge_control_cases()
    assert len(controls) == len({c.case_id for c in controls}) == 100
    assert sum(c.expected_grade == 0 for c in controls) == 50
    assert sum(c.expected_grade == 2 for c in controls) == 50
    for case in controls:
        assert case.constraints
        assert not case.constraints.providers
        assert not case.canonical_sequence
        assert (
            case.constraints.max_year is not None
            or case.constraints.max_runtime is not None
            or case.constraints.media_type is not None
        )


def test_zero_controls_fail_instead_of_loading_different_defaults(monkeypatch):
    runner = JudgeQualificationRunner()
    evidence = ItemEvidence(TypedId("movie", 1), "Film", "Plot")
    monkeypatch.setattr(
        runner, "build_pilot_items_for_family", lambda *a, **k: [evidence] * 4
    )
    report = runner.run_candidate_pilot(
        StubJudgeAdapter(), ["family"], control_cases=[]
    )
    assert report["control_tests"] == 0
    assert report["control_accuracy"] == 0
    assert not report["qualified"]


def test_panel_selection_rejects_stale_pass_flag():
    judge = StubJudgeAdapter(model_name="bespoke-nimble-9b")
    report = valid_record(judge)
    report["qualification_protocol"] = "v2.1"
    primary, secondary, tie, mode = JudgeQualificationRunner().select_judge_panel(
        {judge.model_name: report}, {judge.model_name: judge}
    )
    assert mode == "unqualified"
    assert primary is secondary is tie is None


@pytest.mark.parametrize("failure_location", ["repeat", "control", "permutation"])
def test_failures_in_all_qualification_phases_count(monkeypatch, failure_location):
    judge = StubJudgeAdapter(fixed_grade=0)
    runner = JudgeQualificationRunner()
    evidence = ItemEvidence(TypedId("movie", 1), "Film", "Full synopsis")
    monkeypatch.setattr(
        runner, "build_pilot_items_for_family", lambda *a, **k: [evidence] * 4
    )
    original = judge.judge_pair
    calls = 0

    def run(inp):
        nonlocal calls
        calls += 1
        if (failure_location == "repeat" and calls in (5, 6)) or (
            failure_location == "control" and inp.query == "control"
        ):
            judge.simulate_error = True
        else:
            judge.simulate_error = False
        return original(inp)

    monkeypatch.setattr(judge, "judge_pair", run)
    if failure_location == "permutation":
        monkeypatch.setattr(
            judge,
            "test_option_order_permutation",
            lambda inp: LocalJudgeAdapter.test_option_order_permutation(judge, inp),
        )
    report = runner.run_candidate_pilot(
        judge,
        ["family"],
        control_cases=[
            TestCase(
                case_id="control",
                family_id="control",
                track="product",
                split="dev",
                task="constraint",
                slice_tags=[],
                query="control",
                expected_grade=0,
            )
        ],
        run_option_order_diagnostic=True,
    )
    if failure_location == "permutation":
        assert report["permutation_execution_failures"] == 1
        assert report["execution_failures"] == 0
        assert not report["option_permutation_pass"]
    else:
        assert report["execution_failures"] >= 1
    assert not report["qualified"]
    if failure_location == "repeat":
        assert report["repeatability_rate"] == 0
    if failure_location == "control":
        assert report["control_accuracy"] == 0


@pytest.mark.parametrize("rate", [0.0, 0.83, 0.95])
def test_option_order_diagnostic_cannot_veto_current_record(rate):
    judge = make_nimble()
    record = valid_record(judge)
    record.update(option_permutation_rate=rate, option_permutation_pass=False)
    assert qualification_record_matches(judge, record)
    del record["option_permutation_rate"]
    del record["permutation_tests"]
    assert qualification_record_matches(judge, record)


@pytest.mark.parametrize("diagnostic", ["skip", "disagree", "fail"])
def test_full_fixed_prompt_qualification_separates_diagnostics(monkeypatch, diagnostic):
    runner = JudgeQualificationRunner()
    monkeypatch.setattr(
        runner,
        "build_pilot_items_for_family",
        lambda fam, **kw: [
            ItemEvidence(TypedId("movie", i), f"{fam} Film", "Plot") for i in range(20)
        ],
    )
    judge = StubJudgeAdapter(auto_qualify=True)

    def permutation(inp):
        if diagnostic == "skip":
            pytest.fail("Skipped diagnostics must not make model requests")
        if diagnostic == "fail":
            raise RuntimeError("Alternative prompt unavailable")
        return {}, {}, False

    monkeypatch.setattr(judge, "test_option_order_permutation", permutation)
    report = runner.run_candidate_pilot(
        judge,
        [f"family-{i}" for i in range(20)],
        run_option_order_diagnostic=diagnostic != "skip",
    )
    assert report["qualified"]
    assert qualification_record_matches(judge, report)
    assert report["option_permutation_role"] == "diagnostic"
    assert not report["option_permutation_pass"]
    assert report["permutation_tests"] == (0 if diagnostic == "skip" else 100)
    assert report["permutation_execution_failures"] == (
        100 if diagnostic == "fail" else 0
    )
    assert report["execution_failures"] == 0


def test_weak_positive_control_is_not_correct_and_evidence_is_neutral(monkeypatch):
    runner = JudgeQualificationRunner()
    monkeypatch.setattr(runner, "build_pilot_items_for_family", lambda *a, **k: [])
    judge = StubJudgeAdapter(fixed_grade=1)
    original = judge.judge_pair
    presented = []

    def collect(inp):
        presented.append(inp.evidence)
        return original(inp)

    monkeypatch.setattr(judge, "judge_pair", collect)
    report = runner.run_candidate_pilot(judge, [])
    assert report["control_accuracy"] == 0
    assert len(presented) == 100
    assert len({ev.synopsis for ev in presented}) == 1
    assert not any(
        "satisfying" in ev.synopsis or "exceeding" in ev.synopsis for ev in presented
    )
    assert all(ev.release_year <= 2026 for ev in presented)


def test_pilot_uses_actual_queries_not_family_identifiers(monkeypatch):
    runner = JudgeQualificationRunner()
    evidence = ItemEvidence(TypedId("movie", 1), "Film", "Full synopsis")
    monkeypatch.setattr(
        runner, "build_pilot_items_for_family", lambda *a, **k: [evidence] * 4
    )
    case = TestCase(
        "q", "opaque-id", "product", "dev", "search", [], "cozy autumn mystery"
    )
    judge = StubJudgeAdapter(auto_qualify=True)
    original = judge.judge_pair
    queries = []

    def record(inp):
        queries.append(inp.query)
        return original(inp)

    monkeypatch.setattr(judge, "judge_pair", record)
    runner.run_candidate_pilot(judge, ["opaque-id"], catalog_cases=[case])
    assert queries[:5] == [case.query] * 5
    with pytest.raises(ValueError, match="No real query"):
        runner.run_candidate_pilot(judge, ["nonexistent"], catalog_cases=[case])


def test_runtime_failure_stops_without_fabricating_completed_pilot(monkeypatch):
    runner = JudgeQualificationRunner()
    evidence = ItemEvidence(TypedId("movie", 1), "Film", "Full synopsis")
    monkeypatch.setattr(
        runner, "build_pilot_items_for_family", lambda *a, **k: [evidence] * 20
    )
    judge = StubJudgeAdapter(simulate_error=True)
    report = runner.run_candidate_pilot(judge, ["family"])
    assert report["stopped_early"]
    assert report["total_judgments"] == 1
    assert report["execution_failures"] == 1
    assert not report["qualified"]
