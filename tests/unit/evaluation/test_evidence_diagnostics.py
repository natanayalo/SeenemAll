"""Context delivery, immutable evidence identities and sufficiency diagnostics."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from evaluation import evidence, evidence_assessment
from evaluation.judge.consensus import JudgmentCache, PoolAdjudicator
from evaluation.judge.qualification import (
    JudgeQualificationRunner,
    qualification_record_matches,
)
from evaluation.judge.stub import StubJudgeAdapter
from evaluation.models import ItemEvidence, JudgeInput, TypedId
from tests.unit.evaluation.judge_helpers import make_nimble
from tests.unit.evaluation.test_systemone_contract import reply, response
from tests.unit.evaluation.test_qualification_contract import valid_record


def test_legacy_evidence_and_snapshot_fingerprints_are_preserved():
    snapshot = json.loads(
        Path("evaluation/baseline_v2.json").read_text(encoding="utf-8")
    )
    catalog = evidence.load_catalog_metadata(
        Path("evaluation/fixtures/catalog_evidence_v2.2.json")
    )
    for row in snapshot["per_query_results"]:
        for record in row["judgments"]:
            tid = TypedId.parse(record["typed_id"])
            if str(tid) in catalog:
                assert (
                    evidence.build_item_evidence(tid, catalog[str(tid)]).content_hash()
                    == record["provenances"][0]["evidence_hash"]
                )


def test_contextual_facts_are_sourced_country_specific_and_not_absence():
    catalog = evidence.load_catalog_metadata(evidence_version="v2.3")
    item = evidence.build_item_evidence(TypedId("movie", 8392), catalog["movie:8392"])
    facts = item.contextual_facts
    assert facts["availability"]["country"] == "IL"
    assert facts["availability"]["captured_at"]
    assert facts["availability"]["source"].endswith("/8392/watch/providers")
    assert any(p["provider_name"] == "Netflix" for p in facts["availability"]["offers"])
    assert "unlisted providers remain unknown" in facts["availability"]["coverage"]
    assert any(p["name"] == "Studio Ghibli" for p in facts["studios"]["companies"])
    assert "Studio Ghibli" in item.to_evidence_text()
    assert facts["awards"] is None
    other = evidence.build_item_evidence(TypedId("movie", 13), catalog["movie:13"])
    assert other.contextual_facts["awards"]["wins"][0]["category"] == "Best Picture"


def test_missing_context_and_empty_offers_remain_unknown():
    catalog = evidence.load_catalog_metadata(evidence_version="v2.3")
    absent = next(
        row
        for row in catalog.values()
        if row["_contextual_facts"]["availability"] is None
    )
    assert absent["_contextual_facts"]["availability"] is None
    item = evidence.pool_item_evidence(
        TypedId("tv", 9999999),
        {"watch_options": [{"service": "Netflix"}]},
        catalog,
        evidence_version="v2.3",
    )
    assert item.evidence_version == "v2.3"
    assert item.contextual_facts == {
        "availability": None,
        "studios": None,
        "awards": None,
    }
    assert "Netflix" not in item.to_evidence_text()
    empty = next(
        row
        for row in catalog.values()
        if row["_contextual_facts"]["availability"]
        and not row["_contextual_facts"]["availability"]["offers"]
    )
    assert "unknown" in empty["_contextual_facts"]["availability"]["coverage"]


@pytest.mark.parametrize("version", ["invalid", "v2.1"])
def test_unsupported_profile_fails(version):
    with pytest.raises(ValueError, match="Unsupported"):
        evidence.contextualize_catalog({}, version)


def test_incompatible_context_fixture_fails(tmp_path, monkeypatch):
    path = tmp_path / "context.json"
    path.write_text(json.dumps({"evidence_version": "wrong"}))
    monkeypatch.setattr(evidence, "CONTEXT_EVIDENCE_PATH", path)
    with pytest.raises(ValueError, match="version mismatch"):
        evidence.contextualize_catalog({}, "v2.3")


def test_profile_has_distinct_cache_and_qualification_identity():
    judge = make_nimble()
    original = judge.qualification_fingerprint()
    judge.evidence_version = "v2.3"
    assert judge.qualification_fingerprint() != original
    assert not qualification_record_matches(
        judge, {"qualified": True, "fingerprint": original}
    )
    inp = JudgeInput("movie", ItemEvidence(TypedId("movie", 1), "Film"))
    with pytest.raises(ValueError, match="versions must match"):
        judge.judge_pair(inp)
    assert judge.input_character_limit() == 8192
    judge.evidence_version = None
    assert judge.input_character_limit() == 4096


@pytest.mark.parametrize("probability", [0.0, 0.49, 0.5, 1.0])
def test_native_probability_survives_output_cache_and_adjudication(
    tmp_path, probability
):
    judge = make_nimble()
    inp = JudgeInput("movie", ItemEvidence(TypedId("movie", 1), "Film"))
    payload = reply([0, 0, 1, 0])
    payload["answers"]["evidence_sufficient"]["noul"] = probability
    cache = JudgmentCache(tmp_path / "cache.json")
    adjudicator = PoolAdjudicator(judge, cache=cache)
    with patch.object(judge, "is_available", return_value=True), patch(
        "urllib.request.urlopen", return_value=response(payload)
    ) as service:
        first = adjudicator.adjudicate_pair(
            inp.query, inp.evidence, mode="single_judge"
        )
        second = adjudicator.adjudicate_pair(
            inp.query, inp.evidence, mode="single_judge"
        )
    assert service.call_count == 1
    assert (
        first.evidence_sufficiency_probabilities
        == second.evidence_sufficiency_probabilities
        == [probability]
    )
    assert (first.grade is not None) == (probability >= 0.5)
    assert cache.get(inp, judge)["evidence_sufficiency_probability"] == probability


def test_legacy_cache_probability_is_unknown_not_invented(tmp_path):
    judge = StubJudgeAdapter("test", fixed_grade=2)
    inp = JudgeInput("movie", ItemEvidence(TypedId("movie", 1), "Film"))
    cache = JudgmentCache(tmp_path / "cache.json")
    old = judge.judge_pair(inp).to_dict()
    old.pop("evidence_sufficiency_probability")
    cache.set(inp, judge, old)
    record = PoolAdjudicator(judge, cache=cache).adjudicate_pair(
        inp.query, inp.evidence
    )
    assert record.evidence_sufficiency_probabilities == [None]


def test_qualification_pilot_uses_selected_profile(monkeypatch):
    row = {
        "media_type": "movie",
        "tmdb_id": 1,
        "title": "Film",
        "overview": "A journey.",
        "_evidence_version": "v2.3",
        "_contextual_facts": {"studios": None},
    }
    monkeypatch.setattr(
        "evaluation.judge.qualification.load_catalog_metadata",
        lambda **kw: {"movie:1": row},
    )
    from types import SimpleNamespace

    item = JudgeQualificationRunner(
        evidence_version="v2.3"
    ).build_pilot_items_for_family(
        "family",
        1,
        [SimpleNamespace(family_id="family", golden_set=["movie:1"], golden_ids=None)],
    )[
        0
    ]
    assert item.evidence_version == "v2.3"


@pytest.mark.parametrize("repeats", [0, 4])
def test_assessment_rejects_invalid_workload(tmp_path, repeats):
    with pytest.raises(ValueError, match="repeats"):
        evidence_assessment.assess(tmp_path / "report.json", repeats)


def test_assessment_is_uncached_and_expectations_stay_out_of_inputs(
    tmp_path, monkeypatch
):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "id",
                        "query": "journey",
                        "typed_id": "movie:1",
                        "expected_sufficient": {"v2.2": True, "v2.3": False},
                        "minimum_grade": 2,
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(evidence_assessment, "CASES_PATH", path)
    monkeypatch.setattr(
        evidence_assessment,
        "load_catalog_metadata",
        lambda **kw: {
            "movie:1": {"title": "Film", "_evidence_version": kw["evidence_version"]}
        },
    )
    judges = []

    def discover(version):
        judge = StubJudgeAdapter("test", fixed_grade=2)
        judge.evidence_version = version
        judges.append(judge)
        return {"bespoke-nimble-9b": judge}

    monkeypatch.setattr(evidence_assessment, "discover_ollama_judges", discover)
    destination = tmp_path / "report.json"
    report = evidence_assessment.assess(destination, 2)
    assert report["native_requests"] == 4 and not report["production_cache_used"]
    assert report["summary"]["v2.2"]["correct"] == 2
    assert report["summary"]["v2.3"]["false_sufficiency"] == 2
    manifest = json.loads(destination.with_suffix(".inputs.json").read_text())
    assert all(
        "expected" not in row and "minimum_grade" not in row["evidence_text"]
        for row in manifest
    )
    with pytest.raises(ValueError, match="immutable"):
        evidence_assessment.assess(destination)
    monkeypatch.setattr(
        "sys.argv", ["assess", "--output", str(tmp_path / "second.json")]
    )
    assert evidence_assessment.main() == 1
    monkeypatch.setattr(StubJudgeAdapter, "is_available", lambda self: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        evidence_assessment.assess(tmp_path / "third.json")


@pytest.mark.parametrize(
    "controls,accuracy,valid",
    [
        (14, 1.0, True),
        (14, 13 / 14, False),
        (13, 1.0, False),
        (True, 1.0, False),
        (14, True, False),
        (14, float("nan"), False),
    ],
)
def test_enriched_qualification_requires_focused_sufficiency_controls(
    controls, accuracy, valid
):
    judge = make_nimble()
    judge.evidence_version = "v2.3"
    record = valid_record(judge)
    record.update(
        sufficiency_control_cases=controls, sufficiency_control_accuracy=accuracy
    )
    assert qualification_record_matches(judge, record) is valid


@pytest.mark.parametrize("false_sufficiency", [False, True])
def test_new_controls_are_executed_in_the_enriched_qualification(
    monkeypatch, false_sufficiency
):
    from types import SimpleNamespace
    from evaluation.models import JudgeOutput

    cases = json.loads(evidence_assessment.CASES_PATH.read_text())["cases"]
    catalog = evidence.load_catalog_metadata(
        Path("evaluation/fixtures/catalog_evidence_v2.2.json"), evidence_version="v2.3"
    )
    monkeypatch.setattr(
        "evaluation.judge.qualification.load_catalog_metadata", lambda **kw: catalog
    )
    runner = JudgeQualificationRunner(evidence_version="v2.3")
    monkeypatch.setattr(
        runner,
        "build_pilot_items_for_family",
        lambda *a, **kw: [
            evidence.build_item_evidence(TypedId("movie", 603), catalog["movie:603"])
        ],
    )
    judge = StubJudgeAdapter("test", fixed_grade=2)
    judge.evidence_version = "v2.3"

    def judge_pair(inp):
        case = next(
            (
                c
                for c in cases
                if c["query"] == inp.query
                and c["typed_id"] == str(inp.evidence.typed_id)
            ),
            None,
        )
        expected = case["expected_sufficient"]["v2.3"] if case else True
        if false_sufficiency:
            expected = True
        grade = case.get("maximum_grade", 2) if case else 2
        return JudgeOutput(
            grade,
            {grade: 1.0},
            expected,
            "success",
            judge.get_provenance(inp.evidence.content_hash()),
        )

    monkeypatch.setattr(judge, "judge_pair", judge_pair)
    report = runner.run_candidate_pilot(
        judge,
        ["family"],
        control_cases=[],
        catalog_cases=[SimpleNamespace(family_id="family", query="journey")],
        items_per_family=1,
    )
    assert report["sufficiency_control_cases"] == 14
    assert report["sufficiency_control_accuracy"] == (
        12 / 14 if false_sufficiency else 1.0
    )
    assert report["sufficiency_control_pass"] is (not false_sufficiency)
    assert not report[
        "qualified"
    ]  # Full 400-pair / repeat / control gates still apply.
