"""Unit tests for Milestone 0 judge qualification and hardware inspection."""

from evaluation.judge.qualification import (
    JudgeQualificationRunner,
    inspect_local_hardware,
)
from evaluation.judge.stub import StubJudgeAdapter


def test_hardware_inspection():
    hw = inspect_local_hardware()
    assert "CPU" in hw["devices"]
    assert hw["ram_gb"] >= 16.0
    assert isinstance(hw["gpu_available"], bool)
    assert isinstance(hw["npu_available"], bool)
    assert isinstance(hw["openvino_devices"], list)


def test_judge_qualification_runner():
    runner = JudgeQualificationRunner()
    # Stub with auto_qualify=True achieves >=99% repeatability, >=95% perm, >=95% control accuracy
    qualified_stub = StubJudgeAdapter(
        model_name="bespoke-nimble-9b",
        auto_qualify=True,
    )

    pilot_report = runner.run_candidate_pilot(
        qualified_stub,
        query_families=[f"fam_{i}" for i in range(20)],
        run_option_order_diagnostic=True,
    )
    assert pilot_report["repeatability_pass"] is True
    assert pilot_report["option_permutation_pass"] is True
    assert pilot_report["control_accuracy_pass"] is True
    assert pilot_report["qualified"] is True

    # Test disqualified stub
    unqualified_stub = StubJudgeAdapter(model_name="failing-stub", fixed_grade=1)
    fail_report = runner.run_candidate_pilot(
        unqualified_stub,
        query_families=["fam_1"],
    )
    # Fixed grade 1 will fail control traps (expected grade 0)
    assert fail_report["qualified"] is False


def test_judge_selection_never_falls_back_to_another_model_or_stub():
    runner = JudgeQualificationRunner()
    candidates_map = {
        "bespoke-nimble-9b": StubJudgeAdapter("bespoke-nimble-9b"),
        "unused-model": StubJudgeAdapter("unused-model"),
        "stub": StubJudgeAdapter("stub"),
    }

    # Only Nimble can be selected, even if another model has a passing record.
    def report(key):
        return {
            "qualified": True,
            "fingerprint": candidates_map[key].qualification_fingerprint(),
            "qualification_protocol": "v2.8",
            "scope": "qualification",
            "pilot_pairs": 400,
            "pilot_distinct_pairs": 400,
            "pilot_families": 20,
            "repeat_tests": 100,
            "permutation_tests": 100,
            "control_tests": 100,
            "repeatability_rate": 1.0,
            "option_permutation_rate": 1.0,
            "control_accuracy": 1.0,
            "execution_failures": 0,
            "malformed_count": 0,
        }

    all_qualified = {k: report(k) for k in candidates_map}
    p, s, t, mode = runner.select_judge_panel(all_qualified, candidates_map)
    assert p.model_name == "bespoke-nimble-9b"
    assert s is t is None
    assert mode == "single_judge"

    # An unrelated model cannot substitute for unqualified Nimble.
    partial_qualified = {
        "unused-model": report("unused-model"),
    }
    p, s, t, mode = runner.select_judge_panel(partial_qualified, candidates_map)
    assert p is s is t is None
    assert mode == "unqualified"

    # No qualification cannot fabricate a selected production judge.
    none_qualified = {}
    p, s, t, mode = runner.select_judge_panel(none_qualified, candidates_map)
    assert p is s is t is None
    assert mode == "unqualified"


def test_judge_qualification_catalog_cases_and_unavailable(monkeypatch):
    from unittest.mock import MagicMock

    runner = JudgeQualificationRunner()

    # Test build_pilot_items_for_family with catalog cases
    fake_case = MagicMock()
    fake_case.family_id = "fam_1"
    fake_case.golden_set = [
        {"id": 42, "title": "Catalog Hit", "synopsis": "Hit synopsis"}
    ]
    from evaluation.evidence import index_catalog_metadata
    from evaluation.judge import qualification

    original_catalog_loader = qualification.load_catalog_metadata

    monkeypatch.setattr(
        "evaluation.judge.qualification.load_catalog_metadata",
        lambda: index_catalog_metadata(
            [
                {
                    "tmdb_id": 42,
                    "media_type": "movie",
                    "title": "Actual Catalog Title",
                    "overview": "Actual catalog synopsis",
                }
            ]
        ),
    )
    items = runner.build_pilot_items_for_family(
        "fam_1", count=1, catalog_cases=[fake_case]
    )
    assert len(items) == 1
    assert items[0].title == "Actual Catalog Title"
    assert items[0].synopsis == "Actual catalog synopsis"
    assert items[0].typed_id.id == 42
    monkeypatch.setattr(qualification, "load_catalog_metadata", original_catalog_loader)

    # Test unavailable judge pilot reporting
    unavail_judge = MagicMock()
    unavail_judge.is_available.return_value = False
    report = runner.run_candidate_pilot(unavail_judge, query_families=["fam_1"])
    assert report["qualified"] is False
    assert report["repeatability_pct"] == 0.0

    # Test malformed and retry logic in pilot
    def mock_out(grade: int, status: str):
        m = MagicMock()
        m.grade = grade
        m.execution_status = status
        return m

    flaky_judge = MagicMock()
    flaky_judge.is_available.return_value = True
    flaky_judge.judge_pair.side_effect = [
        mock_out(1, "malformed"),
        mock_out(0, "failed"),
        mock_out(0, "malformed"),
        mock_out(0, "abstain"),
    ] + [mock_out(2, "success")] * 200
    flaky_judge.test_option_order_permutation.side_effect = RuntimeError(
        "Permutation error"
    )

    flaky_report = runner.run_candidate_pilot(flaky_judge, query_families=["fam_1"])
    assert flaky_report["malformed_count"] > 0
    assert flaky_report["abstention_count"] > 0
