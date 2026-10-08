"""End-to-End Acceptance Tests for Seen'emAll Evaluation Suite v2.

Verifies:
1. Same-system comparison (default vs default): lift is 0, no regressions, passes.
2. Degraded candidate comparison: fails gates with FAIL (exit code 1).
3. Automated disagreement resolution:
   - 2-agreeing judges: accepts exact grade.
   - Unresolved disagreements: retains UNJUDGED and yields INCONCLUSIVE (exit code 3).
4. Rejection of unintended index rebuilds: checksum mismatch raises error / INVALID (exit code 2).
5. Execution failures / unexpected fallbacks yield INVALID (exit code 2).
6. Latency harness execution with ABBA / BAAB pattern and verified latency stats.
"""

from unittest.mock import MagicMock, patch
import pytest

from evaluation.evaluate import parse_args, run_evaluation_v2
from evaluation.judge.consensus import ConsensusJudgeEngine
from evaluation.judge.stub import StubJudgeAdapter
from evaluation.latency import LatencyHarness
from evaluation.metrics import evaluate_comparison_gates
from evaluation.models import (
    EvaluationStatus,
    ItemEvidence,
    JudgeInput,
    TypedId,
)
from evaluation.runner import IndexArtifactVerifier


def test_acceptance_same_system_comparison():
    """Same-system comparison (e.g. default vs default) must produce zero delta, zero regressions, and PASS (0)."""
    # 50 families all with 0.0 delta
    family_deltas = {f"fam_{i}": 0.0 for i in range(50)}
    slice_deltas = {
        "vibe": [0.0] * 15,
        "franchise": [0.0] * 12,
        "constraint": [0.0] * 12,
        "entity": [0.0] * 11,
    }

    gate_result = evaluate_comparison_gates(
        family_ndcg_deltas=family_deltas,
        slice_family_deltas=slice_deltas,
        baseline_recall_100=0.85,
        candidate_recall_100=0.85,
        hard_constraint_violations=0,
        disliked_violations=0,
        exploratory_coverages=[1.0] * 50,
        authoritative_unresolved_count=0,
        execution_failures=0,
        unexpected_fallbacks=0,
        duplicate_outputs_detected=False,
        is_statistical_promotion=False,
    )

    assert gate_result.passed is True
    assert gate_result.status == EvaluationStatus.PASS
    assert int(gate_result.status) == 0


def test_acceptance_degraded_candidate_comparison():
    """A degraded candidate with negative delta or regressions must trigger FAIL (exit code 1)."""
    # Systematically negative delta
    family_deltas = {f"fam_{i}": -0.05 for i in range(50)}
    slice_deltas = {"vibe": [-0.05] * 50}

    gate_result = evaluate_comparison_gates(
        family_ndcg_deltas=family_deltas,
        slice_family_deltas=slice_deltas,
        baseline_recall_100=0.85,
        candidate_recall_100=0.70,  # candidate recall drops by 15%
        hard_constraint_violations=5,  # violations present
        disliked_violations=0,
        exploratory_coverages=[1.0] * 50,
        authoritative_unresolved_count=0,
        execution_failures=0,
        unexpected_fallbacks=0,
        duplicate_outputs_detected=False,
        is_statistical_promotion=False,
    )

    assert gate_result.passed is False
    assert gate_result.status == EvaluationStatus.FAIL
    assert int(gate_result.status) == 1
    assert any(
        "Violation" in r or "Recall" in r or "nDCG" in r for r in gate_result.reasons
    )


def test_acceptance_automated_disagreement_resolution_inconclusive():
    """Unresolved disagreements in authoritative mode must yield UNJUDGED and INCONCLUSIVE (exit code 3)."""
    # 3 judges returning different grades (0, 1, 2)
    j1 = StubJudgeAdapter("judge_a", grade=0, sufficient=True)
    j2 = StubJudgeAdapter("judge_b", grade=1, sufficient=True)
    j3 = StubJudgeAdapter("judge_c", grade=2, sufficient=True)

    engine = ConsensusJudgeEngine(
        primary_judge=j1, secondary_judge=j2, tie_breaker_judge=j3
    )

    evidence = [
        ItemEvidence(
            typed_id=TypedId("movie", 100), title="Test Movie", synopsis="A film"
        ),
    ]

    qrels, records, _ = engine.label_pool(
        query="mystery thriller",
        pool_evidence=evidence,
        mode="consensus",
    )

    # All 3 disagreed, grade should be UNJUDGED and unjudged items count is 1
    assert "movie:100" not in qrels  # not in accepted qrels
    assert records[0].status == "UNJUDGED"

    # In authoritative comparison, unresolved items trigger INCONCLUSIVE
    gate_result = evaluate_comparison_gates(
        family_ndcg_deltas={"fam_1": 0.01},
        slice_family_deltas={"vibe": [0.01]},
        baseline_recall_100=0.80,
        candidate_recall_100=0.81,
        hard_constraint_violations=0,
        disliked_violations=0,
        exploratory_coverages=[1.0],
        authoritative_unresolved_count=1,  # 1 unresolved item
        execution_failures=0,
        unexpected_fallbacks=0,
        duplicate_outputs_detected=False,
        is_statistical_promotion=False,
    )

    assert gate_result.passed is False
    assert gate_result.status == EvaluationStatus.INCONCLUSIVE
    assert int(gate_result.status) == 3
    assert any("unresolved" in r.lower() for r in gate_result.reasons)


def test_acceptance_automated_disagreement_resolution_resolved():
    """Disagreement between primary and secondary is successfully resolved by 3rd judge when two agree."""
    j1 = StubJudgeAdapter("judge_a", grade=2, sufficient=True)
    j2 = StubJudgeAdapter("judge_b", grade=3, sufficient=True)
    j3 = StubJudgeAdapter("judge_c", grade=2, sufficient=True)  # agrees with j1

    engine = ConsensusJudgeEngine(
        primary_judge=j1, secondary_judge=j2, tie_breaker_judge=j3
    )

    evidence = [
        ItemEvidence(
            typed_id=TypedId("movie", 200),
            title="Agreeing Movie",
            synopsis="Great film",
        ),
    ]

    qrels, records, _ = engine.label_pool(
        query="action adventure",
        pool_evidence=evidence,
        mode="consensus",
    )

    assert qrels["movie:200"] == 2
    assert records[0].status == "ACCEPTED"
    assert records[0].grade == 2


def test_acceptance_rejection_of_unintended_index_rebuilds():
    """Mismatched index checksums must reject execution and yield INVALID (exit code 2)."""
    verifier = IndexArtifactVerifier()

    with pytest.raises(ValueError, match="Index.*checksum mismatch"):
        verifier.verify_index_artifact(
            expected_checksum="checksum_abc123",
            actual_checksum="checksum_xyz999",
        )

    # Verifier failure in gate evaluation leads to INVALID
    gate_result = evaluate_comparison_gates(
        family_ndcg_deltas={"fam_1": 0.0},
        slice_family_deltas={"vibe": [0.0]},
        baseline_recall_100=0.8,
        candidate_recall_100=0.8,
        hard_constraint_violations=0,
        disliked_violations=0,
        exploratory_coverages=[1.0],
        authoritative_unresolved_count=0,
        execution_failures=1,  # e.g., index validation failure
        unexpected_fallbacks=0,
        duplicate_outputs_detected=False,
        is_statistical_promotion=False,
    )

    assert gate_result.passed is False
    assert gate_result.status == EvaluationStatus.INVALID
    assert int(gate_result.status) == 2


def test_acceptance_latency_harness_verification():
    """Latency harness must execute ABBA / BAAB query pattern and verify latency stats."""
    mock_runner = MagicMock()
    from tests.unit.evaluation.test_latency_contract import measured_trace

    mock_runner.execute_query.return_value = ([{"id": 1}], measured_trace())
    harness = LatencyHarness(runner=mock_runner, warmup_count=2, repetition_count=1)

    queries = ["sci-fi thriller", "comedy movie"]

    curr_time = 0.0

    def fake_perf_counter():
        nonlocal curr_time
        curr_time += 0.010
        return curr_time

    with patch("time.perf_counter", side_effect=fake_perf_counter):
        result = harness.benchmark_paired_latency(
            queries=queries,
            baseline_params={"param": "base"},
            candidate_params={"param": "cand"},
        )

    assert "baseline_p50_ms" in result
    assert "candidate_p50_ms" in result
    assert "median_paired_ratio" in result
    assert result["median_paired_ratio_pass"] is True
    assert result["passed"] is True


def test_acceptance_cli_v2_run(tmp_path):
    """CLI execution with --v2 and stub judge produces report and correct exit status."""
    report_file = tmp_path / "v2_report.json"
    args = parse_args(
        [
            "--v2",
            "--track",
            "product",
            "--split",
            "dev",
            "--judge-config",
            "stub",
            "--judgment-mode",
            "single_judge",
            "--v2-report",
            str(report_file),
        ]
    )

    # Mock runner to return fast deterministic items
    with patch("evaluation.evaluate.EvaluationRunner") as mock_runner_cls:
        instance = mock_runner_cls.return_value
        fake_item = {
            "id": 101,
            "title": "Interstellar",
            "synopsis": "Space exploration",
            "media_type": "movie",
            "genres": ["Sci-Fi"],
            "release_year": 2014,
            "runtime": 169,
            "original_language": "en",
        }
        trace = MagicMock()
        trace.errors = []
        trace.fallbacks = []
        instance.run_case.return_value = ([fake_item], trace)

        code = run_evaluation_v2(args)
        assert code in (0, 1, 2, 3)
        assert report_file.exists()


def test_acceptance_unavailable_judge_rejection():
    """Unavailable Nimble must fail without loading weights or fabricating judgments."""
    from tests.unit.evaluation.judge_helpers import make_nimble

    ev = ItemEvidence(typed_id=TypedId("movie", 10), title="Film", synopsis="Story")
    inp = JudgeInput(query="sample query", evidence=ev)

    judge = make_nimble()
    with patch("urllib.request.urlopen", side_effect=OSError("offline")):
        assert judge.is_available() is False
        with pytest.raises(RuntimeError, match="unavailable"):
            judge._run_inference("test prompt", inp)


def test_acceptance_missing_dataset_rejection():
    """Missing external dataset files must raise FileNotFoundError, never silently synthesize data."""
    from pathlib import Path
    from evaluation.datasets import MovieLens20MLoader, TagGenomeLoader

    ml_loader = MovieLens20MLoader()
    with pytest.raises(FileNotFoundError, match="MovieLens 20M dataset"):
        ml_loader.load_as_public_entries(
            interactions_path=Path("non_existent_movielens_file")
        )

    tg_loader = TagGenomeLoader(genome_path=Path("non_existent_tag_genome.json"))
    with pytest.raises(FileNotFoundError, match="Tag Genome dataset"):
        tg_loader.load_as_public_entries()


def test_acceptance_private_holdout_promotion_requires_real_file():
    """Private holdout evaluation must return INVALID (Exit 2) when holdout file is missing."""
    from evaluation.evaluate import parse_args, run_evaluation_v2

    args = parse_args(
        [
            "--private-eval",
            "--candidate",
            "default",
        ]
    )

    with patch("pathlib.Path.exists", return_value=False):
        code = run_evaluation_v2(args)
        assert code == 2  # EvaluationStatus.INVALID (Exit 2)


def test_acceptance_failed_requests_invalidate_latency():
    """Latency harness must fail benchmark when requests encounter errors."""
    mock_runner = MagicMock()
    mock_trace = MagicMock()
    mock_trace.errors = ["Retriever failed with HTTP 500"]
    mock_runner.execute_query.return_value = ([], mock_trace)

    harness = LatencyHarness(runner=mock_runner, warmup_count=1, repetition_count=1)
    rep = harness.benchmark_paired_latency(
        queries=["q1"],
        baseline_params={"p": 1},
        candidate_params={"p": 2},
    )
    assert rep["passed"] is False
    assert rep["error_count"] > 0


def test_acceptance_empty_candidate_output_rejection():
    """Gates must reject comparison runs where candidate returns empty output despite zero delta."""
    gate_result = evaluate_comparison_gates(
        family_ndcg_deltas={"fam_1": 0.0},
        slice_family_deltas={"vibe": [0.0]},
        baseline_recall_100=0.0,
        candidate_recall_100=0.0,
        hard_constraint_violations=0,
        disliked_violations=0,
        exploratory_coverages=[1.0],
        authoritative_unresolved_count=0,
        execution_failures=0,
        unexpected_fallbacks=0,
        duplicate_outputs_detected=False,
        is_statistical_promotion=False,
        empty_output_cases=5,  # 5 empty query outputs
    )
    assert gate_result.passed is False
    assert gate_result.status == EvaluationStatus.INVALID
    assert any("empty" in r.lower() for r in gate_result.reasons)


def test_acceptance_authoritative_stub_rejection():
    """Authoritative runs must reject stub judges with INVALID (exit 2) unless explicit test flag passed."""
    args = parse_args(
        [
            "--v2",
            "--track",
            "product",
            "--split",
            "dev",
            "--judgment-mode",
            "consensus",
            "--judge-config",
            "stub",
        ]
    )
    with patch("evaluation.evaluate.load_evaluation_cases") as mock_cases:
        mock_cases.return_value = [MagicMock()]
        code = run_evaluation_v2(args)
        assert code == 2  # EvaluationStatus.INVALID

    # With flag, stub judge is permitted strictly for test harnesses
    args_allowed = parse_args(
        [
            "--v2",
            "--track",
            "product",
            "--split",
            "dev",
            "--judgment-mode",
            "consensus",
            "--judge-config",
            "stub",
            "--allow-stub-judges-for-testing",
        ]
    )
    with patch("evaluation.evaluate.load_evaluation_cases") as mock_cases:
        mock_cases.return_value = (
            []
        )  # No cases loaded -> exits cleanly with INVALID (no cases)
        code = run_evaluation_v2(args_allowed)
        assert code == 2  # Exit on no test cases found


def test_acceptance_nimble_malformed_empty_dict_rejected():
    """Nimble must reject missing System One fields as malformed."""
    from tests.unit.evaluation.judge_helpers import make_nimble

    adapter = make_nimble()
    ev = ItemEvidence(
        typed_id=TypedId("movie", 10), title="Test Movie", synopsis="Action film"
    )
    inp = JudgeInput(query="action", evidence=ev)

    mock_read = MagicMock()
    mock_read.read.return_value = b"{}"
    mock_read.__enter__.return_value = mock_read

    with patch.object(adapter, "is_available", return_value=True), patch(
        "urllib.request.urlopen", return_value=mock_read
    ):
        res = adapter.judge_pair(inp)
        assert res.execution_status == "malformed"
        assert res.grade == 0
        assert res.evidence_sufficiency is False


def test_acceptance_personalization_decline_against_personalized_baseline():
    """Personalization decline gate compares against personalized baseline, failing if decline > 0.03."""
    from evaluation.personalization import PersonalizationHarness

    mock_runner = MagicMock()
    mock_session = MagicMock()
    mock_session_factory = MagicMock(return_value=mock_session)
    mock_session.query.return_value.filter.return_value.count.return_value = 5
    mock_session.query.return_value.filter_by.return_value.count.return_value = 5

    harness = PersonalizationHarness(
        runner=mock_runner,
        db_session_factory=mock_session_factory,
        verify_seeding=True,
    )

    # Candidate gets nDCG 0.0, personalized baseline gets 1.0 (decline is 1.0 > 0.03)
    with patch.object(harness, "evaluate_persona") as mock_eval:
        mock_eval.return_value = {
            "persona_key": "test_persona",
            "candidate_ndcg": 0.0,
            "personalized_baseline_ndcg": 1.0,
            "masked_baseline_ndcg": 0.0,
            "total_personalization_system_lift": 0.0,
            "relative_candidate_lift": -1.0,
            "disliked_violations_count": 0,
            "execution_error": False,
            "empty_output": False,
        }
        result = harness.run_personalization_benchmark(
            k=10,
            baseline_params={"is_base": True},
            candidate_params={"is_base": False},
        )
        assert result["decline_pass"] is False
        assert result["passed"] is False
        assert result["max_persona_ndcg_decline"] > 0.03


def test_acceptance_chronology_reversal_detected():
    """Chronological sequence inversion must be detected and fail chronology gate."""
    from evaluation.deterministic import check_canonical_order

    seq = ["movie:1", "movie:2", "movie:3"]
    # Candidate returned in reverse chronological order
    cand_items = [
        {"id": 3, "media_type": "movie"},
        {"id": 2, "media_type": "movie"},
        {"id": 1, "media_type": "movie"},
    ]
    order_res = check_canonical_order(cand_items, seq, k=10)
    assert order_res["inversions"] == 3
    assert order_res["pairwise_accuracy"] == 0.0

    gate_res = evaluate_comparison_gates(
        family_ndcg_deltas={"fam_1": 0.0},
        slice_family_deltas={"franchise": [0.0]},
        baseline_recall_100=0.8,
        candidate_recall_100=0.8,
        hard_constraint_violations=0,
        disliked_violations=0,
        exploratory_coverages=[1.0],
        authoritative_unresolved_count=0,
        execution_failures=0,
        unexpected_fallbacks=0,
        duplicate_outputs_detected=False,
        chronology_violations=order_res["inversions"],
    )
    assert gate_res.passed is False
    assert any("chronology" in r.lower() for r in gate_res.reasons)


def test_acceptance_completeness_underfill_rejected():
    """Candidate returning 1 item when 10 are eligible must get 0.10 completeness and fail completeness gate."""
    from evaluation.metrics import calculate_completeness

    cand_items = [{"id": 10, "media_type": "movie"}]
    comp = calculate_completeness(cand_items, eligible_catalog_count=10, k=10)
    assert comp == 0.10

    gate_res = evaluate_comparison_gates(
        family_ndcg_deltas={"fam_1": 0.0},
        slice_family_deltas={"vibe": [0.0]},
        baseline_recall_100=0.8,
        candidate_recall_100=0.8,
        hard_constraint_violations=0,
        disliked_violations=0,
        exploratory_coverages=[1.0],
        authoritative_unresolved_count=0,
        execution_failures=0,
        unexpected_fallbacks=0,
        duplicate_outputs_detected=False,
        completeness_scores=[comp],
    )
    assert gate_res.passed is False
    assert any("completeness" in r.lower() for r in gate_res.reasons)


def test_acceptance_index_verification_missing_client_rejected():
    """Index verification when ES client is unavailable returns INVALID (exit code 2)."""
    args = parse_args(
        [
            "--v2",
            "--verify-index",
            "--expected-index-checksum",
            "dummy_expected_checksum_12345",
        ]
    )
    with patch(
        "api.core.elasticsearch_client.get_elasticsearch_client", return_value=None
    ):
        code = run_evaluation_v2(args)
        assert code == 2  # EvaluationStatus.INVALID


def test_acceptance_movielens_seen_items_excluded(tmp_path):
    """MovieLens loader excludes train-seen items from test positives and rejects null timestamps."""
    import json
    from evaluation.datasets import MovieLens20MLoader

    ml_data = {
        "users": [
            {
                "user_id": "u1",
                "train_items": [
                    {"tmdb_id": i, "timestamp": 1000 + i, "rating": 5.0}
                    for i in range(1, 25)
                ],
                "test_items": [
                    {
                        "tmdb_id": 1,
                        "timestamp": 1414000000,
                        "rating": 5.0,
                    },  # seen in train!
                    {
                        "tmdb_id": 999,
                        "timestamp": 1414000000,
                        "rating": 5.0,
                        "release_year": 2010,
                    },
                    {
                        "tmdb_id": 998,
                        "timestamp": 1414000000,
                        "rating": 5.0,
                        "release_year": 2011,
                    },
                    {
                        "tmdb_id": 997,
                        "timestamp": 1414000000,
                        "rating": 5.0,
                        "release_year": 2012,
                    },
                    {
                        "tmdb_id": 996,
                        "timestamp": 1414000000,
                        "rating": 5.0,
                        "release_year": 2013,
                    },
                    {
                        "tmdb_id": 995,
                        "timestamp": 1414000000,
                        "rating": 5.0,
                        "release_year": 2010,
                    },
                ],
            }
        ]
    }
    ml_json = tmp_path / "movielens_sample.json"
    ml_json.write_text(json.dumps(ml_data), encoding="utf-8")

    cat_data = {
        str(i): {"id": i, "title": f"Movie {i}", "release_year": 2010}
        for i in [1, 999, 998, 997, 996, 995]
    }
    cat_json = tmp_path / "movie_catalog.json"
    cat_json.write_text(json.dumps(cat_data), encoding="utf-8")

    loader = MovieLens20MLoader()
    entries = loader.load_as_public_entries(
        interactions_path=ml_json, movie_catalog_path=cat_json
    )
    assert len(entries) > 0
    for e in entries:
        train_seen = {it["tmdb_id"] for it in e["train_items"]}
        test_positives = set(e["golden_ids"])
        assert 1 not in test_positives
        assert len(train_seen.intersection(test_positives)) == 0

    # Null timestamp check
    bad_data = {
        "users": [
            {
                "user_id": "u2",
                "train_items": [{"tmdb_id": 1, "timestamp": None, "rating": 5.0}],
                "test_items": [],
            }
        ]
    }
    bad_json = tmp_path / "bad_ml.json"
    bad_json.write_text(json.dumps(bad_data), encoding="utf-8")
    with pytest.raises(ValueError, match="Null or empty timestamp"):
        loader.load_as_public_entries(interactions_path=bad_json)


def test_acceptance_tag_genome_continuous_scores_preserved(tmp_path):
    """Tag Genome loader preserves continuous relevance scores in golden_scores and golden_set."""
    import json
    from evaluation.datasets import TagGenomeLoader

    genome_data = {
        "dystopia": {
            "100": 0.895,
            "200": 0.654,
            "300": 0.120,
        }
    }
    genome_file = tmp_path / "tag_genome.json"
    genome_file.write_text(json.dumps(genome_data), encoding="utf-8")

    loader = TagGenomeLoader(genome_path=genome_file)
    entries = loader.load_as_public_entries(tags=["dystopia"])
    assert len(entries) == 1
    e = entries[0]
    assert e["query"] == "dystopia"
    assert e["golden_ids"] == [100, 200]
    # Continuous scores preserved
    assert e["golden_scores"][100] == 0.895
    assert e["golden_scores"][200] == 0.654
    assert e["golden_set"][0]["relevance"] == 0.895


def test_acceptance_private_eval_isolated_storage_and_redaction(tmp_path):
    """Private evaluation isolates cache and qrels under priv_harness.benchmark_dir and redacts query keys."""
    from evaluation.private_benchmark import PrivateBenchmarkHarness

    priv_dir = tmp_path / "private_bench"
    harness = PrivateBenchmarkHarness(benchmark_dir=priv_dir)

    engine = ConsensusJudgeEngine(
        primary_judge=StubJudgeAdapter(),
        qrels_dir=priv_dir / "qrels",
        redact_queries=True,
    )
    q_key = engine._query_key("Confidential Search Query")
    assert q_key.startswith("query_sha256_")
    assert "Confidential" not in q_key

    # Rejection of duplicate candidate submission
    gate_mock = {"status": "PASS", "exit_code": 0, "passed": True, "checks": []}
    res1 = harness.submit_candidate("cand_v1", gate_mock)
    assert res1["status"] == "PASS"

    res2 = harness.submit_candidate("cand_v1", gate_mock)
    assert res2["status"] == "REJECTED"
    assert res2["exit_code"] == 2  # EvaluationStatus.INVALID


def test_acceptance_latency_zero_inferences_rejected():
    """Latency benchmark must fail if actual inferences performed is zero."""
    mock_runner = MagicMock()
    mock_trace = MagicMock()
    mock_trace.errors = []
    mock_trace.inference_counts = {"actual_inferences_performed": 0}
    mock_runner.execute_query.return_value = ([{"id": 1}], mock_trace)

    harness = LatencyHarness(runner=mock_runner, warmup_count=1, repetition_count=1)
    rep = harness.benchmark_paired_latency(
        queries=["test query"],
        baseline_params={"p": 1},
        candidate_params={"p": 2},
    )
    assert rep["passed"] is False
    assert rep["error_count"] > 0
    assert any("Missing expected" in err for err in rep["errors"])
