from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from evaluation.evaluate import (
    EvaluationEntry,
    EvaluationStatus,
    load_title_based_entries,
    load_id_based_entries,
    load_entries_from_args,
    inject_report_explanations,
    evaluate_entries,
    call_recommendation_api_items,
    call_recommendation_api,
    save_csv,
    save_scores_csv,
    maybe_generate_report,
    run_evaluation_v2,
    main,
)


def test_load_title_based_entries_branches(tmp_path):
    # 1. Invalid titles (not a list)
    p_invalid = tmp_path / "invalid.json"
    p_invalid.write_text(json.dumps([{"query": "q1", "golden_titles": "not_a_list"}]))
    assert load_title_based_entries(p_invalid, "sqlite:///:memory:") == []

    # 2. Titles with some non-dict elements & unresolvable titles
    p_mixed = tmp_path / "mixed.json"
    p_mixed.write_text(
        json.dumps(
            [
                {
                    "query": "q2",
                    "golden_titles": ["not_a_dict", {"title": "Unknown Film"}],
                }
            ]
        )
    )
    with patch("evaluation.evaluate.resolve_titles_to_tmdb_ids", return_value=[]):
        assert load_title_based_entries(p_mixed, "sqlite:///:memory:") == []

    # 3. Successful resolution
    with patch(
        "evaluation.evaluate.resolve_titles_to_tmdb_ids", return_value=[101, 102]
    ):
        entries = load_title_based_entries(p_mixed, "sqlite:///:memory:")
        assert len(entries) == 1
        assert entries[0].golden_ids == [101, 102]


def test_call_recommendation_api_branches(monkeypatch):
    # 1. in_process=True
    with patch("evaluation.evaluate._call_in_process", return_value=[{"tmdb_id": 99}]):
        items = call_recommendation_api_items("query", {}, in_process=True)
        assert len(items) == 1
        assert items[0]["tmdb_id"] == 99

        tmdb_ids = call_recommendation_api("query", {}, in_process=True)
        assert tmdb_ids == [99]

    # 2. httpx connect error falling back to in_process
    import httpx

    mock_client = MagicMock()
    mock_client.get.side_effect = httpx.ConnectError("connection refused")
    monkeypatch.setattr(
        "evaluation.evaluate._get_http_client", lambda timeout: mock_client
    )

    with patch("evaluation.evaluate._call_in_process", return_value=[{"tmdb_id": 77}]):
        items_fallback = call_recommendation_api_items("query", {}, in_process=False)
        assert len(items_fallback) == 1
        assert items_fallback[0]["tmdb_id"] == 77

    # 3. httpx ReadTimeout and RequestError
    mock_client.get.side_effect = httpx.ReadTimeout("read timed out")
    assert call_recommendation_api_items("query", {}, in_process=False) == []

    mock_client.get.side_effect = httpx.RequestError("other request error")
    assert call_recommendation_api_items("query", {}, in_process=False) == []


def test_save_csv_and_scores_csv_branches(tmp_path):
    # 1. Empty rows
    p_csv = tmp_path / "out.csv"
    save_csv([], p_csv)
    assert not p_csv.exists()

    save_scores_csv([], p_csv, 10)
    assert not p_csv.exists()

    # 2. Normal write
    save_csv([{"a": 1, "b": 2}], p_csv)
    assert p_csv.exists()

    p_scores = tmp_path / "scores.csv"
    save_scores_csv(
        [
            {
                "query": "q",
                "backend": "b",
                "params_name": "p",
                "query_id": "qid",
                "category": "cat",
            }
        ],
        p_scores,
        10,
    )
    assert p_scores.exists()

    # 3. PermissionError branch
    with patch.object(Path, "open", side_effect=PermissionError("denied")):
        save_csv([{"a": 1}], p_csv)
        save_scores_csv([{"a": 1}], p_scores, 10)


def test_maybe_generate_report_branches(tmp_path):
    p_rep = tmp_path / "report.html"
    # 1. Empty rows
    maybe_generate_report([], p_rep, 10, "dsn")

    # 2. Non-empty with PermissionError
    rows = [
        {
            "item_id": 1,
            "query": "q",
            "backend": "b",
            "params_name": "p",
            "rank": 1,
            "relevant": 1,
        }
    ]
    with patch(
        "evaluation.evaluate._run_evidently_report",
        side_effect=PermissionError("denied"),
    ):
        maybe_generate_report(rows, p_rep, 10, "dsn")


def test_qualification_enforcement_in_authoritative_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    qual_path = tmp_path / "evaluation" / ".judge_qualification_ollama.json"

    args = argparse.Namespace(
        v2=True,
        track="authoritative",
        split="regression",
        judgment_mode="single_judge",
        judge_config="nimble",
        allow_stub_judges_for_testing=False,
        hardware_inspect=False,
        qualify_judges=False,
        ab_compare=False,
        benchmark=False,
        latency_benchmark=False,
        personalization_test=False,
        private_eval=False,
        rescore_only=False,
        dataset=None,
        titles_set=Path("evaluation/golden_queries_titles.json"),
        set=Path("evaluation/golden_queries.json"),
        resolve_titles=False,
        category=None,
        gain_mode="exponential",
        candidate="cross_encoder",
        baseline="default",
        k=10,
        backend="elasticsearch",
        dsn="postgresql://seenemall:seenemall@localhost:5432/seenemall",
        v2_report=tmp_path / "v2_report.json",
    )

    from evaluation import evaluate as ev
    from tests.unit.evaluation.judge_helpers import make_nimble

    monkeypatch.setattr(
        ev, "discover_ollama_judges", lambda: {"bespoke-nimble-9b": make_nimble()}
    )
    # 1. Missing qualification file -> INVALID
    code_missing = run_evaluation_v2(args)
    assert code_missing == int(EvaluationStatus.INVALID)

    # 2. Malformed qualification JSON -> INVALID
    qual_path.parent.mkdir(parents=True, exist_ok=True)
    qual_path.write_text("not_valid_json{")
    assert run_evaluation_v2(args) == int(EvaluationStatus.INVALID)

    # 3. Missing reports mapping -> INVALID
    qual_path.write_text(json.dumps({"some_key": 123}))
    assert run_evaluation_v2(args) == int(EvaluationStatus.INVALID)

    # 4. Panel judge missing from reports -> INVALID
    qual_path.write_text(json.dumps({"reports": {"unused-model": {"qualified": True}}}))
    assert run_evaluation_v2(args) == int(EvaluationStatus.INVALID)

    # 5. Panel judge unqualified -> INVALID
    qual_path.write_text(
        json.dumps(
            {
                "reports": {
                    "bespoke-nimble-9b": {"qualified": False},
                }
            }
        )
    )
    assert run_evaluation_v2(args) == int(EvaluationStatus.INVALID)


def test_main_cli_dispatch(monkeypatch, tmp_path):
    mock_entry = EvaluationEntry("q", [1], raw={"query": "q", "golden_ids": [1]})

    # 1. Hardware inspect
    with pytest.raises(SystemExit) as exc_info:
        main(["--hardware-inspect"])
    assert exc_info.value.code == 0

    # 2. Benchmark gate (mocked)
    with patch("evaluation.evaluate.load_entries_from_args", return_value=[mock_entry]):
        with patch("evaluation.evaluate.run_regression_gate", return_value=True):
            main(["--benchmark", "--k", "10"])

        # Benchmark gate failed
        with patch("evaluation.evaluate.run_regression_gate", return_value=False):
            with pytest.raises(SystemExit) as exc:
                main(["--benchmark", "--k", "10"])
            assert exc.value.code == 1

    # 3. AB comparison (mocked)
    with patch("evaluation.evaluate.load_entries_from_args", return_value=[mock_entry]):
        with patch(
            "evaluation.evaluate.run_ab_comparison",
            return_value={"quality_gate": {"passed": True}},
        ):
            main(["--ab-compare"])

        with patch(
            "evaluation.evaluate.run_ab_comparison",
            return_value={"quality_gate": {"passed": False}},
        ):
            with pytest.raises(SystemExit) as exc:
                main(["--ab-compare"])
            assert exc.value.code == 1

    # 4. Unknown config
    with patch("evaluation.evaluate.load_entries_from_args", return_value=[mock_entry]):
        main(["--config", "non_existent_config_xyz"])

    # 5. Standard evaluation execution with save-baseline
    base_file = tmp_path / "baseline_out.json"
    agg_mock = {
        ("elasticsearch", "default"): {
            "precision": 0.5,
            "recall": 0.4,
            "map": 0.45,
            "ndcg": 0.55,
            "ild": 0.6,
            "unique_count": 10,
            "intent_alignment": 0.9,
        }
    }
    sum_mock = [
        {
            "query": "q",
            "backend": "elasticsearch",
            "params_name": "default",
            "query_id": "1",
            "category": "cat",
        }
    ]
    with patch("evaluation.evaluate.load_entries_from_args", return_value=[mock_entry]):
        with patch(
            "evaluation.evaluate.evaluate_entries",
            return_value=([], sum_mock, agg_mock),
        ):
            with patch("evaluation.evaluate.save_csv"):
                with patch("evaluation.evaluate.save_scores_csv"):
                    with patch("evaluation.evaluate.maybe_generate_report"):
                        with patch("evaluation.evaluate.save_baseline_snapshot"):
                            main(
                                [
                                    "--config",
                                    "default",
                                    "--save-baseline",
                                    str(base_file),
                                ]
                            )


def test_run_evaluation_v2_qualify_judges(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from evaluation import evaluate as ev
    from tests.unit.evaluation.judge_helpers import make_nimble

    monkeypatch.setattr(
        ev, "discover_ollama_judges", lambda: {"bespoke-nimble-9b": make_nimble()}
    )
    args = argparse.Namespace(
        hardware_inspect=False,
        qualify_judges=True,
    )
    mock_runner = MagicMock()
    mock_rep = {
        "fingerprint": "test-contract",
        "malformed_count": 0,
        "qualified": True,
        "qualification_protocol": "v2.8",
        "scope": "qualification",
        "pilot_pairs": 400,
        "pilot_distinct_pairs": 400,
        "pilot_families": 20,
        "repeat_tests": 100,
        "permutation_tests": 100,
        "control_tests": 100,
        "repeatability_rate": 0.99,
        "option_permutation_rate": 0.99,
        "control_accuracy": 0.99,
        "execution_failures": 0,
    }
    mock_runner.run_candidate_pilot.return_value = mock_rep
    stub_judge = MagicMock()
    stub_judge.model_name = "stub"
    mock_runner.select_judge_panel.return_value = (
        stub_judge,
        None,
        None,
        "single_judge",
    )

    with patch(
        "evaluation.evaluate.JudgeQualificationRunner", return_value=mock_runner
    ):
        with patch(
            "evaluation.evaluate.generate_factual_control_cases", return_value=[]
        ):
            with patch("evaluation.evaluate.load_evaluation_cases", return_value=[]):
                code = run_evaluation_v2(args)
                assert code == 0
                assert (
                    tmp_path / "evaluation" / ".judge_qualification_ollama.json"
                ).exists()
                saved = json.loads(
                    (
                        tmp_path / "evaluation" / ".judge_qualification_ollama.json"
                    ).read_text()
                )
                assert all(report == mock_rep for report in saved["reports"].values())
                assert (
                    mock_runner.run_candidate_pilot.call_args.kwargs[
                        "run_option_order_diagnostic"
                    ]
                    is False
                )


def test_run_evaluation_v2_latency_benchmark():
    args = argparse.Namespace(
        hardware_inspect=False,
        qualify_judges=False,
        latency_benchmark=True,
        baseline="default",
        candidate="cross_encoder",
        backend="elasticsearch",
    )
    mock_harness = MagicMock()
    mock_harness.benchmark_paired_latency.return_value = {
        "baseline_p50_ms": 20,
        "baseline_p95_ms": 40,
        "candidate_p50_ms": 22,
        "candidate_p95_ms": 42,
        "median_paired_ratio": 1.05,
        "median_paired_ratio_pass": True,
        "p95_ratio": 1.05,
        "p95_ratio_pass": True,
        "passed": True,
    }
    with patch("evaluation.evaluate.EvaluationRunner"):
        with patch("evaluation.evaluate.LatencyHarness", return_value=mock_harness):
            code = run_evaluation_v2(args)
            assert code == 0
            _, baseline, candidate = (
                mock_harness.benchmark_paired_latency.call_args.args
            )
            assert baseline["ann_backend_override"] == "elasticsearch"
            assert candidate["ann_backend_override"] == "elasticsearch"


def test_run_evaluation_v2_personalization_test():
    args = argparse.Namespace(
        hardware_inspect=False,
        qualify_judges=False,
        latency_benchmark=False,
        personalization_test=True,
        baseline="default",
        candidate="cross_encoder",
        backend="pgvector",
        k=10,
    )
    mock_pers_harness = MagicMock()
    mock_pers_harness.run_personalization_benchmark.return_value = {
        "mean_personalization_lift": 0.05,
        "mean_lift_pass": True,
        "total_disliked_violations": 0,
        "disliked_pass": True,
        "max_persona_ndcg_decline": 0.01,
        "decline_pass": True,
        "persona_results": [
            {
                "persona_key": "p1",
                "candidate_ndcg": 0.8,
                "masked_baseline_ndcg": 0.7,
                "total_personalization_system_lift": 0.1,
                "disliked_violations_count": 0,
            }
        ],
        "passed": True,
    }
    with patch("evaluation.evaluate.EvaluationRunner"):
        with patch(
            "evaluation.evaluate.PersonalizationHarness", return_value=mock_pers_harness
        ):
            code = run_evaluation_v2(args)
            assert code == 0
            params = mock_pers_harness.run_personalization_benchmark.call_args.kwargs
            assert params["baseline_params"]["ann_backend_override"] == "pgvector"
            assert params["candidate_params"]["ann_backend_override"] == "pgvector"


def test_run_evaluation_v2_verify_index_and_private_eval(tmp_path):
    # 1. verify_index without expected checksum -> INVALID
    args_idx_err = argparse.Namespace(
        hardware_inspect=False,
        qualify_judges=False,
        latency_benchmark=False,
        personalization_test=False,
        verify_index=True,
        expected_index_checksum=None,
    )
    assert run_evaluation_v2(args_idx_err) == int(EvaluationStatus.INVALID)

    # 2. verify_index with ES client None -> INVALID
    args_idx_ok = argparse.Namespace(
        hardware_inspect=False,
        qualify_judges=False,
        latency_benchmark=False,
        personalization_test=False,
        verify_index=True,
        expected_index_checksum="hash123",
    )
    with patch(
        "api.core.elasticsearch_client.get_elasticsearch_client", return_value=None
    ):
        assert run_evaluation_v2(args_idx_ok) == int(EvaluationStatus.INVALID)

    # 3. private_eval with missing holdout file -> INVALID
    args_priv = argparse.Namespace(
        hardware_inspect=False,
        qualify_judges=False,
        latency_benchmark=False,
        personalization_test=False,
        verify_index=False,
        expected_index_checksum=None,
        private_eval=True,
    )
    mock_priv = MagicMock()
    mock_priv.benchmark_dir = tmp_path / "priv_missing"
    with patch("evaluation.evaluate.PrivateBenchmarkHarness", return_value=mock_priv):
        assert run_evaluation_v2(args_priv) == int(EvaluationStatus.INVALID)


def test_run_evaluation_v2_full_flow(tmp_path):
    from evaluation.metrics import evaluate_comparison_gates

    v2_report_file = tmp_path / "v2_report.json"
    args = argparse.Namespace(
        v2=True,
        track="dev",
        split="dev",
        judgment_mode="primary_only",
        judge_config="stub",
        allow_stub_judges_for_testing=True,
        hardware_inspect=False,
        qualify_judges=False,
        ab_compare=False,
        benchmark=False,
        latency_benchmark=False,
        personalization_test=False,
        private_eval=False,
        rescore_only=False,
        dataset=None,
        titles_set=Path("evaluation/golden_queries_titles.json"),
        set=Path("evaluation/golden_queries.json"),
        resolve_titles=False,
        category=None,
        gain_mode="graded_exponential",
        candidate="cross_encoder",
        baseline="default",
        k=10,
        backend="elasticsearch",
        dsn="postgresql://seenemall:seenemall@localhost:5432/seenemall",
        v2_report=v2_report_file,
    )

    mock_gate = evaluate_comparison_gates(
        family_ndcg_deltas={f"fam_{i}": 0.05 for i in range(60)},
        slice_family_deltas={
            "vibe": [0.05] * 15,
            "franchise": [0.05] * 15,
            "constraint": [0.05] * 15,
            "entity": [0.05] * 15,
        },
        baseline_recall_100=0.8,
        candidate_recall_100=0.85,
        hard_constraint_violations=0,
        disliked_violations=0,
        exploratory_coverages=[1.0] * 60,
        authoritative_unresolved_count=0,
        execution_failures=0,
        unexpected_fallbacks=0,
        duplicate_outputs_detected=False,
        is_statistical_promotion=True,
    )

    mock_case = MagicMock()
    mock_case.case_id = "c1"
    mock_case.family_id = "f1"
    mock_case.slice_tags = ["vibe"]
    mock_case.query = "query 1"
    mock_case.constraints = None
    mock_case.canonical_sequence = None
    mock_case.eligible_catalog_count = 10
    mock_case.expected_empty = False

    mock_trace = MagicMock(errors=[], fallbacks=[])
    mock_items = [
        {"id": 1, "title": "Movie 1", "synopsis": "Overview", "media_type": "movie"}
    ]

    mock_engine = MagicMock()
    mock_rec = MagicMock(deterministic_override=False, status="CONFIRMED")
    mock_engine.label_pool.return_value = ({"movie:1": 3}, [mock_rec], {})

    with patch("evaluation.evaluate.load_evaluation_cases", return_value=[mock_case]):
        with patch("evaluation.evaluate.EvaluationRunner") as mock_runner_cls:
            mock_runner = mock_runner_cls.return_value
            mock_runner.run_case.return_value = (mock_items, mock_trace)
            with patch(
                "evaluation.evaluate.ConsensusJudgeEngine", return_value=mock_engine
            ):
                with patch(
                    "evaluation.evaluate.evaluate_comparison_gates",
                    return_value=mock_gate,
                ):
                    code = run_evaluation_v2(args)
                    assert code == int(EvaluationStatus.PASS)
                    assert v2_report_file.exists()

    # With private_eval promotion submit
    holdout_file = tmp_path / "holdout_cases.json"
    holdout_file.write_text(json.dumps([{"query": "q", "golden_ids": [1]}]))
    args.private_eval = True
    mock_priv_harness = MagicMock()
    mock_priv_harness.benchmark_dir = tmp_path
    mock_priv_harness.submit_candidate.return_value = {
        "submission_index": 1,
        "status": "APPROVED",
        "passed": True,
        "refresh_required": False,
        "exit_code": 0,
    }
    with patch(
        "evaluation.evaluate.PrivateBenchmarkHarness", return_value=mock_priv_harness
    ):
        with patch(
            "evaluation.evaluate.load_evaluation_cases", return_value=[mock_case]
        ):
            with patch("evaluation.evaluate.EvaluationRunner") as mock_runner_cls:
                mock_runner = mock_runner_cls.return_value
                mock_runner.run_case.return_value = (mock_items, mock_trace)
                with patch(
                    "evaluation.evaluate.ConsensusJudgeEngine", return_value=mock_engine
                ):
                    with patch(
                        "evaluation.evaluate.evaluate_comparison_gates",
                        return_value=mock_gate,
                    ):
                        code_priv = run_evaluation_v2(args)
                        assert code_priv == 0


def test_inject_report_explanations(tmp_path):
    # 1. File does not exist
    missing_file = tmp_path / "missing.html"
    inject_report_explanations(missing_file)
    assert not missing_file.exists()

    # 2. File already has marker
    marker_file = tmp_path / "marker.html"
    marker_file.write_text(
        "<html><section data-report-explainer>Old</section><body>Content</body></html>"
    )
    inject_report_explanations(marker_file)
    assert "Old" in marker_file.read_text()

    # 3. File with <body> tag
    body_file = tmp_path / "body.html"
    body_file.write_text("<html><body><div>Content</div></body></html>")
    inject_report_explanations(body_file)
    content = body_file.read_text()
    assert "<section data-report-explainer" in content
    assert "<body>\n    <section data-report-explainer" in content

    # 4. File without <body> tag
    no_body_file = tmp_path / "no_body.html"
    no_body_file.write_text("<div>Just a div</div>")
    inject_report_explanations(no_body_file)
    content_nb = no_body_file.read_text()
    assert "data-report-explainer" in content_nb
    assert content_nb.endswith("<div>Just a div</div>")


def test_load_entries_from_args_branches(tmp_path):
    # 1. args.dataset != "none" normal
    args_ds = argparse.Namespace(
        dataset="movielens_small",
        resolve_titles=False,
        titles_set=Path("titles.json"),
        set=Path("ids.json"),
        category=None,
    )
    mock_public = [{"query": "q", "golden_ids": [10, 20], "user_id": "u99"}]
    with patch("evaluation.evaluate.load_public_dataset", return_value=mock_public):
        res = load_entries_from_args(args_ds)
        assert len(res) == 1
        assert res[0].user_id == "u99"

    # 2. args.dataset != "none" with NotImplementedError
    with patch(
        "evaluation.evaluate.load_public_dataset",
        side_effect=NotImplementedError("not implemented"),
    ):
        with pytest.raises(SystemExit) as exc:
            load_entries_from_args(args_ds)
        assert exc.value.code == 1

    # 3. args.dataset == "none", resolve_titles with existing titles_set
    titles_file = tmp_path / "t.json"
    titles_file.write_text("[]")
    args_titles = argparse.Namespace(
        dataset="none",
        resolve_titles=True,
        titles_set=titles_file,
        set=Path("ids.json"),
        category="comedy",
        dsn="dsn",
    )
    mock_entry1 = EvaluationEntry("q1", [1], {}, category="comedy")
    mock_entry2 = EvaluationEntry("q2", [2], {}, category="action")
    with patch(
        "evaluation.evaluate.load_title_based_entries",
        return_value=[mock_entry1, mock_entry2],
    ):
        filtered = load_entries_from_args(args_titles)
        assert len(filtered) == 1
        assert filtered[0].category == "comedy"

    # 4. args.dataset == "none", resolve_titles=False, id set not existing
    args_no_id = argparse.Namespace(
        dataset="none",
        resolve_titles=False,
        titles_set=Path("none.json"),
        set=tmp_path / "nonexistent_ids.json",
        category=None,
    )
    assert load_entries_from_args(args_no_id) == []


def test_load_id_based_entries_skip_branches(tmp_path):
    # Missing golden keys (KeyError) and empty golden ids
    bad_items = [
        {"query": "bad1"},  # no golden_ids or golden_set
        {"query": "bad2", "golden_ids": []},  # empty
        {"query": "good", "golden_ids": ["1", "not_an_int", 2], "category": "cat"},
    ]
    p = tmp_path / "test_ids.json"
    p.write_text(json.dumps(bad_items))
    loaded = load_id_based_entries(p)
    assert len(loaded) == 1
    assert loaded[0].golden_ids == [1, 2]

    # golden_set with valid and invalid dict entries
    good_set_items = [
        {
            "query": "set_good",
            "golden_set": [{"id": 42}, {"id": "invalid"}, "string_row"],
            "category": "cat",
        }
    ]
    p_set = tmp_path / "test_set.json"
    p_set.write_text(json.dumps(good_set_items))
    loaded_set = load_id_based_entries(p_set)
    assert len(loaded_set) == 1
    assert loaded_set[0].golden_ids == [42]


def test_call_recommendation_api_more_branches():
    # 1. _call_in_process with status != 200
    mock_app = MagicMock()
    mock_resp = MagicMock(status_code=500, text="Internal Error")
    with patch("starlette.testclient.TestClient.get", return_value=mock_resp):
        res = call_recommendation_api_items(
            "q", {}, in_process=True, app_instance=mock_app
        )
        assert res == []

    # 2. _call_in_process raising Exception
    with patch(
        "starlette.testclient.TestClient.get", side_effect=RuntimeError("crash")
    ):
        res = call_recommendation_api_items(
            "q", {}, in_process=True, app_instance=mock_app
        )
        assert res == []

    # 3. HTTP with api_key and ReadTimeout
    with patch("evaluation.evaluate._get_http_client") as mock_client_factory:
        mock_client = mock_client_factory.return_value
        import httpx

        mock_client.get.side_effect = httpx.ReadTimeout("timeout")
        res = call_recommendation_api_items(
            "q", {}, api_key="secret_key", in_process=False
        )
        assert res == []

    # 4. HTTP with HTTPStatusError
    with patch("evaluation.evaluate._get_http_client") as mock_client_factory:
        mock_client = mock_client_factory.return_value
        import httpx

        req = httpx.Request("GET", "http://test")
        resp = httpx.Response(500, request=req)
        mock_client.get.side_effect = httpx.HTTPStatusError(
            "500", request=req, response=resp
        )
        res = call_recommendation_api_items("q", {}, in_process=False)
        assert res == []


def test_verify_index_manifest_and_metadata_branches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Manifest exists
    fixtures_dir = tmp_path / "evaluation" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = fixtures_dir / "frozen_index_manifest.json"
    manifest_file.write_text(json.dumps({"checksum": "expected_hash_123"}))

    args = argparse.Namespace(
        v2=True,
        track="product",
        split="dev",
        judgment_mode="consensus",
        judge_config="stub",
        allow_stub_judges_for_testing=True,
        hardware_inspect=False,
        qualify_judges=False,
        latency_benchmark=False,
        personalization_test=False,
        verify_index=True,
        expected_index_checksum=None,
        private_eval=False,
        rescore_only=False,
        dataset=None,
        titles_set=Path("titles.json"),
        set=Path("ids.json"),
        resolve_titles=False,
        category=None,
        gain_mode="graded_exponential",
        candidate="cross_encoder",
        baseline="default",
        k=10,
        backend="elasticsearch",
        dsn="postgresql://seenemall:seenemall@localhost:5432/seenemall",
        v2_report=tmp_path / "rep.json",
    )

    mock_es = MagicMock()
    mock_verifier = MagicMock()
    mock_verifier.fetch_live_index_metadata.return_value = {
        "fingerprint": "expected_hash_123"
    }
    mock_verifier.verify_reuse.return_value = True

    mock_case = MagicMock(
        case_id="c1",
        family_id="fam1",
        slice_tags=["tag1"],
        query="q",
        constraints=None,
        canonical_sequence=None,
        expected_empty=False,
        eligible_catalog_count=1,
    )
    mock_runner = MagicMock()
    mock_runner.run_case.return_value = (
        [{"id": 1, "title": "M1", "synopsis": "S1", "media_type": "movie"}],
        MagicMock(errors=[], fallbacks=[]),
    )
    mock_engine = MagicMock()
    mock_engine.label_pool.return_value = (
        {"movie:1": 3},
        [MagicMock(deterministic_override=False, status="CONFIRMED")],
        {},
    )

    with patch(
        "api.core.elasticsearch_client.get_elasticsearch_client", return_value=mock_es
    ):
        with patch(
            "evaluation.evaluate.IndexArtifactVerifier", return_value=mock_verifier
        ):
            with patch(
                "evaluation.evaluate.load_evaluation_cases", return_value=[mock_case]
            ):
                with patch(
                    "evaluation.evaluate.EvaluationRunner", return_value=mock_runner
                ):
                    with patch(
                        "evaluation.evaluate.ConsensusJudgeEngine",
                        return_value=mock_engine,
                    ):
                        code = run_evaluation_v2(args)
                        assert code == int(EvaluationStatus.PASS)

    # Live verification fails (verify_reuse returns False)
    mock_verifier.verify_reuse.return_value = False
    with patch(
        "api.core.elasticsearch_client.get_elasticsearch_client", return_value=mock_es
    ):
        with patch(
            "evaluation.evaluate.IndexArtifactVerifier", return_value=mock_verifier
        ):
            code_fail = run_evaluation_v2(args)
            assert code_fail == int(EvaluationStatus.INVALID)

    # Verification raises exception
    mock_verifier.fetch_live_index_metadata.side_effect = RuntimeError("ES error")
    with patch(
        "api.core.elasticsearch_client.get_elasticsearch_client", return_value=mock_es
    ):
        with patch(
            "evaluation.evaluate.IndexArtifactVerifier", return_value=mock_verifier
        ):
            code_err = run_evaluation_v2(args)
            assert code_err == int(EvaluationStatus.INVALID)


def test_evaluate_entries_personalization_and_continuous_identity(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "evaluation.evaluate.fetch_embeddings_for_tmdb_ids", lambda *a: {}
    )
    mock_entry_seeded = EvaluationEntry(
        query="test query",
        golden_ids=[101],
        raw={
            "query": "test query",
            "seed_history": [201, 202],
            "known_negatives": [999],
            "golden_scores": {"101": 3.0},
        },
        user_id="user_test",
    )

    mock_session = MagicMock()
    mock_session_factory = MagicMock(return_value=mock_session)
    mock_session.__enter__.return_value = mock_session

    with patch("api.db.session.get_sessionmaker", return_value=mock_session_factory):
        with patch("evaluation.personalization.seed_persona_fixtures") as mock_seed:
            with patch(
                "evaluation.evaluate.call_recommendation_api_items",
                return_value=[{"tmdb_id": 101, "title": "Test"}],
            ):
                with patch(
                    "evaluation.evaluate.calculate_ndcg_at_k", return_value=1.0
                ) as mock_ndcg:
                    per_rank, summary, agg = evaluate_entries(
                        entries=[mock_entry_seeded],
                        k=10,
                        backends=["elasticsearch"],
                        param_grid={"default": lambda r: {}},
                        in_process=True,
                    )
                    assert len(per_rank) > 0
                    assert mock_seed.call_count == 1
                    assert mock_ndcg.call_count == 1


def test_run_evaluation_v2_panel_judges_and_loop_branches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    qual_path = tmp_path / "evaluation" / ".judge_qualification_ollama.json"
    qual_path.parent.mkdir(parents=True, exist_ok=True)
    from evaluation import evaluate as ev

    from tests.unit.evaluation.judge_helpers import make_nimble

    factories = {"bespoke-nimble-9b": make_nimble}
    judge = make_nimble()
    monkeypatch.setattr(
        ev, "discover_ollama_judges", lambda: {"bespoke-nimble-9b": judge}
    )
    qual_path.write_text(
        json.dumps(
            {
                "reports": {
                    name: {
                        "qualified": True,
                        "qualification_protocol": "v2.8",
                        "scope": "qualification",
                        "pilot_pairs": 400,
                        "pilot_distinct_pairs": 400,
                        "pilot_families": 20,
                        "repeat_tests": 100,
                        "permutation_tests": 100,
                        "control_tests": 100,
                        "fingerprint": factory().qualification_fingerprint(),
                        "repeatability_rate": 1.0,
                        "option_permutation_rate": 1.0,
                        "control_accuracy": 1.0,
                        "execution_failures": 0,
                        "malformed_count": 0,
                    }
                    for name, factory in factories.items()
                }
            }
        )
    )

    for cfg in ["nimble"]:
        for mode in ["single_judge"]:
            args = argparse.Namespace(
                v2=True,
                track="product",
                split="dev",
                judgment_mode=mode,
                judge_config=cfg,
                allow_stub_judges_for_testing=False,
                hardware_inspect=False,
                qualify_judges=False,
                ab_compare=False,
                benchmark=False,
                latency_benchmark=False,
                personalization_test=False,
                private_eval=False,
                rescore_only=False,
                dataset=None,
                titles_set=Path("titles.json"),
                set=Path("ids.json"),
                resolve_titles=False,
                category=None,
                gain_mode="graded_exponential",
                candidate="cross_encoder",
                baseline="default",
                k=10,
                backend="elasticsearch",
                dsn="postgresql://seenemall:seenemall@localhost:5432/seenemall",
                v2_report=tmp_path / f"report_{cfg}_{mode}.json",
            )

            # Complex case hitting loop branches:
            # - disliked_ids match
            # - canonical_sequence inversions & missing
            # - errors & fallbacks in traces
            # - rec.status == UNJUDGED
            # - rec.deterministic_override == True
            # - unmatched item evidence fallback
            mock_case = MagicMock()
            mock_case.case_id = "c_complex"
            mock_case.family_id = "fam_1"
            mock_case.slice_tags = ["franchise"]
            mock_case.query = "complex query"
            from evaluation.models import DeterministicConstraint

            mock_case.constraints = DeterministicConstraint(disliked_ids=[101])
            mock_case.canonical_sequence = [101, 102, 103]
            mock_case.eligible_catalog_count = None
            mock_case.expected_empty = True

            mock_trace_with_err = MagicMock(
                errors=["trace error"], fallbacks=["trace fallback"]
            )
            mock_items = [
                {
                    "id": 101,
                    "title": "Movie 101",
                    "synopsis": "Overview",
                    "media_type": "movie",
                }
            ]

            mock_rec1 = MagicMock(deterministic_override=True, status="UNJUDGED")
            mock_engine = MagicMock()
            mock_engine.label_pool.return_value = (
                {"movie:101": 2, "movie:999": 0},
                [mock_rec1],
                {},
            )

            mock_runner = MagicMock()
            mock_runner.run_case.return_value = (mock_items, mock_trace_with_err)

            with patch(
                "evaluation.evaluate.load_evaluation_cases", return_value=[mock_case]
            ):
                with patch(
                    "evaluation.evaluate.EvaluationRunner", return_value=mock_runner
                ):
                    with patch(
                        "evaluation.evaluate.ConsensusJudgeEngine",
                        return_value=mock_engine,
                    ):
                        with patch(
                            "evaluation.evaluate.check_for_duplicates",
                            return_value=True,
                        ):
                            with patch(
                                "evaluation.evaluate.check_canonical_order",
                                return_value={
                                    "inversions": 1,
                                    "missing_prefix_count": 1,
                                    "exact_prefix_match": False,
                                },
                            ):
                                code = run_evaluation_v2(args)
                                assert isinstance(code, int)
                                assert mock_runner.run_case.call_count == 2


def test_slice_promotion_counts_independent_families(tmp_path, monkeypatch):
    from evaluation import evaluate
    from evaluation.models import TestCase
    from evaluation.trace import EvaluationTrace

    cases = [
        TestCase(
            f"vibe-{i}",
            "one-vibe-family",
            "product",
            "regression",
            "search",
            ["vibe"],
            f"vibe {i}",
        )
        for i in range(12)
    ]
    for tag in ["franchise", "entity", "constraint"]:
        cases.extend(
            TestCase(
                f"{tag}-{i}",
                f"{tag}-family-{i}",
                "product",
                "regression",
                "search",
                [tag],
                f"{tag} {i}",
            )
            for i in range(17)
        )
    runner = MagicMock()
    runner.run_case.side_effect = lambda case, **kw: (
        [
            {
                "tmdb_id": 1,
                "media_type": "movie",
                "title": "Film",
                "overview": "Complete synopsis",
                "genres": ["Drama"],
            }
        ],
        EvaluationTrace(case.query, case.user_id),
    )
    monkeypatch.setattr(evaluate, "load_evaluation_cases", lambda **kw: cases)
    monkeypatch.setattr(evaluate, "EvaluationRunner", lambda **kw: runner)
    report_file = tmp_path / "report.json"
    args = evaluate.parse_args(
        [
            "--v2",
            "--split",
            "regression",
            "--judge-config",
            "stub",
            "--allow-stub-judges-for-testing",
            "--v2-report",
            str(report_file),
        ]
    )
    evaluate.run_evaluation_v2(args)
    gate = json.loads(report_file.read_text())["gate_result"]
    assert gate["family_count"] == 52
    assert gate["slice_counts"]["vibe"] == 1
    assert not gate["passed"]
    assert any(
        "vibe" in reason and "fewer than 10 families" in reason
        for reason in gate["reasons"]
    )
