import json
from unittest.mock import MagicMock

import pytest

from evaluation.judge.ollama import (
    MODEL_TAGS,
    OllamaJudgeAdapter,
    discover_ollama_judges,
)
from evaluation.judge.qualification import qualification_record_matches
from evaluation.models import ItemEvidence, JudgeInput, TypedId


def metadata(tag, digest="digest-a"):
    return {
        "name": tag,
        "digest": digest,
        "details": {"quantization_level": "Q8_0"},
        "capabilities": ["decision"],
    }


@pytest.fixture
def runtime(monkeypatch):
    tags = {"models": [metadata(tag) for tag in MODEL_TAGS.values()]}
    version = {"version": "0.35.1"}
    monkeypatch.setattr(
        OllamaJudgeAdapter,
        "_get",
        lambda self, path: version if path == "/api/version" else tags,
    )
    return tags, version


def test_pins_native_artifact_and_runtime(runtime):
    judges = discover_ollama_judges()
    assert set(judges) == set(MODEL_TAGS)
    judge = judges["bespoke-nimble-9b"]
    assert judge.is_available()
    assert judge.checkpoint_revision == judge.tokenizer_revision == "digest-a"
    assert judge.quantization == "Q8_0"
    assert judge.runtime == "systemone_ollama"
    assert judge.service_model == "nimble:latest"
    record = {
        "qualified": True,
        "qualification_protocol": "v2.8",
        "scope": "qualification",
        "pilot_pairs": 400,
        "pilot_distinct_pairs": 400,
        "pilot_families": 20,
        "repeat_tests": 100,
        "permutation_tests": 100,
        "control_tests": 100,
        "fingerprint": judge.qualification_fingerprint(),
        "repeatability_rate": 1,
        "option_permutation_rate": 1,
        "control_accuracy": 1,
        "execution_failures": 0,
        "malformed_count": 0,
    }
    assert qualification_record_matches(judge, record)
    tags, version = runtime
    tags["models"][0]["digest"] = "digest-b"
    assert not judge.is_available()
    replacement = discover_ollama_judges()["bespoke-nimble-9b"]
    assert not qualification_record_matches(replacement, record)
    tags["models"][0]["digest"] = "digest-a"
    version["version"] = "0.36.0"
    assert not judge.is_available()
    assert not qualification_record_matches(
        discover_ollama_judges()["bespoke-nimble-9b"], record
    )


def test_exact_tags_and_decision_capability_required(runtime):
    tags, _ = runtime
    tags["models"][0]["capabilities"] = ["completion"]
    assert not discover_ollama_judges()["bespoke-nimble-9b"].is_available()
    tags["models"][0]["name"] = "nimble:other"
    assert not discover_ollama_judges()["bespoke-nimble-9b"].is_available()


def test_offline_and_configured_transport(monkeypatch):
    monkeypatch.setenv("OLLAMA_JUDGE_URL", "http://configured:11434/")
    monkeypatch.setenv("OLLAMA_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_JUDGE_TIMEOUT", "99")
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = b'{"version":"0.35.1"}'
    request = MagicMock(return_value=response)
    monkeypatch.setattr("urllib.request.urlopen", request)
    judge = OllamaJudgeAdapter("nimble", "nimble:latest", {}, "unknown", "http://test")
    assert judge._get("/api/version") == {"version": "0.35.1"}
    assert request.call_args.args[0].get_header("Authorization") == "Bearer test-key"
    request.side_effect = OSError("offline")
    candidates = discover_ollama_judges()
    for candidate in candidates.values():
        assert candidate.endpoint_url == "http://configured:11434/v1/systemone"
        assert candidate.timeout_seconds == 99
        assert not candidate.is_available()


def test_native_score_and_reversed_semantics(runtime, monkeypatch):
    judge = discover_ollama_judges()["bespoke-nimble-9b"]
    inputs = JudgeInput(
        "space adventure", ItemEvidence(TypedId("movie", 1), "Arrival", "Aliens visit")
    )
    requests = []

    def response(request, **kwargs):
        payload = json.loads(request.data)
        requests.append(payload)
        probs = [0, 0, 1, 0] if len(requests) % 2 else [0, 1, 0, 0]
        result = MagicMock()
        result.__enter__.return_value = result
        result.read.return_value = json.dumps(
            {
                "answers": {
                    "relevance": {"probabilities": probs},
                    "evidence_sufficient": {"noul": 0.9},
                }
            }
        ).encode()
        return result

    monkeypatch.setattr("urllib.request.urlopen", response)
    forward, reverse, consistent = judge.test_option_order_permutation(inputs)
    assert consistent and forward == reverse
    assert requests[0]["model"] == "nimble:latest"
    from evaluation.judge.rubric import RELEVANCE_INSTRUCTIONS, SUFFICIENCY_INSTRUCTIONS

    assert (
        requests[0]["questions"]["relevance"]["instructions"] == RELEVANCE_INSTRUCTIONS
    )
    assert (
        requests[0]["questions"]["evidence_sufficient"]["instructions"]
        == SUFFICIENCY_INSTRUCTIONS
    )
    assert requests[1]["questions"]["relevance"]["criteria"] == list(
        reversed(requests[0]["questions"]["relevance"]["criteria"])
    )


def test_nimble_single_judge_wiring_and_consensus_rejection(
    runtime, tmp_path, monkeypatch
):
    from evaluation import evaluate as ev

    monkeypatch.chdir(tmp_path)
    candidates = discover_ollama_judges()
    monkeypatch.setattr(ev, "discover_ollama_judges", lambda: candidates)
    monkeypatch.setattr(ev, "load_evaluation_cases", lambda **kw: [MagicMock()])
    qual_path = tmp_path / "evaluation" / ".judge_qualification_ollama.json"
    qual_path.parent.mkdir()
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
                        "fingerprint": judge.qualification_fingerprint(),
                        "repeatability_rate": 1,
                        "option_permutation_rate": 1,
                        "control_accuracy": 1,
                        "execution_failures": 0,
                        "malformed_count": 0,
                    }
                    for name, judge in candidates.items()
                }
            }
        )
    )
    args = ev.parse_args(
        [
            "--v2",
            "--judge-runtime",
            "ollama",
            "--judge-config",
            "nimble",
            "--judgment-mode",
            "single_judge",
            "--split",
            "regression",
        ]
    )
    # Stop at construction: only panel wiring and qualification are under test.
    engine = MagicMock(side_effect=RuntimeError("panel reached"))
    monkeypatch.setattr(ev, "ConsensusJudgeEngine", engine)
    with pytest.raises(RuntimeError, match="panel reached"):
        ev.run_evaluation_v2(args)
    assert (
        engine.call_args.kwargs["adjudicator"].primary_judge
        is candidates["bespoke-nimble-9b"]
    )
    assert engine.call_args.kwargs["adjudicator"].secondary_judge is None
    assert engine.call_args.kwargs["adjudicator"].tie_breaker_judge is None
    args.judgment_mode = "consensus"
    engine.reset_mock()
    assert ev.run_evaluation_v2(args) == int(ev.EvaluationStatus.INVALID)
    engine.assert_not_called()


def test_cli_defaults_to_nimble_single_judge():
    from evaluation.evaluate import parse_args

    args = parse_args(["--v2"])
    assert args.judge_config == "nimble"
    assert args.judge_runtime == "ollama"
    assert args.judgment_mode == "single_judge"


@pytest.mark.parametrize("retired", ["clef", "kev", "clm"])
def test_cli_rejects_retired_judge_configs(retired):
    from evaluation.evaluate import parse_args

    with pytest.raises(SystemExit) as error:
        parse_args(["--v2", "--judge-config", retired])
    assert error.value.code == 2


def test_ollama_qualification_namespace(runtime, tmp_path, monkeypatch):
    from evaluation import evaluate as ev

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        ev, "load_evaluation_cases", lambda **kw: [MagicMock(family_id="actual-family")]
    )
    monkeypatch.setattr(
        ev.JudgeQualificationRunner,
        "run_candidate_pilot",
        lambda self, judge, **kw: {
            "qualified": False,
            "fingerprint": judge.qualification_fingerprint(),
            "repeatability_rate": 0,
            "option_permutation_rate": 0,
            "control_accuracy": 0,
            "execution_failures": 1,
            "malformed_count": 0,
        },
    )
    args = ev.parse_args(["--v2", "--qualify-judges", "--judge-runtime", "ollama"])
    assert ev.run_evaluation_v2(args) == 0
    data = json.loads(
        (tmp_path / "evaluation" / ".judge_qualification_ollama.json").read_text()
    )
    assert set(data["reports"]) == set(MODEL_TAGS)
    assert not (tmp_path / "evaluation" / ".judge_qualification.json").exists()
