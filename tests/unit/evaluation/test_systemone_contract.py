import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import pytest

from tests.unit.evaluation.judge_helpers import make_nimble
from evaluation.judge.base import LocalJudgeAdapter
from evaluation.judge.rubric import GRADE_CRITERIA, normalize_probabilities
from evaluation.models import ItemEvidence, JudgeInput, TypedId


@pytest.fixture
def judge_input():
    return JudgeInput(
        "space adventure",
        ItemEvidence(TypedId("movie", 1), "Film", "Space adventure", ["Adventure"]),
    )


def reply(probabilities, **extra):
    return {
        "answers": {
            "relevance": {"probabilities": probabilities},
            "evidence_sufficient": {"noul": 0.9},
        },
        **extra,
    }


def response(payload):
    result = MagicMock()
    result.__enter__.return_value = result
    result.read.return_value = json.dumps(payload).encode()
    return result


def test_published_rubric_permutation(judge_input):
    judge = make_nimble()
    with patch.object(judge, "is_available", return_value=True), patch(
        "urllib.request.urlopen",
        side_effect=[response(reply([0, 0, 2, 0])), response(reply([0, 2, 0, 0]))],
    ) as request:
        forward, reverse, consistent = judge.test_option_order_permutation(judge_input)
    assert consistent and forward == reverse == {0: 0, 1: 0, 2: 1, 3: 0}
    payloads = [json.loads(c.args[0].data) for c in request.call_args_list]
    assert payloads[0]["state"] == payloads[1]["state"]
    assert payloads[0]["questions"]["relevance"]["criteria"] == list(GRADE_CRITERIA)
    assert payloads[1]["questions"]["relevance"]["criteria"] == list(
        reversed(GRADE_CRITERIA)
    )
    assert "scale" not in payloads[0]["questions"]["relevance"]


def test_permutation_disagreement_and_failure(judge_input):
    judge = make_nimble()
    with patch.object(judge, "is_available", return_value=True):
        with patch(
            "urllib.request.urlopen",
            side_effect=[response(reply([0, 0, 1, 0])), response(reply([0, 0, 1, 0]))],
        ):
            assert judge.test_option_order_permutation(judge_input)[2] is False
        with patch(
            "urllib.request.urlopen",
            side_effect=[response(reply([0, 0, 1, 0])), OSError("offline")],
        ):
            with pytest.raises(OSError):
                judge.test_option_order_permutation(judge_input)

        with patch("urllib.request.urlopen", return_value=response({})):
            with pytest.raises(RuntimeError):
                judge.test_option_order_permutation(judge_input)


@pytest.mark.parametrize("reverse_sufficient,consistent", [(0.1, True), (0.9, False)])
def test_abstentions_ignore_hypothetical_grade_but_not_sufficiency_flip(
    judge_input, reverse_sufficient, consistent
):
    judge = make_nimble()
    first, second = reply([1, 0, 0, 0]), reply([1, 0, 0, 0])
    first["answers"]["evidence_sufficient"]["noul"] = 0.1
    second["answers"]["evidence_sufficient"]["noul"] = reverse_sufficient
    with patch.object(judge, "is_available", return_value=True), patch(
        "urllib.request.urlopen", side_effect=[response(first), response(second)]
    ):
        forward, reverse, result = judge.test_option_order_permutation(judge_input)
    assert forward != reverse  # Raw distributions remain available for audit.
    assert result is consistent


@pytest.mark.parametrize(
    "media_type,query", [("tv", "feature movie"), ("movie", "TV series")]
)
def test_known_mismatch_does_not_fabricate_model_sufficiency(media_type, query):
    """A prompt repair must not turn the model's abstention into a usable control label."""
    judge = make_nimble()
    inp = JudgeInput(query, ItemEvidence(TypedId(media_type, 1), "Candidate"))
    payload = reply([1, 0, 0, 0])
    payload["answers"]["evidence_sufficient"]["noul"] = 0.1
    with patch.object(judge, "is_available", return_value=True), patch(
        "urllib.request.urlopen", return_value=response(payload)
    ) as request:
        out = judge.judge_pair(inp)
    assert out.execution_status == "success" and out.grade == 0
    assert out.evidence_sufficiency is False
    sent = json.loads(request.call_args.args[0].data)
    assert sent["state"]["query"] == query
    assert f"Media Type: {media_type}" in sent["state"]["evidence"]


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        [1, 0, 0],
        [1, 0, 0, 0, 0],
        {"0": 1, "1": 0, "2": 0, "4": 0},
        [0] * 4,
        [-1, 2, 0, 0],
        [float("nan"), 1, 0, 0],
        [float("inf"), 0, 0, 0],
        [True, 0, 0, 0],
        [None, 1, 0, 0],
    ],
)
def test_invalid_probabilities(invalid):
    with pytest.raises(ValueError):
        normalize_probabilities(invalid)


def test_ties_overflow_and_score_not_rounded(judge_input):
    judge = make_nimble()
    answer = reply({"0": 0, "1": 2, "2": 2, "3": 1})
    answer["answers"]["relevance"]["score"] = 2.9
    with patch.object(judge, "is_available", return_value=True), patch(
        "urllib.request.urlopen", return_value=response(answer)
    ):
        out = judge.judge_pair(judge_input)
    assert out.grade == 1 and out.execution_status == "success"
    assert normalize_probabilities([1e308] * 4) == {g: 0.25 for g in range(4)}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        {"answers": []},
        {"answers": {}},
        reply([0, 0, 1, 0], truncated=True),
        {"answers": {"relevance": {"probabilities": [0, 0, 1, 0]}}},
    ],
)
def test_malformed_or_truncated_is_unjudged(payload, judge_input):
    judge = make_nimble()
    with patch.object(judge, "is_available", return_value=True), patch(
        "urllib.request.urlopen", return_value=response(payload)
    ):
        out = judge.judge_pair(judge_input)
    assert out.execution_status == "malformed" and not out.evidence_sufficiency


@pytest.mark.parametrize("value", [True, "yes", float("nan"), 2, -0.1])
def test_invalid_evidence_sufficiency(value, judge_input):
    payload = reply([0, 0, 1, 0])
    payload["answers"]["evidence_sufficient"]["noul"] = value
    judge = make_nimble()
    with patch.object(judge, "is_available", return_value=True), patch(
        "urllib.request.urlopen", return_value=response(payload)
    ):
        assert judge.judge_pair(judge_input).execution_status == "malformed"


def test_discovery_and_base_failures(judge_input):
    judge = make_nimble()
    with patch("urllib.request.urlopen", side_effect=OSError("offline")):
        assert not judge.is_available()
        assert judge.judge_pair(judge_input).execution_status == "failed"
    with pytest.raises(NotImplementedError):
        LocalJudgeAdapter.test_option_order_permutation(judge, judge_input)


@pytest.mark.parametrize("truncated", [False, True])
def test_nimble_local_http_server_auth_discovery_and_scoring(judge_input, truncated):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            calls.append((self.path, self.headers.get("Authorization"), None))
            self.send_response(200)
            self.end_headers()
            payload = (
                {"version": "test-version"}
                if self.path == "/api/version"
                else {
                    "models": [
                        {
                            "name": "nimble:latest",
                            "digest": "test-digest",
                            "capabilities": ["decision"],
                        }
                    ]
                }
            )
            self.wfile.write(json.dumps(payload).encode())

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, self.headers.get("Authorization"), payload))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps(reply([0, 0, 1, 0], truncated=truncated)).encode()
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        judge = make_nimble(url=f"http://127.0.0.1:{server.server_port}")
        judge.api_key = "test-key"
        assert judge.is_available()
        out = judge.judge_pair(judge_input)
        assert out.provenance.runtime == "systemone_ollama"
        assert out.execution_status == ("malformed" if truncated else "success")
        assert out.grade == (0 if truncated else 2)
        assert all(auth == "Bearer test-key" for _, auth, _ in calls)
        assert (
            calls[-1][0] == "/v1/systemone" and calls[-1][2]["model"] == "nimble:latest"
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert not judge.is_available()  # The same local service is now offline.
