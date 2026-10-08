"""Unit tests for runner, index artifact verifier, and tracing."""

from unittest.mock import MagicMock, patch
import pytest

from evaluation.models import TestCase
from evaluation.runner import EvaluationRunner, IndexArtifactVerifier
from evaluation.trace import TraceCollector


def test_index_artifact_verifier():
    verifier = IndexArtifactVerifier()
    settings = {"mappings": {"properties": {"vector": {"type": "dense_vector"}}}}
    checksum1 = verifier.compute_index_checksum(settings)
    assert len(checksum1) == 64

    # Without expected checksum, raises ValueError
    with pytest.raises(ValueError, match="requires an expected checksum"):
        verifier.verify_reuse(settings)

    # Matches same settings when expected_checksum is set
    verifier.expected_checksum = checksum1
    assert verifier.verify_reuse(settings) is True

    # Rejects modified settings (unintended rebuild)
    verifier_locked = IndexArtifactVerifier(expected_checksum=checksum1)
    modified_settings = {
        "mappings": {"properties": {"vector": {"type": "dense_vector", "dims": 768}}}
    }
    assert verifier_locked.verify_reuse(modified_settings) is False


def test_trace_collector_and_stages():
    collector = TraceCollector()
    trace = collector.start_trace(query="action thriller", user_id="u2")
    assert trace.query == "action thriller"
    assert trace.user_id == "u2"

    collector.record_stage(
        trace, "retrieval_merged", ["movie:1", "movie:2"], timing_ms=12.5
    )
    collector.record_stage(trace, "final", ["movie:2", "movie:1"], timing_ms=5.0)

    assert trace.depths["retrieval"] == 2
    assert trace.depths["final"] == 2
    assert trace.merged_candidates == ["movie:1", "movie:2"]
    assert trace.final_order == ["movie:2", "movie:1"]
    assert trace.timings_ms["retrieval_merged"] == 12.5
    assert trace.timings_ms["final"] == 5.0

    d = trace.to_dict()
    assert d["user_id"] == "u2"


def test_runner_query_execution():
    runner = EvaluationRunner(in_process=True)
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "items": [{"tmdb_id": 10, "media_type": "movie", "title": "Test Title"}],
        "debug": {"initial_candidates_count": 5, "post_filter_candidates_count": 1},
    }
    mock_client.get.return_value = mock_response

    with patch.object(runner, "_get_test_client", return_value=mock_client):
        items, trace = runner.execute_query("sci fi", user_id="u1", limit=5)
        assert len(items) == 1
        assert items[0]["tmdb_id"] == 10
        assert trace.prefilter_outcome["initial_candidates_count"] == 5
        assert len(trace.errors) == 0


def test_runner_error_handling():
    runner = EvaluationRunner(in_process=True)
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_client.get.return_value = mock_response

    with patch.object(runner, "_get_test_client", return_value=mock_client):
        items, trace = runner.execute_query("error query")
        assert len(items) == 0
        assert len(trace.errors) == 1
        assert "HTTP 500" in trace.errors[0]


def test_runner_fallback_detection_and_exceptions():
    runner = EvaluationRunner(in_process=True)

    # 1. Fallback applied
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "items": [{"id": 1}],
        "debug": {
            "fallback_applied": True,
            "fallback_reason": "pgvector_fallback",
        },
    }
    mock_client.get.return_value = mock_response

    with patch.object(runner, "_get_test_client", return_value=mock_client):
        case = TestCase(
            case_id="c1",
            family_id="f1",
            track="product",
            split="dev",
            task="search",
            slice_tags=["vibe"],
            query="fallback test",
        )
        items, trace = runner.run_case(case)
        assert len(items) == 1
        assert "pgvector_fallback" in trace.fallbacks

    # 2. Client raises exception
    mock_client_err = MagicMock()
    mock_client_err.get.side_effect = RuntimeError("connection drop")
    with patch.object(runner, "_get_test_client", return_value=mock_client_err):
        items, trace = runner.execute_query("fail query")
        assert len(items) == 0
        assert any("Execution exception" in e for e in trace.errors)

    # 3. TestClient init error
    with patch("starlette.testclient.TestClient", side_effect=Exception("no app")):
        bad_runner = EvaluationRunner()
        with pytest.raises(
            RuntimeError, match="Failed to initialize FastAPI TestClient"
        ):
            bad_runner._get_test_client()


def test_fetch_live_index_metadata_and_verifier_uuid():
    from evaluation.runner import IndexArtifactVerifier

    verifier = IndexArtifactVerifier(
        expected_checksum="chk_123", expected_uuid="uuid_abc"
    )

    # Mismatched UUID check
    meta_wrong_uuid = {"mappings": {}, "settings": {}, "uuid": "uuid_xyz"}
    with patch.object(verifier, "compute_index_checksum", return_value="chk_123"):
        assert verifier.verify_reuse(meta_wrong_uuid) is False

    # fetch_live_index_metadata with None es_client raises RuntimeError
    with pytest.raises(RuntimeError, match="client is None"):
        verifier.fetch_live_index_metadata(None)

    # fetch_live_index_metadata with mocked client
    mock_es = MagicMock()
    mock_es.open_point_in_time.return_value = {"id": "test-pit"}
    mock_es.indices.get_settings.return_value = {
        "items": {
            "settings": {"index": {"uuid": "uuid_abc", "creation_date": "1700000000"}}
        }
    }
    mock_es.indices.get_mapping.return_value = {
        "items": {"mappings": {"properties": {}}}
    }
    mock_es.count.return_value = {"count": 1}
    mock_es.search.return_value = {
        "hits": {
            "hits": [{"_source": {"id": 1, "title": "Movie 1", "media_type": "movie"}}]
        }
    }

    res = verifier.fetch_live_index_metadata(mock_es, index_name="items")
    assert res["uuid"] == "uuid_abc"
    assert res["doc_count"] == 1
    assert len(res["content_fingerprint"]) == 64

    # When doc count is 0, raises RuntimeError
    mock_es.count.return_value = {"count": 0}
    with pytest.raises(RuntimeError, match="contains 0 documents"):
        verifier.fetch_live_index_metadata(mock_es, index_name="items")


def test_index_vectors_beyond_first_page_and_incomplete_search():
    import copy
    from evaluation.runner import IndexArtifactVerifier

    es = MagicMock()
    es.open_point_in_time.return_value = {"id": "test-pit"}
    es.indices.get_settings.return_value = {
        "items": {"settings": {"index": {"uuid": "frozen"}}}
    }
    es.indices.get_mapping.return_value = {"items": {"mappings": {}}}
    es.indices.stats.return_value = {}
    es.count.return_value = {"count": 1001}
    page = {
        "hits": {
            "hits": [
                {"_source": {"id": i, "vector": [1.0]}, "sort": [i]}
                for i in range(1000)
            ]
        }
    }
    final_page = {
        "hits": {"hits": [{"_source": {"id": 1000, "vector": [1.0]}, "sort": [1000]}]}
    }
    verifier = IndexArtifactVerifier(expected_checksum="")
    es.search.side_effect = [page, final_page]
    original = verifier.fetch_live_index_metadata(es)
    verifier.expected_checksum = verifier.compute_index_checksum(original)
    changed = copy.deepcopy(final_page)
    changed["hits"]["hits"][0]["_source"]["vector"] = [2.0]
    es.search.side_effect = [page, changed]
    assert not verifier.verify_reuse(verifier.fetch_live_index_metadata(es))
    assert es.search.call_args.kwargs["search_after"] == [999]
    assert es.search.call_args.kwargs["sort"] == [{"_shard_doc": "asc"}]
    assert es.search.call_args.kwargs["pit"]["id"] == "test-pit"
    es.close_point_in_time.assert_called_with(id="test-pit")
    es.search.side_effect = [page, {"hits": {"hits": []}}]
    with pytest.raises(RuntimeError, match="pagination is incomplete"):
        verifier.fetch_live_index_metadata(es)
    es.search.side_effect = [{"timed_out": True}]
    with pytest.raises(RuntimeError, match="Incomplete index search"):
        verifier.fetch_live_index_metadata(es)


@pytest.mark.parametrize(
    "field,value",
    [
        ("actual_inferences_performed", False),
        ("cache_hits", False),
        ("actual_inferences_performed", "1"),
        ("actual_inferences_performed", MagicMock()),
    ],
)
def test_runner_rejects_non_integer_inference_telemetry(field, value):
    runner = EvaluationRunner()
    response = MagicMock(status_code=200)
    response.json.return_value = {"items": [{"id": 1}], "debug": {field: value}}
    client = MagicMock()
    client.get.return_value = response
    with patch.object(runner, "_get_test_client", return_value=client):
        items, trace = runner.execute_query("films", bypass_cache=False)
    assert not items
    assert any("Malformed inference telemetry" in error for error in trace.errors)
