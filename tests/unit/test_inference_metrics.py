import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.core.inference_metrics import (
    inference_request,
    record_success,
    record_cache_hit,
    record_failure,
    submit_with_inference_context,
)
from api.pipeline.models import RecommendParams
from api.pipeline.runner import RecommendationPipeline


def test_collector_lifecycle_and_worker_context():
    record_success("cross_encoder", 20)  # No request owns this operation.
    record_cache_hit("small")
    record_failure("small")
    with ThreadPoolExecutor(max_workers=2) as pool:
        with inference_request() as first:
            submit_with_inference_context(
                pool, record_success, "cross_encoder", 20
            ).result()
            record_cache_hit("small")
            record_failure("small")
            stats = first.snapshot()
            assert stats["actual_inferences_performed"] == 1
            assert stats["providers"]["cross_encoder"]["scored_items"] == 20
            assert stats["cache_hits"] == stats["inference_failures"] == 1
        with inference_request() as second:
            assert second.snapshot()["actual_inferences_performed"] == 0
            assert first.closed
            first.record("cross_encoder", "successful_calls")
            assert first.snapshot() == stats


def test_late_executor_completion_cannot_mutate_closed_request():
    ready, finish = Event(), Event()

    def delayed():
        ready.set()
        assert finish.wait(5)
        record_success("small", 9)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with inference_request() as old:
            future = submit_with_inference_context(pool, delayed)
            assert ready.wait(5)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.01)
            record_failure("small")
        with inference_request() as new:
            finish.set()
            future.result()
            assert new.snapshot()["actual_inferences_performed"] == 0
            assert old.snapshot()["actual_inferences_performed"] == 0


def test_async_task_isolation_and_exception_cleanup():
    async def worker(count):
        with inference_request() as collector:
            record_success("cross_encoder", count)
            await asyncio.sleep(0)
            return collector.snapshot()["providers"]["cross_encoder"]["scored_items"]

    async def run():
        return await asyncio.gather(worker(2), worker(7))

    assert asyncio.run(run()) == [2, 7]
    with pytest.raises(RuntimeError):
        with inference_request() as failed:
            raise RuntimeError("request failed")
    assert failed.closed
    with inference_request() as clean:
        assert clean.snapshot()["providers"] == {}


@pytest.mark.parametrize("debug", [True, False])
def test_pipeline_request_scope_non_debug_and_early_return(debug):
    item = {"id": 1, "title": "Film"}
    context = SimpleNamespace(profile_meta={}, cold_start=True)
    intent = SimpleNamespace(
        prefilter_kwargs={},
        prefer_top_rated=False,
        intent_filters=None,
        matched_collection_ids=(),
        collection_item_ids=(),
    )
    pool = SimpleNamespace(
        ids=[1], prefilter=SimpleNamespace(allowed_ids=[]), boost_ids=[]
    )
    scored = SimpleNamespace(ordered=[item], serendipity_context=None)

    def rerank(*args, **kwargs):
        if kwargs["rerank"]:
            record_success("cross_encoder", 1)
        return [item]

    with patch("api.pipeline.runner.load_user_context", return_value=context), patch(
        "api.pipeline.runner.resolve_query_intent", new=AsyncMock(return_value=intent)
    ), patch("api.pipeline.runner.retrieve_candidates", return_value=pool), patch(
        "api.pipeline.runner.score_candidates", return_value=scored
    ), patch(
        "api.pipeline.runner.apply_diversity_policies", return_value=[item]
    ), patch(
        "api.pipeline.runner.rerank_candidates", side_effect=rerank
    ):
        pipeline = RecommendationPipeline()
        asyncio.run(
            pipeline.run(
                MagicMock(),
                RecommendParams(query="query", rerank=True, debug=debug),
                MagicMock(),
            )
        )
        result = asyncio.run(
            pipeline.run(
                MagicMock(),
                RecommendParams(query="query", rerank=False, debug=True),
                MagicMock(),
            )
        )
        assert result.debug_context["actual_inferences_performed"] == 0
        pool.ids = []
        assert (
            asyncio.run(pipeline.run(MagicMock(), RecommendParams(), MagicMock())).items
            == []
        )
        pool.ids = [1]
        scored.ordered = []
        assert (
            asyncio.run(pipeline.run(MagicMock(), RecommendParams(), MagicMock())).items
            == []
        )


def test_result_cache_clear_keeps_models_loaded():
    from api.core import cross_encoder, reranker

    model = object()
    with patch.dict(cross_encoder._model_cache, {"test": model}, clear=True):
        reranker._CROSS_ENCODER_CACHE["test"] = []
        reranker._SMALL_RERANK_CACHE["test"] = []
        reranker.clear_reranker_result_caches()
        assert cross_encoder._model_cache["test"] is model
        assert not reranker._CROSS_ENCODER_CACHE and not reranker._SMALL_RERANK_CACHE


@pytest.mark.parametrize("failure", [False, True])
def test_actual_cross_encoder_scoring_in_executor(monkeypatch, failure):
    from api.core import cross_encoder

    model = MagicMock()
    if failure:
        model.predict.side_effect = RuntimeError("scoring failed")
    else:
        model.predict.return_value = [0.3, 0.9]
    monkeypatch.setattr(cross_encoder, "get_cross_encoder_model", lambda *a: model)
    candidates = [{"id": 1, "title": "Film"}, {"id": 2, "title": "Other film"}]
    with ThreadPoolExecutor(max_workers=1) as executor:
        with inference_request() as collector:
            result = submit_with_inference_context(
                executor, cross_encoder.score_query_candidates, "films", candidates
            ).result()
            assert len(result) == 2
            stats = collector.snapshot()["providers"]["cross_encoder"]
            assert stats["successful_calls"] == int(not failure)
            assert stats["scored_items"] == (0 if failure else 2)
            assert stats["failures"] == int(failure)


def test_pipeline_exception_releases_request(monkeypatch):
    from api.core import inference_metrics

    captured = []

    def fail(*args, **kwargs):
        captured.append(inference_metrics._collector.get())
        record_success("small", 2)
        raise RuntimeError("failed pipeline")

    monkeypatch.setattr("api.pipeline.runner.load_user_context", fail)
    with pytest.raises(RuntimeError, match="failed pipeline"):
        asyncio.run(
            RecommendationPipeline().run(MagicMock(), RecommendParams(), MagicMock())
        )
    assert captured[0].closed
    assert inference_metrics._collector.get() is None


@pytest.mark.parametrize("provider", ["small", "cross_encoder"])
def test_reranker_scoring_calls_items_and_result_cache_hits(monkeypatch, provider):
    import numpy as np
    from api.core import cross_encoder, reranker

    monkeypatch.setattr(
        reranker,
        "encode_texts",
        lambda texts: np.ones((len(texts), 2), dtype=np.float32),
    )
    model = MagicMock()
    model.predict.return_value = [0.3, 0.9]
    monkeypatch.setattr(cross_encoder, "get_cross_encoder_model", lambda *a: model)
    items = [{"id": 1, "title": "Film"}, {"id": 2, "title": "Other film"}]
    call = (
        reranker._call_small_reranker
        if provider == "small"
        else reranker._call_cross_encoder_reranker
    )
    reranker.clear_reranker_result_caches()
    try:
        with inference_request() as collector:
            assert (
                len(
                    call(
                        SimpleNamespace(timeout=5, model=None),
                        items,
                        None,
                        "films",
                        None,
                    )
                )
                == 2
            )
            assert (
                len(
                    call(
                        SimpleNamespace(timeout=5, model=None),
                        items,
                        None,
                        "films",
                        None,
                    )
                )
                == 2
            )
            stats = collector.snapshot()["providers"][provider]
            assert stats == dict(
                successful_calls=1, scored_items=2, cache_hits=1, failures=0
            )
    finally:
        reranker.clear_reranker_result_caches()
