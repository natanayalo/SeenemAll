"""Recommendation execution runner, trace interception, and environment isolation."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from evaluation.models import TestCase
from evaluation.trace import EvaluationTrace, TraceCollector


class IndexArtifactVerifier:
    """Verifies that baseline and candidate use the exact same frozen search index artifact."""

    def __init__(
        self,
        expected_checksum: Optional[str] = None,
        expected_uuid: Optional[str] = None,
    ) -> None:
        self.expected_checksum = expected_checksum
        self.expected_uuid = expected_uuid
        self.verified_checksum: Optional[str] = None

    def compute_index_checksum(self, index_metadata: Dict[str, Any]) -> str:
        """Compute sha256 checksum over frozen index UUID, doc count, creation date, mappings/settings, and content fingerprint."""
        # Normalize fields to guarantee identical hash across environments
        payload = {
            "uuid": index_metadata.get("uuid") or index_metadata.get("index_uuid"),
            "doc_count": index_metadata.get("doc_count")
            or index_metadata.get("docs_count", 0),
            "creation_date": index_metadata.get("creation_date")
            or index_metadata.get("index_creation_date"),
            "settings": index_metadata.get("settings", {}),
            "mappings": index_metadata.get("mappings", {}),
            "content_fingerprint": str(index_metadata.get("content_fingerprint") or ""),
        }
        serialized = json.dumps(payload, sort_keys=True)
        checksum = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        self.verified_checksum = checksum
        return checksum

    def verify_reuse(self, candidate_index_metadata: Dict[str, Any]) -> bool:
        """Verify that index settings, UUID, and content match expected frozen artifact; reject rebuilds."""
        if not self.expected_checksum:
            raise ValueError(
                "Index verification requires an expected checksum or frozen reference artifact; "
                "cannot verify reuse without a reference standard."
            )
        candidate_checksum = self.compute_index_checksum(candidate_index_metadata)
        if candidate_checksum != self.expected_checksum:
            return False
        candidate_uuid = candidate_index_metadata.get(
            "uuid"
        ) or candidate_index_metadata.get("index_uuid")
        if (
            self.expected_uuid
            and candidate_uuid
            and candidate_uuid != self.expected_uuid
        ):
            return False
        return True

    def verify_index_artifact(
        self, expected_checksum: str, actual_checksum: str
    ) -> bool:
        """Verify that artifact checksum matches; raises ValueError if mismatch."""
        if not expected_checksum or expected_checksum != actual_checksum:
            raise ValueError(
                f"Index artifact checksum mismatch: expected {expected_checksum}, got {actual_checksum}"
            )
        return True

    def fetch_live_index_metadata(
        self, es_client: Any, index_name: str = "items"
    ) -> Dict[str, Any]:
        """Query Elasticsearch for index UUID, doc count, creation date, settings, mappings, and vector content fingerprint."""
        if es_client is None:
            raise RuntimeError(
                "Elasticsearch client is None; cannot fetch live index metadata."
            )
        try:
            settings_res = es_client.indices.get_settings(index=index_name)
            mappings_res = es_client.indices.get_mapping(index=index_name)
            count_res = es_client.count(index=index_name)

            idx_settings = (
                settings_res.get(index_name, {}).get("settings", {}).get("index", {})
            )
            # Collect complete content and vector fingerprint across all documents in index
            try:
                hasher = hashlib.sha256()
                # Include index store stats if available
                try:
                    stats_res = es_client.indices.stats(index=index_name)
                    total_store = (
                        stats_res.get("indices", {})
                        .get(index_name, {})
                        .get("total", {})
                        .get("store", {})
                        .get("size_in_bytes", 0)
                    )
                    hasher.update(f"store_bytes:{total_store}:".encode("utf-8"))
                except Exception:
                    pass

                # A PIT supplies a stable snapshot and _shard_doc requires no application ID mapping.
                pit_id = es_client.open_point_in_time(
                    index=index_name, keep_alive="1m"
                )["id"]
                document_hashes = []
                search_after = None
                try:
                    while True:
                        kwargs = {
                            "pit": {"id": pit_id, "keep_alive": "1m"},
                            "query": {"match_all": {}},
                            "size": 1000,
                            "sort": [{"_shard_doc": "asc"}],
                            "_source": True,
                        }
                        if search_after is not None:
                            kwargs["search_after"] = search_after
                        page = es_client.search(**kwargs)
                        pit_id = page.get("pit_id", pit_id)
                        if page.get("timed_out") or page.get("_shards", {}).get(
                            "failed", 0
                        ):
                            raise RuntimeError("Incomplete index search")
                        hits = page.get("hits", {}).get("hits", [])
                        for hit in hits:
                            encoded = json.dumps(
                                {
                                    "id": hit.get("_id"),
                                    "source": hit.get("_source", {}),
                                },
                                sort_keys=True,
                            ).encode("utf-8")
                            document_hashes.append(hashlib.sha256(encoded).hexdigest())
                        if len(hits) < 1000:
                            break
                        next_sort = hits[-1].get("sort")
                        if next_sort is None or next_sort == search_after:
                            raise RuntimeError("Index pagination did not advance")
                        search_after = next_sort
                finally:
                    es_client.close_point_in_time(id=pit_id)
                if (
                    count_res.get("count", 0) > 0
                    and len(document_hashes) != count_res["count"]
                ):
                    raise RuntimeError(
                        "Index document count changed or pagination is incomplete"
                    )
                # Canonical per-document digests avoid depending on shard or segment iteration order.
                for digest in sorted(document_hashes):
                    hasher.update(digest.encode("ascii"))
                content_fingerprint = hasher.hexdigest()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to collect complete content and vector fingerprint for index '{index_name}': {exc}"
                ) from exc

            doc_cnt = count_res.get("count", 0)
            if doc_cnt == 0:
                raise RuntimeError(f"Live index '{index_name}' contains 0 documents.")

            return {
                "uuid": idx_settings.get("uuid"),
                "creation_date": idx_settings.get("creation_date"),
                "doc_count": doc_cnt,
                "settings": idx_settings,
                "mappings": mappings_res.get(index_name, {}).get("mappings", {}),
                "content_fingerprint": content_fingerprint,
            }
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch live index metadata from Elasticsearch: {exc}"
            ) from exc


class EvaluationRunner:
    """Direct and HTTP evaluation runner with trace collection and cache bypass."""

    def __init__(
        self,
        in_process: bool = True,
        dsn: Optional[str] = None,
        base_url: str = "http://localhost:8000/recommend",
        reference_time: Optional[str] = None,
        trace_collector: Optional[TraceCollector] = None,
    ) -> None:
        self.in_process = in_process
        self.dsn = dsn
        self.base_url = base_url
        self.reference_time = reference_time
        self.trace_collector = trace_collector or TraceCollector()
        self._test_client = None

    def _get_test_client(self) -> Any:
        if self._test_client is None:
            try:
                from starlette.testclient import TestClient
                from api.main import app

                self._test_client = TestClient(app)
            except Exception as exc:
                raise RuntimeError(f"Failed to initialize FastAPI TestClient: {exc}")
        return self._test_client

    def execute_query(
        self,
        query: str,
        user_id: str = "u1",
        params: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        bypass_cache: bool = True,
    ) -> Tuple[List[Dict[str, Any]], EvaluationTrace]:
        """Execute a single query, bypass query-caches, capture trace and return items."""
        trace = self.trace_collector.start_trace(query=query, user_id=user_id)
        merged_params = dict(params or {})
        merged_params["user_id"] = user_id
        merged_params["query"] = query
        merged_params["limit"] = limit
        merged_params["debug"] = True

        if bypass_cache:
            merged_params["bypass_cache"] = True
            merged_params["_bypass_cache"] = "1"
            # Invalidate all recommendation and reranking caches to ensure true query-cache-cold
            try:
                from api.core.entity_linker import ENTITY_LINKER_CACHE
                from api.core.llm_parser import (
                    INTENT_CACHE,
                    _persistent_intent_store,
                )
                from api.core.reranker import clear_reranker_result_caches
                from api.pipeline.context import (
                    clear_recommend_cache_for_tests,
                    clear_user_cache,
                )

                clear_user_cache(user_id)
                clear_recommend_cache_for_tests()
                clear_reranker_result_caches()
                ENTITY_LINKER_CACHE.clear()
                INTENT_CACHE.clear()
                p_store = _persistent_intent_store()
                if p_store and hasattr(p_store, "clear"):
                    p_store.clear()
            except Exception as exc:
                trace.errors.append(f"Cache invalidation failed: {exc}")
                return [], trace

        t0 = time.perf_counter()
        try:
            client = self._get_test_client()
            resp = client.get("/recommend", params=merged_params)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            trace.timings_ms["total_latency_ms"] = elapsed_ms

            if resp.status_code != 200:
                err_msg = f"HTTP {resp.status_code}: {resp.text[:300]}"
                trace.errors.append(err_msg)
                return [], trace

            payload = resp.json()
            raw_items = payload.get("items", [])
            debug_info = payload.get("debug") or {}

            # Populate trace stages
            trace.effective_context = {
                "user_id": user_id,
                "cold_start": debug_info.get("cold_start", False),
            }
            trace.prefilter_outcome = {
                "initial_candidates_count": debug_info.get(
                    "initial_candidates_count", 0
                ),
                "post_filter_candidates_count": debug_info.get(
                    "post_filter_candidates_count", 0
                ),
            }
            self.trace_collector.record_stage(
                trace, "final", raw_items, timing_ms=elapsed_ms
            )

            # Record actual-inference execution counters strictly from reported telemetry
            cand_cnt = len(raw_items)
            actual_inf: Optional[int] = None
            if "actual_inferences_performed" in debug_info:
                actual_inf = debug_info["actual_inferences_performed"]
            elif (
                "metrics" in debug_info
                and isinstance(debug_info["metrics"], dict)
                and "actual_inferences_performed" in debug_info["metrics"]
            ):
                actual_inf = debug_info["metrics"]["actual_inferences_performed"]
            elif "inferences_performed" in debug_info:
                actual_inf = debug_info["inferences_performed"]
            elif "actual_inferences" in debug_info:
                actual_inf = debug_info["actual_inferences"]

            trace.inference_counts["candidate_count"] = cand_cnt
            trace.inference_counts["actual_inferences_performed"] = (
                actual_inf if actual_inf is not None else 0
            )
            metrics = debug_info.get("metrics", {})
            trace.inference_counts["cache_hits"] = debug_info.get(
                "cache_hits",
                metrics.get("cache_hits", 0) if isinstance(metrics, dict) else 0,
            )

            if any(
                not isinstance(trace.inference_counts[name], int)
                or isinstance(trace.inference_counts[name], bool)
                or trace.inference_counts[name] < 0
                for name in ("actual_inferences_performed", "cache_hits")
            ):
                raise ValueError("Malformed inference telemetry")

            inference_stats = debug_info.get("inference_stats", {})
            trace.inference_providers = inference_stats.get("providers", {})
            if inference_stats.get("inference_failures", 0):
                trace.errors.append("Configured reranker inference failed")

            # Check if fallbacks were reported in debug
            if debug_info.get("fallback_applied"):
                trace.fallbacks.append(
                    str(debug_info.get("fallback_reason", "unknown_fallback"))
                )

            return raw_items, trace

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            trace.timings_ms["total_latency_ms"] = elapsed_ms
            trace.errors.append(f"Execution exception: {exc}")
            return [], trace

    def run_case(
        self,
        case: TestCase,
        params: Optional[Dict[str, Any]] = None,
        k: int = 10,
    ) -> Tuple[List[Dict[str, Any]], EvaluationTrace]:
        """Execute a structured TestCase and return ranked item dictionaries."""
        return self.execute_query(
            query=case.query,
            user_id=case.user_id,
            params=params,
            limit=k,
            bypass_cache=True,
        )
