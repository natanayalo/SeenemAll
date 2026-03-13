"""Tests for the lightweight metrics module."""

from __future__ import annotations

import threading

import pytest

from api.core.metrics import Counter, Histogram, MetricsRegistry, timer, METRICS


class TestCounter:
    def test_starts_at_zero(self):
        c = Counter()
        assert c.value == 0
        assert c.snapshot() == 0

    def test_increments(self):
        c = Counter()
        c.inc()
        assert c.value == 1
        c.inc(5)
        assert c.value == 6

    def test_thread_safety(self):
        c = Counter()
        threads = [threading.Thread(target=c.inc) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert c.value == 100


class TestHistogram:
    def test_empty_snapshot(self):
        h = Histogram()
        snap = h.snapshot()
        assert snap["count"] == 0
        assert snap["avg"] == 0.0

    def test_observe_single(self):
        h = Histogram()
        h.observe(42.0)
        snap = h.snapshot()
        assert snap["count"] == 1
        assert snap["min"] == 42.0
        assert snap["max"] == 42.0
        assert snap["avg"] == 42.0

    def test_observe_multiple(self):
        h = Histogram()
        h.observe(10.0)
        h.observe(20.0)
        h.observe(30.0)
        snap = h.snapshot()
        assert snap["count"] == 3
        assert snap["min"] == 10.0
        assert snap["max"] == 30.0
        assert snap["avg"] == 20.0
        assert snap["sum"] == 60.0


class TestMetricsRegistry:
    @pytest.fixture(autouse=True)
    def _reset(self):
        METRICS.reset()
        yield
        METRICS.reset()

    def test_counter_created_on_access(self):
        c = METRICS.counter("test.counter")
        assert c.value == 0
        c.inc()
        assert METRICS.counter("test.counter").value == 1

    def test_histogram_created_on_access(self):
        h = METRICS.histogram("test.hist")
        h.observe(5.0)
        snap = METRICS.histogram("test.hist").snapshot()
        assert snap["count"] == 1

    def test_snapshot_structure(self):
        METRICS.counter("a").inc(3)
        METRICS.histogram("b").observe(10.0)
        snap = METRICS.snapshot()
        assert "counters" in snap
        assert "histograms" in snap
        assert snap["counters"]["a"] == 3
        assert snap["histograms"]["b"]["count"] == 1

    def test_reset_clears_all(self):
        METRICS.counter("x").inc()
        METRICS.histogram("y").observe(1.0)
        METRICS.reset()
        snap = METRICS.snapshot()
        assert snap["counters"] == {}
        assert snap["histograms"] == {}


class TestTimer:
    @pytest.fixture(autouse=True)
    def _reset(self):
        METRICS.reset()
        yield
        METRICS.reset()

    def test_records_latency(self):
        with timer("test.timer"):
            pass  # instant operation
        snap = METRICS.histogram("test.timer").snapshot()
        assert snap["count"] == 1
        assert snap["min"] >= 0
