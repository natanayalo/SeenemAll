"""Lightweight in-process metrics for the recommendation pipeline.

Provides thread-safe Counter and Histogram collectors, plus a singleton
MetricsRegistry that can be snapshotted as JSON via ``/healthz/metrics``.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict


class Counter:
    """Monotonically-increasing integer counter."""

    __slots__ = ("_value", "_lock")

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, n: int = 1) -> None:
        with self._lock:
            self._value += n

    @property
    def value(self) -> int:
        return self._value

    def snapshot(self) -> int:
        return self._value


class Histogram:
    """Tracks count, sum, min, max for a numeric measurement (e.g. latency)."""

    __slots__ = ("_count", "_sum", "_min", "_max", "_lock")

    def __init__(self) -> None:
        self._count = 0
        self._sum = 0.0
        self._min = float("inf")
        self._max = float("-inf")
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._count += 1
            self._sum += value
            if value < self._min:
                self._min = value
            if value > self._max:
                self._max = value

    def snapshot(self) -> Dict[str, float | None]:
        with self._lock:
            if self._count == 0:
                return {"count": 0, "sum": 0.0, "avg": 0.0, "min": None, "max": None}
            s = float(self._sum)
            c = float(self._count)
            return {
                "count": float(self._count),
                "sum": round(s, 3),
                "avg": round(s / c, 3),
                "min": round(float(self._min), 3),
                "max": round(float(self._max), 3),
            }


class MetricsRegistry:
    """Singleton registry that holds all application metrics."""

    _instance: MetricsRegistry | None = None
    _init_lock = threading.Lock()
    _counters: Dict[str, Counter]
    _histograms: Dict[str, Histogram]

    def __new__(cls) -> MetricsRegistry:
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    # We cast to avoid mypy errors during initialization of the singleton
                    from typing import cast

                    inst = super().__new__(cls)
                    # Initialize attributes that mypy expects to exist
                    object.__setattr__(inst, "_counters", {})
                    object.__setattr__(inst, "_histograms", {})
                    cls._instance = cast(MetricsRegistry, inst)
        assert cls._instance is not None
        return cls._instance

    # -- Counters --

    def counter(self, name: str) -> Counter:
        if name not in self._counters:
            self._counters[name] = Counter()
        return self._counters[name]

    # -- Histograms --

    def histogram(self, name: str) -> Histogram:
        if name not in self._histograms:
            self._histograms[name] = Histogram()
        return self._histograms[name]

    # -- Snapshot --

    def snapshot(self) -> Dict[str, Any]:
        return {
            "counters": {k: v.snapshot() for k, v in self._counters.items()},
            "histograms": {k: v.snapshot() for k, v in self._histograms.items()},
        }

    def reset(self) -> None:
        """Reset all metrics — useful for testing."""
        self._counters.clear()
        self._histograms.clear()


# -- Convenience singleton --
METRICS = MetricsRegistry()


class timer:
    """Context manager that records elapsed time into a Histogram.

    Usage::

        with timer("recommend.ann_latency_ms"):
            ids = ann_candidates(...)
    """

    __slots__ = ("_name", "_start")

    def __init__(self, name: str) -> None:
        self._name = name
        self._start = 0.0

    def __enter__(self) -> "timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        METRICS.histogram(self._name).observe(elapsed_ms)
