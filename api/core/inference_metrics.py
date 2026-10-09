"""Request-owned inference diagnostics, propagated explicitly into model workers."""

from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict


@dataclass
class InferenceCollector:
    providers: Dict[str, Dict[str, int]] = field(default_factory=dict)
    lock: Any = field(default_factory=Lock)
    closed: bool = False

    def record(
        self, provider: str, event: str, count: int = 1, scored_items: int = 0
    ) -> None:
        with self.lock:
            if self.closed:
                return
            stats = self.providers.setdefault(
                provider,
                {
                    "successful_calls": 0,
                    "scored_items": 0,
                    "cache_hits": 0,
                    "failures": 0,
                },
            )
            stats[event] += count
            stats["scored_items"] += scored_items

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            providers = {name: dict(stats) for name, stats in self.providers.items()}
        return {
            "actual_inferences_performed": sum(
                s["successful_calls"] for s in providers.values()
            ),
            "cache_hits": sum(s["cache_hits"] for s in providers.values()),
            "inference_failures": sum(s["failures"] for s in providers.values()),
            "providers": providers,
        }

    def close(self) -> None:
        with self.lock:
            self.closed = True


_collector: ContextVar[InferenceCollector | None] = ContextVar(
    "inference_collector", default=None
)


@contextmanager
def inference_request():
    collector = InferenceCollector()
    token = _collector.set(collector)
    try:
        yield collector
    finally:
        collector.close()
        _collector.reset(token)


def record_success(provider: str, items: int) -> None:
    collector = _collector.get()
    if collector is not None:
        collector.record(provider, "successful_calls", scored_items=items)


def record_cache_hit(provider: str) -> None:
    collector = _collector.get()
    if collector is not None:
        collector.record(provider, "cache_hits")


def record_failure(provider: str) -> None:
    collector = _collector.get()
    if collector is not None:
        collector.record(provider, "failures")


def submit_with_inference_context(executor, function, *args):
    return executor.submit(copy_context().run, function, *args)
