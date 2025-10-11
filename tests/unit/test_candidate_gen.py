from __future__ import annotations

import numpy as np

from api.core.candidate_gen import ann_candidates


class DummyResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class DummySession:
    def __init__(self):
        self.calls = []
        self.rows = [(42,), (7,)]

    def execute(self, statement, params):
        self.calls.append((statement, params))
        return DummyResult(self.rows)


def test_ann_candidates_returns_empty_when_vector_missing():
    db = DummySession()
    assert ann_candidates(db, None, exclude_ids=[1, 2]) == []
    assert db.calls == []


def test_ann_candidates_executes_query_and_returns_ids():
    db = DummySession()
    vec = np.array([0.1, 0.9, 0.0], dtype="float32")

    result = ann_candidates(db, vec, exclude_ids=[1], limit=5)

    assert result == [42, 7]
    assert len(db.calls) == 1
    _, params = db.calls[0]
    assert params["exclude"] == [1]
    assert params["lim"] == 5
    assert params["uvec"] == [float(x) for x in vec]
