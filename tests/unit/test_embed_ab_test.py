"""Tests for embedding A/B comparison utilities."""

import math

import numpy as np
import pytest

from api.db.models import ItemEmbedding
from scripts import embed_ab_test as ab


class DummyQuery:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return list(self._rows)


class DummySession:
    def __init__(self, rows):
        self._rows = rows

    def query(self, model):
        assert model is ItemEmbedding
        return DummyQuery(self._rows)


def test_load_embeddings_returns_expected_vectors():
    rows = [
        ItemEmbedding(item_id=1, version="v1", vector=[1.0, 0.0]),
        ItemEmbedding(item_id=2, version="v1", vector=[0.0, 1.0]),
        ItemEmbedding(item_id=1, version="v2", vector=[0.5, 0.5]),
    ]
    session = DummySession(rows)

    embeddings = ab.load_embeddings(session, "v1", [1, 2])

    assert embeddings == {1: [1.0, 0.0], 2: [0.0, 1.0]}


def test_load_embeddings_missing_items_raises():
    rows = [ItemEmbedding(item_id=1, version="v1", vector=[1.0, 0.0])]
    session = DummySession(rows)

    with pytest.raises(ValueError, match="Missing embeddings"):
        ab.load_embeddings(session, "v1", [1, 2])


def test_compare_versions_computes_metrics():
    embeddings_a = {
        1: [1.0, 0.0],
        2: [0.0, 1.0],
    }
    embeddings_b = {
        1: [np.sqrt(0.5), np.sqrt(0.5)],
        2: [np.sqrt(0.5), np.sqrt(0.5)],
    }
    pairs = [(1, 2, 0.0), (1, 1, 1.0)]

    report = ab.compare_versions(embeddings_a, embeddings_b, pairs, "base", "alt")

    assert math.isclose(report.mae_a, 0.0, rel_tol=1e-4)
    # Candidate embeddings produce identical vectors, so cross-item similarity is 1.0.
    assert math.isclose(report.mae_b, 0.5, rel_tol=1e-4)
    assert report.win_rate_b == 0.0
    assert report.tie_rate == 0.0
    assert len(report.pair_results) == 2
    first = report.pair_results[0]
    assert first.item_a == 1 and first.item_b == 2


def test_run_ab_test_invokes_helpers(monkeypatch):
    pairs = [(1, 2, 0.5)]
    monkeypatch.setattr(ab, "load_test_pairs", lambda: list(pairs))

    captured = []

    def fake_load(session, version, item_ids):
        captured.append((version, sorted(item_ids)))
        return {1: [1.0, 0.0], 2: [0.0, 1.0]}

    def fake_compare(a, b, pair_list, version_a, version_b):
        assert pair_list == pairs
        assert version_a == "base"
        assert version_b == "cand"
        assert a == b == {1: [1.0, 0.0], 2: [0.0, 1.0]}
        return "report"

    monkeypatch.setattr(ab, "load_embeddings", fake_load)
    monkeypatch.setattr(ab, "compare_versions", fake_compare)

    session = object()
    result = ab.run_ab_test(session, "base", "cand")

    assert result == "report"
    assert captured == [("base", [1, 2]), ("cand", [1, 2])]
