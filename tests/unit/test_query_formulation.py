from __future__ import annotations

from types import SimpleNamespace

from api.core.query_formulation import select_retrieval_query


def test_raw_query_is_the_default_and_preserves_user_text(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_QUERY_FORMULATION", raising=False)

    query = "  hopeful sci-fi movies under 2 hours on Netflix  "
    text, strategy = select_retrieval_query(query)

    assert text == query.strip()
    assert strategy == "raw"


def test_strategy_f_remains_an_explicit_experiment(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_QUERY_FORMULATION", "strategy_f")
    filters = SimpleNamespace(
        reference_titles=("Arrival",),
        cast=("Amy Adams",),
        directors=(),
    )

    text, strategy = select_retrieval_query(
        "cold_start: sci-fi movies under 2 hours from the 2010s on Netflix with Amy Adams",
        filters,
    )

    assert strategy == "strategy_f"
    assert "Netflix" not in text
    assert "2 hours" not in text
    assert "2010s" not in text
    assert "movies" not in text
    assert "sci-fi" in text
    assert "Arrival" in text
    assert "Amy Adams" in text


def test_unknown_formulation_falls_back_to_raw(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_QUERY_FORMULATION", "strategy_c")

    text, strategy = select_retrieval_query("mystery films on Hulu")

    assert text == "mystery films on Hulu"
    assert strategy == "raw"
