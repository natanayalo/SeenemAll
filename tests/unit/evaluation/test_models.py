"""Unit tests for evaluation data models, schemas, and contracts."""

import pytest

from evaluation.models import (
    EvaluationStatus,
    ItemEvidence,
    JudgeProvenance,
    JudgeOutput,
    RubricGrade,
    TypedId,
    select_grade_from_probabilities,
)


def test_typed_id_parsing_and_equality():
    tid1 = TypedId.parse("movie:1893")
    assert tid1.media_type == "movie"
    assert tid1.id == 1893
    assert str(tid1) == "movie:1893"

    tid2 = TypedId.parse({"media_type": "tv", "id": 456})
    assert tid2.media_type == "tv"
    assert tid2.id == 456
    assert str(tid2) == "tv:456"

    tid3 = TypedId.parse(789)
    assert tid3.media_type == "movie"
    assert tid3.id == 789

    assert tid1 == TypedId("movie", 1893)
    assert hash(tid1) == hash(TypedId("movie", 1893))
    assert tid1 != tid2

    with pytest.raises(ValueError):
        TypedId.parse({})

    with pytest.raises(TypeError):
        TypedId.parse(None)


def test_item_evidence_formatting_and_missing_fields():
    ev = ItemEvidence(
        typed_id=TypedId("movie", 10),
        title="Test Movie",
        synopsis="A short description of a dramatic journey.",
        genres=["Drama"],
        media_type="movie",
    )
    # Missing fields auto-detected
    assert "keywords" in ev.missing_fields
    assert "cast" in ev.missing_fields
    assert "crew" in ev.missing_fields
    assert "release_year" in ev.missing_fields

    text = ev.to_evidence_text(max_chars=2000)
    assert "Title: Test Movie" in text
    assert "Media Type: movie" in text
    assert "Genres: Drama" in text
    assert "Explicitly Missing Metadata:" in text

    # Deterministic content hash is stable
    h1 = ev.content_hash()
    h2 = ev.content_hash()
    assert h1 == h2
    assert len(h1) == 64


def test_item_evidence_truncation_preserves_structure():
    long_synopsis = "word " * 1000
    ev = ItemEvidence(
        typed_id=TypedId("movie", 99),
        title="Long Synopsis Movie",
        synopsis=long_synopsis,
        genres=["Action"],
    )
    truncated = ev.to_evidence_text(max_chars=650)
    assert len(truncated) <= 650
    assert "[TRUNCATED]" in truncated
    assert "Title: Long Synopsis Movie" in truncated


def test_judge_grade_selection_and_ties():
    # Normal argmax
    probs = {0: 0.1, 1: 0.2, 2: 0.6, 3: 0.1}
    assert select_grade_from_probabilities(probs) == 2

    # Exact ties break toward lower grade
    tied_probs = {0: 0.1, 1: 0.4, 2: 0.4, 3: 0.1}
    assert select_grade_from_probabilities(tied_probs) == 1

    tied_zero_one = {0: 0.5, 1: 0.5, 2: 0.0, 3: 0.0}
    assert select_grade_from_probabilities(tied_zero_one) == 0

    assert select_grade_from_probabilities({}) == 0


def test_rubric_grades_and_status_from_legacy():
    assert RubricGrade.is_positive(3) is True
    assert RubricGrade.is_positive(2) is True
    assert RubricGrade.is_positive(1) is False
    assert RubricGrade.is_positive(0) is False
    assert RubricGrade.is_positive(None) is False

    assert EvaluationStatus.from_legacy("PASS") == EvaluationStatus.PASS
    assert EvaluationStatus.from_legacy("FAIL") == EvaluationStatus.FAIL
    assert EvaluationStatus.from_legacy("INVALID") == EvaluationStatus.INVALID
    assert (
        EvaluationStatus.from_legacy("NEEDS_ADJUDICATION")
        == EvaluationStatus.INCONCLUSIVE
    )
    assert EvaluationStatus.from_legacy("UNKNOWN") == EvaluationStatus.INCONCLUSIVE


def test_judge_output_expected_score():
    prov = JudgeProvenance(
        model_name="test-model",
        checkpoint_revision="rev1",
        tokenizer_revision="rev1",
    )
    out = JudgeOutput(
        grade=2,
        probabilities={0: 0.1, 1: 0.2, 2: 0.5, 3: 0.2},
        evidence_sufficiency=True,
        execution_status="success",
        provenance=prov,
    )
    # Expected score = 0*0.1 + 1*0.2 + 2*0.5 + 3*0.2 = 0 + 0.2 + 1.0 + 0.6 = 1.8
    assert pytest.approx(out.expected_score, abs=1e-3) == 1.8
    d = out.to_dict()
    assert d["grade"] == 2
    assert d["evidence_sufficiency"] is True
    assert d["provenance"]["model_name"] == "test-model"
