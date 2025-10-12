"""Tests for embedding template formats."""

import pytest
from etl.embedding_templates import format_with_template

TEST_ITEM = {
    "id": 1,
    "title": "The Matrix",
    "overview": "A computer programmer discovers a mysterious world beneath our reality.",
    "genres": [{"id": 878, "name": "Science Fiction"}, {"id": 28, "name": "Action"}],
    "release_year": 1999,
    "media_type": "movie",
    "runtime": 136,
}


def test_basic_template():
    text = format_with_template("basic", TEST_ITEM)
    assert "[Science Fiction, Action] The Matrix" in text
    assert "mysterious world" in text


def test_structured_template():
    text = format_with_template("structured", TEST_ITEM)
    assert "Title: The Matrix" in text
    assert "Genres: Science Fiction, Action" in text
    assert "Overview:" in text


def test_natural_template():
    text = format_with_template("natural", TEST_ITEM)
    assert "The Matrix is a Science Fiction and Action movie/show" in text
    assert "mysterious world" in text


def test_emphasized_template():
    text = format_with_template("emphasized", TEST_ITEM)
    assert "The Matrix #Science Fiction #Action" in text
    assert "mysterious world" in text


def test_unknown_template():
    with pytest.raises(ValueError, match="Unknown template"):
        format_with_template("nonexistent", TEST_ITEM)


def test_missing_fields():
    incomplete_item = {"title": "Test"}
    # Should work with missing fields
    text = format_with_template("basic", incomplete_item)
    assert "Test" in text
