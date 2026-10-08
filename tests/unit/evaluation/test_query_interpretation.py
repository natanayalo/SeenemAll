import json
from unittest.mock import patch

import pytest

from tests.unit.evaluation.judge_helpers import make_nimble
from evaluation.models import ItemEvidence, JudgeInput, TypedId
from evaluation.query_interpretation import interpret_query


@pytest.mark.parametrize(
    "query,expected",
    [
        (
            "English-language drama movies from the 1990s over two hours.",
            "English-language drama movies released between 1990 and 1999 with runtime longer than 120 minutes.",
        ),
        (
            "films from 2000s under one hour",
            "films released between 2000 and 2009 with runtime shorter than 60 minutes",
        ),
        (
            "dramas from the 1980s at least 1.5 hours",
            "dramas released between 1980 and 1989 with runtime at least 90 minutes",
        ),
        ("films at most three hours", "films with runtime at most 180 minutes"),
        ("films longer than 2 hours", "films with runtime longer than 120 minutes"),
        ("films shorter than 1 hour", "films with runtime shorter than 60 minutes"),
        ("Movies From The 2010s", "Movies released between 2010 and 2019"),
        (
            "a story set in the 1990s that unfolds over two hours",
            "a story set in the 1990s that unfolds over two hours",
        ),
        ("actors from the 1990s", "actors from the 1990s"),
        (
            "movies released after 1990 under 120 minutes",
            "movies released after 1990 under 120 minutes",
        ),
    ],
)
def test_conservative_interpretation(query, expected):
    assert interpret_query(query) == expected
    assert interpret_query(expected) == expected


def test_all_prompt_paths_use_interpretation_without_mutating_audit_input():
    from tests.unit.evaluation.test_systemone_contract import response, reply

    inp = JudgeInput(
        "drama movies from the 1990s over two hours",
        ItemEvidence(
            TypedId("movie", 13),
            "Forrest Gump",
            "Plot",
            ["Drama"],
            release_year=1994,
            runtime=142,
        ),
    )
    original_hash = inp.input_hash()
    judge = make_nimble()
    assert interpret_query(inp.query) in judge.build_prompt(inp)
    with patch.object(judge, "is_available", return_value=True), patch(
        "urllib.request.urlopen", return_value=response(reply([0, 0, 0, 1]))
    ) as request:
        assert judge.judge_pair(inp).grade == 3
    payload = json.loads(request.call_args.args[0].data)
    assert payload["state"]["query"] == interpret_query(inp.query)
    assert inp.input_hash() == original_hash


def test_interpretation_version_invalidates_fingerprint(monkeypatch):
    judge = make_nimble()
    old = judge.qualification_fingerprint()
    monkeypatch.setattr("evaluation.judge.base.INTERPRETATION_VERSION", "v2")
    assert judge.qualification_fingerprint() != old
