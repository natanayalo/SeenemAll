from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from api.core.fast_intent_parser import (
    DeterministicRuleParser,
    FastIntentParser,
    needs_neural_inference,
    parse_fast_intent,
)
from api.core.llm_parser import parse_intent, _get_settings


def test_deterministic_runtime_parsing():
    parser = DeterministicRuleParser()

    res1 = parser.parse("action movies under 90 minutes")
    assert res1["runtime_minutes_max"] == 90
    assert "Action" in res1["include_genres"]
    assert res1["media_types"] == ["movie"]

    res2 = parser.parse("thrillers less than 2 hours")
    assert res2["runtime_minutes_max"] == 120

    res3 = parser.parse("sci-fi movies over 150 mins")
    assert res3["runtime_minutes_min"] == 150


def test_deterministic_year_and_decade_parsing():
    parser = DeterministicRuleParser()

    res_decade = parser.parse("90s action movies")
    assert res_decade["year_min"] == 1990
    assert res_decade["year_max"] == 1999

    res_exact = parser.parse("comedies from 1999")
    assert res_exact["year_min"] == 1999
    assert res_exact["year_max"] == 1999

    res_range = parser.parse("dramas between 1980 and 1990")
    assert res_range["year_min"] == 1980
    assert res_range["year_max"] == 1990

    res_after = parser.parse("horror after 2015")
    assert res_after["year_min"] == 2015


def test_deterministic_negation_parsing():
    parser = DeterministicRuleParser()

    res = parser.parse("romantic comedy without horror and no drama")
    assert "Romance" in res["include_genres"]
    assert "Comedy" in res["include_genres"]
    assert "Horror" in res["exclude_genres"]
    assert "Drama" in res["exclude_genres"]


def test_deterministic_genres_and_synonyms():
    parser = DeterministicRuleParser()

    res_romcom = parser.parse("funny rom-com")
    assert "Romance" in res_romcom["include_genres"]
    assert "Comedy" in res_romcom["include_genres"]

    res_superhero = parser.parse("superhero film")
    assert "Action" in res_superhero["include_genres"]
    assert "Science Fiction" in res_superhero["include_genres"]


def test_deterministic_streaming_providers():
    parser = DeterministicRuleParser()

    res_netflix = parser.parse("sci-fi movies on netflix")
    assert "netflix" in res_netflix["streaming_providers"]

    res_apple = parser.parse("dramas on apple tv+")
    assert "apple tv+" in res_apple["streaming_providers"]


def test_deterministic_languages():
    parser = DeterministicRuleParser()

    res_fr = parser.parse("french romantic comedy")
    assert "fr" in res_fr["languages"]

    res_ko = parser.parse("korean thrillers")
    assert "ko" in res_ko["languages"]


def test_deterministic_maturity_rating():
    parser = DeterministicRuleParser()

    res_r = parser.parse("action movies rated R")
    assert res_r["maturity_rating_max"] == "R"

    res_pg = parser.parse("family movies PG-13")
    assert res_pg["maturity_rating_max"] == "PG-13"


def test_fast_intent_parser_empty():
    parser = FastIntentParser()
    res = parser.parse("")
    assert res["include_genres"] is None
    assert res["include_people"] is None
    assert res["include_actors"] is None
    assert res["include_directors"] is None
    assert res["include_producers"] is None
    assert res["include_writers"] is None
    assert res["reference_titles"] is None
    assert res["franchises"] is None


def test_fast_intent_parser_role_extraction():
    parser = FastIntentParser()
    mock_gliner = MagicMock()
    mock_gliner.predict_entities.return_value = [
        {"label": "director", "text": "Christopher Nolan"},
        {"label": "actor", "text": "Keanu Reeves"},
        {"label": "producer", "text": "Steven Spielberg"},
        {"label": "writer", "text": "Quentin Tarantino"},
        {"label": "person", "text": "Pedro Pascal"},
    ]

    with patch.object(parser, "_ensure_gliner", return_value=mock_gliner):
        res = parser.parse(
            "films by Nolan with Keanu produced by Spielberg written by Tarantino with Pedro"
        )
        assert res["include_directors"] == ["Christopher Nolan"]
        assert res["include_actors"] == ["Keanu Reeves"]
        assert res["include_producers"] == ["Steven Spielberg"]
        assert res["include_writers"] == ["Quentin Tarantino"]
        assert "Christopher Nolan" in res["include_people"]
        assert "Keanu Reeves" in res["include_people"]
        assert "Steven Spielberg" in res["include_people"]
        assert "Quentin Tarantino" in res["include_people"]
        assert "Pedro Pascal" in res["include_people"]


def test_fast_intent_parser_reference_titles_and_franchises():
    parser = FastIntentParser()
    mock_gliner = MagicMock()
    mock_gliner.predict_entities.return_value = [
        {"label": "movie", "text": "Arrival"},
        {"label": "tv_show", "text": "Severance"},
        {"label": "franchise", "text": "Star Wars"},
        {"label": "movie", "text": "movie"},  # Should be filtered out as generic
    ]

    with patch.object(parser, "_ensure_gliner", return_value=mock_gliner):
        res = parser.parse("movies like Arrival and Severance in Star Wars universe")
        assert res["franchises"] == ["Star Wars"]
        assert "Arrival" in res["reference_titles"]
        assert "Severance" in res["reference_titles"]
        assert "Star Wars" in res["reference_titles"]
        assert "movie" not in res["reference_titles"]


def test_fast_intent_parser_gliner_entities():
    parser = FastIntentParser()

    mock_gliner = MagicMock()
    mock_gliner.predict_entities.return_value = [
        {"label": "person", "text": "Pedro Pascal"},
        {"label": "genre", "text": "cyberpunk"},
        {"label": "streaming_service", "text": "hulu"},
        {"label": "language", "text": "spanish"},
    ]

    with patch.object(parser, "_ensure_gliner", return_value=mock_gliner):
        res = parser.parse("Pedro Pascal cyberpunk movies on hulu in spanish")
        assert "Pedro Pascal" in res["include_people"]
        assert "hulu" in res["streaming_providers"]
        assert "es" in res["languages"]


def test_fast_intent_parser_gliner_failure_fallback():
    parser = FastIntentParser()

    mock_gliner = MagicMock()
    mock_gliner.predict_entities.side_effect = [
        RuntimeError("GPU out of memory"),
        [{"label": "person", "text": "Pedro Pascal"}],
    ]
    counters = {}

    def get_counter(name):
        counters.setdefault(name, MagicMock())
        return counters[name]

    metrics = MagicMock()
    metrics.counter.side_effect = get_counter
    with patch.object(parser, "_ensure_gliner", return_value=mock_gliner), patch(
        "api.core.fast_intent_parser.METRICS", metrics
    ):
        res = parser.parse("Pedro Pascal sci-fi movies under 90 minutes")
        # Should gracefully fall back to rule parser outputs
        assert res["runtime_minutes_max"] == 90
        assert "Science Fiction" in res["include_genres"]
        assert parser.gliner_failed
        assert parser._gliner_error is not None

        recovered = parser.parse("Pedro Pascal sci-fi movies under 90 minutes")

    assert recovered["include_people"] == ["Pedro Pascal"]
    assert not parser.gliner_failed
    assert parser._gliner_error is None
    assert counters["intent.parser.neural_error"].inc.call_count == 1
    assert counters["intent.parser.neural_fallback"].inc.call_count == 1
    assert counters["intent.parser.path.rules"].inc.call_count == 1
    assert counters["intent.parser.path.neural"].inc.call_count == 1


def test_require_gliner_fails_when_model_cannot_load():
    parser = FastIntentParser()
    parser._gliner_error = FileNotFoundError("model.xml is missing")

    with patch.object(parser, "_ensure_gliner", return_value=None):
        with pytest.raises(RuntimeError, match="could not be loaded"):
            parser.require_gliner()


def test_require_gliner_runs_startup_inference():
    parser = FastIntentParser()
    model = MagicMock()

    with patch.object(parser, "_ensure_gliner", return_value=model):
        assert parser.require_gliner() is model

    model.predict_entities.assert_called_once()


def test_fast_intent_parser_singleton():
    inst1 = FastIntentParser.get_instance()
    inst2 = FastIntentParser.get_instance()
    assert inst1 is inst2

    res = parse_fast_intent("90s action movies")
    assert res["year_min"] == 1990
    assert "Action" in res["include_genres"]


def test_llm_parser_hybrid_fast_provider(monkeypatch):
    monkeypatch.setenv("INTENT_PROVIDER", "hybrid_fast")
    monkeypatch.setenv("INTENT_CACHE_PERSIST", "0")
    _get_settings.cache_clear()
    from api.core.llm_parser import (
        _persistent_intent_store,
        _persistent_cache_for_namespace,
    )

    _persistent_intent_store.cache_clear()
    _persistent_cache_for_namespace.cache_clear()

    settings = _get_settings()
    assert settings.provider == "hybrid_fast"
    assert settings.enabled is True

    intent = parse_intent(
        "90s action thrillers under 105 minutes without horror",
        {"user_id": "u_test_fresh"},
    )
    assert intent.year_min == 1990
    assert intent.year_max == 1999
    assert intent.runtime_minutes_max == 105
    assert "Action" in intent.include_genres
    assert "Horror" in intent.exclude_genres


def test_needs_neural_inference_logic():
    # Fully explained queries should NOT need neural inference
    assert not needs_neural_inference(
        "action movies from the 90s under 100 minutes on Netflix", {}
    )
    assert not needs_neural_inference("romantic comedy without horror", {})
    assert not needs_neural_inference("PG-13 animated movies", {})
    assert not needs_neural_inference("disney plus movies", {})
    assert not needs_neural_inference(
        "French dramas between 1990 and 2000 on HBO Max", {}
    )

    # Queries with unresolved person/title/thematic text MUST trigger neural inference
    assert needs_neural_inference("Pedro Pascal movies", {})
    assert needs_neural_inference("movies like Arrival with Amy Adams", {})
    assert needs_neural_inference("gritty cyberpunk thrillers", {})


def test_needs_neural_inference_adversarial():
    # 1. Single-token surnames
    assert needs_neural_inference("Nolan films", {})
    assert needs_neural_inference("Tarantino movies", {})
    assert needs_neural_inference("Kubrick", {})
    assert needs_neural_inference("Scorsese crime dramas", {})
    assert needs_neural_inference("Fincher thrillers", {})

    # 2. Modern actors / mononyms
    assert needs_neural_inference("with Zendaya", {})
    assert needs_neural_inference("starring Chalamet", {})
    assert needs_neural_inference("Denis Villeneuve sci-fi", {})

    # 3. Short surnames (>= 2 chars)
    assert needs_neural_inference("films with Ed", {})
    assert needs_neural_inference("starring Jet Li", {})

    # 4. Surnames that overlap ordinary vocabulary
    assert needs_neural_inference("films with Ford", {})
    assert needs_neural_inference("Bale thrillers", {})
    assert needs_neural_inference("Emma Stone comedies", {})

    # 5. Pure vibe / unconstrained semantic descriptors
    assert needs_neural_inference("surreal slow-burn psychological horror", {})
    assert needs_neural_inference("melancholic neo-noir detective mystery", {})


def test_adaptive_gating_skips_gliner():
    parser = FastIntentParser()
    mock_gliner = MagicMock()

    with patch.object(parser, "_ensure_gliner", return_value=mock_gliner):
        # 1. Structured query -> GLiNER NOT called
        res_structured = parser.parse(
            "action movies from the 90s under 100 minutes on Netflix"
        )
        assert mock_gliner.predict_entities.call_count == 0
        assert res_structured["runtime_minutes_max"] == 100
        assert "Action" in res_structured["include_genres"]

        # 2. Entity query -> GLiNER called
        parser.parse("Pedro Pascal movies")
        assert mock_gliner.predict_entities.call_count == 1


def test_openvino_runtime_loader(monkeypatch):
    monkeypatch.setenv("FAST_INTENT_RUNTIME", "openvino")
    monkeypatch.setenv("FAST_INTENT_OPENVINO_DEVICE", "CPU")
    parser = FastIntentParser()
    parser._gliner_loaded = False
    parser._gliner_failed = False
    parser._runtime = "openvino"

    mock_ov_model = MagicMock()
    mock_gliner_mod = MagicMock()
    mock_gliner_mod.GLiNER.from_pretrained.return_value = mock_ov_model

    real_exists = os.path.exists

    def fake_exists(path):
        if "model.xml" in str(path):
            return True
        return real_exists(path)

    with patch("os.path.exists", side_effect=fake_exists), patch.dict(
        "sys.modules", {"gliner": mock_gliner_mod}
    ):
        model = parser._ensure_gliner()
        assert model is mock_ov_model
        mock_gliner_mod.GLiNER.from_pretrained.assert_called_once()


def test_comparative_person_routed_to_reference_titles():
    parser = FastIntentParser()
    mock_gliner = MagicMock()

    mock_gliner.predict_entities.return_value = [
        {"start": 12, "end": 29, "text": "Quentin Tarantino", "label": "director"}
    ]

    with patch.object(parser, "_ensure_gliner", return_value=mock_gliner):
        res_comp = parser.parse("movies like Quentin Tarantino")
        assert res_comp.get("include_directors") is None
        assert res_comp.get("reference_titles") == ["Quentin Tarantino"]

    mock_gliner.predict_entities.return_value = [
        {"start": 10, "end": 27, "text": "Quentin Tarantino", "label": "director"}
    ]

    with patch.object(parser, "_ensure_gliner", return_value=mock_gliner):
        res_filter = parser.parse("movies by Quentin Tarantino")
        assert res_filter.get("include_directors") == ["Quentin Tarantino"]
        assert res_filter.get("reference_titles") is None


def test_deterministic_vibe_lexicon_parsing():
    parser = DeterministicRuleParser()

    queries = [
        ("existential dread indie thrillers", "Thriller", ["existentialism", "dread"]),
        ("cozy autumn mystery", "Mystery", ["autumn", "whodunit"]),
        ("neon cyberpunk anime noir", "Animation", ["cyberpunk", "neo-noir"]),
        (
            "surreal slow-burn psychological horror",
            "Horror",
            ["slow burn", "psychological horror"],
        ),
        ("feel-good road trip indie comedy", "Comedy", ["road trip", "friendship"]),
        ("dark satirical dystopian black comedy", "Comedy", ["satire", "dystopia"]),
        (
            "claustrophobic isolated survival thrillers",
            "Thriller",
            ["survival", "isolation"],
        ),
        (
            "whimsical magical realism romance",
            "Romance",
            ["magical realism", "whimsical"],
        ),
        ("gritty neo-western crime", "Western", ["neo-western", "desert"]),
        (
            "philosophical hard science fiction",
            "Science Fiction",
            ["hard science fiction", "philosophical"],
        ),
        ("gripping courtroom legal drama", "Drama", ["courtroom", "legal drama"]),
    ]

    for q, expected_genre, expected_kws in queries:
        res = parser.parse(q)
        assert res["is_vibe"] is True, f"Query '{q}' should be flagged as is_vibe"
        assert (
            res["include_genres"] is not None
            and expected_genre in res["include_genres"]
        ), f"Query '{q}' should include genre '{expected_genre}', got {res['include_genres']}"
        assert res["keywords"] is not None, f"Query '{q}' should have keywords"
        for kw in expected_kws:
            assert (
                kw in res["keywords"]
            ), f"Query '{q}' should contain keyword '{kw}', got {res['keywords']}"


def test_media_types_detects_both_movie_and_tv_when_present():
    parser = DeterministicRuleParser()
    res = parser.parse("sci-fi movies and tv shows")
    assert "movie" in res["media_types"]
    assert "tv" in res["media_types"]
