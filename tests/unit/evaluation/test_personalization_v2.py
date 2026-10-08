"""Unit tests for synthetic personalization behavioral gates."""

from unittest.mock import MagicMock, patch
import pytest

from evaluation.personalization import PersonalizationHarness, load_synthetic_personas
from evaluation.runner import EvaluationRunner


@pytest.fixture(autouse=True)
def isolate_persona_database(monkeypatch):
    session = MagicMock()
    session.query.return_value.filter.return_value.count.return_value = 1
    monkeypatch.setattr("api.db.session.get_sessionmaker", lambda: lambda: session)
    monkeypatch.setattr(
        "api.core.user_profile.upsert_user_vectors", lambda *a, **kw: None
    )


def test_load_synthetic_personas():
    personas = load_synthetic_personas()
    assert "persona_scifi_enthusiast" in personas
    assert "persona_prestige_drama" in personas
    assert "persona_family_animation" in personas
    assert "persona_neutral_empty" in personas

    scifi = personas["persona_scifi_enthusiast"]
    assert len(scifi["seed_history"]) > 0
    assert len(scifi["known_negatives"]) > 0
    assert len(scifi["hidden_targets"]) > 0

    neutral = personas["persona_neutral_empty"]
    assert len(neutral["seed_history"]) == 0
    assert len(neutral["known_negatives"]) == 0


def test_personalization_harness_behavioral_gates():
    runner = EvaluationRunner(in_process=True)
    harness = PersonalizationHarness(runner=runner)

    mock_runner = MagicMock()

    # Candidate returns hidden target (e.g. 335984), masked baseline returns general items
    def mock_exec(query, user_id, params=None, limit=10, bypass_cache=True):
        if user_id == "eval_p_scifi":
            # Returns target item 335984 (Blade Runner 2049)
            items = [
                {"tmdb_id": 335984, "media_type": "movie", "title": "Blade Runner 2049"}
            ]
        else:
            # Masked baseline returns general item 11
            items = [{"tmdb_id": 11, "media_type": "movie", "title": "Star Wars"}]
        mock_trace = MagicMock()
        mock_trace.errors = []
        mock_trace.fallbacks = []
        return items, mock_trace

    mock_runner.execute_query.side_effect = mock_exec
    harness.runner = mock_runner

    res = harness.run_personalization_benchmark(k=5)
    assert res["passed"] is True
    assert res["mean_personalization_lift"] >= 0.0
    assert res["total_disliked_violations"] == 0
    assert res["max_persona_ndcg_decline"] <= 0.03


def test_seed_persona_fixtures():
    from evaluation.personalization import seed_persona_fixtures

    mock_db = MagicMock()
    persona = {
        "persona_id": "test_user_1",
        "seed_history": [{"tmdb_id": 100, "rating": 5.0}],
        "known_negatives": [{"tmdb_id": 200}],
    }
    with patch("api.core.user_profile.upsert_user_vectors") as mock_upsert:
        seed_persona_fixtures(mock_db, persona)
        assert mock_db.commit.called
        assert mock_upsert.called


def test_personalization_harness_error_failure():
    runner = EvaluationRunner(in_process=True)
    harness = PersonalizationHarness(runner=runner)

    mock_runner = MagicMock()

    def mock_exec(query, user_id, params=None, limit=10, bypass_cache=True):
        items = [{"tmdb_id": 11, "media_type": "movie", "title": "Star Wars"}]
        mock_trace = MagicMock()
        mock_trace.errors = ["HTTP 500 error in downstream retriever"]
        mock_trace.fallbacks = []
        return items, mock_trace

    mock_runner.execute_query.side_effect = mock_exec
    harness.runner = mock_runner

    res = harness.run_personalization_benchmark(k=5)
    assert res["passed"] is False
    assert res["execution_failed"] is True


def test_personalization_harness_empty_items_failure():
    runner = EvaluationRunner(in_process=True)
    harness = PersonalizationHarness(runner=runner)

    mock_runner = MagicMock()

    def mock_exec(query, user_id, params=None, limit=10, bypass_cache=True):
        items = []
        mock_trace = MagicMock()
        mock_trace.errors = []
        mock_trace.fallbacks = []
        return items, mock_trace

    mock_runner.execute_query.side_effect = mock_exec
    harness.runner = mock_runner

    res = harness.run_personalization_benchmark(k=5)
    assert res["passed"] is False
    assert res["execution_failed"] is True


def test_personalization_harness_disliked_violation():
    runner = EvaluationRunner(in_process=True)
    harness = PersonalizationHarness(runner=runner)

    mock_runner = MagicMock()

    def mock_exec(query, user_id, params=None, limit=10, bypass_cache=True):
        # 597 is Titanic, a known negative for persona_scifi_enthusiast
        items = [{"tmdb_id": 597, "media_type": "movie", "title": "Titanic"}]
        mock_trace = MagicMock()
        mock_trace.errors = []
        mock_trace.fallbacks = []
        return items, mock_trace

    mock_runner.execute_query.side_effect = mock_exec
    harness.runner = mock_runner

    res = harness.run_personalization_benchmark(k=5)
    assert res["passed"] is False
    assert res["total_disliked_violations"] > 0


@pytest.mark.parametrize("failure", ["commit", "empty_history"])
def test_failed_history_seeding_aborts_before_recommendation(failure):
    db = MagicMock()
    db.query.return_value.filter.return_value.count.return_value = 0
    if failure == "commit":
        db.commit.side_effect = RuntimeError("database write failed")
    runner = MagicMock()
    harness = PersonalizationHarness(
        runner,
        personas={
            "p": {
                "persona_id": "test-p",
                "seed_history": [{"tmdb_id": 1, "rating": 4}],
                "known_negatives": [],
            }
        },
        db_session_factory=lambda: db,
    )
    with pytest.raises(RuntimeError, match="seeding verification failed"):
        harness.seed_all_personas()
    assert not harness.seeding_verified and harness.seeding_error
    runner.execute_query.assert_not_called()
    db.close.assert_called_once()
