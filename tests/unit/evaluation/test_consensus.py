"""Unit tests for pooling, consensus adjudication, and UNJUDGED handling."""

import pytest

from evaluation.judge.consensus import (
    ConsensusJudgeEngine,
    JudgmentCache,
    PoolAdjudicator,
)
from evaluation.judge.stub import StubJudgeAdapter
from evaluation.models import ItemEvidence, TypedId


def test_pool_deduplication(tmp_path):
    cache = JudgmentCache(path=tmp_path / "cache.json")
    primary = StubJudgeAdapter()
    adjudicator = PoolAdjudicator(primary_judge=primary, cache=cache)

    cand_list1 = ["movie:1", "movie:2", "movie:3"]
    cand_list2 = ["movie:2", "movie:4", "movie:5"]
    cand_list3 = [{"tmdb_id": 1, "media_type": "movie"}, {"tmdb_id": 6}]

    pooled = adjudicator.deduplicate_pool(
        [cand_list1, cand_list2, cand_list3], max_pool_size=10
    )
    assert pooled == ["movie:1", "movie:2", "movie:3", "movie:4", "movie:5", "movie:6"]
    assert len(pooled) == 6


def test_exploratory_single_judge_mode(tmp_path):
    cache = JudgmentCache(path=tmp_path / "cache.json")
    primary = StubJudgeAdapter(fixed_grade=2, fixed_sufficiency=True)
    adjudicator = PoolAdjudicator(primary_judge=primary, cache=cache)

    ev = ItemEvidence(
        typed_id=TypedId("movie", 10), title="Test Title", synopsis="Overview"
    )
    rec = adjudicator.adjudicate_pair(
        query="test query", evidence=ev, mode="single_judge"
    )
    assert rec.grade == 2
    assert rec.status == "ACCEPTED"
    assert rec.consensus_model_count == 1


def test_consensus_exact_agreement(tmp_path):
    cache = JudgmentCache(path=tmp_path / "cache.json")
    primary = StubJudgeAdapter(model_name="p1", fixed_grade=2, fixed_sufficiency=True)
    secondary = StubJudgeAdapter(model_name="p2", fixed_grade=2, fixed_sufficiency=True)
    tie_breaker = StubJudgeAdapter(
        model_name="p3", fixed_grade=1, fixed_sufficiency=True
    )

    adjudicator = PoolAdjudicator(
        primary_judge=primary,
        secondary_judge=secondary,
        tie_breaker_judge=tie_breaker,
        cache=cache,
    )
    ev = ItemEvidence(
        typed_id=TypedId("movie", 20), title="Agree Title", synopsis="Synopsis"
    )
    rec = adjudicator.adjudicate_pair(
        query="agree query", evidence=ev, mode="consensus"
    )

    # Both primary and secondary agree on grade 2 -> accepted directly
    assert rec.grade == 2
    assert rec.status == "ACCEPTED"
    assert rec.consensus_model_count == 2


def test_consensus_disagreement_resolved_by_tie_breaker(tmp_path):
    cache = JudgmentCache(path=tmp_path / "cache.json")
    primary = StubJudgeAdapter(model_name="p1", fixed_grade=2, fixed_sufficiency=True)
    secondary = StubJudgeAdapter(model_name="p2", fixed_grade=1, fixed_sufficiency=True)
    # Tie-breaker agrees with primary on grade 2
    tie_breaker = StubJudgeAdapter(
        model_name="p3", fixed_grade=2, fixed_sufficiency=True
    )

    adjudicator = PoolAdjudicator(
        primary_judge=primary,
        secondary_judge=secondary,
        tie_breaker_judge=tie_breaker,
        cache=cache,
    )
    ev = ItemEvidence(
        typed_id=TypedId("movie", 30), title="Dispute Title", synopsis="Synopsis"
    )
    rec = adjudicator.adjudicate_pair(
        query="dispute query", evidence=ev, mode="consensus"
    )

    # 2 out of 3 models voted grade 2 -> accepted
    assert rec.grade == 2
    assert rec.status == "ACCEPTED"
    assert rec.consensus_model_count == 3


def test_consensus_unresolved_retains_unjudged(tmp_path):
    cache = JudgmentCache(path=tmp_path / "cache.json")
    # All 3 models assign distinct grades (0, 1, 2)
    primary = StubJudgeAdapter(model_name="p1", fixed_grade=0, fixed_sufficiency=True)
    secondary = StubJudgeAdapter(model_name="p2", fixed_grade=1, fixed_sufficiency=True)
    tie_breaker = StubJudgeAdapter(
        model_name="p3", fixed_grade=2, fixed_sufficiency=True
    )

    adjudicator = PoolAdjudicator(
        primary_judge=primary,
        secondary_judge=secondary,
        tie_breaker_judge=tie_breaker,
        cache=cache,
    )
    ev = ItemEvidence(
        typed_id=TypedId("movie", 40), title="Conflict Title", synopsis="Synopsis"
    )
    rec = adjudicator.adjudicate_pair(
        query="conflict query", evidence=ev, mode="consensus"
    )

    # No 2 models agree -> must retain UNJUDGED (grade is None, not forced to 0)
    assert rec.grade is None
    assert rec.status == "UNJUDGED"
    assert rec.consensus_model_count == 3


def test_consensus_engine_qrels_saving(tmp_path):
    cache = JudgmentCache(path=tmp_path / "cache.json")
    primary = StubJudgeAdapter(model_name="p1", fixed_grade=3, fixed_sufficiency=True)
    adjudicator = PoolAdjudicator(primary_judge=primary, cache=cache)
    engine = ConsensusJudgeEngine(
        adjudicator=adjudicator,
        qrels_dir=tmp_path / "qrels",
        version="test_v1",
    )

    ev1 = ItemEvidence(typed_id=TypedId("movie", 101), title="Film 1", synopsis="Syn 1")
    ev2 = ItemEvidence(typed_id=TypedId("movie", 102), title="Film 2", synopsis="Syn 2")

    qrels, records, had_changes = engine.label_pool(
        query="space drama",
        pool_evidence=[ev1, ev2],
        mode="single_judge",
    )
    assert had_changes is True
    assert qrels["movie:101"] == 3.0
    assert qrels["movie:102"] == 3.0
    assert engine.qrels_file.exists()


def test_cache_contract_and_immutable_qrels(tmp_path, monkeypatch):
    import evaluation.judge.base as base
    from evaluation.models import JudgeInput

    judge = StubJudgeAdapter(fixed_grade=2)
    evidence = ItemEvidence(TypedId("movie", 7), "Film", "Full synopsis")
    inp = JudgeInput(query="film", evidence=evidence)
    cache_file = tmp_path / "cache.json"
    cache_file.write_text("{broken")
    cache = JudgmentCache(cache_file)
    adjudicator = PoolAdjudicator(judge, cache=cache)
    assert adjudicator.adjudicate_pair("film", evidence).grade == 2
    calls = judge.call_count
    assert adjudicator.adjudicate_pair("film", evidence).grade == 2
    assert judge.call_count == calls
    monkeypatch.setattr(base, "ADAPTER_CONTRACT_VERSION", "next-contract")
    assert cache.get(inp, judge) is None
    assert adjudicator.deduplicate_pool([["unparseable", "movie:1", "movie:2"]], 2) == [
        "unparseable",
        "movie:1",
    ]
    old_file = tmp_path / "qrels_v2.0.json"
    old_file.write_text('{"old": {"movie:7": 3}}')
    engine = ConsensusJudgeEngine(primary_judge=judge, cache=cache, qrels_dir=tmp_path)
    assert engine.version == "v2.7"
    engine.label_pool("film", [evidence])
    assert old_file.read_text() == '{"old": {"movie:7": 3}}'
    assert engine.get_query_qrels("film") == {"movie:7": 2.0}
    published = engine.publish_version("approved")
    before = published.read_bytes()
    with pytest.raises(FileExistsError):
        engine.publish_version("approved")
    assert published.read_bytes() == before
    judge.fixed_sufficiency = False
    judge.checkpoint_revision = "new-checkpoint"
    qrels, records, changed = engine.label_pool("film", [evidence])
    assert changed and not qrels and records[0].grade is None
    engine.qrels_file.write_text("{broken")
    engine.load_qrels()
    assert not engine.get_query_qrels("film")
    with pytest.raises(ValueError):
        ConsensusJudgeEngine()
