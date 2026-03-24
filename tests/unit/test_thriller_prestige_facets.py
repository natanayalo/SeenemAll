from api.routes.recommend import (
    _constraint_bonus_for_thriller_facets,
    _constraint_bonus_for_prestige_and_superhero_facets,
    ConstraintBonusContext,
)


class MockIntentSignals:
    def __init__(self, facets=None, mood=None, prestige_indicator=False):
        self.semantic_facets = facets or []
        self.mood = mood
        self.prestige_indicator = prestige_indicator


class MockSignals:
    def __init__(self, **kwargs):
        self.money_psychology_thriller = False
        self.temporal_thriller = False
        self.high_concept_thriller = False
        self.cerebral_temporal_thriller = False
        self.serialized_prestige_tv = False
        self.street_level_superhero = False
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestThrillerFacets:
    def test_psychological_facet_bonus(self):
        """Psychological facet should trigger bonus even without exact keyword in query."""
        intent_signals = MockIntentSignals(facets=["psychological"])
        signals = MockSignals()
        ctx = ConstraintBonusContext(
            haystack="a tense drama about obsession",  # matches 'obsession' keyword in item metadata
            genre_names={"thriller", "drama"},
            original_language="en",
            media_type="movie",
            runtime=110,
            release_year=2020,
            maturity_rating="R",
            vote_count=1000,
        )
        bonus = _constraint_bonus_for_thriller_facets(intent_signals, signals, ctx)
        # 1. is_thriller_query (psychological facet) -> True
        # 2. thriller genre (+0.2)
        # 3. psychology_thriller check: drama (+0.12) + obsession keyword (+0.08)
        # 4. vote_count >= 500 (+0.1)
        # Total: 0.2 + 0.12 + 0.08 + 0.1 = 0.5
        assert abs(bonus - 0.5) < 0.01

    def test_temporal_facet_bonus(self):
        """Temporal/Time-loop facets should trigger Sci-Fi thriller bonuses."""
        intent_signals = MockIntentSignals(facets=["time-loop"])
        signals = MockSignals()
        ctx = ConstraintBonusContext(
            haystack="trapped in a day",
            genre_names={"thriller", "science fiction"},
            original_language="en",
            media_type="movie",
            runtime=95,
            release_year=2015,
            maturity_rating="PG-13",
            vote_count=5000,
        )
        bonus = _constraint_bonus_for_thriller_facets(intent_signals, signals, ctx)
        # 1. is_thriller_query (time-loop facet) -> True
        # 2. thriller genre (+0.2)
        # 3. temporal_thriller check: sci-fi (+0.28)
        # 4. vote_count >= 500 (+0.1)
        # Total: 0.2 + 0.28 + 0.1 = 0.58
        assert abs(bonus - 0.58) < 0.01

    def test_dark_mood_thriller_boost(self):
        """Mood='dark' should boost thriller genres and penalize comedy."""
        intent_signals = MockIntentSignals(facets=["thriller"], mood="dark")
        signals = MockSignals()

        # Boost case
        ctx_dark = ConstraintBonusContext(
            haystack="",
            genre_names={"thriller"},
            original_language="en",
            media_type="movie",
            runtime=120,
            release_year=2022,
            maturity_rating="R",
            vote_count=1000,
        )
        bonus_dark = _constraint_bonus_for_thriller_facets(
            intent_signals, signals, ctx_dark
        )
        # Thriller (+0.2) + dark mood boost (+0.16) + vote count (+0.1) = 0.46
        assert abs(bonus_dark - 0.46) < 0.01

        # Penalty case
        ctx_comedy = ConstraintBonusContext(
            haystack="",
            genre_names={"comedy", "family"},
            original_language="en",
            media_type="movie",
            runtime=90,
            release_year=2022,
            maturity_rating="PG",
            vote_count=1000,
        )
        bonus_comedy = _constraint_bonus_for_thriller_facets(
            intent_signals, signals, ctx_comedy
        )
        # dark mood penalty (-0.24) + vote count (+0.1) = -0.14
        assert abs(bonus_comedy - (-0.14)) < 0.01


class TestPrestigeFacets:
    def test_prestige_indicator_quality_bonus(self):
        """Prestige indicator + high vote count + recent release should give max bonus."""
        intent_signals = MockIntentSignals(facets=[], prestige_indicator=True)
        signals = MockSignals()
        ctx = ConstraintBonusContext(
            haystack="masterpiece",
            genre_names={"drama", "history"},
            original_language="en",
            media_type="movie",
            runtime=150,
            release_year=2022,
            maturity_rating="R",
            vote_count=10000,
        )
        bonus = _constraint_bonus_for_prestige_and_superhero_facets(
            intent_signals, signals, ctx
        )
        # Drama/History (+0.15) + recent (+0.16) + high votes (+0.25) + masterpiece keyword (+0.08)
        # Total: 0.15 + 0.16 + 0.25 + 0.08 = 0.64
        assert abs(bonus - 0.64) < 0.01

    def test_prestige_penalty_for_old_low_vote(self):
        """Old movies with low vote counts should get penalties even if 'prestige'."""
        intent_signals = MockIntentSignals(facets=[], prestige_indicator=True)
        signals = MockSignals()
        ctx = ConstraintBonusContext(
            haystack="drama",
            genre_names={"drama"},
            original_language="en",
            media_type="movie",
            runtime=90,
            release_year=1990,
            maturity_rating="PG",
            vote_count=50,
        )
        bonus = _constraint_bonus_for_prestige_and_superhero_facets(
            intent_signals, signals, ctx
        )
        # Drama (+0.15) - old (-0.18) - low votes (-0.1)
        # Total: 0.15 - 0.18 - 0.1 = -0.13
        assert abs(bonus - (-0.13)) < 0.01


class TestSuperheroFacets:
    def test_gritty_superhero_bonus(self):
        """Gritty/Street-level superhero facet should trigger crime/thriller boost."""
        intent_signals = MockIntentSignals(facets=["superhero", "gritty"])
        signals = MockSignals()
        ctx = ConstraintBonusContext(
            haystack="vigilante justice",
            genre_names={"action", "crime"},
            original_language="en",
            media_type="movie",
            runtime=120,
            release_year=2021,
            maturity_rating="R",
            vote_count=2000,
        )
        bonus = _constraint_bonus_for_prestige_and_superhero_facets(
            intent_signals, signals, ctx
        )
        # Base superhero: Action (+0.2) + Crime (+0.16) = 0.36
        # Gritty facet: Crime boost (+0.16)
        # Total: 0.36 + 0.16 = 0.52
        assert abs(bonus - 0.52) < 0.01

    def test_superhero_fantasy_penalty(self):
        """Superhero with fantasy/family genres should get a penalty (preferring 'gritty' logic when asked)."""
        intent_signals = MockIntentSignals(facets=["superhero"])
        signals = MockSignals()
        ctx = ConstraintBonusContext(
            haystack="magical hero",
            genre_names={
                "action & adventure",
                "fantasy",
            },  # fixed genre name to match logic
            original_language="en",
            media_type="movie",
            runtime=100,
            release_year=2018,
            maturity_rating="PG",
            vote_count=1000,
        )
        bonus = _constraint_bonus_for_prestige_and_superhero_facets(
            intent_signals, signals, ctx
        )
        # Base superhero: Action (+0.2)
        # Fantasy penalty (-0.18)
        # Total: 0.2 - 0.18 = 0.02
        assert abs(bonus - 0.02) < 0.01
