from api.routes.recommend import (
    _constraint_bonus_for_audience_facets,
    _constraint_bonus_for_scifi_fantasy_facets,
    ConstraintBonusContext,
)


class MockIntentSignals:
    def __init__(self, facets=None, mood=None, prestige_indicator=False):
        self.semantic_facets = facets or []
        self.mood = mood
        self.prestige_indicator = prestige_indicator


class MockSignals:
    def __init__(self, **kwargs):
        pass  # All signals are now deprecated


class TestAudienceFacets:
    def test_kids_animation_adventure_bonus(self):
        """'Kids' facet should trigger large bonuses for Animation/Adventure/Family genres."""
        intent_signals = MockIntentSignals(facets=["kids", "adventure", "quest"])
        ctx = ConstraintBonusContext(
            haystack="",
            genre_names={"animation", "adventure", "family"},
            original_language="en",
            media_type="tv",
            runtime=30,
            release_year=2022,
            maturity_rating="TV-Y7",
            vote_count=500,
        )
        bonus = _constraint_bonus_for_audience_facets(intent_signals, None, ctx, None)
        # is_kids_query: adventure (+0.24) + family (+0.26) + animation (+0.3) + TV (+0.22) + rating (+0.12)
        # + quest facet (+0.18)
        # Total = 1.32
        assert abs(bonus - 1.32) < 0.01

    def test_multilingual_family_bonus(self):
        """'Family' facet with non-english language should trigger international bonuses."""
        intent_signals = MockIntentSignals(facets=["family", "magic"])
        ctx = ConstraintBonusContext(
            haystack="",
            genre_names={"family", "animation", "fantasy"},
            original_language="ja",
            media_type="movie",
            runtime=120,
            release_year=2019,
            maturity_rating="PG",
            vote_count=2000,
        )
        bonus = _constraint_bonus_for_audience_facets(intent_signals, None, ctx, None)
        # is_intl_family: family (+0.22) + animation (+0.24) + fantasy (+0.20) + non-english (+0.18)
        # + kids (+0.08) + magic facet (+0.18)
        # Total = 1.10
        assert abs(bonus - 1.10) < 0.01


class TestScifiFantasyFacets:
    def test_anime_scifi_movie_bonus(self):
        """'Anime' and 'sci-fi' facets should trigger specific movie bonuses."""
        intent_signals = MockIntentSignals(facets=["anime", "sci-fi", "cyberpunk"])
        ctx = ConstraintBonusContext(
            haystack="futuristic tokyo mecha battle",
            genre_names={"animation", "science fiction"},
            original_language="ja",
            media_type="movie",
            runtime=110,
            release_year=2023,
            maturity_rating="R",
            vote_count=1500,
        )
        bonus = _constraint_bonus_for_scifi_fantasy_facets(intent_signals, None, ctx)
        # anime sci-fi movie: animation (+0.22) + sci-fi (+0.22) + japanese (+0.2)
        # + cyberpunk/mecha/future keywords (+0.16)
        # Total = 0.80
        assert abs(bonus - 0.80) < 0.01

    def test_epic_fantasy_tv_bonus(self):
        """'Epic' and 'fantasy' facets should trigger specific TV bonuses."""
        intent_signals = MockIntentSignals(
            facets=["epic", "fantasy", "dragon", "kingdom"]
        )
        ctx = ConstraintBonusContext(
            haystack="a saga of thrones and dragons",
            genre_names={"fantasy", "action & adventure"},
            original_language="en",
            media_type="tv",
            runtime=55,
            release_year=2021,
            maturity_rating="TV-MA",
            vote_count=10000,
        )
        bonus = _constraint_bonus_for_scifi_fantasy_facets(intent_signals, None, ctx)
        # epic fantasy tv: fantasy (+0.32) + adventure (+0.18)
        # + kingdom/throne/dragon keywords (+0.16)
        # Total = 0.66
        assert abs(bonus - 0.66) < 0.01

    def test_optimistic_space_opera_bonus(self):
        """'Optimistic' and 'sci-fi' facets for TV should trigger space opera bonuses."""
        intent_signals = MockIntentSignals(
            facets=["optimistic", "sci-fi", "exploration"]
        )
        ctx = ConstraintBonusContext(
            haystack="the crew of the starship explores the galaxy",
            genre_names={"science fiction", "adventure", "comedy"},
            original_language="en",
            media_type="tv",
            runtime=45,
            release_year=2018,
            maturity_rating="TV-PG",
            vote_count=2500,
        )
        bonus = _constraint_bonus_for_scifi_fantasy_facets(intent_signals, None, ctx)
        # optimistic sci-fi tv: sci-fi (+0.3) + adventure (+0.18)
        # + space/starship/crew/galaxy/exploration keywords (+0.16)
        # Total = 0.64
        assert abs(bonus - 0.64) < 0.01
