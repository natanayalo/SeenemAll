from __future__ import annotations

from typing import Optional
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from functools import lru_cache


_TEEN_QUERY_PATTERN = re.compile(
    r"\bteen(?:s|agers?)?\b|\bteen[\s-]?(?:safe|friendly)\b",
    flags=re.IGNORECASE,
)
_FAMILY_FRIENDLY_QUERY_PATTERN = re.compile(
    r"\bfamily([\s-]?(?:friendly|safe|movies?|filters?))?\b",
    flags=re.IGNORECASE,
)
_KIDS_PROFILE_QUERY_PATTERN = re.compile(
    r"\bkids?\s+profile\b",
    flags=re.IGNORECASE,
)
_KIDS_PROFILE_ADVENTURE_QUERY_PATTERN = re.compile(
    r"(?:\bkids?\s+profile\b.*\badventure\b)|(?:\badventure\b.*\bkids?\s+profile\b)",
    flags=re.IGNORECASE,
)
_NOT_TOO_DARK_QUERY_PATTERN = re.compile(
    r"\bnot[\s-]?too[\s-]?dark\b",
    flags=re.IGNORECASE,
)
_LAST_DECADE_QUERY_PATTERN = re.compile(
    r"\blast\s+decade\b",
    flags=re.IGNORECASE,
)
_FANTASY_TV_QUERY_PATTERN = re.compile(
    r"(?=.*\bfantasy\b)(?=.*\btv\b)",
    flags=re.IGNORECASE,
)
_SHORT_BINGEABLE_SCIFI_TV_QUERY_PATTERN = re.compile(
    r"(?=.*\bshort\b)(?=.*\bbingeable\b)(?=.*\bsci[\s-]?fi\b|\bscience[\s-]?fiction\b)(?=.*\btv\b)",
    flags=re.IGNORECASE,
)
_COMPARISON_SIGNAL_PATTERN = re.compile(
    r"\b(?:like|similar to|in the vein of|vibes?)\b",
    flags=re.IGNORECASE,
)
_FANTASY_WORLDS_CROSS_MEDIA_QUERY_PATTERN = re.compile(
    r"(?=.*\bfantasy\b)(?=.*\bworlds?\b)(?=.*\btv\b)(?=.*\bmovies?\b)",
    flags=re.IGNORECASE,
)
_ADULT_FEELGOOD_COMEDY_QUERY_PATTERN = re.compile(
    r"(?=.*\bfeel[\s-]?good\b)(?=.*\bcomed(?:y|ies)\b)",
    flags=re.IGNORECASE,
)
_TEEN_FRIENDLY_ADVENTURE_PROVIDER_QUERY_PATTERN = re.compile(
    r"(?=.*\bteen[\s-]?(?:safe|friendly)|\bteens?\b)(?=.*\badventure\b)(?=.*\bnetflix\b)",
    flags=re.IGNORECASE,
)
_DATE_NIGHT_QUERY_PATTERN = re.compile(
    r"\bdate[\s-]?night\b",
    flags=re.IGNORECASE,
)
_FAMILY_ADVENTURE_MOVIE_QUERY_PATTERN = re.compile(
    r"(?=.*\bfamily\b)(?=.*\badventure\b)(?=.*\bmovies?\b)",
    flags=re.IGNORECASE,
)
_MODERN_QUERY_PATTERN = re.compile(
    r"\bmodern\b|\bcontemporary\b|\bpresent[\s-]?day\b",
    flags=re.IGNORECASE,
)
_ANIMATION_AUDIENCE_PATTERN = re.compile(
    r"\b(animation|animated|anime|cartoon|cartoons|pixar|disney)\b",
    flags=re.IGNORECASE,
)
_AUDIENCE_SIGNAL_PATTERN = re.compile(
    r"\b(teen(?:s|agers?)?|teen[\s-]?(?:safe|friendly)|family|kids?|children|animation|animated|anime|cartoon|cartoons|pixar|disney)\b",
    flags=re.IGNORECASE,
)


def _normalize_query_text(query: str | None) -> str:
    return " ".join((query or "").lower().replace("-", " ").split())


def _query_has_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _ordered_tuple(values: list[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _is_adult_feelgood_movieish_query(query: str | None) -> bool:
    text = _normalize_query_text(query)
    if not text or not _ADULT_FEELGOOD_COMEDY_QUERY_PATTERN.search(query or ""):
        return False
    if "minute" not in text:
        return False
    if _AUDIENCE_SIGNAL_PATTERN.search(query or ""):
        return False
    return not _query_has_any(
        text,
        ("tv", "series", "show", "episode", "episodes", "season", "seasons"),
    )


def _is_optimistic_scifi_tv_query(query: str | None) -> bool:
    text = _normalize_query_text(query)
    if not _query_has_any(text, ("tv", "series", "show")):
        return False
    has_space_adventure_signal = _query_has_any(
        text,
        (
            "space opera",
            "space adventure",
            "space exploration",
            "starship",
            "crew",
            "exploration",
        ),
    )
    if has_space_adventure_signal and _query_has_any(
        text, ("space", "starship", "crew", "exploration")
    ):
        return True
    return _query_has_any(
        text, ("sci fi", "science fiction", "space")
    ) and _query_has_any(
        text,
        (
            "optimistic",
            "hopeful",
            "uplifting",
            "adventure",
            "crew",
            "starship",
            "exploration",
            "not too dark",
        ),
    )


def _is_serialized_prestige_tv_query(query: str | None) -> bool:
    text = _normalize_query_text(query)
    if not _query_has_any(text, ("tv", "series", "show")):
        return False
    prestige_signal = _query_has_any(
        text,
        (
            "serialized",
            "prestige",
            "high stakes",
            "high-stakes",
            "power struggle",
            "survival",
            "antihero",
            "political",
            "dynasty",
            "dystopian",
            "apocalypse",
        ),
    )
    drama_signal = _query_has_any(
        text,
        ("drama", "survival", "political", "war", "fantasy", "antihero"),
    )
    return prestige_signal and drama_signal


def _is_fantasy_epic_tv_query(query: str | None) -> bool:
    text = _normalize_query_text(query)
    has_tv_signal = _query_has_any(text, ("tv", "series", "show"))
    has_comparison_signal = bool(_COMPARISON_SIGNAL_PATTERN.search(text))
    return (
        (has_tv_signal or has_comparison_signal)
        and "fantasy" in text
        and _query_has_any(
            text,
            ("epic", "epics", "quest", "kingdom", "prophecy", "monster"),
        )
    )


def _is_temporal_thriller_query(query: str | None) -> bool:
    text = _normalize_query_text(query)
    temporal_signal = _query_has_any(
        text, ("time travel", "time bending", "time loop", "temporal", "paradox")
    )
    thriller_signal = _query_has_any(
        text, ("thriller", "thrillers", "brainy", "cerebral", "mind bending")
    )
    return temporal_signal and thriller_signal


def _is_high_concept_thriller_query(query: str | None) -> bool:
    text = _normalize_query_text(query)
    return _query_has_any(
        text, ("high concept", "brainy", "cerebral")
    ) and _query_has_any(text, ("thriller", "thrillers", "movie", "movies"))


def _is_cerebral_temporal_thriller_query(query: str | None) -> bool:
    text = _normalize_query_text(query)
    return _is_temporal_thriller_query(text) and _query_has_any(
        text, ("brainy", "cerebral", "mind bending", "mind-bending", "high concept")
    )


def _is_caper_crime_tv_query(query: str | None) -> bool:
    text = _normalize_query_text(query)
    return (
        _query_has_any(text, ("tv", "series", "show"))
        and "crime" in text
        and _query_has_any(
            text,
            (
                "short episode",
                "short episodes",
                "heist",
                "caper",
                "con artist",
                "con artists",
                "thief",
                "thieves",
                "grifter",
            ),
        )
    )


def _is_cross_media_international_crime_query(query: str | None) -> bool:
    text = _normalize_query_text(query)
    return (
        "crime" in text
        and _query_has_any(
            text,
            (
                "international",
                "europe",
                "european",
                "foreign",
                "non english",
                "global",
                "world cinema",
            ),
        )
        and _query_has_any(
            text, ("movie", "movies", "film", "films", "tv", "series", "show")
        )
    )


def _is_heist_tv_query(query: str | None) -> bool:
    text = _normalize_query_text(query)
    return _query_has_any(text, ("tv", "series", "show")) and _query_has_any(
        text,
        (
            "heist",
            "caper",
            "con artist",
            "con artists",
            "conman",
            "grifter",
            "thief",
            "thieves",
            "robbery",
            "robber",
        ),
    )


def _is_multilingual_family_adventure_query(query: str | None) -> bool:
    text = _normalize_query_text(query)
    has_audience_signal = _query_has_any(
        text, ("kids profile", "kid", "kids", "children", "family")
    )
    has_language_signal = _query_has_any(
        text,
        (
            "bilingual",
            "multilingual",
            "non english",
            "foreign language",
            "international",
        ),
    )
    has_adventure_signal = _query_has_any(
        text, ("adventure", "fantasy", "quest", "magic")
    )
    has_media_signal = _query_has_any(
        text, ("movie", "movies", "film", "films", "tv", "series", "show")
    )
    return (
        has_audience_signal
        and has_language_signal
        and has_adventure_signal
        and has_media_signal
    )


def _is_street_level_superhero_query(query: str | None) -> bool:
    text = _normalize_query_text(query)
    return (
        _query_has_any(text, ("tv", "series", "show"))
        and _query_has_any(text, ("superhero", "vigilante"))
        and _query_has_any(text, ("street level", "gritty", "grounded", "urban"))
    )


def _is_anime_scifi_movie_query(query: str | None) -> bool:
    text = _normalize_query_text(query)
    return (
        _query_has_any(text, ("movie", "movies", "film", "films"))
        and _query_has_any(text, ("anime", "animated", "japanese"))
        and _query_has_any(
            text,
            (
                "sci fi",
                "science fiction",
                "cyberpunk",
                "mecha",
                "future",
                "futuristic",
                "android",
                "space",
            ),
        )
    )


def _is_noir_movie_query(query: str | None) -> bool:
    text = _normalize_query_text(query)
    return (
        _query_has_any(text, ("noir", "neo noir", "film noir"))
        and (
            _query_has_any(text, ("movie", "movies", "film", "films"))
            or _query_has_any(text, ("city", "cities", "urban", "modern"))
        )
        and _query_has_any(
            text,
            ("crime", "mystery", "mysteries", "thriller", "detective", "investigation"),
        )
    )


def _is_money_psychology_thriller_query(query: str | None) -> bool:
    text = _normalize_query_text(query)
    if not _query_has_any(
        text,
        (
            "thriller",
            "thrillers",
            "psychological",
            "mind game",
            "mind games",
            "obsession",
        ),
    ):
        return False
    return _query_has_any(
        text,
        (
            "money",
            "greed",
            "wealth",
            "wealthy",
            "rich",
            "status",
            "power",
            "ambition",
            "billionaire",
            "finance",
            "financial",
            "banking",
            "broker",
            "wall street",
        ),
    )


def _is_rom_com_query(query: str | None) -> bool:
    text = _normalize_query_text(query)
    if _query_has_any(
        text,
        ("rom com", "rom-com", "romcom", "romantic comedy", "romantic comedies"),
    ):
        return True
    return _query_has_any(text, ("romance", "romantic")) and _query_has_any(
        text,
        ("comedy", "comedies", "date night", "meet cute"),
    )


class SignalsForQuery:
    def __init__(self):
        self.crime_query_signal: bool = False
        self.scifi_query_signal: bool = False
        self.fantasy_query_signal: bool = False
        self.audience_signal: bool = False
        self.family_friendly: bool = False
        self.teen_or_family_safe: bool = False
        self.kids_profile: bool = False
        self.kids_profile_adventure: bool = False
        self.animation_signal: bool = False
        self.comparison_signal: bool = False
        self.last_decade: bool = False
        self.modern_setting: bool = False
        self.city_setting: bool = False
        self.feel_good_comedy: bool = False
        self.adult_feelgood_movieish: bool = False
        self.not_too_dark_fantasy_tv: bool = False
        self.teen_friendly_adventure_provider: bool = False
        self.family_adventure_movie: bool = False
        self.date_night: bool = False
        self.optimistic_scifi_tv: bool = False
        self.short_bingeable_scifi_tv: bool = False
        self.fantasy_epic_tv: bool = False
        self.fantasy_worlds_cross_media: bool = False
        self.serialized_prestige_tv: bool = False
        self.temporal_thriller: bool = False
        self.high_concept_thriller: bool = False
        self.cerebral_temporal_thriller: bool = False
        self.caper_crime_tv: bool = False
        self.heist_tv: bool = False
        self.international_crime: bool = False
        self.multilingual_family_adventure: bool = False
        self.street_level_superhero: bool = False
        self.anime_scifi_movie: bool = False
        self.noir_movie: bool = False
        self.money_psychology_thriller: bool = False
        self.rom_com: bool = False


@dataclass(frozen=True)
class QueryHardConstraints:
    year_min: int | None = None
    year_max: int | None = None
    min_runtime: int | None = None
    max_runtime: int | None = None
    maturity_rating_max: str | None = None


@dataclass(frozen=True)
class QueryProfile:
    raw_query: str = ""
    normalized_query: str = ""
    audience: tuple[str, ...] = field(default_factory=tuple)
    tone: tuple[str, ...] = field(default_factory=tuple)
    structure: tuple[str, ...] = field(default_factory=tuple)
    semantic_domains: tuple[str, ...] = field(default_factory=tuple)
    semantic_facets: tuple[str, ...] = field(default_factory=tuple)
    hard_constraints: QueryHardConstraints = field(default_factory=QueryHardConstraints)
    signals: SignalsForQuery = field(default_factory=SignalsForQuery)

    def to_debug_dict(self) -> dict[str, object]:
        active_signals = {
            key: value for key, value in self.signals.__dict__.items() if value
        }
        return {
            "normalized_query": self.normalized_query,
            "audience": list(self.audience),
            "tone": list(self.tone),
            "structure": list(self.structure),
            "semantic_domains": list(self.semantic_domains),
            "semantic_facets": list(self.semantic_facets),
            "hard_constraints": asdict(self.hard_constraints),
            "signals": active_signals,
        }


@lru_cache(maxsize=1024)
def build_query_profile(query: str | None) -> QueryProfile:
    normalized = _normalize_query_text(query)
    print(f"DEBUG PROFILE: query='{query}' normalized='{normalized}'")
    if not normalized:
        return QueryProfile()

    raw_query = query or ""
    kids_profile = bool(_KIDS_PROFILE_QUERY_PATTERN.search(raw_query))
    kids_profile_adventure = bool(
        _KIDS_PROFILE_ADVENTURE_QUERY_PATTERN.search(raw_query)
    )
    family_friendly = bool(_FAMILY_FRIENDLY_QUERY_PATTERN.search(raw_query))
    teen_or_family_safe = bool(_TEEN_QUERY_PATTERN.search(raw_query) or family_friendly)
    animation_signal = bool(_ANIMATION_AUDIENCE_PATTERN.search(raw_query))
    audience_signal = bool(_AUDIENCE_SIGNAL_PATTERN.search(raw_query))
    comparison_signal = bool(_COMPARISON_SIGNAL_PATTERN.search(normalized))
    last_decade = bool(_LAST_DECADE_QUERY_PATTERN.search(raw_query))
    modern_setting = bool(_MODERN_QUERY_PATTERN.search(raw_query))
    city_setting = _query_has_any(normalized, ("city", "cities", "urban"))
    feel_good_comedy = bool(_ADULT_FEELGOOD_COMEDY_QUERY_PATTERN.search(raw_query))
    adult_feelgood_movieish = _is_adult_feelgood_movieish_query(raw_query)
    not_too_dark_fantasy_tv = bool(
        _NOT_TOO_DARK_QUERY_PATTERN.search(raw_query)
        and _FANTASY_TV_QUERY_PATTERN.search(raw_query)
    )
    teen_friendly_adventure_provider = bool(
        _TEEN_FRIENDLY_ADVENTURE_PROVIDER_QUERY_PATTERN.search(raw_query)
    )
    family_adventure_movie = bool(
        _FAMILY_ADVENTURE_MOVIE_QUERY_PATTERN.search(raw_query)
    )
    date_night = bool(_DATE_NIGHT_QUERY_PATTERN.search(raw_query))
    optimistic_scifi_tv = _is_optimistic_scifi_tv_query(raw_query)
    short_bingeable_scifi_tv = bool(
        _SHORT_BINGEABLE_SCIFI_TV_QUERY_PATTERN.search(raw_query)
    )
    fantasy_epic_tv = _is_fantasy_epic_tv_query(raw_query)
    fantasy_worlds_cross_media = bool(
        _FANTASY_WORLDS_CROSS_MEDIA_QUERY_PATTERN.search(raw_query)
    )
    serialized_prestige_tv = _is_serialized_prestige_tv_query(raw_query)
    temporal_thriller = _is_temporal_thriller_query(raw_query)
    high_concept_thriller = _is_high_concept_thriller_query(raw_query)
    cerebral_temporal_thriller = _is_cerebral_temporal_thriller_query(raw_query)
    caper_crime_tv = _is_caper_crime_tv_query(raw_query)
    heist_tv = _is_heist_tv_query(raw_query)
    international_crime = _is_cross_media_international_crime_query(raw_query)
    multilingual_family_adventure = _is_multilingual_family_adventure_query(raw_query)
    street_level_superhero = _is_street_level_superhero_query(raw_query)
    anime_scifi_movie = _is_anime_scifi_movie_query(raw_query)
    noir_movie = _is_noir_movie_query(raw_query)
    money_psychology_thriller = _is_money_psychology_thriller_query(raw_query)
    _crime_signal_val = "crime" in normalized or _query_has_any(
        normalized, ("heist", "caper", "noir", "detective", "thief", "grifter")
    )
    _scifi_signal_val = _query_has_any(
        normalized, ("sci fi", "science fiction", "scifi")
    )
    _fantasy_signal_val = "fantasy" in normalized
    rom_com = _is_rom_com_query(raw_query)
    print(
        f"DEBUG LOCAL: crime={_crime_signal_val} scifi={_scifi_signal_val} fantasy={_fantasy_signal_val}"
    )

    audience: list[str] = []
    tone: list[str] = []
    structure: list[str] = []
    semantic_domains: list[str] = []
    semantic_facets: list[str] = []

    if kids_profile:
        audience.append("kids_profile")
    if kids_profile or family_friendly:
        audience.append("family")
    if teen_or_family_safe:
        audience.append("teen")
    if audience_signal and not (kids_profile or family_friendly or teen_or_family_safe):
        audience.append("broad_audience")
    if animation_signal:
        audience.append("animation")

    if feel_good_comedy:
        tone.append("feel_good")
    if adult_feelgood_movieish:
        tone.append("adult")
    if optimistic_scifi_tv:
        tone.append("optimistic")
    if not_too_dark_fantasy_tv:
        tone.append("not_too_dark")
    if serialized_prestige_tv:
        tone.append("prestige")
    if cerebral_temporal_thriller or high_concept_thriller:
        tone.append("cerebral")
    if street_level_superhero:
        tone.append("grounded")
    if noir_movie:
        tone.append("noir")

    if comparison_signal:
        structure.append("comparison")
    if short_bingeable_scifi_tv:
        structure.extend(["short_form", "bingeable"])
    if serialized_prestige_tv:
        structure.append("serialized")
    if fantasy_worlds_cross_media or international_crime:
        structure.append("cross_media")
    if date_night:
        structure.append("occasion")
    if adult_feelgood_movieish:
        structure.append("movieish")

    if optimistic_scifi_tv or short_bingeable_scifi_tv or anime_scifi_movie:
        semantic_domains.append("science_fiction")
    if fantasy_epic_tv or fantasy_worlds_cross_media or multilingual_family_adventure:
        semantic_domains.append("fantasy")
    if caper_crime_tv or heist_tv or international_crime or noir_movie:
        semantic_domains.append("crime")
    if temporal_thriller or high_concept_thriller or money_psychology_thriller:
        semantic_domains.append("thriller")
    if street_level_superhero:
        semantic_domains.append("superhero")
    if anime_scifi_movie or animation_signal or kids_profile_adventure:
        semantic_domains.append("animation")
    if (
        multilingual_family_adventure
        or kids_profile_adventure
        or family_adventure_movie
    ):
        semantic_domains.append("family_adventure")
    if rom_com or date_night or feel_good_comedy:
        semantic_domains.append("romance")
    if serialized_prestige_tv:
        semantic_domains.append("prestige_tv")

    if temporal_thriller:
        semantic_facets.append("temporal_thriller")
    if high_concept_thriller:
        semantic_facets.append("high_concept_thriller")
    if cerebral_temporal_thriller:
        semantic_facets.append("cerebral_temporal")
    if noir_movie:
        semantic_facets.append("noir")
    if caper_crime_tv or heist_tv:
        semantic_facets.append("caper_crime")
    if street_level_superhero:
        semantic_facets.append("street_level_superhero")
    if serialized_prestige_tv:
        semantic_facets.append("serialized_prestige_tv")
    if optimistic_scifi_tv:
        semantic_facets.append("optimistic_scifi_tv")
    if fantasy_epic_tv:
        semantic_facets.append("fantasy_epic_tv")
    if international_crime:
        semantic_facets.append("international_crime")
    if multilingual_family_adventure:
        semantic_facets.append("multilingual_family_adventure")
    if anime_scifi_movie:
        semantic_facets.append("anime_scifi_movie")
    if comparison_signal:
        semantic_facets.append("comparison")
    if _query_has_any(
        normalized,
        (
            "detective",
            "investigation",
            "courtroom",
            "political",
            "survival",
            "noir",
            "neo noir",
            "film noir",
            "heist",
            "caper",
            "grifter",
            "con artist",
            "robbery",
            "vigilante",
            "street level",
            "gritty",
            "grounded",
            "urban",
            "crew",
            "quest",
            "bilingual",
            "international",
            "paradox",
            "mind bending",
        ),
    ):
        semantic_facets.append("fine_grained_theme")

    year_min: int | None = None
    if noir_movie and modern_setting:
        year_min = 1990
    elif temporal_thriller:
        year_min = 2000
    elif last_decade:
        year_min = date.today().year - 10

    max_runtime: int | None = None
    if short_bingeable_scifi_tv:
        max_runtime = 50
    elif high_concept_thriller and "short" in normalized:
        max_runtime = 125

    maturity_rating_max: str | None = None
    if kids_profile or teen_or_family_safe or not_too_dark_fantasy_tv:
        maturity_rating_max = "PG-13"

    import inspect

    print(f"DEBUG SOURCE:\n{inspect.getsource(SignalsForQuery)}")
    print(f"DEBUG ID: {id(SignalsForQuery)}")
    print(f"DEBUG LOCALS: _crime_signal_val={_crime_signal_val}")
    signals = SignalsForQuery()
    signals.crime_query_signal = _crime_signal_val
    signals.scifi_query_signal = _scifi_signal_val
    signals.fantasy_query_signal = _fantasy_signal_val
    signals.audience_signal = audience_signal
    signals.family_friendly = family_friendly
    signals.teen_or_family_safe = teen_or_family_safe
    signals.kids_profile = kids_profile
    signals.kids_profile_adventure = kids_profile_adventure
    signals.animation_signal = animation_signal
    signals.comparison_signal = comparison_signal
    signals.last_decade = last_decade
    signals.modern_setting = modern_setting
    signals.city_setting = city_setting
    signals.feel_good_comedy = feel_good_comedy
    signals.adult_feelgood_movieish = adult_feelgood_movieish
    signals.not_too_dark_fantasy_tv = not_too_dark_fantasy_tv
    signals.teen_friendly_adventure_provider = teen_friendly_adventure_provider
    signals.family_adventure_movie = family_adventure_movie
    signals.date_night = date_night
    signals.optimistic_scifi_tv = optimistic_scifi_tv
    signals.short_bingeable_scifi_tv = short_bingeable_scifi_tv
    signals.fantasy_epic_tv = fantasy_epic_tv
    signals.fantasy_worlds_cross_media = fantasy_worlds_cross_media
    signals.serialized_prestige_tv = serialized_prestige_tv
    signals.temporal_thriller = temporal_thriller
    signals.high_concept_thriller = high_concept_thriller
    signals.cerebral_temporal_thriller = cerebral_temporal_thriller
    signals.caper_crime_tv = caper_crime_tv
    signals.heist_tv = heist_tv
    signals.international_crime = international_crime
    signals.multilingual_family_adventure = multilingual_family_adventure
    signals.street_level_superhero = street_level_superhero
    signals.anime_scifi_movie = anime_scifi_movie
    signals.noir_movie = noir_movie
    signals.money_psychology_thriller = money_psychology_thriller
    signals.rom_com = rom_com

    print(f"DEBUG DIRECT ATTRIB: {signals.crime_query_signal}")
    return QueryProfile(
        raw_query=raw_query,
        normalized_query=normalized,
        audience=_ordered_tuple(audience),
        tone=_ordered_tuple(tone),
        structure=_ordered_tuple(structure),
        semantic_domains=_ordered_tuple(semantic_domains),
        semantic_facets=_ordered_tuple(semantic_facets),
        hard_constraints=QueryHardConstraints(
            year_min=year_min,
            max_runtime=max_runtime,
            maturity_rating_max=maturity_rating_max,
        ),
        signals=signals,
    )


@dataclass(frozen=True)
class IntentSignals:
    """
    LLM-extracted intent signals for robust, paraphrase-safe query understanding.
    Created by extract_intent_signals() in llm_parser.py; immune to query wording variations.
    """

    mood: Optional[str] = None
    """
    Emotional tone: "romantic", "dark", "light", "intense", "heartwarming", "suspenseful", etc.
    Extracted from sentiment + genre cues; paraphrase-safe (e.g., "feel-good" → "uplifting").
    """

    intent_type: Optional[str] = None
    """
    Query intent: "discovery", "specific_title", "exploration", "comfort", "challenge".
    Derived from query structure and content.
    """

    semantic_facets: tuple[str, ...] = field(default_factory=tuple)
    """
    Semantic themes from query: ("heist", "caper", "crime") or ("superhero", "powered", "vigilante").
    LLM-extracted; immune to keyword variations ("heist" vs "robbery" → same facet set).
    """

    prestige_indicator: bool = False
    """
    True if query seeks high-award-winning titles. Extracted from intent (not query keywords).
    Safe alternative to checking query for "oscar"/"academy" keywords.
    """

    quality_mode: Optional[str] = None
    """
    Ranking preference: "high_quality" (vote_average > 7.5), "newest" (recent releases), "trending", or None for balanced mix.
    Derived from query language and intent.
    """

    temporal_preference: Optional[str] = None
    """
    Temporal scope: "recent" (last 5 years), "classic" (pre-2015), "timeless" (all eras), or None for unspecified.
    Extracted from query hints or implicitly from mood/tone.
    """
