"""Templates for generating text input for embeddings."""

from typing import List, Dict, Any, Optional


def _get_decade(year: Optional[int]) -> Optional[str]:
    """Convert year to decade string (e.g., 2020 -> '2020s')."""
    if not year:
        return None
    decade = (year // 10) * 10
    return f"{decade}s"


def _get_era(year: Optional[int]) -> Optional[str]:
    """Convert year to era description."""
    if not year:
        return None
    current_year = 2025
    if year >= current_year:
        return "Upcoming"
    elif year >= 2020:
        return "Contemporary"
    elif year >= 2010:
        return "Recent"
    elif year >= 1990:
        return "Modern"  # Changed: 1990-2009 is now Modern
    elif year >= 1970:
        return "Classic"  # Changed: 1970-1989 is Classic
    return "Vintage"  # Changed: pre-1970 is Vintage


def format_basic(
    title: str,
    overview: str,
    genres: List[str],
    year: Optional[int] = None,
) -> str:
    """Basic format: [genres] [era] (year • decade) title :: overview"""
    genre_text = f"[{', '.join(genres)}] " if genres else ""
    era = _get_era(year)
    era_text = f"[{era}] " if era else ""
    time_text = ""
    if year:
        decade = _get_decade(year)
        time_bits = [str(year)]
        if decade:
            time_bits.append(decade)
        time_text = f"({' • '.join(time_bits)}) "
    return f"{genre_text}{era_text}{time_text}{title} :: {overview}"


def format_structured(
    title: str, overview: str, genres: List[str], year: Optional[int] = None
) -> str:
    """Structured format with clear sections:
    Title: {title}
    Year: {year}
    Era: {era}
    Decade: {decade}
    Genres: {genres}
    Overview: {overview}
    """
    genre_text = ", ".join(genres) if genres else "None"
    era = _get_era(year) or "Unknown"
    decade = _get_decade(year) or "Unknown"
    return (
        f"Title: {title}\n"
        f"Year: {year or 'Unknown'}\n"
        f"Era: {era}\n"
        f"Decade: {decade}\n"
        f"Genres: {genre_text}\n"
        f"Overview: {overview}"
    )


def format_natural(
    title: str, overview: str, genres: List[str], year: Optional[int] = None
) -> str:
    """Natural language format describing the movie/show with temporal context"""
    genre_desc = " and ".join(genres) if genres else "no specific genre"
    era = _get_era(year)
    era_desc = f" from the {era} era" if era else ""
    return f"{title} is a {genre_desc} {era_desc} movie/show. {overview}"


def format_emphasized(
    title: str, overview: str, genres: List[str], year: Optional[int] = None
) -> str:
    """Format emphasizing genre keywords and temporal context"""
    genre_tags = " ".join(f"#{g}" for g in genres) if genres else ""
    decade = _get_decade(year)
    era = _get_era(year)
    time_tags = []
    if decade:
        time_tags.append(f"#{decade}")
    if era:
        time_tags.append(f"#{era}")
    time_text = " " + " ".join(time_tags) if time_tags else ""
    return f"{title} {genre_tags}{time_text} :: {overview}"


# Map of template names to their functions
TEMPLATES = {
    "basic": format_basic,
    "structured": format_structured,
    "natural": format_natural,
    "emphasized": format_emphasized,
}


def format_with_template(template: str, item: Dict[str, Any]) -> str:
    """Format item data using the specified template"""
    if template not in TEMPLATES:
        raise ValueError(f"Unknown template: {template}")

    genres = [g["name"] for g in item.get("genres", []) if g.get("name")]
    year = item.get("release_year")

    return TEMPLATES[template](
        title=item.get("title", ""),
        overview=item.get("overview", ""),
        genres=genres,
        year=year,
    )
