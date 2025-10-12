"""Templates for generating text input for embeddings."""

from typing import List, Dict, Any


def format_basic(title: str, overview: str, genres: List[str]) -> str:
    """Basic format: [genres] title :: overview"""
    genre_text = f"[{', '.join(genres)}] " if genres else ""
    return f"{genre_text}{title} :: {overview}"


def format_structured(title: str, overview: str, genres: List[str]) -> str:
    """Structured format with clear sections:
    Title: {title}
    Genres: {genres}
    Overview: {overview}
    """
    genre_text = ", ".join(genres) if genres else "None"
    return f"Title: {title}\nGenres: {genre_text}\nOverview: {overview}"


def format_natural(title: str, overview: str, genres: List[str]) -> str:
    """Natural language format describing the movie/show"""
    genre_desc = " and ".join(genres) if genres else "no specific genre"
    return f"{title} is a {genre_desc} movie/show. {overview}"


def format_emphasized(title: str, overview: str, genres: List[str]) -> str:
    """Format emphasizing genre keywords in the text"""
    genre_tags = " ".join(f"#{g}" for g in genres) if genres else ""
    return f"{title} {genre_tags} :: {overview}"


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
    return TEMPLATES[template](
        title=item.get("title", ""), overview=item.get("overview", ""), genres=genres
    )
