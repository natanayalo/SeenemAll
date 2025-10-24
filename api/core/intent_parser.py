from pydantic import BaseModel, Field
from typing import List, Optional


class Intent(BaseModel):
    """
    Represents the user's intent for movie recommendations, parsed from their query.
    """

    include_genres: Optional[List[str]] = Field(
        None, description="List of genres to include."
    )
    exclude_genres: Optional[List[str]] = Field(
        None, description="List of genres to exclude."
    )
    runtime_minutes_min: Optional[int] = Field(
        None, description="Minimum runtime in minutes."
    )
    runtime_minutes_max: Optional[int] = Field(
        None, description="Maximum runtime in minutes."
    )
    languages: Optional[List[str]] = Field(None, description="List of languages.")
    year_min: Optional[int] = Field(None, description="Minimum release year.")
    year_max: Optional[int] = Field(None, description="Maximum release year.")
    maturity_rating_max: Optional[str] = Field(
        None, description="Maximum maturity rating (e.g., 'PG-13')."
    )
    boost_genres: Optional[List[str]] = Field(
        None, description="List of genres to boost in recommendations."
    )
    media_types: Optional[List[str]] = Field(
        None, description="Optional list of media types (e.g., 'tv', 'movie')."
    )
    include_people: Optional[List[str]] = Field(
        None, description="List of person names or IDs to include."
    )
    streaming_providers: Optional[List[str]] = Field(
        None, description="Preferred streaming providers (normalized identifiers)."
    )
    ann_description: Optional[str] = Field(
        None,
        description="Optional natural-language summary for ANN query enrichment.",
    )
