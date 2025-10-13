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
