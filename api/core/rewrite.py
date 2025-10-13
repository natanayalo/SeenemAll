from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class Rewrite(BaseModel):
    """
    Represents the rewritten user query, optimized for embedding.
    """

    rewritten_text: str = Field(
        ..., description="A rewritten, concise version of the user query (≤8 words)."
    )
    facet_allow: Optional[List[str]] = Field(
        None, description="Facets to allow in the search."
    )
    facet_block: Optional[List[str]] = Field(
        None, description="Facets to block in the search."
    )

    @field_validator("rewritten_text")
    def check_word_count(cls, v):
        if len(v.split()) > 8:
            raise ValueError("must be a maximum of 8 words")
        return v
