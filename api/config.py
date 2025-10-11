import os
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_PAGE_LIMIT = int(
    os.getenv("TMDB_PAGE_LIMIT", "25")
)  # ~20 items/page → 25 pages ≈ 500 titles per endpoint
COUNTRY_DEFAULT = os.getenv("JUSTWATCH_COUNTRY", "IL")
