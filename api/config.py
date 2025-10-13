import os
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_PAGE_LIMIT = int(
    os.getenv("TMDB_PAGE_LIMIT", "25")
)  # ~20 items/page → 25 pages ≈ 500 titles per endpoint
COUNTRY_DEFAULT = os.getenv("JUSTWATCH_COUNTRY", "IL")
JUSTWATCH_LANGUAGE = os.getenv("JUSTWATCH_LANGUAGE", "en")
JUSTWATCH_PLATFORM = os.getenv("JUSTWATCH_PLATFORM", "WEB")
USER_PROFILE_DECAY_HALF_LIFE = float(os.getenv("USER_PROFILE_DECAY_HALF_LIFE", "10"))
