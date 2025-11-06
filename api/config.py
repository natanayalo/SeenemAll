import os
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_PAGE_LIMIT = int(
    os.getenv("TMDB_PAGE_LIMIT", "25")
)  # ~20 items/page → 25 pages ≈ 500 titles per endpoint
COUNTRY_DEFAULT = os.getenv("JUSTWATCH_COUNTRY", "IL")
JUSTWATCH_LANGUAGE = os.getenv("JUSTWATCH_LANGUAGE", "en")
JUSTWATCH_PLATFORM = os.getenv("JUSTWATCH_PLATFORM", "WEB")
USER_PROFILE_DECAY_HALF_LIFE = float(os.getenv("USER_PROFILE_DECAY_HALF_LIFE", "10"))
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200")
ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME", "")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", "")
ELASTICSEARCH_VERIFY_CERTS = _env_bool("ELASTICSEARCH_VERIFY_CERTS", "false")
ELASTICSEARCH_TIMEOUT = float(os.getenv("ELASTICSEARCH_TIMEOUT", "15"))
ELASTICSEARCH_ITEMS_INDEX = os.getenv("ELASTICSEARCH_ITEMS_INDEX", "items")
ELASTICSEARCH_KNN_K = int(os.getenv("ELASTICSEARCH_KNN_K", "40"))
ELASTICSEARCH_KNN_NUM_CANDIDATES = int(
    os.getenv("ELASTICSEARCH_KNN_NUM_CANDIDATES", "200")
)
ANN_BACKEND = os.getenv("ANN_BACKEND", "elasticsearch").strip().lower()
