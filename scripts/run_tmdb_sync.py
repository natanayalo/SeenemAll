from etl.tmdb_sync import run
import os

# Allow TMDB_PAGE_LIMIT override, default to 50
if "TMDB_PAGE_LIMIT" not in os.environ:
    os.environ["TMDB_PAGE_LIMIT"] = "50"

if __name__ == "__main__":
    run()
