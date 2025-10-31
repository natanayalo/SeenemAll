from etl.tmdb_sync import run
import os

# Override TMDB_PAGE_LIMIT to fetch many more pages
os.environ["TMDB_PAGE_LIMIT"] = "200"  # 200 pages = ~4000 titles per list type

if __name__ == "__main__":
    run()
