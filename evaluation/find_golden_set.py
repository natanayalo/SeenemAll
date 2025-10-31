"""Script to find candidate items for the evaluation set."""

import asyncio
import argparse
import logging
from typing import List

from sqlalchemy import select

from api.db.session import get_sessionmaker
from api.db.models import Item
from api.config import TMDB_API_KEY
from etl.tmdb_client import TMDBClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def discover_and_find(
    media_type: str,
    genre_ids: List[int] | None = None,
    keyword_ids: List[int] | None = None,
    person_ids: List[int] | None = None,
):
    """Discover movies or TV shows by genre, keyword, or person and find them in the local database."""
    logger.info(f"TMDB_API_KEY starts with: {TMDB_API_KEY[:4]}")

    tmdb_client = TMDBClient(TMDB_API_KEY)
    SessionLocal = get_sessionmaker()

    params = {}
    if genre_ids:
        params["with_genres"] = ",".join(map(str, genre_ids))
    if keyword_ids:
        params["with_keywords"] = ",".join(map(str, keyword_ids))
    if person_ids:
        params["with_people"] = ",".join(map(str, person_ids))

    try:
        logger.info(f"Discovering TMDB for: media_type='{media_type}', params={params}")
        discover_results = await tmdb_client.discover(
            media_type=media_type, params=params
        )
        logger.info(f"TMDB API response: {discover_results}")
    except Exception as e:
        logger.error(f"Error discovering from TMDB: {e}", exc_info=True)
        return
    finally:
        await tmdb_client.aclose()

    if not discover_results.get("results"):
        logger.warning("No results found on TMDB.")
        return

    tmdb_ids = [result["id"] for result in discover_results["results"]]

    with SessionLocal() as db:
        items = (
            db.execute(select(Item).where(Item.tmdb_id.in_(tmdb_ids))).scalars().all()
        )

    if not items:
        logger.warning("No matching items found in the local database.")
        return

    logger.info("Found the following items in the local database:")
    for item in items:
        logger.info(
            f"  - id: {item.id}, tmdb_id: {item.tmdb_id}, "
            f'title: "{item.title}", release_year: {item.release_year}'
        )


def main():
    """Parse command-line arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Find candidate items for the evaluation set."
    )
    parser.add_argument(
        "--genre-ids",
        type=int,
        nargs="+",
        help="A list of genre IDs to discover.",
    )
    parser.add_argument(
        "--keyword-ids",
        type=int,
        nargs="+",
        help="A list of keyword IDs to discover.",
    )
    parser.add_argument(
        "--person-ids",
        type=int,
        nargs="+",
        help="A list of person IDs to discover.",
    )
    parser.add_argument(
        "--media-type",
        type=str,
        choices=["movie", "tv"],
        required=True,
        help="The media type to search for.",
    )
    args = parser.parse_args()

    if not args.genre_ids and not args.keyword_ids and not args.person_ids:
        parser.error(
            "At least one of --genre-ids, --keyword-ids, or --person-ids is required."
        )

    asyncio.run(
        discover_and_find(
            args.media_type, args.genre_ids, args.keyword_ids, args.person_ids
        )
    )


if __name__ == "__main__":
    main()
