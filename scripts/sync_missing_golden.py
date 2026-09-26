"""Sync missing golden set titles from TMDB directly into PostgreSQL."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
from sqlalchemy import select

from api.config import TMDB_API_KEY
from api.db.models import Item
from api.db.session import get_sessionmaker
from etl.tmdb_client import TMDBClient
from etl.tmdb_sync import _upsert_items

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("sync_missing_golden")


async def fetch_item_details(
    client: TMDBClient, tmdb_id: int, preferred_media: str = "movie"
) -> Dict[str, Any] | None:
    """Try fetching details for tmdb_id as preferred_media, then fallback to other media type."""
    media_types = [preferred_media, "tv" if preferred_media == "movie" else "movie"]
    for media in media_types:
        try:
            data = await client.details(media, tmdb_id)
            if data and data.get("id"):
                data["media_type"] = media
                return data
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                continue
            logger.warning(f"Error fetching TMDB {media} {tmdb_id}: {exc}")
        except Exception as exc:
            logger.warning(f"Unexpected error fetching {media} {tmdb_id}: {exc}")
    return None


async def sync_missing_golden_items(evaluation_set_path: Path) -> int:
    with evaluation_set_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    # Collect needed golden items
    golden_targets: Dict[int, Tuple[str, str]] = (
        {}
    )  # tmdb_id -> (title, preferred_media)
    for entry in entries:
        preferred_media = "movie"
        constraints = entry.get("constraints") or {}
        if constraints.get("media_type") == "tv":
            preferred_media = "tv"
        elif (
            "series" in entry.get("query", "").lower()
            or "tv" in entry.get("query", "").lower()
            or "shows" in entry.get("query", "").lower()
        ):
            preferred_media = "tv"

        for item in entry.get("golden_set", []):
            tmdb_id = int(item["id"])
            title = item.get("title", "")
            if tmdb_id not in golden_targets:
                golden_targets[tmdb_id] = (title, preferred_media)

    SessionLocal = get_sessionmaker()
    with SessionLocal() as db:
        existing_ids = set(
            db.execute(
                select(Item.tmdb_id).where(
                    Item.tmdb_id.in_(list(golden_targets.keys()))
                )
            )
            .scalars()
            .all()
        )

    missing_ids = [gid for gid in golden_targets if gid not in existing_ids]
    logger.info(f"Total golden targets: {len(golden_targets)}")
    logger.info(f"Already present in DB: {len(existing_ids)}")
    logger.info(f"Missing from DB:       {len(missing_ids)}")

    if not missing_ids:
        logger.info("All golden items already present in DB!")
        return 0

    client = TMDBClient(TMDB_API_KEY, rate_per_sec=4.0)
    fetched_payloads: List[Dict[str, Any]] = []

    try:
        for idx, gid in enumerate(missing_ids, start=1):
            title, pref_media = golden_targets[gid]
            logger.info(
                f"[{idx}/{len(missing_ids)}] Fetching TMDB {gid} ('{title}', media={pref_media})..."
            )
            details = await fetch_item_details(client, gid, preferred_media=pref_media)
            if details:
                fetched_payloads.append(details)
                logger.info(
                    f"  -> Found: '{details.get('title') or details.get('name')}' ({details.get('media_type')})"
                )
            else:
                logger.error(f"  -> FAILED to find TMDB item {gid} ('{title}')")

        if fetched_payloads:
            logger.info(
                f"Upserting {len(fetched_payloads)} newly fetched items into PostgreSQL..."
            )
            with SessionLocal() as db:
                _upsert_items(db, fetched_payloads)
                db.commit()
            logger.info("Upsert complete!")
    finally:
        await client._client.aclose()

    return len(fetched_payloads)


def main() -> None:
    eval_path = Path("evaluation/evaluation_set.json")
    if not eval_path.exists():
        logger.error(f"Evaluation set not found at {eval_path}")
        sys.exit(1)

    added = asyncio.run(sync_missing_golden_items(eval_path))
    logger.info(f"Successfully synced {added} missing golden titles into the database.")


if __name__ == "__main__":
    main()
