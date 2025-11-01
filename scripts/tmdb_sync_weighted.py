from __future__ import annotations
import asyncio
from typing import List, Dict, Any, Optional, Tuple, cast
import logging
from api.db.session import get_sessionmaker
from api.config import TMDB_API_KEY, TMDB_PAGE_LIMIT
from etl.tmdb_client import TMDBClient
from etl.tmdb_sync import _upsert_items

MEDIA_TYPES = ("movie", "tv")
logger = logging.getLogger(__name__)
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False


async def run(*, list_weights: Optional[Dict[str, float]] = None) -> None:
    """Run TMDB sync with configurable list weights.

    Args:
        list_weights: Dictionary of list name to weight ratio.
                     Default is {"popular": 1.0, "top_rated": 0.5, "trending": 0.5}
    """
    client = TMDBClient(TMDB_API_KEY)
    pages = TMDB_PAGE_LIMIT

    # Default weights favor popular content
    weights = list_weights or {"popular": 1.0, "top_rated": 0.5, "trending": 0.5}

    sessionmaker = get_sessionmaker()

    try:
        candidate_signals: Dict[Tuple[str, int], Dict[str, Optional[int]]] = {}

        def record_signal(media: str, id: Any, field: str, rank: int) -> None:
            if not isinstance(id, int):
                return
            info = candidate_signals.setdefault(
                (media, id),
                {"popular_rank": None, "top_rated_rank": None, "trending_rank": None},
            )
            current = info.get(field)
            if current is None or rank < current:
                info[field] = rank

        for media in MEDIA_TYPES:
            collected = 0

            # Fetch from each list according to configured weights
            for list_type, weight in weights.items():
                pages_for_list = int(pages * weight)
                if pages_for_list < 1:
                    continue

                rank = 0
                logger.info(f"Fetching {pages_for_list} pages from {list_type} {media}")
                async for it in client.iter_list(media, list_type, pages_for_list):
                    rank += 1
                    collected += 1
                    record_signal(media, it.get("id"), f"{list_type}_rank", rank)

            logger.info("Collected %d candidate IDs for %s.", collected, media)

        candidate_ids = list(candidate_signals.keys())
        logger.info("Total unique candidate IDs: %d", len(candidate_ids))

        # Fetch details concurrently in batches
        BATCH = 20
        for i in range(0, len(candidate_ids), BATCH):
            batch = candidate_ids[i : i + BATCH]
            details_list = await asyncio.gather(
                *[client.details(m, tid) for (m, tid) in batch],
                return_exceptions=True,
            )
            # Attach media_type back
            enriched: List[Dict[str, Any]] = []
            for (m, tid), det in zip(batch, details_list):
                if isinstance(det, BaseException):
                    continue
                payload = cast(Dict[str, Any], det)
                payload["media_type"] = m
                payload["_list_ranks"] = candidate_signals.get((m, tid), {})
                enriched.append(payload)

            # Upsert
            SessionLocal = sessionmaker()
            with SessionLocal as db:
                _upsert_items(db, enriched)
                db.commit()
            logger.info(
                "Processed batch %d/%d.",
                (i // BATCH) + 1,
                (len(candidate_ids) + BATCH - 1) // BATCH,
            )
        logger.info("TMDB sync completed successfully.")
    finally:
        await client.aclose()
