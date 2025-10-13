from __future__ import annotations
import asyncio
from typing import List, Dict, Any, Optional, cast
from sqlalchemy.orm import Session
from sqlalchemy import select
from api.db.session import get_engine, get_sessionmaker
from api.db.models import Item
from api.config import TMDB_API_KEY, TMDB_PAGE_LIMIT
from etl.tmdb_client import TMDBClient

MEDIA_TYPES = ("movie", "tv")


def _extract_release_year(data: Dict[str, Any]) -> Optional[int]:
    for key in ("release_date", "first_air_date"):
        value = data.get(key)
        if isinstance(value, str) and len(value) >= 4:
            year_part = value[:4]
            if year_part.isdigit():
                return int(year_part)
    return None


def map_item_payload(d: Dict[str, Any]) -> Dict[str, Any]:
    media_type = d.get("media_type") or ("tv" if "name" in d else "movie")
    title = d.get("title") or d.get("name")
    overview = d.get("overview")
    # runtime: only in details (minutes). For TV, use episode_run_time[0] if present.
    runtime = None
    if "runtime" in d and isinstance(d["runtime"], int):
        runtime = d["runtime"]
    elif media_type == "tv":
        ert = d.get("episode_run_time") or []
        runtime = ert[0] if ert else None
    poster_url = None
    if d.get("poster_path"):
        poster_url = f"https://image.tmdb.org/t/p/w500{d['poster_path']}"
    genres = None
    if d.get("genres"):
        genres = [{"id": g.get("id"), "name": g.get("name")} for g in d["genres"]]

    popularity = None
    raw_popularity = d.get("popularity")
    if raw_popularity is not None:
        try:
            popularity = float(raw_popularity)
        except (TypeError, ValueError):
            popularity = None

    vote_average = None
    if d.get("vote_average") is not None:
        try:
            vote_average = float(d["vote_average"])
        except (TypeError, ValueError):
            vote_average = None

    vote_count = None
    if d.get("vote_count") is not None:
        try:
            vote_count = int(d["vote_count"])
        except (TypeError, ValueError):
            vote_count = None

    list_ranks = d.pop("_list_ranks", None) or {}

    return dict(
        tmdb_id=int(d["id"]),
        media_type=media_type,
        title=title or "",
        overview=overview,
        runtime=runtime,
        original_language=d.get("original_language"),
        genres=genres,
        poster_url=poster_url,
        release_year=_extract_release_year(d),
        popularity=popularity,
        vote_average=vote_average,
        vote_count=vote_count,
        popular_rank=list_ranks.get("popular_rank"),
        trending_rank=list_ranks.get("trending_rank"),
        top_rated_rank=list_ranks.get("top_rated_rank"),
    )


async def _fetch_and_upsert(sessionmaker, pages: int):
    client = TMDBClient(TMDB_API_KEY)
    try:
        # Pools to collect IDs then fetch details (for reliable fields)
        candidate_signals: Dict[tuple[str, int], Dict[str, Optional[int]]] = (
            {}
        )  # {(media, id): rank info}

        def record_signal(media: str, raw_id: Any, field: str, rank: int):
            if raw_id is None:
                return
            try:
                tmdb_id = int(raw_id)
            except (TypeError, ValueError):
                return
            key = (media, tmdb_id)
            info = candidate_signals.setdefault(
                key,
                {"popular_rank": None, "top_rated_rank": None, "trending_rank": None},
            )
            current = info.get(field)
            if current is None or rank < current:
                info[field] = rank

        for media in MEDIA_TYPES:
            rank = 0
            async for it in client.iter_list(media, "popular", pages):
                rank += 1
                record_signal(media, it.get("id"), "popular_rank", rank)
            rank = 0
            async for it in client.iter_list(media, "top_rated", pages // 2):
                rank += 1
                record_signal(media, it.get("id"), "top_rated_rank", rank)
            rank = 0
            async for it in client.iter_list(media, "trending", pages // 2):
                rank += 1
                record_signal(media, it.get("id"), "trending_rank", rank)

        candidate_ids = list(candidate_signals.keys())

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
    finally:
        await client.aclose()


def _upsert_items(db: Session, items: List[Dict[str, Any]]):
    # fetch existing tmdb_ids to avoid duplicate insert
    ids = [int(x["id"]) for x in items if "id" in x]
    if not ids:
        return
    existing = set(
        [
            row[0]
            for row in db.execute(
                select(Item.tmdb_id).where(Item.tmdb_id.in_(ids))
            ).all()
        ]
    )
    to_insert = []
    to_update = []

    for d in items:
        mapped = map_item_payload(d)
        if mapped["tmdb_id"] in existing:
            to_update.append(mapped)
        else:
            to_insert.append(mapped)

    if to_insert:
        db.bulk_insert_mappings(Item, to_insert)

    for u in to_update:
        db.execute(
            select(Item).where(Item.tmdb_id == u["tmdb_id"])
        )  # touch to ensure table exists (paranoia)

        db.query(Item).filter(Item.tmdb_id == u["tmdb_id"]).update(u)


def run(pages: int = TMDB_PAGE_LIMIT):
    engine = get_engine()
    SessionLocal = get_sessionmaker()
    # Ensure connection OK
    with engine.connect() as _:
        pass
    asyncio.run(_fetch_and_upsert(SessionLocal, pages))


if __name__ == "__main__":
    run()
