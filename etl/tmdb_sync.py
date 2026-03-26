from __future__ import annotations
import asyncio
from typing import List, Dict, Any, Optional, cast
import logging
from tqdm.auto import tqdm
from sqlalchemy.orm import Session
from sqlalchemy import select
from api.db.session import get_engine, get_sessionmaker
from api.db.models import Item
from api.config import TMDB_API_KEY, TMDB_PAGE_LIMIT
from etl.tmdb_client import TMDBClient
from api.core.maturity import normalize_rating

MEDIA_TYPES = ("movie", "tv")
logger = logging.getLogger(__name__)
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False


def _extract_release_year(data: Dict[str, Any]) -> Optional[int]:
    for key in ("release_date", "first_air_date"):
        value = data.get(key)
        if isinstance(value, str) and len(value) >= 4:
            year_part = value[:4]
            if year_part.isdigit():
                return int(year_part)
    return None


_PREFERRED_RATING_COUNTRIES = ("US", "CA", "GB", "AU")


def _pick_rating(
    entries: List[Dict[str, Any]],
    value_iterator,
) -> Optional[str]:
    by_country: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        country = entry.get("iso_3166_1")
        if isinstance(country, str):
            by_country.setdefault(country, []).append(entry)
    for country in _PREFERRED_RATING_COUNTRIES:
        for entry in by_country.get(country, []):
            for value in value_iterator(entry):
                normalized = normalize_rating(value)
                if normalized:
                    return normalized
    for entry in entries:
        for value in value_iterator(entry):
            normalized = normalize_rating(value)
            if normalized:
                return normalized
    return None


def _extract_movie_rating(data: Dict[str, Any]) -> Optional[str]:
    payload = data.get("release_dates") or {}
    results = payload.get("results") or []
    if not isinstance(results, list):
        return None

    def iterator(entry: Dict[str, Any]):
        release_dates = entry.get("release_dates") or []
        if not isinstance(release_dates, list):
            return []
        return [rel.get("certification") for rel in release_dates]

    return _pick_rating(results, iterator)


def _extract_tv_rating(data: Dict[str, Any]) -> Optional[str]:
    payload = data.get("content_ratings") or {}
    results = payload.get("results") or []
    if not isinstance(results, list):
        return None

    def iterator(entry: Dict[str, Any]):
        return [entry.get("rating")]

    return _pick_rating(results, iterator)


def _extract_maturity_rating(media_type: str, data: Dict[str, Any]) -> Optional[str]:
    if media_type == "movie":
        return _extract_movie_rating(data)
    if media_type == "tv":
        return _extract_tv_rating(data)
    return None


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_cast(
    data: Dict[str, Any], limit: int = 5
) -> Optional[List[Dict[str, Any]]]:
    credits = data.get("credits") or {}
    raw_cast = credits.get("cast") or []
    if not isinstance(raw_cast, list):
        return None

    def sort_key(entry: Dict[str, Any]):
        order = entry.get("order")
        order = order if isinstance(order, int) else 10_000
        return (order, -_safe_float(entry.get("popularity")))

    cast_members: List[Dict[str, Any]] = []
    seen_ids: set[int] = set()
    for entry in sorted(raw_cast, key=sort_key):
        person_id = _safe_int(entry.get("id"))
        name = entry.get("name")
        if person_id is None or not name:
            continue
        if person_id in seen_ids:
            continue
        cast_members.append(
            {
                "id": person_id,
                "name": name,
                "character": entry.get("character"),
                "order": entry.get("order"),
            }
        )
        seen_ids.add(person_id)
        if len(cast_members) >= limit:
            break
    return cast_members or None


def _filter_crew(
    data: Dict[str, Any],
    *,
    jobs: set[str],
    departments: set[str],
    limit: int,
) -> Optional[List[Dict[str, Any]]]:
    credits = data.get("credits") or {}
    raw_crew = credits.get("crew") or []
    if not isinstance(raw_crew, list):
        return None

    def matches(entry: Dict[str, Any]) -> bool:
        job = entry.get("job")
        department = entry.get("department")
        job_match = job in jobs if jobs else False
        dept_match = department in departments if departments else False
        return job_match or dept_match

    def sort_key(entry: Dict[str, Any]):
        order = entry.get("order")
        order = order if isinstance(order, int) else 10_000
        return (order, -_safe_float(entry.get("popularity")))

    selected: List[Dict[str, Any]] = []
    seen_ids: set[int] = set()
    for entry in sorted(raw_crew, key=sort_key):
        if not matches(entry):
            continue
        person_id = _safe_int(entry.get("id"))
        name = entry.get("name")
        if person_id is None or not name or person_id in seen_ids:
            continue
        selected.append(
            {
                "id": person_id,
                "name": name,
                "job": entry.get("job"),
                "department": entry.get("department"),
            }
        )
        seen_ids.add(person_id)
        if len(selected) >= limit:
            break
    return selected or None


def _extract_keywords(
    data: Dict[str, Any], limit: int = 30
) -> Optional[List[Dict[str, Any]]]:
    payload = data.get("keywords") or {}
    if isinstance(payload, dict):
        raw_keywords = payload.get("keywords") or payload.get("results") or []
    else:
        raw_keywords = []
    if not isinstance(raw_keywords, list):
        return None
    keywords: List[Dict[str, Any]] = []
    seen_ids: set[int] = set()
    for entry in raw_keywords:
        keyword_id = _safe_int(entry.get("id"))
        name = entry.get("name")
        if keyword_id is None or not name or keyword_id in seen_ids:
            continue
        keywords.append({"id": keyword_id, "name": name})
        seen_ids.add(keyword_id)
        if len(keywords) >= limit:
            break
    return keywords or None


def _extract_spoken_languages(data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    payload = data.get("spoken_languages") or []
    if not isinstance(payload, list):
        return None
    languages: List[Dict[str, Any]] = []
    for entry in payload:
        code = entry.get("iso_639_1")
        name = entry.get("english_name") or entry.get("name")
        if not code and not name:
            continue
        languages.append({"iso_639_1": code, "name": name})
    return languages or None


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

    collection = d.get("belongs_to_collection")
    collection_id = None
    collection_name = None
    if collection and isinstance(collection, dict):
        collection_id = collection.get("id")
        collection_name = collection.get("name")

    cast_members = _extract_cast(d, limit=5)
    directors = _filter_crew(
        d,
        jobs={"Director", "Co-Director"},
        departments={"Directing"},
        limit=2,
    )
    producers = _filter_crew(
        d,
        jobs={
            "Producer",
            "Executive Producer",
            "Co-Producer",
            "Associate Producer",
            "Line Producer",
        },
        departments={"Production"},
        limit=2,
    )
    writers = _filter_crew(
        d,
        jobs={
            "Writer",
            "Screenplay",
            "Story",
            "Author",
            "Teleplay",
            "Adaptation",
            "Novel",
        },
        departments={"Writing"},
        limit=2,
    )
    keywords = _extract_keywords(d)
    spoken_languages = _extract_spoken_languages(d)

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
        maturity_rating=_extract_maturity_rating(media_type, d),
        collection_id=collection_id,
        collection_name=collection_name,
        popularity=popularity,
        vote_average=vote_average,
        vote_count=vote_count,
        popular_rank=list_ranks.get("popular_rank"),
        trending_rank=list_ranks.get("trending_rank"),
        top_rated_rank=list_ranks.get("top_rated_rank"),
        cast=cast_members,
        directors=directors,
        producers=producers,
        writers=writers,
        keywords=keywords,
        spoken_languages=spoken_languages,
    )


async def _fetch_and_upsert(sessionmaker, pages: int):
    client = TMDBClient(TMDB_API_KEY)
    try:
        logger.info("Starting TMDB sync for %d pages per media type.", pages)
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
            collected = 0
            rank = 0
            async for it in client.iter_list(media, "popular", pages):
                rank += 1
                collected += 1
                record_signal(media, it.get("id"), "popular_rank", rank)
            rank = 0
            async for it in client.iter_list(media, "top_rated", pages // 2):
                rank += 1
                collected += 1
                record_signal(media, it.get("id"), "top_rated_rank", rank)
            rank = 0
            async for it in client.iter_list(media, "trending", pages // 2):
                rank += 1
                collected += 1
                record_signal(media, it.get("id"), "trending_rank", rank)
            logger.info("Collected %d candidate IDs for %s.", collected, media)

        candidate_ids = list(candidate_signals.keys())
        logger.info("Total unique candidate IDs: %d", len(candidate_ids))

        # Fetch details concurrently in batches
        BATCH = 20
        batch_indices = range(0, len(candidate_ids), BATCH)
        progress = tqdm(
            batch_indices,
            desc="TMDB sync",
            unit="batch",
            total=(len(candidate_ids) + BATCH - 1) // BATCH,
            leave=False,
        )
        for i in progress:
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
        progress.close()
        logger.info("TMDB sync completed successfully.")
    finally:
        await client.aclose()


def _upsert_items(db: Session, items: List[Dict[str, Any]]):
    # fetch existing tmdb_ids to avoid duplicate insert
    ids = [int(x["id"]) for x in items if "id" in x]
    if not ids:
        return
    result = db.execute(
        select(Item.tmdb_id, Item.media_type).where(Item.tmdb_id.in_(ids))
    ).all()
    existing = {(row.tmdb_id, row.media_type) for row in result}
    to_insert = []
    to_update = []

    for d in items:
        mapped = map_item_payload(d)
        key = (mapped["tmdb_id"], mapped["media_type"])
        if key in existing:
            to_update.append(mapped)
        else:
            to_insert.append(mapped)

    if to_insert:
        db.bulk_insert_mappings(Item, to_insert)

    for u in to_update:
        db.execute(
            select(Item).where(
                Item.tmdb_id == u["tmdb_id"], Item.media_type == u["media_type"]
            )
        )
        (
            db.query(Item)
            .filter(
                Item.tmdb_id == u["tmdb_id"],
                Item.media_type == u["media_type"],
            )
            .update(u)
        )


def run(pages: int = TMDB_PAGE_LIMIT):
    engine = get_engine()
    SessionLocal = get_sessionmaker()
    # Ensure connection OK
    with engine.connect() as _:
        pass
    asyncio.run(_fetch_and_upsert(SessionLocal, pages))


if __name__ == "__main__":
    run()
