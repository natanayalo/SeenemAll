from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Optional, Tuple

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from api.config import (
    COUNTRY_DEFAULT,
    JUSTWATCH_LANGUAGE,
    JUSTWATCH_PLATFORM,
)
from api.db.models import Availability, Item
from api.db.session import get_engine, get_sessionmaker
from etl.justwatch_client import JustWatchClient, flatten_offers

CONCURRENCY = 6
CHUNK_SIZE = 25


@dataclass(slots=True)
class ItemRow:
    item_id: int
    tmdb_id: int
    media_type: str
    title: str
    release_year: Optional[int]


def _fetch_catalog_items(db: Session, limit: Optional[int]) -> List[ItemRow]:
    stmt = (
        select(
            Item.id,
            Item.tmdb_id,
            Item.media_type,
            Item.title,
            Item.release_year,
        )
        .where(Item.title.isnot(None))
        .order_by(Item.id)
    )
    if limit is not None:
        stmt = stmt.limit(limit)
    rows = db.execute(stmt).all()
    result: List[ItemRow] = []
    for item_id, tmdb_id, media_type, title, release_year in rows:
        if not title:
            continue
        result.append(
            ItemRow(
                item_id=int(item_id),
                tmdb_id=int(tmdb_id),
                media_type=media_type,
                title=title,
                release_year=int(release_year) if release_year is not None else None,
            )
        )
    return result


def _pick_deeplink(urls: Dict[str, Any]) -> Optional[str]:
    for key in (
        "deeplink_ios",
        "deeplink_android",
        "deeplink_tvos",
        "deeplink_web",
        "deeplink",
        "alternate_web",
    ):
        link = urls.get(key)
        if link:
            return link
    return None


def _normalise_offer(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "service" in raw and "offer_type" in raw:
        return {
            "service": raw["service"],
            "offer_type": str(raw["offer_type"]).upper(),
            "deeplink": raw.get("deeplink"),
            "web_url": raw.get("web_url"),
        }

    provider = raw.get("package_short_name") or raw.get("provider_id")
    monetization = raw.get("monetization_type") or raw.get("presentation_type")
    if provider is None or monetization is None:
        return None

    urls = raw.get("urls") or {}
    web_url = urls.get("standard_web") or urls.get("web") or urls.get("disney_web_url")
    deeplink = _pick_deeplink(urls)

    return {
        "service": str(provider),
        "offer_type": str(monetization).upper(),
        "deeplink": deeplink,
        "web_url": web_url,
    }


def _prepare_offers(raw_offers: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[Tuple[str, str, Optional[str]]] = set()
    prepared: List[Dict[str, Any]] = []
    for offer in raw_offers:
        normalised = _normalise_offer(offer)
        if not normalised:
            continue
        key = (
            normalised["service"],
            normalised["offer_type"],
            normalised["web_url"],
        )
        if key in seen:
            continue
        seen.add(key)
        prepared.append(normalised)
    return prepared


def _replace_availability(
    db: Session,
    item_id: int,
    country: str,
    offers: Iterable[Dict[str, Any]],
) -> None:
    db.execute(
        delete(Availability).where(
            Availability.item_id == item_id, Availability.country == country
        )
    )
    payload = []
    for offer in offers:
        payload.append(
            {
                "item_id": item_id,
                "country": country,
                "service": offer["service"],
                "offer_type": offer["offer_type"],
                "deeplink": offer.get("deeplink"),
                "web_url": offer.get("web_url"),
            }
        )
    if payload:
        db.bulk_insert_mappings(Availability, payload)


async def _fetch_chunk(client: JustWatchClient, rows: List[ItemRow]) -> List[Any]:
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _worker(row: ItemRow) -> Any:
        async with sem:
            match, offers = await client.resolve_offers_for_item(
                title=row.title,
                media_type=row.media_type,
                release_year=row.release_year,
            )
            return {"match": match, "offers": offers}

    tasks = [asyncio.create_task(_worker(row)) for row in rows]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def _sync_availability(
    country: str,
    limit: Optional[int],
) -> None:
    SessionLocal = get_sessionmaker()
    client = JustWatchClient(
        country=country,
        language=JUSTWATCH_LANGUAGE,
        platform=JUSTWATCH_PLATFORM,
    )
    try:
        with SessionLocal() as db:
            items = _fetch_catalog_items(db, limit)

        total_processed = 0
        total_updated = 0
        for idx in range(0, len(items), CHUNK_SIZE):
            chunk = items[idx : idx + CHUNK_SIZE]
            responses = await _fetch_chunk(client, chunk)

            with SessionLocal() as db:
                for row, resp in zip(chunk, responses):
                    if isinstance(resp, Exception):
                        continue
                    raw_offers = resp.get("offers") or []
                    prepared = _prepare_offers(
                        flatten_offers(raw_offers) if raw_offers else []
                    )
                    if not prepared:
                        continue
                    total_updated += len(prepared)
                    _replace_availability(db, row.item_id, country, prepared)
                db.commit()
            total_processed += len(chunk)
            print(
                f"[justwatch] processed {total_processed} titles "
                f"(updated {total_updated} offers so far)"
            )
    finally:
        await client.aclose()


def run(country: Optional[str] = None, limit: Optional[int] = None) -> None:
    """
    Entry point for `python -m etl.justwatch_sync`.
    """
    country = country or COUNTRY_DEFAULT
    engine = get_engine()
    with engine.connect():
        pass
    asyncio.run(_sync_availability(country=country, limit=limit))


if __name__ == "__main__":  # pragma: no cover
    run()
