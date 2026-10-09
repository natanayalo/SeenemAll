from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from sqlalchemy import select

from api.db.models import Availability
from api.db.session import get_db
from api.config import COUNTRY_DEFAULT
from api.pipeline.context import _DEFAULT_STREAMING_PROVIDER_ALIASES

router = APIRouter(tags=["watch"])


@router.get("/watch-link/{item_id}")
def get_watch_link(
    item_id: int,
    service: str = Query(
        ...,
        description="Streaming service slug or short name (e.g., 'netflix' or 'nfx')",
    ),
    country: str = Query(
        COUNTRY_DEFAULT, description="ISO 3166-1 alpha-2 country code"
    ),
    db: Session = Depends(get_db),
):
    """Redirect to the deep link for a given item, service, and country."""
    service_clean = service.strip().lower()
    service_variants = {service_clean}
    for canonical, variants in _DEFAULT_STREAMING_PROVIDER_ALIASES.items():
        if service_clean in variants or service_clean == canonical:
            service_variants.update(variants)
            break

    stmt = (
        select(Availability.deeplink, Availability.web_url)
        .where(
            Availability.item_id == item_id,
            Availability.service.in_(service_variants),
            Availability.country == country,
        )
        .limit(1)
    )
    result = db.execute(stmt).first()
    if not result:
        raise HTTPException(status_code=404, detail="Link not found")

    deeplink, web_url = result
    url = deeplink or web_url
    if not url:
        raise HTTPException(status_code=404, detail="Link not found")

    return RedirectResponse(url=url)
