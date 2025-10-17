from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from cachetools import TTLCache

import httpx

from etl.tmdb_client import TMDBClient

logger = logging.getLogger(__name__)

ENTITY_LINKER_CACHE: TTLCache[str, Dict[str, Any]] = TTLCache(maxsize=1000, ttl=300)
_LINKER_CACHE_LOCK = asyncio.Lock()


class EntityLinker:
    def __init__(self, tmdb_client: TMDBClient):
        self._tmdb = tmdb_client

    async def link_entities(self, query: str) -> Dict[str, Any]:
        """Resolves entities in a query to TMDB IDs."""
        if not query:
            return {}

        cached = ENTITY_LINKER_CACHE.get(query)
        if cached is not None:
            return cached

        async with _LINKER_CACHE_LOCK:
            cached = ENTITY_LINKER_CACHE.get(query)
            if cached is not None:
                return cached

            try:
                results = await self._tmdb.search(query, media_type="multi")
            except httpx.HTTPError:  # pragma: no cover - network failure guard
                logger.warning("Entity linker failed to fetch results for %s", query)
                ENTITY_LINKER_CACHE[query] = {}
                return {}

            payload = results.get("results") or []
            movies: List[int] = []
            shows: List[int] = []
            persons: List[int] = []
            media_map = {
                "movie": movies,
                "tv": shows,
                "person": persons,
            }

            for result in payload:
                media_type = result.get("media_type")
                tmdb_id = result.get("id")
                if tmdb_id is not None and media_type in media_map:
                    media_map[media_type].append(tmdb_id)

            linked_entities: Dict[str, Any] = {
                "movie": movies,
                "tv": shows,
                "person": persons,
            }

            ENTITY_LINKER_CACHE[query] = linked_entities
            return linked_entities
