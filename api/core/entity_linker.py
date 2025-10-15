from __future__ import annotations

from typing import Dict, Any

from cachetools import TTLCache

from etl.tmdb_client import TMDBClient

ENTITY_LINKER_CACHE: TTLCache[str, Dict[str, Any]] = TTLCache(maxsize=1000, ttl=300)


class EntityLinker:
    def __init__(self, tmdb_client: TMDBClient):
        self._tmdb = tmdb_client

    async def link_entities(self, query: str) -> Dict[str, Any]:
        """Resolves entities in a query to TMDB IDs."""
        if query in ENTITY_LINKER_CACHE:
            return ENTITY_LINKER_CACHE[query]

        results = await self._tmdb.search(query, media_type="multi")
        if not results.get("results"):
            return {}

        linked_entities: Dict[str, Any] = {"movie": [], "person": []}
        for result in results["results"]:
            media_type = result.get("media_type")
            if media_type == "movie":
                linked_entities["movie"].append(result["id"])
            elif media_type == "person":
                linked_entities["person"].append(result["id"])

        ENTITY_LINKER_CACHE[query] = linked_entities
        return linked_entities
