from __future__ import annotations
import asyncio
import time
from typing import AsyncIterator, Dict, Any
import httpx

TMDB_BASE = "https://api.themoviedb.org/3"


class TMDBClient:
    def __init__(self, api_key: str, timeout: float = 15.0, rate_per_sec: float = 3.0):
        self.api_key = api_key
        self.timeout = timeout
        self.rate = rate_per_sec
        self._last = 0.0
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def _throttle(self):
        dt = time.time() - self._last
        min_gap = 1.0 / max(self.rate, 1e-6)
        if dt < min_gap:
            await asyncio.sleep(min_gap - dt)
        self._last = time.time()

    async def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        await self._throttle()
        q = dict(params)
        q["api_key"] = self.api_key
        r = await self._client.get(f"{TMDB_BASE}{path}", params=q)
        r.raise_for_status()
        return r.json()

    async def iter_list(
        self, media: str, list_name: str, pages: int
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        list_name in {"popular","top_rated","trending"}  (for trending we’ll use /trending/<media>/day)
        """
        if list_name == "trending":
            for page in range(1, pages + 1):
                data = await self._get(f"/trending/{media}/day", {"page": page})
                for it in data.get("results", []):
                    it["media_type"] = media
                    yield it
        else:
            for page in range(1, pages + 1):
                data = await self._get(f"/{media}/{list_name}", {"page": page})
                for it in data.get("results", []):
                    it["media_type"] = media
                    yield it

    async def details(self, media: str, tmdb_id: int) -> Dict[str, Any]:
        return await self._get(f"/{media}/{tmdb_id}", {"language": "en-US"})

    async def aclose(self):
        await self._client.aclose()
