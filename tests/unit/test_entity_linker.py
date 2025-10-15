from __future__ import annotations

import asyncio
from unittest.mock import Mock

from api.core.entity_linker import EntityLinker


def test_entity_linker_links_movies():
    async def scenario():
        from api.core.entity_linker import ENTITY_LINKER_CACHE

        ENTITY_LINKER_CACHE.clear()

        tmdb_client = Mock()
        tmdb_client.search = Mock(return_value=asyncio.Future())
        tmdb_client.search.return_value.set_result(
            {"results": [{"id": 123, "title": "The Matrix", "media_type": "movie"}]}
        )

        linker = EntityLinker(tmdb_client)
        result = await linker.link_entities("the matrix")

        assert result == {"movie": [123], "person": []}
        tmdb_client.search.assert_called_once_with("the matrix", media_type="multi")

    asyncio.run(scenario())


def test_entity_linker_caches_results():
    async def scenario():
        from api.core.entity_linker import ENTITY_LINKER_CACHE

        ENTITY_LINKER_CACHE.clear()

        tmdb_client = Mock()
        tmdb_client.search = Mock(return_value=asyncio.Future())
        tmdb_client.search.return_value.set_result(
            {"results": [{"id": 123, "title": "The Matrix", "media_type": "movie"}]}
        )

        linker = EntityLinker(tmdb_client)
        result1 = await linker.link_entities("the matrix")
        result2 = await linker.link_entities("the matrix")

        assert result1 == {"movie": [123], "person": []}
        assert result2 == {"movie": [123], "person": []}
        tmdb_client.search.assert_called_once_with("the matrix", media_type="multi")

    asyncio.run(scenario())


def test_entity_linker_links_people():
    async def scenario():
        from api.core.entity_linker import ENTITY_LINKER_CACHE

        ENTITY_LINKER_CACHE.clear()

        tmdb_client = Mock()
        tmdb_client.search = Mock(return_value=asyncio.Future())
        tmdb_client.search.return_value.set_result(
            {
                "results": [
                    {"id": 123, "name": "Denis Villeneuve", "media_type": "person"}
                ]
            }
        )

        linker = EntityLinker(tmdb_client)
        result = await linker.link_entities("denis villeneuve")

        assert result == {"movie": [], "person": [123]}
        tmdb_client.search.assert_called_once_with(
            "denis villeneuve", media_type="multi"
        )

    asyncio.run(scenario())
