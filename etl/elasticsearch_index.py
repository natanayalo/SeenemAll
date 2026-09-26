from __future__ import annotations

from typing import Any, Dict

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConflictError, NotFoundError

from api import config

ITEMS_INDEX_NAME = config.ELASTICSEARCH_ITEMS_INDEX


def build_items_index_body() -> Dict[str, Any]:
    """
    Build the items index settings + mappings for Elasticsearch.
    """
    return {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "analysis": {
                "analyzer": {
                    "english_with_stop": {
                        "tokenizer": "standard",
                        "filter": ["lowercase", "porter_stem", "english_stop"],
                    }
                },
                "filter": {
                    "english_stop": {"type": "stop", "stopwords": "_english_"},
                },
            },
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "item_id": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "english_with_stop",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                },
                "overview": {
                    "type": "text",
                    "analyzer": "english_with_stop",
                },
                "genres": {"type": "keyword"},
                "media_type": {"type": "keyword"},
                "runtime": {"type": "short"},
                "release_year": {"type": "short"},
                "maturity": {"type": "keyword"},
                "streaming_providers": {"type": "keyword"},
                "popularity": {"type": "float"},
                "cast": {"type": "keyword"},
                "directors": {"type": "keyword"},
                "producers": {"type": "keyword"},
                "writers": {"type": "keyword"},
                "keywords": {"type": "keyword"},
                "spoken_languages": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine",
                    "index_options": {
                        "type": "hnsw",
                        "m": 16,
                        "ef_construction": 128,
                    },
                },
                "updated_at": {"type": "date"},
            },
        },
    }


def ensure_items_index(client: Elasticsearch, *, force: bool = False) -> None:
    """
    Ensure the items index exists with the desired mapping. Optionally recreate.
    """
    index_body = build_items_index_body()

    if force:
        try:
            client.indices.delete(index=ITEMS_INDEX_NAME)
        except NotFoundError:
            pass

    if client.indices.exists(index=ITEMS_INDEX_NAME):
        return

    try:
        client.indices.create(
            index=ITEMS_INDEX_NAME,
            settings=index_body["settings"],
            mappings=index_body["mappings"],
        )
    except ConflictError:
        # Another process created it between exists check and create
        pass
