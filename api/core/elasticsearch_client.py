from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

from elasticsearch import Elasticsearch

from api import config


def _parse_hosts(raw_hosts: str) -> List[str]:
    return [host.strip() for host in raw_hosts.split(",") if host.strip()]


def _client_kwargs() -> Dict:
    kwargs: Dict = {
        "hosts": _parse_hosts(config.ELASTICSEARCH_URL),
        "request_timeout": config.ELASTICSEARCH_TIMEOUT,
        "verify_certs": config.ELASTICSEARCH_VERIFY_CERTS,
    }

    if config.ELASTICSEARCH_USERNAME:
        kwargs["basic_auth"] = (
            config.ELASTICSEARCH_USERNAME,
            config.ELASTICSEARCH_PASSWORD,
        )

    return kwargs


def create_elasticsearch_client() -> Elasticsearch:
    """
    Create a fresh Elasticsearch client.
    Separated from the cached accessor so scripts can control lifecycle.
    """
    return Elasticsearch(**_client_kwargs())


@lru_cache(maxsize=1)
def get_elasticsearch_client() -> Elasticsearch:
    """
    Cached Elasticsearch client for application usage.
    """
    return create_elasticsearch_client()
