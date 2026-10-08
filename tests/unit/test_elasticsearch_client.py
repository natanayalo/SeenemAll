from unittest.mock import MagicMock
import socket
import pytest
from api.core import elasticsearch_client as client


@pytest.mark.parametrize("resolvable", [True, False])
def test_container_hostname_resolution_and_explicit_hosts(monkeypatch, resolvable):
    def resolve(host):
        if not resolvable:
            raise socket.gaierror("container name unavailable")
        return "10.0.0.1"

    monkeypatch.setattr(socket, "gethostbyname", resolve)
    hosts = client._parse_hosts(
        " http://elasticsearch:9200, , https://example.test:9201 "
    )
    assert hosts == [
        "http://elasticsearch:9200" if resolvable else "http://localhost:9200",
        "https://example.test:9201",
    ]


def test_client_auth_configuration_and_cache(monkeypatch):
    monkeypatch.setattr(client.config, "ELASTICSEARCH_URL", "http://localhost:9200")
    monkeypatch.setattr(client.config, "ELASTICSEARCH_USERNAME", "")
    kwargs = client._client_kwargs()
    assert "basic_auth" not in kwargs
    monkeypatch.setattr(client.config, "ELASTICSEARCH_USERNAME", "test-user")
    monkeypatch.setattr(client.config, "ELASTICSEARCH_PASSWORD", "test-password")
    factory = MagicMock()
    monkeypatch.setattr(client, "Elasticsearch", factory)
    client.get_elasticsearch_client.cache_clear()
    try:
        assert client.get_elasticsearch_client() is client.get_elasticsearch_client()
        assert factory.call_count == 1
        assert factory.call_args.kwargs["basic_auth"] == ("test-user", "test-password")
        assert factory.call_args.kwargs["hosts"] == ["http://localhost:9200"]
    finally:
        client.get_elasticsearch_client.cache_clear()
