import pytest
from elasticsearch.exceptions import TransportError

from api.core import elasticsearch_search
from api.core.elasticsearch_search import (
    ElasticsearchSearchError,
    SearchFilters,
    knn_search,
)


class _DummyTransportError(TransportError):
    def __init__(self, message="boom"):
        super().__init__(500, message)


def test_knn_search_builds_body(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self):
            self.calls = []
            self._responses = iter(
                [
                    {
                        "hits": {
                            "hits": [
                                {
                                    "_id": "1",
                                    "_score": 0.5,
                                    "_source": {"item_id": "1"},
                                }
                            ]
                        }
                    },
                    {"hits": {"hits": []}},
                ]
            )

        def search(self, **kwargs):
            self.calls.append(kwargs)
            try:
                return next(self._responses)
            except StopIteration:
                return {"hits": {"hits": []}}

    fake_client = FakeClient()
    monkeypatch.setattr(
        elasticsearch_search, "get_elasticsearch_client", lambda: fake_client
    )

    results = knn_search(
        [0.1, 0.2, 0.3],
        k=5,
        num_candidates=10,
        text_query="space opera",
        filters=SearchFilters(genres=["sci-fi"], runtime_lte=150),
        source_includes=["title"],
    )

    assert results[0]["item_id"] == "1"
    assert len(fake_client.calls) == 2

    knn_call = fake_client.calls[0]
    knn_body = knn_call["body"]
    assert knn_body["knn"]["k"] == 5
    assert knn_body["knn"]["num_candidates"] == 10
    knn_filter = knn_body["knn"]["filter"]["bool"]
    assert {"terms": {"genres": ["sci-fi"]}} in knn_filter["filter"]
    assert {"range": {"runtime": {"lte": 150}}} in knn_filter["filter"]
    assert knn_call["_source"]["includes"] == ["title"]

    text_call = fake_client.calls[1]
    text_body = text_call["body"]
    assert "knn" not in text_body
    text_bool = text_body["query"]["bool"]
    assert any("multi_match" in clause for clause in text_bool["must"])
    assert {"terms": {"genres": ["sci-fi"]}} in text_bool["filter"]
    assert text_call["_source"]["includes"] == ["title"]


def test_knn_search_applies_exclusions(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self):
            self.calls = []

        def search(self, **kwargs):
            self.calls.append(kwargs)
            return {"hits": {"hits": []}}

    fake_client = FakeClient()
    monkeypatch.setattr(
        elasticsearch_search, "get_elasticsearch_client", lambda: fake_client
    )

    knn_search(
        [0.1, 0.2],
        k=2,
        filters=SearchFilters(exclude_item_ids=["123"]),
    )

    call = fake_client.calls[0]
    knn_filter = call["body"]["knn"]["filter"]["bool"]
    assert knn_filter["must_not"] == [{"terms": {"item_id": ["123"]}}]


def test_knn_search_raises_on_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def search(self, **kwargs):
            raise _DummyTransportError("failed")

    monkeypatch.setattr(
        elasticsearch_search, "get_elasticsearch_client", lambda: FakeClient()
    )

    with pytest.raises(ElasticsearchSearchError):
        knn_search([0.0, 0.1])


def test_knn_search_defaults_num_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self):
            self.calls = []

        def search(self, **kwargs):
            self.calls.append(kwargs)
            return {"hits": {"hits": []}}

    fake_client = FakeClient()
    monkeypatch.setattr(
        elasticsearch_search, "get_elasticsearch_client", lambda: fake_client
    )

    results = knn_search([0.0, 0.1], k=10)
    assert results == []
    knn = fake_client.calls[0]["body"]["knn"]
    assert knn["num_candidates"] >= knn["k"]
