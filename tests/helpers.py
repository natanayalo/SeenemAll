from __future__ import annotations

from typing import Any, Iterable, List, Sequence


from api.db.models import ItemEmbedding, User, UserHistory


class _HistoryWeightsQuery:
    def __init__(self, rows: Sequence[tuple[int, float]]):
        self._rows = list(rows)
        self._limit: int | None = None

    def filter(self, *args: Any, **kwargs: Any) -> "_HistoryWeightsQuery":
        return self

    def order_by(self, *args: Any, **kwargs: Any) -> "_HistoryWeightsQuery":
        return self

    def limit(self, n: int) -> "_HistoryWeightsQuery":
        self._limit = n
        return self

    def all(self) -> List[tuple[int, float]]:
        data = self._rows
        if self._limit is not None:
            data = data[: self._limit]
        return list(data)


class _EmbeddingQuery:
    def __init__(self, vectors: Iterable[Iterable[float]]):
        self._vectors = [tuple(vec) for vec in vectors]

    def filter(self, *args: Any, **kwargs: Any) -> "_EmbeddingQuery":
        return self

    def all(self) -> List[tuple[List[float]]]:
        return [([float(v) for v in vec],) for vec in self._vectors]


class _HistoryIdsQuery:
    def __init__(self, item_ids: Sequence[int]):
        self._ids = list(item_ids)

    def filter(self, *args: Any, **kwargs: Any) -> "_HistoryIdsQuery":
        return self

    def all(self) -> List[tuple[int]]:
        return [(iid,) for iid in self._ids]


class _UserQuery:
    def __init__(self, session: "FakeSession"):
        self._session = session

    def filter(self, *args: Any, **kwargs: Any) -> "_UserQuery":
        return self

    def one_or_none(self) -> User | None:
        return self._session.user


class FakeSession:
    """
    Lightweight stub to emulate SQLAlchemy session behaviour for unit tests.
    """

    def __init__(
        self,
        history_rows: Sequence[tuple[int, float]] | None = None,
        embedding_vectors: Iterable[Iterable[float]] | None = None,
        user: User | None = None,
        history_ids: Sequence[int] | None = None,
    ):
        self.history_rows = list(history_rows or [])
        self.embeddings = list(embedding_vectors or [])
        self.user = user
        if history_ids is not None:
            self.history_ids = list(history_ids)
        else:
            self.history_ids = [row[0] for row in self.history_rows]
        self.added: List[Any] = []
        self.commits: int = 0

    # SQLAlchemy API stubs -------------------------------------------------
    def query(self, *entities: Any):
        if (
            len(entities) == 2
            and entities[0] is UserHistory.item_id
            and entities[1] is UserHistory.weight
        ):
            return _HistoryWeightsQuery(self.history_rows)
        if len(entities) == 1 and entities[0] is ItemEmbedding.vector:
            return _EmbeddingQuery(self.embeddings)
        if len(entities) == 1 and entities[0] is UserHistory.item_id:
            return _HistoryIdsQuery(self.history_ids)
        if len(entities) == 1 and entities[0] is User:
            return _UserQuery(self)
        raise AssertionError(f"Unsupported query entities: {entities!r}")

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        pass


class FakeResult:
    """
    Minimal result wrapper to emulate SQLAlchemy scalar result contract.
    """

    def __init__(self, rows: Sequence[Any]):
        self._rows = list(rows)
        self._scalar_mode = False

    def scalars(self) -> "FakeResult":
        result = FakeResult(self._rows)
        result._scalar_mode = True
        return result

    def all(self) -> List[Any]:
        if self._scalar_mode:
            return [
                row[0] if isinstance(row, (tuple, list)) else row for row in self._rows
            ]
        return list(self._rows)

    def first(self) -> Any | None:
        if not self._rows:
            return None
        if self._scalar_mode and isinstance(self._rows[0], (tuple, list)):
            return self._rows[0][0]
        return self._rows[0]

    def __iter__(self):
        if self._scalar_mode:
            return iter(
                row[0] if isinstance(row, (tuple, list)) else row for row in self._rows
            )
        return iter(self._rows)
