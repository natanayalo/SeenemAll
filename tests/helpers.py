from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Dict, cast
from collections.abc import Iterable as IterableABC


from api.db.models import ItemEmbedding, Item, User, UserHistory


class _HistoryQuery:
    def __init__(self, rows: Sequence[tuple[int, float, str]]):
        self._rows = list(rows)
        self._limit: int | None = None

    def filter(self, *args: Any, **kwargs: Any) -> "_HistoryQuery":
        return self

    def order_by(self, *args: Any, **kwargs: Any) -> "_HistoryQuery":
        return self

    def limit(self, n: int) -> "_HistoryQuery":
        self._limit = n
        return self

    def all(self) -> List[tuple[int, float, str]]:
        data = self._rows
        if self._limit is not None:
            data = data[: self._limit]
        return list(data)


class _EmbeddingVectorQuery:
    def __init__(self, rows: Iterable[tuple[int, Iterable[float]]]):
        self._rows = [(item_id, [float(v) for v in vec]) for item_id, vec in rows]

    def filter(self, *args: Any, **kwargs: Any) -> "_EmbeddingVectorQuery":
        return self

    def all(self) -> List[tuple[List[float]]]:
        return [(vec,) for _, vec in self._rows]


class _EmbeddingWithIdQuery:
    def __init__(self, rows: Iterable[tuple[int, Iterable[float]]]):
        self._rows = [(item_id, [float(v) for v in vec]) for item_id, vec in rows]

    def filter(self, *args: Any, **kwargs: Any) -> "_EmbeddingWithIdQuery":
        return self

    def all(self) -> List[tuple[int, List[float]]]:
        return list(self._rows)


class _HistoryIdsQuery:
    def __init__(self, item_ids: Sequence[int]):
        self._ids = list(item_ids)

    def filter(self, *args: Any, **kwargs: Any) -> "_HistoryIdsQuery":
        return self

    def all(self) -> List[tuple[int]]:
        return [(iid,) for iid in self._ids]


class _HistoryItemEventQuery:
    def __init__(self, rows: Sequence[tuple[int, float, str]]):
        self._rows = list(rows)

    def filter(self, *args: Any, **kwargs: Any) -> "_HistoryItemEventQuery":
        return self

    def all(self) -> List[tuple[int, str]]:
        return [(item_id, event_type) for item_id, _, event_type in self._rows]


class _HistoryUserWeightQuery:
    def __init__(self, rows: Sequence[tuple[int, float, str]]):
        self._rows = list(rows)

    def filter(self, *args: Any, **kwargs: Any) -> "_HistoryUserWeightQuery":
        return self

    def all(self) -> List[tuple[str, float]]:
        # Fake sessions do not model other users, so return an empty list.
        return []


class _ItemQuery:
    def __init__(self, rows: Dict[int, List[dict]]):
        self._rows = rows
        self._filter_ids: List[int] | None = None

    def filter(self, criterion, *args: Any, **kwargs: Any) -> "_ItemQuery":
        # Expect criterion to be Item.id.in_(ids)
        ids = getattr(getattr(criterion, "right", None), "value", None)
        if ids is not None:
            self._filter_ids = list(ids)
        return self

    def all(self) -> List[tuple[int, List[dict]]]:
        if self._filter_ids is None:
            return [(item_id, list(genres)) for item_id, genres in self._rows.items()]
        return [
            (item_id, list(self._rows.get(item_id, []))) for item_id in self._filter_ids
        ]


class _UserQuery:
    def __init__(self, session: "FakeSession"):
        self._session = session
        self._filter_ids: List[str] | None = None

    def filter(self, *criteria: Any, **kwargs: Any) -> "_UserQuery":
        for criterion in criteria:
            right = getattr(getattr(criterion, "right", None), "value", None)
            if right is None:
                continue
            if isinstance(right, list):
                self._filter_ids = [str(v) for v in right]
            else:
                self._filter_ids = [str(right)]
        return self

    def one_or_none(self) -> User | None:
        if self._filter_ids:
            uid = self._filter_ids[0]
            return self._session.user_map.get(uid)
        return self._session.user

    def all(self) -> List[User]:
        if self._filter_ids is None:
            return list(self._session.user_map.values())
        return [
            self._session.user_map[uid]
            for uid in self._filter_ids
            if uid in self._session.user_map
        ]


class FakeSession:
    """
    Lightweight stub to emulate SQLAlchemy session behaviour for unit tests.
    """

    def __init__(
        self,
        history_rows: (
            Sequence[tuple[int, float] | tuple[int, float, str]] | None
        ) = None,
        embedding_vectors: (
            Iterable[Iterable[float]] | Iterable[tuple[int, Iterable[float]]] | None
        ) = None,
        item_rows: Iterable[tuple[int, list[dict]]] | None = None,
        user: User | None = None,
        other_users: Sequence[User] | None = None,
        history_ids: Sequence[int] | None = None,
    ):
        processed_rows: List[tuple[int, float, str]] = []
        for row in history_rows or []:
            if len(row) == 2:
                item_id, weight = row
                event_type = "watched"
            elif len(row) == 3:
                item_id, weight, event_type = row
            else:
                raise ValueError(f"Unexpected history row shape: {row!r}")
            processed_rows.append((item_id, float(weight), event_type))
        self.history_rows = processed_rows

        processed_embeddings: List[tuple[int, List[float]]] = []
        if embedding_vectors is not None:
            for entry in embedding_vectors:
                if (
                    isinstance(entry, tuple)
                    and len(entry) == 2
                    and isinstance(entry[0], (int, float))
                    and isinstance(entry[1], IterableABC)
                ):
                    item_id, vec_iter = entry
                    processed_embeddings.append(
                        (int(item_id), [float(v) for v in vec_iter])
                    )
                elif isinstance(entry, IterableABC):
                    vec_iter = [float(v) for v in cast(Iterable[float], entry)]
                    item_id = len(processed_embeddings)
                    processed_embeddings.append((item_id, vec_iter))
                else:
                    raise TypeError(
                        "embedding_vectors entries must be iterable vectors or (id, vector) tuples"
                    )
        self.embeddings = processed_embeddings
        self.user = user
        self.user_map: Dict[str, User] = {}
        if user is not None and getattr(user, "user_id", None) is not None:
            self.user_map[str(user.user_id)] = user
        for other in other_users or []:
            if getattr(other, "user_id", None) is None:
                continue
            self.user_map[str(other.user_id)] = other
        if history_ids is not None:
            self.history_ids = list(history_ids)
        else:
            self.history_ids = [row[0] for row in self.history_rows]
        self.added: List[Any] = []
        self.commits: int = 0
        self.item_rows = {
            item_id: list(genres or []) for item_id, genres in item_rows or []
        }
        for item_id, _ in self.embeddings:
            self.item_rows.setdefault(item_id, [])

    # SQLAlchemy API stubs -------------------------------------------------
    def query(self, *entities: Any):
        if set(entities) == {
            UserHistory.item_id,
            UserHistory.weight,
            UserHistory.event_type,
        }:
            return _HistoryQuery(self.history_rows)
        if set(entities) == {
            UserHistory.item_id,
            UserHistory.event_type,
        }:
            return _HistoryItemEventQuery(self.history_rows)
        if set(entities) == {
            UserHistory.user_id,
            UserHistory.weight,
        }:
            return _HistoryUserWeightQuery(self.history_rows)
        if (
            len(entities) == 2
            and entities[0] is ItemEmbedding.item_id
            and entities[1] is ItemEmbedding.vector
        ):
            return _EmbeddingWithIdQuery(self.embeddings)
        if len(entities) == 1 and entities[0] is ItemEmbedding.vector:
            return _EmbeddingVectorQuery(self.embeddings)
        if len(entities) == 2 and entities[0] is Item.id and entities[1] is Item.genres:
            return _ItemQuery(self.item_rows)
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
