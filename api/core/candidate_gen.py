from __future__ import annotations
from typing import List, Sequence
from sqlalchemy import text, bindparam
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector
import numpy as np


def ann_candidates(
    db: Session,
    user_vec: np.ndarray,
    exclude_ids: List[int],
    limit: int = 300,
    allowed_ids: Sequence[int] | None = None,
) -> List[int]:
    """
    Returns item_ids ordered by cosine distance to user_vec.
    Bind :uvec as pgvector, not numeric[].
    """
    if user_vec is None:
        return []

    if allowed_ids is not None and len(allowed_ids) == 0:
        return []

    where_clauses = ["NOT (e.item_id = ANY(:exclude))"]
    params = {
        "exclude": exclude_ids or [],
        "uvec": list(map(float, user_vec)),
        "lim": limit,
    }

    if allowed_ids is not None:
        where_clauses.append("e.item_id = ANY(:allowed)")
        params["allowed"] = list(allowed_ids)

    q = text(
        """
        WITH deduped AS (
            SELECT DISTINCT ON (e.item_id)
                e.item_id,
                e.vector <-> :uvec AS distance
            FROM item_embeddings e
            WHERE {where_clause}
            ORDER BY e.item_id, e.vector <-> :uvec
        )
        SELECT d.item_id
        FROM deduped d
        ORDER BY d.distance, d.item_id
        LIMIT :lim
    """.format(
            where_clause=" AND ".join(where_clauses)
        )
    ).bindparams(
        bindparam("uvec", type_=Vector(384))  # <-- important
    )

    rows = db.execute(
        q,
        params,
    ).fetchall()

    ordered: List[int] = []
    seen: set[int] = set()
    for row in rows:
        item_id = int(row[0])
        if item_id in seen:
            continue
        seen.add(item_id)
        ordered.append(item_id)

    return ordered
