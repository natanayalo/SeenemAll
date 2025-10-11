from __future__ import annotations
from typing import List
from sqlalchemy import text, bindparam
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector
import numpy as np


def ann_candidates(
    db: Session, user_vec: np.ndarray, exclude_ids: List[int], limit: int = 300
) -> List[int]:
    """
    Returns item_ids ordered by cosine distance to user_vec.
    Bind :uvec as pgvector, not numeric[].
    """
    if user_vec is None:
        return []

    q = text(
        """
        SELECT e.item_id
        FROM item_embeddings e
        WHERE NOT (e.item_id = ANY(:exclude))
        ORDER BY e.vector <-> :uvec
        LIMIT :lim
    """
    ).bindparams(
        bindparam("uvec", type_=Vector(384))  # <-- important
    )

    rows = db.execute(
        q,
        {
            "exclude": exclude_ids or [],
            # list/np.array are fine; SQLAlchemy + pgvector will adapt this to a vector
            "uvec": list(map(float, user_vec)),
            "lim": limit,
        },
    ).fetchall()

    return [r[0] for r in rows]
