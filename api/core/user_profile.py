from __future__ import annotations
from typing import Optional
import numpy as np
from sqlalchemy.orm import Session
from api.db.models import User, UserHistory, ItemEmbedding

DIM = 384


def _time_decay_weights(n: int) -> np.ndarray:
    # recent -> 1.0, older half-life ~10 items
    if n == 0:
        return np.zeros((0,), dtype="float32")
    idx = np.arange(n, dtype="float32")  # most recent first
    w = 0.5 ** (idx / 10.0)
    w /= w.sum() + 1e-8
    return w


def compute_user_vector(
    db: Session, user_id: str
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    # pull recent history item_ids (watched/liked/rated), newest first
    rows = (
        db.query(UserHistory.item_id, UserHistory.weight)
        .filter(UserHistory.user_id == user_id)
        .order_by(UserHistory.ts.desc())
        .limit(200)
        .all()
    )
    if not rows:
        return None, None
    item_ids = [r[0] for r in rows]

    embs = (
        db.query(ItemEmbedding.vector).filter(ItemEmbedding.item_id.in_(item_ids)).all()
    )
    vecs = np.array([np.array(v[0], dtype="float32") for v in embs], dtype="float32")
    if vecs.size == 0:
        return None, None

    # short_vec: time-decayed mean; long_vec: uniform mean
    td = _time_decay_weights(len(vecs))
    short_vec = (vecs * td[:, None]).sum(axis=0)
    long_vec = vecs.mean(axis=0)

    # L2 normalize
    def norm(x):
        n = np.linalg.norm(x)
        return (x / n).astype("float32") if n > 0 else x.astype("float32")

    return norm(long_vec), norm(short_vec)


def upsert_user_vectors(db: Session, user_id: str) -> None:
    long_vec, short_vec = compute_user_vector(db, user_id)
    user = db.query(User).filter(User.user_id == user_id).one_or_none()
    if user is None:
        user = User(user_id=user_id, long_vec=long_vec, short_vec=short_vec)
        db.add(user)
    else:
        user.long_vec = long_vec
        user.short_vec = short_vec
