from __future__ import annotations
from typing import List, Tuple
import numpy as np
from sqlalchemy.orm import Session
from api.db.models import User, UserHistory


def load_user_state(
    db: Session, user_id: str
) -> Tuple[np.ndarray | None, np.ndarray | None, List[int]]:
    user = db.query(User).filter(User.user_id == user_id).one_or_none()
    if not user or user.short_vec is None:
        return None, None, []
    exclude = [
        r[0]
        for r in db.query(UserHistory.item_id)
        .filter(UserHistory.user_id == user_id)
        .all()
    ]
    long_v = (
        np.array(user.long_vec, dtype="float32") if user.long_vec is not None else None
    )
    short_v = np.array(user.short_vec, dtype="float32")
    return long_v, short_v, exclude
