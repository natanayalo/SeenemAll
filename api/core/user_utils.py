from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
from sqlalchemy.orm import Session
from api.db.models import User, UserHistory
from api.core.user_profile import NEGATIVE_EVENT_TYPES


def canonical_profile_id(user_id: str, profile_id: str | None) -> str:
    profile_id = (profile_id or "").strip()
    if not profile_id:
        return user_id
    return f"{user_id}::{profile_id}"


def load_user_state(
    db: Session, user_id: str
) -> Tuple[np.ndarray | None, np.ndarray | None, List[int], Dict[str, Any]]:
    user = db.query(User).filter(User.user_id == user_id).one_or_none()
    if not user or user.short_vec is None:
        return (
            None,
            None,
            [],
            {
                "genre_prefs": {},
                "neighbors": [],
                "negative_items": [],
            },
        )
    history_rows = (
        db.query(UserHistory.item_id, UserHistory.event_type)
        .filter(UserHistory.user_id == user_id)
        .all()
    )
    exclude: List[int] = []
    negative_items: List[int] = []
    for item_id, event_type in history_rows:
        exclude.append(item_id)
        normalized_event = (event_type or "").lower()
        if normalized_event in NEGATIVE_EVENT_TYPES:
            negative_items.append(item_id)

    # Deduplicate while preserving order
    exclude = list(dict.fromkeys(exclude))
    negative_items = sorted(set(negative_items))

    long_v = (
        np.array(user.long_vec, dtype="float32") if user.long_vec is not None else None
    )
    short_v = np.array(user.short_vec, dtype="float32")
    taste_clusters = getattr(user, "taste_clusters", None) or []
    if not taste_clusters and user.short_vec is not None:
        taste_clusters = [
            {
                "cluster_id": 0,
                "centroid": list(user.short_vec),
                "weight": 1.0,
                "size": len(exclude) if exclude else 1,
                "top_genres": list((user.genre_prefs or {}).keys())[:3],
            }
        ]

    profile_meta = {
        "genre_prefs": user.genre_prefs or {},
        "neighbors": user.neighbors or [],
        "negative_items": negative_items,
        "taste_clusters": taste_clusters,
    }
    return long_v, short_v, exclude, profile_meta
