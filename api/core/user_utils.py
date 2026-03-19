from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
from sqlalchemy.orm import Session
from api.db.models import Item, User, UserHistory
from api.core.user_profile import NEGATIVE_EVENT_TYPES


def canonical_profile_id(user_id: str, profile_id: str | None) -> str:
    profile_id = (profile_id or "").strip()
    if not profile_id:
        return user_id
    return f"{user_id}::{profile_id}"


def _preferred_media_types_for_history(
    db: Session,
    history_rows: List[tuple[int, str]],
) -> List[str]:
    positive_item_ids = [
        int(item_id)
        for item_id, event_type in history_rows
        if (event_type or "").lower() not in NEGATIVE_EVENT_TYPES
    ]
    if not positive_item_ids:
        return []

    media_type_rows = (
        db.query(Item.id, Item.media_type).filter(Item.id.in_(positive_item_ids)).all()
    )
    media_type_by_id = {
        int(item_id): str(media_type).strip().lower()
        for item_id, media_type in media_type_rows
        if media_type in {"movie", "tv"}
    }
    if not media_type_by_id:
        return []

    scores: Dict[str, float] = {}
    for index, item_id in enumerate(positive_item_ids):
        media_type = media_type_by_id.get(item_id)
        if media_type is None:
            continue
        scores[media_type] = scores.get(media_type, 0.0) + (1.0 / (1.0 + index))

    if not scores:
        return []
    return [
        media_type
        for media_type, _ in sorted(scores.items(), key=lambda entry: (-entry[1], entry[0]))
    ]


def load_user_state(
    db: Session, user_id: str
) -> Tuple[np.ndarray | None, np.ndarray | None, List[int], Dict[str, Any]]:
    user = db.query(User).filter(User.user_id == user_id).one_or_none()
    effective_user_id = user_id
    if (not user or user.short_vec is None) and "::" in user_id:
        base_user_id, _, _ = user_id.partition("::")
        fallback_user = db.query(User).filter(User.user_id == base_user_id).one_or_none()
        if fallback_user is not None and fallback_user.short_vec is not None:
            user = fallback_user
            effective_user_id = base_user_id
    if not user or user.short_vec is None:
        return (
            None,
            None,
            [],
            {
                "genre_prefs": {},
                "neighbors": [],
                "negative_items": [],
                "preferred_media_types": [],
            },
        )
    history_rows = (
        db.query(UserHistory.item_id, UserHistory.event_type)
        .filter(UserHistory.user_id == effective_user_id)
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
    preferred_media_types = _preferred_media_types_for_history(db, history_rows)

    long_v = (
        np.array(user.long_vec, dtype="float32") if user.long_vec is not None else None
    )
    short_v = np.array(user.short_vec, dtype="float32")
    profile_meta = {
        "genre_prefs": user.genre_prefs or {},
        "neighbors": user.neighbors or [],
        "negative_items": negative_items,
        "preferred_media_types": preferred_media_types,
    }
    return long_v, short_v, exclude, profile_meta
