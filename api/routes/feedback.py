from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.db.models import Feedback
from api.db.session import get_db

router = APIRouter(tags=["feedback"])


class FeedbackEvent(BaseModel):
    user_id: Optional[str] = None
    item_id: Optional[int] = None
    event_type: str
    meta: Optional[dict] = None


@router.post("/feedback")
def post_feedback(
    event: FeedbackEvent,
    db: Session = Depends(get_db),
):
    """Record a feedback event (e.g., impression, click)."""
    db_feedback = Feedback(
        user_id=event.user_id,
        item_id=event.item_id,
        type=event.event_type,
        meta=event.meta,
    )
    db.add(db_feedback)
    db.commit()
    return {"ok": True}
