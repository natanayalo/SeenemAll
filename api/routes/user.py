from __future__ import annotations
from typing import List
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from api.db.session import get_db
from api.db.models import UserHistory
from api.core.user_profile import upsert_user_vectors

router = APIRouter(prefix="/user", tags=["user"])


class HistoryIn(BaseModel):
    user_id: str
    items: List[int]  # DB Item.id (not tmdb_id)
    event_type: str = "watched"
    weight: int = 1


@router.post("/history")
def post_history(payload: HistoryIn, db: Session = Depends(get_db)):
    for iid in payload.items:
        db.add(
            UserHistory(
                user_id=payload.user_id,
                item_id=iid,
                event_type=payload.event_type,
                weight=payload.weight,
            )
        )
    db.commit()
    upsert_user_vectors(db, payload.user_id)
    db.commit()
    return {"ok": True}
