from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.core.user_profile import NEGATIVE_EVENT_TYPES, _event_weight
from api.db.models import UserHistory
from api.pipeline.hooks import get_hook
from api.pipeline.models import QueryUnderstanding, UserContext
from api.pipeline.retriever.base import BaseRetriever

logger = logging.getLogger("api.routes.recommend")

_COLLAB_HISTORY_LIMIT_MULTIPLIER = 4


def collaborative_candidates(
    db: Session,
    neighbors: List[Dict[str, Any]] | None,
    exclude_ids: List[int],
    limit: int,
    allowed_ids: List[int] | None,
) -> List[Tuple[int, float]]:
    if not neighbors:
        return []

    if allowed_ids is not None and len(allowed_ids) == 0:
        return []

    neighbor_weights: Dict[str, float] = {}
    for entry in neighbors:
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get("user_id")
        if not raw_id:
            continue
        uid = str(raw_id).strip()
        if not uid:
            continue
        weight = float(entry.get("weight", 0.0) or 0.0)
        if weight <= 0:
            continue
        neighbor_weights[uid] = weight

    if not neighbor_weights:
        return []

    exclude_set = set(exclude_ids or [])
    allowed_set = set(allowed_ids) if allowed_ids is not None else None
    history_limit = max(limit * _COLLAB_HISTORY_LIMIT_MULTIPLIER, 200)

    stmt = (
        select(
            UserHistory.item_id,
            UserHistory.user_id,
            UserHistory.weight,
            UserHistory.event_type,
            UserHistory.ts,
        )
        .where(UserHistory.user_id.in_(list(neighbor_weights)))
        .order_by(UserHistory.ts.desc())
        .limit(history_limit)
    )

    try:
        rows = db.execute(stmt).all()
    except SQLAlchemyError:
        logger.debug("Collaborative recall query failed.", exc_info=True)
        return []

    if not rows:
        return []

    aggregated: Dict[int, tuple[float, float]] = {}
    for item_id, uid, base_weight, event_type, ts in rows:
        if item_id is None or uid is None:
            continue
        uid_str = str(uid)
        neighbor_weight = neighbor_weights.get(uid_str)
        if neighbor_weight is None:
            continue
        if item_id in exclude_set:
            continue
        if allowed_set is not None and item_id not in allowed_set:
            continue
        normalized_event = (event_type or "").lower()
        if normalized_event in NEGATIVE_EVENT_TYPES:
            continue
        event_score = _event_weight(normalized_event, base_weight)
        if event_score <= 0:
            continue
        score = neighbor_weight * event_score
        ts_value = 0.0
        if ts is not None:
            ts_value = float(ts.timestamp()) if hasattr(ts, "timestamp") else 0.0
        if item_id in aggregated:
            current_score, current_ts = aggregated[item_id]
            aggregated[item_id] = (
                current_score + score,
                max(current_ts, ts_value),
            )
        else:
            aggregated[item_id] = (score, ts_value)

    if not aggregated:
        return []

    ranked = sorted(
        aggregated.items(),
        key=lambda kv: (-kv[1][0], -kv[1][1], kv[0]),
    )
    top = ranked[:limit]
    if not top:
        return []
    max_score = max(score for _, (score, _) in top)
    if max_score <= 0:
        max_score = 1.0
    scaled: List[Tuple[int, float]] = []
    for item_id, (score, _) in top:
        normalized = score / max_score if max_score > 0 else 0.0
        scaled.append((item_id, normalized))
    return scaled


class CollaborativeGraphRetriever(BaseRetriever):
    def retrieve(
        self,
        db: Session,
        context: UserContext,
        intent: QueryUnderstanding,
        allowlist: List[int] | None,
    ) -> List[Tuple[int, float]]:
        collab_fn = get_hook("_collaborative_candidates", collaborative_candidates)
        return collab_fn(
            db,
            context.profile_meta.get("neighbors"),
            list(context.exclude_set),
            intent.candidate_limit,
            allowed_ids=allowlist,
        )
