from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from sqlalchemy.orm import Session

from api.pipeline.models import QueryUnderstanding, UserContext


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        db: Session,
        context: UserContext,
        intent: QueryUnderstanding,
        allowlist: List[int] | None,
    ) -> Any:
        pass
