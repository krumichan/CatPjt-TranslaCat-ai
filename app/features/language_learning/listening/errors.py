from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.schemas.language_learning_listening import (
    ListeningError,
    ListeningErrorCode,
    ListeningStage,
)


@dataclass
class ListeningStageException(Exception):
    code: ListeningErrorCode
    stage: ListeningStage
    message: str
    retryable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message

    def to_schema(self) -> ListeningError:
        return ListeningError(
            code=self.code,
            failed_stage=self.stage,
            message=self.message,
            retryable=self.retryable,
            details=self.details,
        )
