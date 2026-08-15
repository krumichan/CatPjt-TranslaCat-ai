from __future__ import annotations

from dataclasses import dataclass

from app.schemas.language_learning_speaking import (
    SpeakingError,
    SpeakingErrorCode,
    SpeakingStage,
)


@dataclass
class SpeakingStageException(Exception):
    code: SpeakingErrorCode
    stage: SpeakingStage
    message: str
    retryable: bool = False

    def __str__(self) -> str:
        return self.message

    def to_schema(self) -> SpeakingError:
        return SpeakingError(
            code=self.code,
            stage=self.stage,
            message=self.message,
            retryable=self.retryable,
        )
