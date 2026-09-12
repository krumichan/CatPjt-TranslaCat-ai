"""Policy port used by common orchestration without supplying a default policy."""
from __future__ import annotations

from typing import Generic, Protocol, TypeVar

from app.features.language_learning.difficulty.contracts import (
    DifficultyAcceptanceDecision,
    DifficultyTarget,
)

TargetT = TypeVar("TargetT")
AssessmentT = TypeVar("AssessmentT", contravariant=True)
ContextT = TypeVar("ContextT", contravariant=True)


class DifficultyAcceptancePolicy(Protocol, Generic[TargetT, AssessmentT, ContextT]):
    def decide(
        self,
        *,
        target: DifficultyTarget[TargetT],
        assessment: AssessmentT,
        context: ContextT,
        adjudicated: bool,
    ) -> DifficultyAcceptanceDecision: ...
