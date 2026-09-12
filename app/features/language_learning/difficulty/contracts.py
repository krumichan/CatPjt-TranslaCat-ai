"""Small service-neutral contracts for difficulty-aware generation."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Generic, Literal, Mapping, Protocol, TypeVar, runtime_checkable

TargetT = TypeVar("TargetT")


@dataclass(frozen=True)
class DifficultyTarget(Generic[TargetT]):
    """An immutable service-owned target on a named difficulty scale."""

    service: str
    scale: str
    value: TargetT
    policy_version: str

    def __post_init__(self) -> None:
        for name in ("service", "scale", "policy_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"DifficultyTarget.{name} must be non-blank")


@runtime_checkable
class DifficultySpec(Protocol[TargetT]):
    """A service-defined spec bound to a target without prescribing its fields."""

    @property
    def version(self) -> str: ...

    @property
    def target(self) -> DifficultyTarget[TargetT]: ...


@dataclass(frozen=True)
class DifficultyValidationResult:
    """Result contract only; services retain all deterministic validation rules."""

    passed: bool
    issues: tuple[str, ...] = ()
    measurements: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.passed and self.issues:
            raise ValueError("A passing validation cannot contain rejection issues")
        if not self.passed and not self.issues:
            raise ValueError("A rejected validation requires at least one issue")
        if any(not isinstance(issue, str) or not issue for issue in self.issues):
            raise ValueError("Validation issue codes must be non-blank strings")
        object.__setattr__(self, "measurements", MappingProxyType(dict(self.measurements)))

    @classmethod
    def accept(cls, *, measurements: Mapping[str, Any] | None = None) -> "DifficultyValidationResult":
        return cls(True, measurements=measurements or {})

    @classmethod
    def reject(
        cls, *issues: str, measurements: Mapping[str, Any] | None = None,
    ) -> "DifficultyValidationResult":
        return cls(False, tuple(issues), measurements or {})

    @property
    def primary_issue(self) -> str | None:
        return self.issues[0] if self.issues else None


@dataclass(frozen=True)
class SemanticAssessment(Generic[TargetT]):
    """Normalized assessment produced only after service-owned binding/validation."""

    status: Literal["PASS", "REJECT", "UNSURE"]
    difficulty_status: str
    observed_target: TargetT | None = None
    alternative_target: TargetT | None = None
    issue_codes: tuple[str, ...] = ()
    confidence: float | None = None
    difficulty_confidence: float | None = None
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.difficulty_status:
            raise ValueError("difficulty_status must be non-blank")
        if any(not isinstance(code, str) or not code for code in self.issue_codes):
            raise ValueError("Semantic issue codes must be opaque non-blank strings")
        for name in ("confidence", "difficulty_confidence"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be null or a finite number from 0 to 1")


@dataclass(frozen=True)
class DifficultyAcceptanceDecision:
    """Opaque decision whose action and reason are interpreted only by its service."""

    action: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.action, str) or not self.action:
            raise ValueError("Acceptance action must be a non-blank string")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("Acceptance reason must be a non-blank string")
