"""Request-local deadline accounting with no service-specific defaults."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class RequestBudget:
    deadline: float | None
    clock: Callable[[], float] = field(default=time.monotonic, compare=False, repr=False)

    def remaining_seconds(self) -> float | None:
        return None if self.deadline is None else max(0.0, self.deadline - self.clock())

    def remaining_ms(self) -> float | None:
        remaining = self.remaining_seconds()
        return None if remaining is None else round(remaining * 1000, 2)
