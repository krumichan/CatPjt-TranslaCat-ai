from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


class AiProviderPoolUnavailableError(RuntimeError):
    status_code = 503


@dataclass
class _ProviderSlot:
    name: str
    provider: Any
    max_concurrency: int
    cooldown_seconds: float
    active_calls: int = 0
    cooldown_until: float = 0.0
    semaphore: asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("Provider max_concurrency must be positive")
        self.semaphore = asyncio.Semaphore(self.max_concurrency)

    @property
    def ready(self) -> bool:
        return bool(getattr(self.provider, "ready", True))

    @property
    def cooling_down(self) -> bool:
        return time.monotonic() < self.cooldown_until

    @property
    def load(self) -> float:
        return self.active_calls / self.max_concurrency

    def activate_cooldown(self, retry_after_seconds: float | None = None) -> None:
        duration = self.cooldown_seconds
        if retry_after_seconds is not None:
            duration = max(duration, min(max(0.0, retry_after_seconds), 300.0))
        if duration <= 0:
            return
        self.cooldown_until = max(
            self.cooldown_until,
            time.monotonic() + duration,
        )


class AiProviderPool:
    """Capability-preserving AI provider pool.

    Feature services continue to depend on the existing provider protocols. The
    pool chooses the least-loaded ready provider that implements the requested
    method. Transient provider failures cool only that slot and may fall through
    to another docked provider. Domain/schema validation still happens above this
    layer and therefore never triggers hidden cross-provider retries.
    """

    provider_name = "pool"

    def __init__(self) -> None:
        self._slots: list[_ProviderSlot] = []

    def dock(
        self,
        *,
        name: str,
        provider: Any,
        max_concurrency: int,
        cooldown_seconds: float,
    ) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("Provider name is required")
        if any(slot.name == normalized for slot in self._slots):
            raise ValueError(f"Provider already docked: {normalized}")
        self._slots.append(
            _ProviderSlot(
                name=normalized,
                provider=provider,
                max_concurrency=max_concurrency,
                cooldown_seconds=cooldown_seconds,
            )
        )

    @property
    def ready(self) -> bool:
        return any(slot.ready for slot in self._slots)

    async def warm_up(self) -> None:
        for slot in self._slots:
            warm_up = getattr(slot.provider, "warm_up", None)
            if warm_up is not None:
                await warm_up()

    async def shutdown(self) -> None:
        for slot in self._slots:
            shutdown = getattr(slot.provider, "shutdown", None)
            if shutdown is not None:
                await shutdown()

    def model_name_for(self, type_name: str) -> str:
        slot = self._first_capable_slot("model_name_for")
        if slot is None:
            return "unknown"
        return str(slot.provider.model_name_for(type_name))

    def provider_name_for(self, type_name: str | None = None) -> str:
        slot = self._first_capable_slot("model_name_for") or self._first_capable_slot("call")
        return slot.name if slot is not None else "unknown"

    def supports(self, method_name: str) -> bool:
        """Capability check without making a speculative provider request."""
        return any(callable(getattr(slot.provider, method_name, None)) for slot in self._slots)

    async def call(
        self,
        type_name: str,
        data: str,
        schema: dict | None = None,
    ) -> Any:
        return await self._invoke(
            "call",
            type_name=type_name,
            data=data,
            schema=schema,
        )

    async def call_with_metadata(
        self,
        type_name: str,
        data: str,
        schema: dict | None = None,
    ) -> Any:
        return await self._invoke(
            "call_with_metadata",
            type_name=type_name,
            data=data,
            schema=schema,
        )

    async def call_with_image(
        self,
        type_name: str,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
        schema: dict | None = None,
    ) -> Any:
        return await self._invoke(
            "call_with_image",
            type_name=type_name,
            prompt=prompt,
            image_bytes=image_bytes,
            mime_type=mime_type,
            schema=schema,
        )

    async def translate_chat_message(
        self,
        text: str,
        target_language_code: str,
        source_language_code: str | None = None,
    ) -> str:
        return await self._invoke(
            "translate_chat_message",
            text=text,
            target_language_code=target_language_code,
            source_language_code=source_language_code,
        )

    async def translate_voice_utterance(
        self,
        *,
        source_text: str,
        source_language: str,
        target_language: str,
    ) -> Any:
        return await self._invoke(
            "translate_voice_utterance",
            source_text=source_text,
            source_language=source_language,
            target_language=target_language,
        )

    async def _invoke(self, method_name: str, **kwargs: Any) -> Any:
        candidates = self._candidate_slots(method_name)
        if not candidates:
            cooldown_wait = self._cooldown_wait_seconds(method_name)
            if cooldown_wait is not None and cooldown_wait > 0:
                # A short provider cooldown should queue work rather than making every
                # application-level retry fail immediately with 503. The caller's
                # overall timeout still bounds this wait.
                await asyncio.sleep(cooldown_wait)
                candidates = self._candidate_slots(method_name)

        if not candidates:
            raise AiProviderPoolUnavailableError(
                f"No ready AI provider supports {method_name}"
            )

        last_transient_error: Exception | None = None
        for slot in candidates:
            async with slot.semaphore:
                if not slot.ready or slot.cooling_down:
                    continue
                slot.active_calls += 1
                try:
                    method: Callable[..., Awaitable[Any]] = getattr(
                        slot.provider,
                        method_name,
                    )
                    return await method(**kwargs)
                except Exception as exc:
                    if not self._is_transient_provider_error(exc):
                        raise
                    slot.activate_cooldown(self._retry_after_seconds(exc))
                    last_transient_error = exc
                finally:
                    slot.active_calls = max(0, slot.active_calls - 1)

        if last_transient_error is not None:
            raise last_transient_error
        raise AiProviderPoolUnavailableError(
            f"All AI providers are busy or cooling down for {method_name}"
        )

    def _candidate_slots(self, method_name: str) -> list[_ProviderSlot]:
        candidates = [
            slot
            for slot in self._slots
            if slot.ready
            and not slot.cooling_down
            and callable(getattr(slot.provider, method_name, None))
        ]
        # Stable sort keeps docking order as the tie breaker while preferring the
        # provider with the most remaining capacity.
        return sorted(candidates, key=lambda slot: slot.load)

    def _first_capable_slot(self, method_name: str) -> _ProviderSlot | None:
        for slot in self._slots:
            if callable(getattr(slot.provider, method_name, None)):
                return slot
        return None

    def _cooldown_wait_seconds(self, method_name: str) -> float | None:
        now = time.monotonic()
        waits = [
            max(0.0, slot.cooldown_until - now)
            for slot in self._slots
            if slot.ready
            and callable(getattr(slot.provider, method_name, None))
            and slot.cooldown_until > now
        ]
        return min(waits) if waits else None

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            headers = getattr(exc, "headers", None)
        if headers is None:
            return None

        try:
            retry_after_ms = headers.get("retry-after-ms")
            if retry_after_ms is not None:
                value = float(str(retry_after_ms).strip()) / 1000.0
                return value if value > 0 else None
            retry_after = headers.get("retry-after")
            if retry_after is not None:
                value = float(str(retry_after).strip())
                return value if value > 0 else None
        except (TypeError, ValueError, AttributeError):
            return None
        return None

    @staticmethod
    def _is_transient_provider_error(exc: Exception) -> bool:
        # A provider may explicitly mark refusal/token-limit output as nonretryable.
        # Do not silently retry that operation on another docked provider.
        if getattr(exc, "retryable", None) is False:
            return False
        status_code = getattr(exc, "status_code", None)
        if status_code in {408, 409, 429, 500, 502, 503, 504}:
            return True
        name = type(exc).__name__.lower()
        return "timeout" in name or "connection" in name
