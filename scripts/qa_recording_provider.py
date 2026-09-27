"""범용 QA 호출 예산과 상태 기록. 요청·응답 원문은 보관하지 않는다."""
from __future__ import annotations

import asyncio
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


class QaBudgetExceeded(RuntimeError):
    pass


class QaCallTimeout(TimeoutError):
    pass


class QaOverallTimeout(TimeoutError):
    pass


@dataclass(frozen=True)
class QaCaps:
    provider_calls: int
    input_tokens: int
    output_tokens: int
    call_seconds: float
    seconds: float


@dataclass
class RecordingProvider:
    upstream: Any
    caps: QaCaps
    diagnostic_capture: bool = False
    # 장기 실행 QA 서버의 요청 취소가 전체 검사 중단으로 바뀌지 않게 선택한다.
    stop_on_caller_cancel: bool = field(default=True, kw_only=True)
    started_at: float = field(default_factory=time.perf_counter)
    calls: list[dict[str, Any]] = field(default_factory=list)
    budget_exhausted_reason: str | None = None
    checkpoint: Callable[[], None] | None = None

    def _checkpoint(self) -> None:
        if self.checkpoint is not None:
            self.checkpoint()

    def _check_before_call(self) -> None:
        if self.budget_exhausted_reason is not None:
            raise QaBudgetExceeded(self.budget_exhausted_reason)
        if len(self.calls) >= self.caps.provider_calls:
            self.budget_exhausted_reason = "provider call cap reached"
            raise QaBudgetExceeded(self.budget_exhausted_reason)
        if (
            sum(item["inputTokens"] for item in self.calls) >= self.caps.input_tokens
            or sum(item["outputTokens"] for item in self.calls) >= self.caps.output_tokens
        ):
            self.budget_exhausted_reason = "token cap reached; no further starts"
            raise QaBudgetExceeded(self.budget_exhausted_reason)
        if time.perf_counter() - self.started_at >= self.caps.seconds:
            self.budget_exhausted_reason = "wall-clock cap reached"
            raise QaBudgetExceeded(self.budget_exhausted_reason)

    def _start_call(self, type_name: str, data: str) -> dict[str, Any]:
        self._check_before_call()
        record: dict[str, Any] = {
            "attempt": len(self.calls) + 1,
            "status": "STARTED",
            "task": type_name,
            "provider": None,
            "model": None,
            "latencyMs": 0.0,
            "inputTokens": 0,
            "outputTokens": 0,
            "providerInternalRetries": "NOT_EXPOSED_BY_PROVIDER_INTERFACE",
            "diagnostic": None,
        }
        if self.diagnostic_capture:
            record["diagnosticCapture"] = {"task": type_name}
        # 공급자 진입 전에 기록해 예외·취소도 호출 예산에서 누락하지 않는다.
        self.calls.append(record)
        self._checkpoint()
        return record

    def _finish_timing(self, record: dict[str, Any], started: float) -> None:
        record["latencyMs"] = round((time.perf_counter() - started) * 1000, 3)

    def _check_after_call(self, record: dict[str, Any]) -> None:
        reason = None
        if sum(item["inputTokens"] for item in self.calls) > self.caps.input_tokens:
            reason = "input token cap exceeded after the last completed call"
        elif sum(item["outputTokens"] for item in self.calls) > self.caps.output_tokens:
            reason = "output token cap exceeded after the last completed call"
        elif time.perf_counter() - self.started_at >= self.caps.seconds:
            reason = "wall-clock cap exceeded during the last completed call"
        if reason is None:
            return
        self.budget_exhausted_reason = reason
        record["budgetExceededAfterCompletion"] = reason
        raise QaBudgetExceeded(reason)

    async def call(
        self,
        type_name: str,
        data: str,
        schema: dict | None = None,
    ) -> Any:
        return (await self.call_with_metadata(type_name, data, schema)).data

    async def call_with_metadata(
        self,
        type_name: str,
        data: str,
        schema: dict | None = None,
    ) -> Any:
        # 예산을 확보한 뒤 공급자의 원래 메타데이터 응답을 그대로 전달한다.
        record = self._start_call(type_name, data)
        result = await self._invoke(
            record,
            lambda: self.upstream.call_with_metadata(
                    type_name=type_name,
                    data=data,
                    schema=schema,
            ),
        )
        # 응답 원문 없이 실행 결과와 사용량만 기록한다.
        record.update(
            {
                "status": "SUCCEEDED",
                "provider": str(getattr(result, "provider", "unknown")),
                "model": str(getattr(result, "model", "unknown")),
                "inputTokens": int(getattr(result, "input_tokens", 0) or 0),
                "outputTokens": int(getattr(result, "output_tokens", 0) or 0),
            }
        )
        try:
            self._check_after_call(record)
        finally:
            self._checkpoint()
        return result

    async def _invoke(
        self, record: dict[str, Any], operation: Callable[[], Awaitable[Any]]
    ) -> Any:
        started = time.perf_counter()
        remaining = max(0.0, self.caps.seconds - (started - self.started_at))
        if remaining <= 0:
            # 기록 I/O로 남은 시간이 소진되었으면 SDK에도 진입하지 않는다.
            record["status"] = "FAILED"
            record["failure"] = {
                "type": "QA_OVERALL_TIMEOUT", "source": "BEFORE_PROVIDER_ENTRY"
            }
            self.budget_exhausted_reason = "wall-clock cap reached"
            self._finish_timing(record, started)
            self._checkpoint()
            raise QaOverallTimeout(self.budget_exhausted_reason)
        overall_limited = remaining <= self.caps.call_seconds
        limit = min(remaining, self.caps.call_seconds)
        deadline = asyncio.timeout(limit)
        task = asyncio.current_task()
        initial_cancellations = task.cancelling() if task is not None else 0
        record["effectiveTimeoutSeconds"] = limit
        record["startedAfterMs"] = round((started - self.started_at) * 1000, 3)
        try:
            async with deadline:
                result = await operation()
            # 공급자가 취소를 삼켜도 늦은 결과를 성공으로 인정하지 않는다.
            if deadline.expired():
                raise TimeoutError("QA deadline expired inside provider")
            if task is not None and task.cancelling() > initial_cancellations:
                raise asyncio.CancelledError
            return result
        except TimeoutError as exc:
            record["status"] = "FAILED"
            if not deadline.expired():
                # 공급자 자체 timeout과 QA 제한 초과는 별도로 기록한다.
                record["failure"] = {
                    "type": type(exc).__name__, "source": "PROVIDER"
                }
                raise
            code = "QA_OVERALL_TIMEOUT" if overall_limited else "QA_CALL_TIMEOUT"
            record["failure"] = {"type": code, "timeoutSeconds": limit}
            self.budget_exhausted_reason = (
                "wall-clock cap reached" if overall_limited else
                f"provider call timed out after {self.caps.call_seconds:g} seconds"
            )
            error_type = QaOverallTimeout if overall_limited else QaCallTimeout
            raise error_type(self.budget_exhausted_reason) from exc
        except BaseException as exc:
            cancelled = isinstance(exc, asyncio.CancelledError)
            record["status"] = "CANCELLED" if cancelled else "FAILED"
            failure: dict[str, Any] = {"type": type(exc).__name__}
            if cancelled:
                task = asyncio.current_task()
                failure["source"] = "EXTERNAL_OR_UPSTREAM_UNDETERMINED"
                failure["taskCancellationCount"] = (
                    task.cancelling() if task is not None else None
                )
                if self.stop_on_caller_cancel:
                    self.budget_exhausted_reason = "execution cancelled; no further starts"
            record["failure"] = failure
            raise
        finally:
            self._finish_timing(record, started)
            if record["status"] != "STARTED":
                self._checkpoint()

    async def call_with_image(self, *args: Any, **kwargs: Any) -> Any:
        type_name = str(kwargs.get("type_name", args[0] if args else "IMAGE_CALL"))
        prompt = str(kwargs.get("prompt", args[1] if len(args) > 1 else ""))
        record = self._start_call(type_name, prompt)
        result = await self._invoke(
            record, lambda: self.upstream.call_with_image(*args, **kwargs)
        )
        record["status"] = "SUCCEEDED"
        try:
            self._check_after_call(record)
        finally:
            self._checkpoint()
        return result



def _call_summary(calls: list[dict[str, Any]]) -> dict[str, Any]:
    # 원문을 제외한 호출 상태·토큰·시간만 합쳐 기존 QA 보고서를 만든다.
    return {
        "count": len(calls),
        "attemptsStarted": len(calls),
        "successfulCalls": sum(item["status"] == "SUCCEEDED" for item in calls),
        "failedCalls": sum(item["status"] == "FAILED" for item in calls),
        "cancelledCalls": sum(item["status"] == "CANCELLED" for item in calls),
        "inputTokens": sum(item["inputTokens"] for item in calls),
        "outputTokens": sum(item["outputTokens"] for item in calls),
        "latencyMs": round(sum(item["latencyMs"] for item in calls), 3),
        "byTask": dict(Counter(item["task"] for item in calls)),
        "attemptLedger": calls,
        "providerInternalRetries": "NOT_EXPOSED_BY_PROVIDER_INTERFACE",
    }
