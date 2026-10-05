"""Novel 전용 자격증명으로 한정한 범용 allowlist 실행."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Annotated
from uuid import UUID
from weakref import WeakKeyDictionary

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from openai import APIConnectionError, APITimeoutError
from pydantic import ValidationError

from app.ai.providers.openai.profile_execution import ProfileExecutor
from app.ai.providers.openai.progressive_fields import ProgressiveFieldError
from app.ai.providers.openai.response import OpenAIProviderResponseError
from app.ai.providers.openai.schema import OpenAISchemaConfigurationError
from app.api.dependencies import get_model_execution_provider
from app.common.provider_retry import retry_after_details
from app.core.config import settings
from app.core.novel_auth import authorize_novel_api_key
from app.schemas.model_profile_execution import (
    ModelProfileExecutionRequest,
    ModelProfileExecutionResponse,
)

router = APIRouter(prefix="/model", tags=["Internal Profile Execution"])


class ExecutionGate:
    def __init__(self, concurrency: int, capacity: int):
        self.slots = asyncio.Semaphore(concurrency)
        self.capacity = concurrency + capacity
        self.admitted = 0


class CallerDisconnected(Exception):
    pass


_gates: WeakKeyDictionary = WeakKeyDictionary()


def _gate() -> ExecutionGate:
    loop = asyncio.get_running_loop()
    if loop not in _gates:
        _gates[loop] = ExecutionGate(
            settings.NOVEL_MODEL_PROFILE_MAX_CONCURRENCY, settings.NOVEL_MODEL_PROFILE_MAX_QUEUE
        )
    return _gates[loop]


def get_profile_executor() -> ProfileExecutor:
    return ProfileExecutor(get_model_execution_provider())


def fail(status: int, code: str, retryable: bool = False, **detail) -> HTTPException:
    return HTTPException(status, detail={"code": code, "retryable": retryable, **detail})


async def _execute_connected(
    request: Request,
    gate: ExecutionGate,
    command: ModelProfileExecutionRequest,
    executor: ProfileExecutor,
    started: float,
    parent_trace_id: str | None,
) -> dict:
    # 본문을 모두 읽은 뒤에는 이 감시자만 receive를 소비한다. 대기 중 종료된 요청은 제거한다.
    disconnected = asyncio.Event()
    queued = time.monotonic()

    async def watch_disconnect() -> None:
        while True:
            if (await request.receive())["type"] == "http.disconnect":
                disconnected.set()
                return

    async def execute() -> dict:
        async with gate.slots:
            # 슬롯 해제와 연결 종료가 겹쳐도 이미 도착한 disconnect를 먼저 관찰한다.
            await asyncio.sleep(0)
            if disconnected.is_set():
                raise CallerDisconnected
            queue_ms = (time.monotonic() - queued) * 1000
            remaining = command.remaining_milliseconds - int((time.monotonic() - started) * 1000)
            if remaining <= 0:
                raise TimeoutError
            bounded = command.model_copy(update={"remaining_milliseconds": remaining})
            result = await executor.execute(command=bounded, parent_trace_id=parent_trace_id)
            result["metadata"].update(queueMs=queue_ms, totalMs=(time.monotonic() - started) * 1000)
            return result

    watcher = asyncio.create_task(watch_disconnect())
    execution = asyncio.create_task(execute())
    try:
        done, _ = await asyncio.wait({watcher, execution}, return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            watcher.result()
            raise CallerDisconnected
        return execution.result()
    finally:
        # 이미 전송된 제공자 요청의 비용 취소를 보장하지 않는다. 로컬 작업과 슬롯만 정리한다.
        watcher.cancel()
        execution.cancel()
        await asyncio.gather(watcher, execution, return_exceptions=True)


@router.post("/execute-profile", response_model=ModelProfileExecutionResponse)
async def execute_profile(
    request: Request,
    executor: Annotated[ProfileExecutor, Depends(get_profile_executor)],
) -> ModelProfileExecutionResponse:
    started = time.monotonic()
    # 공통 key middleware 통과만으로는 이 별도 scope를 사용할 수 없다.
    if (
        authorize_novel_api_key(
            settings, request.method, request.url.path, request.headers.get("X-API-KEY")
        )
        is not True
    ):
        raise fail(401, "NOVEL_PROFILE_AUTH_REQUIRED")
    parent_trace_id = request.headers.get("X-Novel-Trace-Id")
    if parent_trace_id:
        try:
            parent_trace_id = str(UUID(parent_trace_id))
        except ValueError:
            raise fail(422, "EXECUTION_TRACE_INVALID") from None
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > 500_000:
            raise fail(413, "EXECUTION_REQUEST_TOO_LARGE")
        raw.extend(chunk)
    try:
        command = ModelProfileExecutionRequest.model_validate(json.loads(raw))
    except (ValidationError, ValueError, TypeError, UnicodeDecodeError, RecursionError):
        raise fail(422, "EXECUTION_REQUEST_INVALID") from None

    # 대기열은 유한하며 timeout/취소가 대기자를 제거한다. 기존 LL 슬롯과 독립한다.
    gate = _gate()
    if gate.admitted >= gate.capacity:
        raise fail(429, "EXECUTION_QUEUE_FULL", True, retryAfterSeconds=1)
    gate.admitted += 1
    try:
        remaining_seconds = command.remaining_milliseconds / 1000 - (time.monotonic() - started)
        if remaining_seconds <= 0:
            raise TimeoutError
        async with asyncio.timeout(remaining_seconds):
            result = await _execute_connected(
                request, gate, command, executor, started, parent_trace_id
            )
        return ModelProfileExecutionResponse.model_validate(result)
    except CallerDisconnected:
        raise fail(499, "EXECUTION_CALLER_DISCONNECTED") from None
    except OpenAISchemaConfigurationError:
        raise fail(422, "EXECUTION_SCHEMA_INVALID") from None
    except OpenAIProviderResponseError as error:
        raise fail(502, error.reason_code, error.retryable) from None
    except (TimeoutError, APITimeoutError, httpx.TimeoutException):
        raise fail(504, "PROVIDER_TIMEOUT", True) from None
    except (ConnectionError, APIConnectionError, httpx.TransportError):
        raise fail(503, "PROVIDER_UNAVAILABLE", True) from None
    except Exception as error:
        status = getattr(error, "status_code", None)
        if status in {408, 409, 429} or isinstance(status, int) and status >= 500:
            raise fail(
                503,
                "PROVIDER_UNAVAILABLE",
                True,
                providerStatus=status,
                **retry_after_details(error),
            ) from None
        raise fail(
            502,
            "PROVIDER_CONFIGURATION_ERROR"
            if isinstance(status, int) and 400 <= status < 500
            else "PROVIDER_EXECUTION_FAILED",
        ) from None
    finally:
        gate.admitted -= 1


@router.post("/execute-profile-stream")
async def execute_profile_stream(
    request: Request,
    executor: Annotated[ProfileExecutor, Depends(get_profile_executor)],
) -> StreamingResponse:
    """Novel 전용 SSE. 유료 작업은 POST 한 번만 시작하고 수신은 같은 연결에서 진행한다."""
    started = time.monotonic()
    if (
        authorize_novel_api_key(
            settings, request.method, request.url.path, request.headers.get("X-API-KEY")
        )
        is not True
    ):
        raise fail(401, "NOVEL_PROFILE_AUTH_REQUIRED")
    parent_trace_id = request.headers.get("X-Novel-Trace-Id")
    if parent_trace_id:
        try:
            parent_trace_id = str(UUID(parent_trace_id))
        except ValueError:
            raise fail(422, "EXECUTION_TRACE_INVALID") from None
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > 500_000:
            raise fail(413, "EXECUTION_REQUEST_TOO_LARGE")
        raw.extend(chunk)
    try:
        command = ModelProfileExecutionRequest.model_validate(json.loads(raw))
    except (ValidationError, ValueError, TypeError, UnicodeDecodeError, RecursionError):
        raise fail(422, "EXECUTION_REQUEST_INVALID") from None
    gate = _gate()
    if gate.admitted >= gate.capacity:
        raise fail(429, "EXECUTION_QUEUE_FULL", True, retryAfterSeconds=1)
    gate.admitted += 1

    async def events():
        queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=32)
        queued = time.monotonic()

        async def produce():
            try:
                async with gate.slots:
                    queue_wait_ms = (time.monotonic() - queued) * 1000
                    remaining = command.remaining_milliseconds - int(
                        (time.monotonic() - started) * 1000
                    )
                    if remaining <= 0:
                        raise TimeoutError
                    bounded = command.model_copy(update={"remaining_milliseconds": remaining})
                    async with asyncio.timeout(remaining / 1000):
                        async for item in executor.stream(
                            command=bounded, parent_trace_id=parent_trace_id
                        ):
                            if item["type"] == "result":
                                item.pop("_rawResponse", None)
                                item["result"]["metadata"].update(
                                    queueMs=queue_wait_ms,
                                    totalMs=(time.monotonic() - started) * 1000,
                                )
                                ModelProfileExecutionResponse.model_validate(item["result"])
                            await queue.put(item)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # SSE header는 이미 전송됐다. 본문·provider 예외는 노출하지 않는다.
                code = (
                    "PROVIDER_TIMEOUT"
                    if isinstance(error, (TimeoutError, APITimeoutError, httpx.TimeoutException))
                    else error.reason_code
                    if isinstance(error, OpenAIProviderResponseError)
                    else str(error)
                    if isinstance(error, ProgressiveFieldError)
                    else "PROGRESSIVE_EXECUTION_FAILED"
                )
                await queue.put({"type": "error", "code": code})
            finally:
                # 연결 해제 때 꽉 찬 큐에 종료 신호를 기다리며 producer 정리가 멈추지 않는다.
                task = asyncio.current_task()
                if task is None or not task.cancelling():
                    await queue.put(None)

        producer = asyncio.create_task(produce())
        try:
            yield ": connected\n\n"
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=10)
                except TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                if item is None:
                    break
                yield (
                    "data: " + json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n\n"
                )
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
            gate.admitted -= 1

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-store"}
    )
