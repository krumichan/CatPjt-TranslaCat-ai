"""Local-only production route harness with the quality campaign's sole paid boundary."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import secrets
import time
from contextvars import ContextVar
from pathlib import Path

from scripts.novel_quality_verification import (
    ROOT,
    QualityLedger,
    preserve_response,
    provider_slot,
    text_reservation,
)
from scripts.qa_campaign_budget import CampaignBudgetExceeded, Reservation, atomic_json

TRACE: ContextVar[str | None] = ContextVar("novel_quality_trace", default=None)
ACTIVE_PROFILE_ATTEMPT: ContextVar[int | None] = ContextVar("novel_profile_attempt", default=None)
ACTIVE_LEGACY_ATTEMPT: ContextVar[int | None] = ContextVar("novel_legacy_attempt", default=None)
LEGACY_CHILD_TRACE: ContextVar[dict | None] = ContextVar("novel_legacy_child_trace", default=None)


class LegacyChildTraceMiddleware:
    """기존 generic DTO 전달을 변경하지 않는 QA 전용 ASGI body 관측기."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/internal/v1/model/execute"
        ):
            return await self.app(scope, receive, send)

        # 본문을 별도로 소비하거나 재전송하지 않는다. 원래 receive 메시지를 그대로 전달한다.
        state = {"childTraceId": None}
        token = LEGACY_CHILD_TRACE.set(state)
        captured = bytearray()
        oversized = False

        async def observe_receive():
            nonlocal oversized
            message = await receive()
            if message["type"] == "http.request":
                chunk = message.get("body", b"")
                if len(captured) + len(chunk) > 500_000:
                    oversized = True
                    captured.clear()
                if not oversized:
                    captured.extend(chunk)
                if not message.get("more_body", False) and not oversized:
                    try:
                        parsed = json.loads(captured)
                        child = parsed.get("traceId") if isinstance(parsed, dict) else None
                        if isinstance(child, str) and re.fullmatch(
                            r"[A-Za-z0-9._:-]{1,100}", child
                        ):
                            state["childTraceId"] = child
                    except (ValueError, UnicodeDecodeError, RecursionError):
                        pass
                    finally:
                        captured.clear()
            return message

        try:
            await self.app(scope, observe_receive, send)
        finally:
            # request별 dict 참조로 middleware 자식 task와 값을 공유한다.
            # 다음 요청으로 이전 trace가 넘어가지 않도록 context를 복원한다.
            LEGACY_CHILD_TRACE.reset(token)


def response_recorder(ledger, decoder):
    # 로컬 QA 프로세스에서만 decoder 앞에 연결한다. 제품 실행 응답/오류 계약은 그대로 유지한다.
    def decode(response, *, structured):
        attempt = ACTIVE_PROFILE_ATTEMPT.get()
        if attempt is not None:
            path = preserve_response(ledger, attempt, response)
            ledger.annotate(attempt, {"rawResponseFile": str(path)})
        return decoder(response, structured=structured)

    return decode


def legacy_response_recorder(ledger, decoder):
    # 기존 client의 decode 결과를 바꾸지 않고 QA 프로세스에서 응답/usage만 관측한다.
    def decode(response, *, structured):
        attempt = ACTIVE_LEGACY_ATTEMPT.get()
        if attempt is not None:
            path = preserve_response(ledger, attempt, response)
            usage = getattr(response, "usage", None)
            inputs = getattr(usage, "input_tokens_details", None)
            outputs = getattr(usage, "output_tokens_details", None)
            ledger.annotate(
                attempt,
                {
                    "rawResponseFile": str(path),
                    "providerRequestId": getattr(response, "_request_id", None),
                    "providerResponseId": getattr(response, "id", None),
                    "actualModel": getattr(response, "model", None),
                    "inputTokens": getattr(usage, "input_tokens", None),
                    "outputTokens": getattr(usage, "output_tokens", None),
                    "cachedInputTokens": getattr(inputs, "cached_tokens", None),
                    "reasoningTokens": getattr(outputs, "reasoning_tokens", None),
                    "providerMs": None,
                    "queueMs": None,
                    "providerFirstTokenMs": None,
                    "measurementBoundary": (
                        "Legacy decoder observation; adapter interval is separate. "
                        "SDK-send/header and route queue are unobserved."
                    ),
                },
            )
        return decoder(response, structured=structured)

    return decode


class BudgetedProfileExecutor:
    def __init__(self, upstream, ledger, phase="explore"):
        self.upstream, self.ledger, self.phase = upstream, ledger, phase

    async def stream(self, *, command, parent_trace_id):
        """스트림도 첫 provider 진입 전에 기존 장부에 예약한다."""
        from app.ai.providers.openai.execution_profiles import EXECUTION_PROFILES

        profile = EXECUTION_PROFILES[command.profile_id]
        body = command.model_dump_json(by_alias=True)
        amount = text_reservation(
            profile.model, len(body.encode()) + 4096, command.max_output_tokens
        )
        async with provider_slot(self.ledger.path.parent):
            attempt = self.ledger.reserve(
                "NOVEL_PROFILE", profile.model, amount,
                metadata={"phase": self.phase, "traceId": parent_trace_id,
                          "childTraceId": command.trace_id, "sdkMaxRetries": 0,
                          "requestSha256": hashlib.sha256(body.encode()).hexdigest(),
                          "transport": "PROGRESSIVE_A"},
            )
            started = time.monotonic()
            ended = None
            try:
                async for item in self.upstream.stream(
                    command=command, parent_trace_id=parent_trace_id
                ):
                    if item["type"] != "result":
                        yield item
                        continue
                    raw = item.pop("_rawResponse", None)
                    result = item["result"]
                    if result["provider"] != "openai" or result["model"] != profile.model:
                        raise CampaignBudgetExceeded("UNEXPECTED_PROVIDER_IDENTITY")
                    if raw is not None:
                        path = preserve_response(self.ledger, attempt, raw)
                        self.ledger.annotate(attempt, {"rawResponseFile": str(path)})
                    self.ledger.annotate(attempt, {
                        "actualModel": result["model"], "inputTokens": result["inputTokens"],
                        "outputTokens": result["outputTokens"],
                        "cachedInputTokens": result["cachedInputTokens"],
                        "reasoningTokens": result["reasoningTokens"], **result["metadata"],
                    })
                    ended = time.monotonic()
                    self.ledger.finish(
                        attempt, status="COMPLETED", elapsed=ended - started,
                        accounted=text_reservation(
                            profile.model, result["inputTokens"], result["outputTokens"]
                        ),
                    )
                    yield item
            except BaseException as exc:
                if ended is None:
                    self.ledger.finish(
                        attempt,
                        status="CANCELLED" if isinstance(exc, asyncio.CancelledError) else "FAILED",
                        elapsed=time.monotonic() - started, failure=type(exc).__name__,
                    )
                raise
            finally:
                self.ledger.annotate(attempt, {
                    "adapterProcessId": os.getpid(),
                    "adapterStartMonotonicMs": started * 1000,
                    "adapterEndMonotonicMs": (ended or time.monotonic()) * 1000,
                })

    async def execute(self, *, command, parent_trace_id):
        from app.ai.providers.openai.execution_profiles import EXECUTION_PROFILES

        profile = EXECUTION_PROFILES[command.profile_id]
        body = command.model_dump_json(by_alias=True)
        amount = text_reservation(
            profile.model, len(body.encode()) + 4096, command.max_output_tokens
        )
        async with provider_slot(self.ledger.path.parent):
            attempt = self.ledger.reserve(
                "NOVEL_PROFILE",
                profile.model,
                amount,
                metadata={
                    "phase": self.phase,
                    "traceId": parent_trace_id,
                    "childTraceId": command.trace_id,
                    "profileId": command.profile_id,
                    "sdkMaxRetries": 0,
                    "requestSha256": hashlib.sha256(body.encode()).hexdigest(),
                },
            )
            started = time.monotonic()
            ended = None
            recording = ACTIVE_PROFILE_ATTEMPT.set(attempt)
            try:
                result = await self.upstream.execute(
                    command=command, parent_trace_id=parent_trace_id
                )
                ended = time.monotonic()
                if result["provider"] != "openai" or result["model"] != profile.model:
                    raise CampaignBudgetExceeded("UNEXPECTED_PROVIDER_IDENTITY")
                self.ledger.annotate(
                    attempt,
                    {
                        "actualModel": result["model"],
                        "inputTokens": result["inputTokens"],
                        "outputTokens": result["outputTokens"],
                        "cachedInputTokens": result["cachedInputTokens"],
                        "reasoningTokens": result["reasoningTokens"],
                        **result["metadata"],
                    },
                )
                self.ledger.finish(
                    attempt,
                    status="COMPLETED",
                    elapsed=time.monotonic() - started,
                    accounted=text_reservation(
                        profile.model, result["inputTokens"], result["outputTokens"]
                    ),
                )
                return result
            except BaseException as exc:
                self.ledger.finish(
                    attempt,
                    status="CANCELLED" if isinstance(exc, asyncio.CancelledError) else "FAILED",
                    elapsed=time.monotonic() - started,
                    failure=type(exc).__name__,
                )
                raise
            finally:
                ACTIVE_PROFILE_ATTEMPT.reset(recording)
                self.ledger.annotate(
                    attempt,
                    {
                        "adapterProcessId": os.getpid(),
                        "adapterStartMonotonicMs": started * 1000,
                        "adapterEndMonotonicMs": (ended or time.monotonic()) * 1000,
                    },
                )


class BudgetedLegacyExecutor:
    def __init__(self, upstream, ledger, phase="explore"):
        self.upstream, self.ledger, self.phase = upstream, ledger, phase

    async def execute_explicit(self, **kwargs):
        from app.ai.model_policy import get_model_name_for_tier

        model = get_model_name_for_tier(kwargs["tier"])
        body = json.dumps(kwargs, ensure_ascii=False, default=str)
        amount = text_reservation(model, len(body.encode()) + 4096, kwargs["max_output_tokens"])
        async with provider_slot(self.ledger.path.parent):
            attempt = self.ledger.reserve(
                "NOVEL_LEGACY",
                model,
                amount,
                metadata={
                    "phase": self.phase,
                    "traceId": TRACE.get(),
                    "childTraceId": (LEGACY_CHILD_TRACE.get() or {}).get("childTraceId"),
                    "sdkMaxRetries": 0,
                    "requestSha256": hashlib.sha256(body.encode()).hexdigest(),
                },
            )
            started = time.monotonic()
            ended = None
            recording = ACTIVE_LEGACY_ATTEMPT.set(attempt)
            try:
                result = await self.upstream.execute_explicit(**kwargs)
                ended = time.monotonic()
                self.ledger.finish(
                    attempt,
                    status="COMPLETED",
                    elapsed=time.monotonic() - started,
                    accounted=text_reservation(model, result.input_tokens, result.output_tokens),
                )
                return result
            except BaseException as exc:
                self.ledger.finish(
                    attempt,
                    status="CANCELLED" if isinstance(exc, asyncio.CancelledError) else "FAILED",
                    elapsed=time.monotonic() - started,
                    failure=type(exc).__name__,
                )
                raise
            finally:
                ACTIVE_LEGACY_ATTEMPT.reset(recording)
                self.ledger.annotate(
                    attempt,
                    {
                        "adapterProcessId": os.getpid(),
                        "adapterStartMonotonicMs": started * 1000,
                        "adapterEndMonotonicMs": (ended or time.monotonic()) * 1000,
                    },
                )


class BudgetedSpeech:
    def __init__(self, upstream, ledger):
        self.upstream, self.ledger = upstream, ledger

    async def synthesize_speech(self, **kwargs):
        from app.ai.providers.openai.speech import OPENAI_SPEECH_INSTRUCTIONS
        from app.core.config import settings

        text = kwargs["text"]
        if settings.OPENAI_SPEECH_MODEL != "gpt-4o-mini-tts-2025-12-15":
            raise CampaignBudgetExceeded("UNPRICED_SPEECH_MODEL")
        if not text.strip() or len((text + OPENAI_SPEECH_INSTRUCTIONS).encode()) > 2000:
            raise CampaignBudgetExceeded("SPEECH_INPUT_OUT_OF_BOUNDS")
        # Binary Speech에는 사용량 영수증이 없어 성공 후에도 보수적 예약을 유지한다.
        amount = Reservation(
            input_tokens=2000, tts_characters=len(text), usd=max(0.5, len(text) * 0.001)
        )
        # 음성은 대기시키지 않는다. 브라우저가 중지한 뒤 슬롯 해제로 늦게 과금되는 것을 막는다.
        async with provider_slot(self.ledger.path.parent, wait=False):
            attempt = self.ledger.reserve(
                "OPENAI_TTS",
                settings.OPENAI_SPEECH_MODEL,
                amount,
                metadata={
                    "phase": "tts",
                    "traceId": TRACE.get(),
                    "sdkMaxRetries": 0,
                    "voice": kwargs["voice"],
                    "language": kwargs["language"],
                    "characters": len(text),
                    "requestSha256": hashlib.sha256(text.encode()).hexdigest(),
                    "pricingBasis": (
                        "$0.60/M input and $12/M audio output; unknown usage held at >=$0.50/call"
                    ),
                },
            )
            started = time.monotonic()
            try:
                result = await self.upstream.synthesize_speech(**kwargs)
                self.ledger.annotate(
                    attempt,
                    {
                        "durationSeconds": result.duration_seconds,
                        "audioSha256": hashlib.sha256(result.audio_bytes).hexdigest(),
                    },
                )
                self.ledger.finish(attempt, status="COMPLETED", elapsed=time.monotonic() - started)
                return result
            except BaseException as exc:
                self.ledger.finish(
                    attempt,
                    status="CANCELLED" if isinstance(exc, asyncio.CancelledError) else "FAILED",
                    elapsed=time.monotonic() - started,
                    failure=type(exc).__name__,
                )
                raise


def serve(port: int, phase: str, stage_budget: str | None = None, stage_sha256: str | None = None,
          ledger_root: str | None = None, host: str = "127.0.0.1"):
    import uvicorn
    from fastapi.responses import JSONResponse
    from pydantic import SecretStr

    from app.ai.providers.openai import client as legacy_client_module
    from app.ai.providers.openai import profile_execution
    from app.ai.providers.openai.client import OpenAIService
    from app.ai.providers.openai.profile_execution import ProfileExecutor
    from app.ai.providers.openai.speech import OpenAISpeechService
    from app.api.dependencies import get_model_execution_provider, get_speech_execution_provider
    from app.api.internal.model_profile_execution import get_profile_executor
    from app.core.config import settings
    from app.main import app

    # 이 로컬 프로세스에만 별도 키를 적용한다. env/운영 설정 파일은 수정하지 않는다.
    runtime_root = Path(ledger_root) if ledger_root is not None else ROOT
    runtime_root.mkdir(parents=True, exist_ok=True)
    key_path = runtime_root / "novel-ai-local.key"
    if not key_path.exists():
        key_path.write_text(secrets.token_urlsafe(48), encoding="utf-8")
    settings.NOVEL_AUTH_ENVIRONMENT = "Development"
    settings.NOVEL_SERVER_API_KEY = SecretStr(key_path.read_text(encoding="utf-8"))
    settings.NOVEL_MODEL_PROFILE_MAX_CONCURRENCY = 10
    settings.AI_VOICE_WARM_UP = False
    settings.OCR_WARM_UP = False
    # 브라우저 검증 단계의 고정 예약 상한은 모든 로컬 제공자 wrapper가 같은 장부에서 강제한다.
    ledger = QualityLedger(
        path=runtime_root / "ledger.json",
        stage_budget_path=Path(stage_budget) if stage_budget is not None else None,
        stage_budget_sha256=stage_sha256,
    )
    profile_execution.decode_profile_response = response_recorder(
        ledger, profile_execution.decode_profile_response
    )
    legacy_client_module.decode_response = legacy_response_recorder(
        ledger, legacy_client_module.decode_response
    )
    text_service = OpenAIService()
    profile = BudgetedProfileExecutor(ProfileExecutor(text_service), ledger, phase)
    legacy = BudgetedLegacyExecutor(text_service, ledger, phase)
    speech = BudgetedSpeech(OpenAISpeechService(), ledger)
    app.dependency_overrides[get_profile_executor] = lambda: profile
    app.dependency_overrides[get_model_execution_provider] = lambda: legacy
    app.dependency_overrides[get_speech_execution_provider] = lambda: speech
    app.add_middleware(LegacyChildTraceMiddleware)

    @app.middleware("http")
    async def restrict(request, call_next):
        if request.method == "GET" and request.url.path == "/qa/stats":
            state = ledger.snapshot()
            return JSONResponse(
                {
                    "totals": state["totalsIncludingReserved"],
                    "calls": [
                        {
                            "attempt": c["attempt"],
                            "task": c["task"],
                            "status": c["status"],
                            "traceId": c["metadata"].get("traceId"),
                            "model": c["model"],
                        }
                        for c in state["calls"]
                    ],
                }
            )
        if (request.method, request.url.path) not in {
            ("POST", "/internal/v1/model/execute"),
            ("POST", "/internal/v1/model/execute-profile"),
            ("POST", "/internal/v1/model/execute-profile-stream"),
            ("POST", "/internal/v1/speech/synthesize"),
            ("GET", "/"),
        }:
            return JSONResponse(status_code=404, content={"code": "QA_ROUTE_DISABLED"})
        token = TRACE.set(request.headers.get("X-Novel-Trace-Id"))
        try:
            return await call_next(request)
        finally:
            TRACE.reset(token)

    atomic_json(
        runtime_root / "live-server.json",
        {
            "pid": os.getpid(),
            "port": port,
            "phase": phase,
            "keyFile": str(key_path),
            "ledger": str(ledger.path),
            "stageBudgetPath": stage_budget,
            "stageBudgetSha256": stage_sha256,
        },
    )
    # 컨테이너 검증 때만 전체 인터페이스를 명시한다. 호스트 publish는 loopback으로 제한한다.
    uvicorn.run(app, host=host, port=port, access_log=False, log_level="warning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=19781)
    parser.add_argument("--phase", choices=["explore", "e2e", "final"], default="explore")
    parser.add_argument("--stage-budget")
    parser.add_argument("--stage-sha256")
    parser.add_argument("--ledger-root")
    parser.add_argument("--host", choices=["127.0.0.1", "0.0.0.0"], default="127.0.0.1")
    args = parser.parse_args()
    serve(args.port, args.phase, args.stage_budget, args.stage_sha256, args.ledger_root, args.host)
