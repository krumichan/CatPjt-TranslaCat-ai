"""Authorized Novel quality campaign: sole persisted admission for real OpenAI calls.

Product code never imports this runner. Unknown usage keeps its full reservation.
The separate 2026-10-04 quality campaign does not reuse the earlier $10 ledger.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from scripts.qa_campaign_budget import (
    CampaignBudgetExceeded,
    CampaignLedger,
    Reservation,
    atomic_json,
)

ROOT = Path(
    r"C:\workspace\project-cat\.codex-workspace\verification\shared\novel-quality-first-20261004\ai"
)
LIMITS = {
    "starts": 5000,
    "usd": 50.0,
    "input_tokens": 250_000_000,
    "output_tokens": 100_000_000,
    "tts_characters": 1_000_000,
    "stt_seconds": 0,
    "wall_seconds": 14 * 86400,
}
PHASE_LIMITS = {"starts": 3500, "usd": 30.0}
PHASES = {"smoke", "authoring", "explore", "judge", "final", "e2e", "tts"}
FINAL_PHASES = {"final", "e2e", "tts"}
# Standard USD per million tokens, verified from official model pages 2026-10-04.
PRICES = {
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-sol": (2.00, 10.00),
}
# 평가 전용 가격이다. 제품 profile allowlist나 공통 모델 설정에는 등록하지 않는다.
EVALUATOR_PRICES = {"gpt-6-astra": (10.00, 50.00)}
SAFE_HEADERS = {
    "x-request-id",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
    "retry-after",
}


def preserve_response(ledger: CampaignLedger, attempt: int, response: Any) -> Path:
    # 합성 QA 응답만 보존한다. 요청 본문/인증 헤더/키는 포함하지 않고 decode 이전 원자료를 남긴다.
    content = response.model_dump(mode="json")
    safe = {
        key: content.get(key)
        for key in ("id", "model", "status", "output", "usage", "incomplete_details", "error")
    }
    path = ledger.path.parent / "provider-responses" / f"attempt-{attempt:06}.json"
    atomic_json(path, safe)
    return path


def text_reservation(
    model: str, inp: int, out: int, *, evaluator_only: bool = False
) -> Reservation:
    prices = {**PRICES, **EVALUATOR_PRICES} if evaluator_only else PRICES
    if model not in prices:
        raise CampaignBudgetExceeded("UNPRICED_MODEL")
    if not 0 <= inp <= 200_000 or not 0 <= out <= 32768:
        raise CampaignBudgetExceeded("TOKEN_RESERVATION_OUT_OF_BOUNDS")
    input_price, output_price = prices[model]
    # 캐시 쓰기 최대 요율과 추가 25% 여유를 반영한다. 실제 청구액이라고 주장하지 않는다.
    usd = (inp * input_price * 1.25 + out * output_price) * 1.25 / 1_000_000
    return Reservation(inp, out, usd=usd)


class QualityLedger(CampaignLedger):
    def __init__(
        self,
        path: Path = ROOT / "ledger.json",
        *,
        stage_budget_path: Path | None = None,
        stage_budget_sha256: str | None = None,
    ) -> None:
        if (stage_budget_path is None) != (stage_budget_sha256 is None):
            raise ValueError("Both stage admission path and hash are required")
        self.stage_budget_path = stage_budget_path
        self.stage_budget_sha256 = stage_budget_sha256
        super().__init__(
            path, "novel-quality-first-20261004", limits=LIMITS, stop_on_caller_cancel=False
        )
        with self.lock:
            state = self._read()
            if "explorationLimits" not in state:
                if state["calls"]:
                    raise ValueError("Cannot retrofit phase limits into active campaign")
                state["explorationLimits"] = PHASE_LIMITS
                state["accountingBoundary"] = (
                    "Conservative reservations, not provider billing. SDK retries=0."
                )
                atomic_json(self.path, state)
            if state["explorationLimits"] != PHASE_LIMITS:
                raise ValueError("Immutable exploration limits mismatch")
            if self.stage_budget_path is not None:
                self._stage_budget(state)

    def _stage_budget(self, state: dict[str, Any]) -> dict[str, Any]:
        # 불변 admission과 종료된 선행 장부를 대조한다. 파일 소실도 허용하지 않는다.
        try:
            raw = self.stage_budget_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != self.stage_budget_sha256:
                raise CampaignBudgetExceeded("STAGE_ADMISSION_HASH_MISMATCH")
            guard = json.loads(raw)
        except (OSError, ValueError, AttributeError) as exc:
            raise CampaignBudgetExceeded("STAGE_ADMISSION_UNREADABLE") from exc
        numeric = ("startUsd", "maxUsd", "protectedRemainingUsd")
        integers = ("startAttempts", "maxStarts")
        if (
            not isinstance(guard, dict)
            or guard.get("campaignId") != self.campaign_id
            or not isinstance(guard.get("stageId"), str)
            or not guard["stageId"]
            or any(
                type(guard.get(key)) not in (int, float)
                or not math.isfinite(guard[key])
                or guard[key] < 0
                for key in numeric
            )
            or any(type(guard.get(key)) is not int or guard[key] < 0 for key in integers)
            # 기존 단독 단계와 명시 승인된 복구+음성 결합 단계만 허용한다. 장부/상한 형식은 같다.
            or (guard.get("allowedPhases"), guard.get("allowedTasks")) not in (
                (["final"], ["NOVEL_PROFILE", "NOVEL_LEGACY"]),
                (["tts"], ["OPENAI_TTS"]),
                (["final", "tts"], ["NOVEL_PROFILE", "NOVEL_LEGACY", "OPENAI_TTS"]),
            )
            or guard["maxUsd"] > self.limits["usd"]
            or guard["maxStarts"] > self.limits["starts"]
        ):
            raise CampaignBudgetExceeded("STAGE_ADMISSION_INVALID")
        boundary = guard["startAttempts"]
        prior = state["calls"][:boundary]
        prior_hash = hashlib.sha256(
            json.dumps(prior, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            len(prior) != boundary
            or prior_hash != guard.get("priorCallsSha256")
            or any(call["status"] == "STARTED" for call in prior)
            or not math.isclose(
                self.totals({"calls": prior})["usd"], guard["startUsd"], abs_tol=1e-10
            )
        ):
            raise CampaignBudgetExceeded("STAGE_PRIOR_LEDGER_MISMATCH")
        for call in state["calls"][boundary:]:
            if (
                call["metadata"].get("stageBudgetId") != guard["stageId"]
                or call["metadata"].get("stageBudgetSha256") != self.stage_budget_sha256
                or call["task"] not in guard["allowedTasks"]
                or call["metadata"].get("phase") not in guard["allowedPhases"]
            ):
                raise CampaignBudgetExceeded("STAGE_FOREIGN_CALL")
        return guard

    def reserve(
        self, task: str, model: str, amount: Reservation, *, metadata: dict[str, Any] | None = None
    ) -> int:
        phase = (metadata or {}).get("phase")
        if phase not in PHASES:
            raise CampaignBudgetExceeded("EXPLICIT_PHASE_REQUIRED")
        # 같은 프로세스의 재진입 가능한 잠금을 유지한 채 공통 장부 admission까지 원자화한다.
        with self.lock:
            state = self._read()
            if self.stage_budget_path is not None:
                # 같은 잠금 안에서 새 예약과 미정산 실패를 합산한다.
                # 단계 차단으로 독립 후속 검증의 전역 장부를 중단하지 않는다.
                guard = self._stage_budget(state)
                if phase not in guard["allowedPhases"] or task not in guard["allowedTasks"]:
                    raise CampaignBudgetExceeded("STAGE_SCOPE_MISMATCH")
                if (task == "OPENAI_TTS") != (phase == "tts"):
                    raise CampaignBudgetExceeded("STAGE_SCOPE_MISMATCH")
                stage = self.totals({"calls": state["calls"][guard["startAttempts"] :]})
                if stage["starts"] + 1 > guard["maxStarts"]:
                    raise CampaignBudgetExceeded("STAGE_START_LIMIT")
                if stage["usd"] + amount.usd > guard["maxUsd"]:
                    raise CampaignBudgetExceeded("STAGE_USD_LIMIT")
                if (
                    self.totals(state)["usd"] + amount.usd + guard["protectedRemainingUsd"]
                    > self.limits["usd"]
                ):
                    raise CampaignBudgetExceeded("STAGE_OTHER_RESERVE_PROTECTED")
                metadata = {
                    **(metadata or {}),
                    "stageBudgetId": guard["stageId"],
                    "stageBudgetSha256": self.stage_budget_sha256,
                }
            if phase not in FINAL_PHASES:
                exploratory = {
                    "calls": [
                        call
                        for call in state["calls"]
                        if call["metadata"]["phase"] not in FINAL_PHASES
                    ]
                }
                used = self.totals(exploratory)
                if (
                    used["starts"] + 1 > PHASE_LIMITS["starts"]
                    or used["usd"] + amount.usd > PHASE_LIMITS["usd"]
                ):
                    raise CampaignBudgetExceeded("FINAL_RESERVE_PROTECTED")
            return super().reserve(task, model, amount, metadata=metadata)

    def annotate(self, attempt: int, evidence: dict[str, Any]) -> None:
        with self.lock:
            state = self._read()
            state["calls"][attempt - 1].setdefault("evidence", {}).update(evidence)
            atomic_json(self.path, state)


class ProviderSlotsFull(Exception):
    status_code = 429


@asynccontextmanager
async def provider_slot(root: Path = ROOT, maximum: int = 10, *, wait: bool = True):
    if not 1 <= maximum <= 10:
        raise ValueError("Concurrency must be 1..10")
    root.mkdir(parents=True, exist_ok=True)
    held = None
    try:
        while held is None:
            for number in range(maximum):
                lock = FileLock(str(root / f"provider-slot-{number}.lock"))
                try:
                    lock.acquire(timeout=0)
                    held = lock
                    break
                except Timeout:
                    continue
            if held is None:
                if not wait:
                    raise ProviderSlotsFull("GLOBAL_PROVIDER_SLOTS_FULL")
                await asyncio.sleep(0.02)
        yield
    finally:
        if held is not None:
            held.release()


class LiveResponses:
    def __init__(self, *, phase: str, concurrency: int = 2, ledger: QualityLedger | None = None):
        from app.ai.providers.openai.client import OpenAIService

        self.service = OpenAIService()
        self.ledger = ledger or QualityLedger()
        self.phase, self.concurrency = phase, concurrency
        self.active = 0
        self.maximum_active = 0

    async def close(self):
        await self.service.shutdown()

    async def create(
        self,
        *,
        task: str,
        model: str,
        effort: str,
        instructions: str,
        messages: list[dict[str, str]],
        max_output_tokens: int,
        schema: dict | None = None,
        schema_name: str = "novel_quality",
        trace_id: str | None = None,
        parent_trace_id: str | None = None,
        timeout: float = 240,
    ) -> dict[str, Any]:
        from app.ai.providers.openai.profile_execution import decode_profile_response
        from app.ai.providers.openai.schema import build_explicit_text_config

        payload = {"instructions": instructions, "messages": messages, "schema": schema}
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        # 최종 평가도 final 예약을 사용한다. 평가 전용 모델 가격을 일반 번역에 열지는 않는다.
        evaluator_pricing = self.phase == "judge" or (
            self.phase == "final" and task == "blind-independent-quality-judge"
        )
        reserve = text_reservation(
            model, len(encoded) + 4096, max_output_tokens, evaluator_only=evaluator_pricing
        )
        text_config = build_explicit_text_config(
            schema=schema,
            schema_name_value=schema_name if schema else None,
            strict=schema is not None,
            verbosity="low",
        )
        # Provider 입장 직전에 영속 예약한다. 호출 실패/취소도 이미 발송한 시도로 남는다.
        queued = time.monotonic()
        async with provider_slot(self.ledger.path.parent, self.concurrency):
            attempt = self.ledger.reserve(
                task,
                model,
                reserve,
                metadata={
                    "phase": self.phase,
                    "reasoningEffort": effort,
                    "serviceTier": "default",
                    "sdkMaxRetries": 0,
                    "traceId": parent_trace_id or trace_id,
                    "childTraceId": trace_id,
                    "requestSha256": hashlib.sha256(encoded).hexdigest(),
                },
            )
            started = time.monotonic()
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            active_at_entry = self.active
            try:
                raw = await self.service.client.with_options(
                    timeout=timeout
                ).responses.with_raw_response.create(
                    model=model,
                    instructions=instructions,
                    input=messages,
                    reasoning={"effort": effort},
                    max_output_tokens=max_output_tokens,
                    text=text_config,
                    store=False,
                    service_tier="default",
                    extra_headers=(
                        {
                            "X-Client-Request-Id": trace_id,
                            **({"X-Novel-Trace-Id": parent_trace_id} if parent_trace_id else {}),
                        }
                        if trace_id
                        else None
                    ),
                )
                response = raw.parse()
                if hasattr(response, "__await__"):
                    response = await response
                raw_path = preserve_response(self.ledger, attempt, response)
                usage = response.usage
                evidence = {
                    "providerRequestId": raw.headers.get("x-request-id"),
                    "headers": {
                        key: raw.headers[key] for key in SAFE_HEADERS if key in raw.headers
                    },
                    "actualModel": response.model,
                    "status": response.status,
                    "usage": usage.model_dump() if usage else None,
                    "queueMs": (started - queued) * 1000,
                    "providerMs": (time.monotonic() - started) * 1000,
                    "providerFirstTokenMs": None,
                    "activeAtProviderEntry": active_at_entry,
                    "rawResponseFile": str(raw_path),
                }
                self.ledger.annotate(attempt, evidence)
                # 실제 usage가 있을 때만 예약을 정산한다. 형식 실패의 usage도 보존한다.
                charge = (
                    text_reservation(
                        model,
                        usage.input_tokens,
                        usage.output_tokens,
                        evaluator_only=evaluator_pricing,
                    )
                    if usage
                    else None
                )
                output = decode_profile_response(response, structured=schema is not None)
                self.ledger.finish(
                    attempt,
                    status="COMPLETED",
                    elapsed=time.monotonic() - started,
                    accounted=charge,
                )
                return {"attempt": attempt, "output": output, **evidence}
            except BaseException as exc:
                headers = getattr(getattr(exc, "response", None), "headers", {})
                self.ledger.annotate(
                    attempt,
                    {
                        "errorType": type(exc).__name__,
                        "statusCode": getattr(exc, "status_code", None),
                        "headers": {key: headers[key] for key in SAFE_HEADERS if key in headers},
                    },
                )
                self.ledger.finish(
                    attempt,
                    status="CANCELLED" if isinstance(exc, asyncio.CancelledError) else "FAILED",
                    elapsed=time.monotonic() - started,
                    failure=type(exc).__name__,
                )
                raise
            finally:
                self.active -= 1


async def smoke():
    runner = LiveResponses(phase="smoke", concurrency=1)
    cases = [
        ("gpt-5.6-luna", "none"),
        ("gpt-5.6-luna", "low"),
        ("gpt-6-luna", "none"),
        ("gpt-6-luna", "low"),
        ("gpt-6-luna", "medium"),
        ("gpt-6.1-sol", "low"),
    ]
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    results = []
    try:
        for model, effort in cases:
            try:
                result = await runner.create(
                    task="model-support-smoke",
                    model=model,
                    effort=effort,
                    instructions=(
                        "Translate the supplied Japanese sentence into Korean. "
                        "Return the requested JSON."
                    ),
                    messages=[{"role": "user", "content": "店は休みではなかった。"}],
                    max_output_tokens=2048,
                    schema=schema,
                )
                results.append({"requestedModel": model, "effort": effort, **result})
            except Exception as exc:
                results.append(
                    {"requestedModel": model, "effort": effort, "error": type(exc).__name__}
                )
            atomic_json(ROOT / "model-support-smoke.json", results)
            print(
                json.dumps(
                    {
                        "model": model,
                        "effort": effort,
                        "status": "ERROR" if "error" in results[-1] else "OK",
                    }
                )
            )
    finally:
        await runner.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["smoke", "stats"])
    args = parser.parse_args()
    if args.command == "smoke":
        asyncio.run(smoke())
    else:
        print(json.dumps(QualityLedger().snapshot()["totalsIncludingReserved"]))
