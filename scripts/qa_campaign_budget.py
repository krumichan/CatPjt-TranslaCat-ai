"""Private, persistent admission ledger for explicitly authorized live QA only.

No production imports this module. Pending/unknown usage retains its entire
reservation across restarts. Dollar accounting is a conservative estimate, not
an account-level billing limit. Never put prompts, credentials or audio in it.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from filelock import FileLock


class CampaignBudgetExceeded(RuntimeError):
    pass


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        # A Windows reader/scanner may briefly hold the existing checkpoint.
        # Retry only that access race, retaining the previous complete JSON.
        for attempt in range(4):
            try:
                os.replace(name, path)
                break
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(0.02 * (attempt + 1))
    finally:
        if os.path.exists(name):
            os.unlink(name)


@dataclass(frozen=True)
class Reservation:
    input_tokens: int = 0
    output_tokens: int = 0
    tts_characters: int = 0
    stt_seconds: float = 0
    usd: float = 0


LIMITS = {"starts": 600, "input_tokens": 2_000_000, "output_tokens": 500_000,
          "tts_characters": 60_000, "stt_seconds": 1800, "usd": 20.0,
          "wall_seconds": 21600}


def _text_cost_reservation(model: str, input_tokens: int, output_tokens: int) -> float:
    """Conservative QA exposure in USD, including an explicit Sol margin.

    Existing small-model receipts retain their historical $1/$4 reservation.
    The Sol $5/$25 guard exceeds the published $4/$20 standard price; this
    deliberately does not claim an exact invoice or a provider-side spend cap.
    """
    if model == "gpt-5.6-sol":
        return (5 * input_tokens + 25 * output_tokens) / 1_000_000
    if model in {"gpt-5.6-luna", "gpt-5-mini", "gpt-5-nano"}:
        return (input_tokens + 4 * output_tokens) / 1_000_000
    raise CampaignBudgetExceeded("UNPRICED_MODEL")


class CampaignLedger:
    def __init__(self, path: Path, campaign_id: str, *, stop_on_caller_cancel: bool = True,
                 limits: dict[str, float] | None = None,
                 prior_ledger_sha256: str | None = None) -> None:
        self.path = path.resolve()
        self.campaign_id = campaign_id
        self.limits = dict(LIMITS if limits is None else limits)
        # Standalone diagnostics stop the run on cancellation. A shared HTTP QA
        # server opts out: a service-owned wait_for cancels its child provider
        # task as part of the existing transient timeout/retry policy.
        self.stop_on_caller_cancel = stop_on_caller_cancel
        # Shared by this run's text/audio wrappers, but deliberately not persisted:
        # a separately authorized follow-up run may reuse the charged ledger.
        self.run_stop_reason: str | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.path) + ".lock")
        with self.lock:
            if self.path.exists() and limits is None:
                # Existing runs use their immutable campaign limits even after a
                # process restart; never silently replace them with defaults.
                self.limits = json.loads(self.path.read_text(encoding="utf-8"))["limits"]
            if not self.path.exists():
                atomic_json(self.path, {
                    "campaignId": campaign_id, "limits": self.limits,
                    "priorLedgerSha256": prior_ledger_sha256,
                    "firstLiveAt": None, "stoppedReason": None, "calls": [],
                    "accountingBoundary": "Conservative reservations, not a billing hard cap; SDK physical attempts may be unobserved.",
                })
            self._read()
            if prior_ledger_sha256 is not None:
                state = self._read()
                if state.get("priorLedgerSha256") != prior_ledger_sha256:
                    raise ValueError("Prior campaign ledger reference mismatch")

    def _read(self) -> dict[str, Any]:
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if value["campaignId"] != self.campaign_id or value["limits"] != self.limits:
            raise ValueError("Campaign identity/limits mismatch; ledger cannot reset")
        return value

    @staticmethod
    def totals(value: dict[str, Any]) -> dict[str, float]:
        totals = {key: 0.0 for key in Reservation.__dataclass_fields__}
        for call in value["calls"]:
            charge = call.get("accounted") or call["reserved"]
            for key in totals:
                totals[key] += charge[key]
        totals["starts"] = len(value["calls"])
        return totals

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            value = self._read()
            value["totalsIncludingReserved"] = self.totals(value)
            return value

    def reserve(self, task: str, model: str, amount: Reservation, *, metadata: dict[str, Any] | None = None) -> int:
        if self.run_stop_reason:
            raise CampaignBudgetExceeded(self.run_stop_reason)
        fields = vars(amount)
        if any(v < 0 for v in fields.values()):
            raise ValueError("Negative reservation")
        with self.lock:
            state = self._read()
            if state["stoppedReason"]:
                raise CampaignBudgetExceeded(state["stoppedReason"])
            now = datetime.now(UTC)
            first = state["firstLiveAt"]
            if first and (now - datetime.fromisoformat(first)).total_seconds() >= self.limits["wall_seconds"]:
                state["stoppedReason"] = "CAMPAIGN_DEADLINE"
                atomic_json(self.path, state)
                raise CampaignBudgetExceeded("CAMPAIGN_DEADLINE")
            if self._remaining_active_seconds(state, now) <= 0:
                state["stoppedReason"] = "CAMPAIGN_ACTIVE_TIME_LIMIT"
                atomic_json(self.path, state)
                raise CampaignBudgetExceeded("CAMPAIGN_ACTIVE_TIME_LIMIT")
            used = self.totals(state)
            denied_reason = None
            if used["starts"] >= self.limits["starts"]:
                denied_reason = "APPLICATION_START_LIMIT"
            else:
                for key, requested in fields.items():
                    if used[key] + requested > self.limits[key]:
                        denied_reason = "RESERVATION_DENIED_" + key.upper()
                        break
            if denied_reason:
                # One denied reservation ends the campaign across every task
                # and process, even if a cheaper call would still fit. The
                # rejected call itself never starts or consumes a new attempt.
                state["stoppedReason"] = denied_reason
                atomic_json(self.path, state)
                raise CampaignBudgetExceeded(denied_reason)
            entry = {"attempt": len(state["calls"]) + 1, "startedAt": now.isoformat(),
                     "task": task, "model": model, "status": "STARTED",
                     "reserved": fields, "accounted": None,
                     "usageStatus": "UNKNOWN_RESERVED", "metadata": metadata or {}}
            state["firstLiveAt"] = first or now.isoformat()
            state["calls"].append(entry)
            atomic_json(self.path, state)
            return entry["attempt"]

    def remaining_seconds(self) -> float:
        state = self.snapshot()
        if not state["firstLiveAt"]:
            return min(float(self.limits["wall_seconds"]), self._remaining_active_seconds(state, datetime.now(UTC)))
        now = datetime.now(UTC)
        approval = self.limits["wall_seconds"] - (now - datetime.fromisoformat(state["firstLiveAt"])).total_seconds()
        return max(0.0, min(approval, self._remaining_active_seconds(state, now)))

    def _remaining_active_seconds(self, state: dict[str, Any], now: datetime) -> float:
        cap = self.limits.get("active_seconds")
        if cap is None:
            return float(self.limits["wall_seconds"])
        used = sum(
            max(0.0, float(call.get("latencyMs", 0))) / 1000
            if call["status"] != "STARTED" else
            max(0.0, (now - datetime.fromisoformat(call["startedAt"])).total_seconds())
            for call in state["calls"]
        )
        return max(0.0, cap - used)

    def halt(self, reason: str) -> None:
        """Persist a QA boundary stop without dropping prior reservations."""
        with self.lock:
            state = self._read()
            if state["stoppedReason"] is None:
                state["stoppedReason"] = reason
                atomic_json(self.path, state)

    def halt_run(self, reason: str) -> None:
        if self.run_stop_reason is None:
            self.run_stop_reason = reason

    def finish(self, attempt: int, *, status: str, elapsed: float,
               accounted: Reservation | None = None, failure: str | None = None) -> None:
        with self.lock:
            state = self._read()
            entry = state["calls"][attempt - 1]
            if entry["status"] != "STARTED":
                raise ValueError("Attempt already finalized")
            entry.update(status=status, latencyMs=round(elapsed * 1000, 3), failureType=failure)
            if accounted is not None:
                entry["accounted"] = vars(accounted)
                entry["usageStatus"] = "OBSERVED_CONSERVATIVE"
                if state["stoppedReason"] is None and any(
                    value > entry["reserved"][key] for key, value in vars(accounted).items()
                ):
                    state["stoppedReason"] = "USAGE_EXCEEDED_RESERVATION"
            if state["stoppedReason"] is None and any(
                value > self.limits[key] for key, value in self.totals(state).items()
            ):
                state["stoppedReason"] = "CAMPAIGN_CAP_EXCEEDED"
            atomic_json(self.path, state)


def check_campaign_admission(ledger: CampaignLedger) -> float:
    """Check before reservation and again immediately before upstream entry."""
    stopped = ledger.run_stop_reason or ledger.snapshot()["stoppedReason"]
    if stopped:
        raise CampaignBudgetExceeded(stopped)
    remaining = ledger.remaining_seconds()
    if remaining <= 0:
        ledger.halt("CAMPAIGN_DEADLINE")
        raise CampaignBudgetExceeded("CAMPAIGN_DEADLINE")
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        if ledger.stop_on_caller_cancel:
            ledger.halt_run("QA_EXECUTION_INTERRUPTED")
        raise asyncio.CancelledError
    return remaining


@asynccontextmanager
async def campaign_call_boundary(ledger: CampaignLedger, call_seconds: float):
    remaining = check_campaign_admission(ledger)
    task = asyncio.current_task()
    initial_cancellations = task.cancelling() if task is not None else 0
    deadline = asyncio.timeout(min(call_seconds, remaining))
    try:
        async with deadline:
            # Reservation/checkpoint writes may consume the last wall-clock time.
            # asyncio.timeout(0) alone still permits the coroutine's sync prefix.
            check_campaign_admission(ledger)
            yield
        if deadline.expired():
            raise TimeoutError("QA call deadline expired inside upstream")
        if task is not None and task.cancelling() > initial_cancellations:
            raise asyncio.CancelledError
    except TimeoutError:
        if deadline.expired():
            if remaining <= call_seconds:
                ledger.halt("CAMPAIGN_DEADLINE")
            else:
                ledger.halt_run("QA_CALL_TIMEOUT")
        # Upstream's own timeout is distinct: its existing application retry
        # policy is unchanged when the QA deadline itself has not expired.
        raise
    except asyncio.CancelledError:
        # This can be a production service's wait_for timeout or actual caller
        # cancellation; the provider boundary cannot infer which. Always
        # propagate it and retain unknown usage, but let HTTP callers own retry.
        if ledger.stop_on_caller_cancel:
            ledger.halt_run("QA_EXECUTION_INTERRUPTED")
        raise
    except KeyboardInterrupt:
        ledger.halt_run("QA_EXECUTION_INTERRUPTED")
        raise


class BudgetedTextProvider:
    """Preserve the configured OpenAI pool, policies and SDK retry=0."""
    def __init__(self, upstream: Any, ledger: CampaignLedger, *, phase: str, call_seconds: float = 90) -> None:
        if not 0 < call_seconds <= 90:
            raise ValueError("QA text call timeout must be within (0, 90] seconds")
        self.upstream, self.ledger = upstream, ledger
        self.phase, self.call_seconds = phase, call_seconds

    @property
    def ready(self) -> bool:
        return self.upstream.ready

    def model_name_for(self, task: str) -> str:
        from app.ai.model_policy import get_model_name_for_task
        return get_model_name_for_task(task)

    def provider_name_for(self, task: str) -> str:
        return str(self.upstream.provider_name_for(task))

    async def call(self, type_name: str, data: str, schema: dict | None = None) -> Any:
        return (await self.call_with_metadata(type_name, data, schema)).data

    async def call_with_metadata(self, type_name: str, data: str, schema: dict | None = None) -> Any:
        from app.ai.model_policy import get_task_model_policy
        from app.ai.prompt_registry import get_prompt_rule
        from app.core.config import settings
        if settings.AI_TEXT_PROVIDER.strip() != "openai":
            raise CampaignBudgetExceeded("UNPRICED_PROVIDER_ROUTING")
        model = self.model_name_for(type_name)
        if model not in {"gpt-5.6-luna", "gpt-5-mini", "gpt-5-nano", "gpt-5.6-sol"}:
            raise CampaignBudgetExceeded("UNPRICED_MODEL")
        policy = get_task_model_policy(type_name)
        # UTF-8 bytes upper-bound ordinary tokenization; extra framing/schema margin.
        text = data + (get_prompt_rule(type_name) or "") + json.dumps(schema, ensure_ascii=True)
        in_cap = len(text.encode("utf-8")) + 4096
        out_cap = policy.max_output_tokens
        reserve = Reservation(in_cap, out_cap, usd=_text_cost_reservation(model, in_cap, out_cap))
        check_campaign_admission(self.ledger)
        attempt = self.ledger.reserve(type_name, model, reserve, metadata={
            "phase": self.phase, "reasoningEffort": policy.reasoning_effort,
            "sdkMaxRetries": 0, "physicalAttemptsObservable": False,
            "requestSha256": hashlib.sha256(text.encode()).hexdigest(),
        })
        started = time.monotonic()
        try:
            async with campaign_call_boundary(self.ledger, self.call_seconds):
                result = await self.upstream.call_with_metadata(type_name=type_name, data=data, schema=schema)
        except BaseException as exc:
            self.ledger.finish(attempt, status="CANCELLED" if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)) else "FAILED", elapsed=time.monotonic() - started, failure=type(exc).__name__)
            raise
        inp, out = int(result.input_tokens), int(result.output_tokens)
        charge = None if inp <= 0 or out <= 0 else Reservation(
            inp, out, usd=_text_cost_reservation(model, inp, out)
        )
        self.ledger.finish(attempt, status="COMPLETED", elapsed=time.monotonic() - started, accounted=charge)
        return result

    async def warm_up(self) -> None:
        await self.upstream.warm_up()

    async def shutdown(self) -> None:
        await self.upstream.shutdown()
