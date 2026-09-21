"""QA-only audio admission wrappers. No production configuration is changed.

Root owns live execution. Missing receipts keep the full reservation, including
possible hidden Gemini network resends. Audio output tokens are dollar-accounted
separately from the campaign's text-output-token allowance.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import time
import wave
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scripts.qa_campaign_budget import (
    CampaignBudgetExceeded, CampaignLedger, Reservation, atomic_json,
    campaign_call_boundary, check_campaign_admission,
)


class BudgetedOpenAISpeechProvider:
    """One OpenAI Speech application start per reservation, without hidden SDK retries.

    The binary Speech response has no guaranteed usage receipt. Even after a
    valid WAV, its conservative reservation remains UNKNOWN_RESERVED; it is
    never settled to zero or inferred from a Gemini token multiplier.
    """

    def __init__(self, upstream: Any, ledger: CampaignLedger, *, phase: str,
                 call_seconds: float = 90, receipt_dir: Path | None = None) -> None:
        if getattr(upstream, "provider_name", None) != "openai":
            raise CampaignBudgetExceeded("GEMINI_GENERATION_DISABLED")
        self.upstream, self.ledger, self.phase = upstream, ledger, phase
        self.call_seconds = _call_seconds(call_seconds)
        self.receipt_dir = receipt_dir or ledger.path.parent / "audio-budget-receipts"

    @property
    def ready(self) -> bool:
        return bool(self.upstream.ready)

    async def warm_up(self) -> None:
        await self.upstream.warm_up()

    async def shutdown(self) -> None:
        await self.upstream.shutdown()

    async def synthesize_speech(self, *, text: str, voice: str, language: str, speed: str):
        from app.ai.providers.openai.speech import OPENAI_SPEECH_INSTRUCTIONS
        from app.core.config import settings

        if settings.OPENAI_SPEECH_MODEL != "gpt-4o-mini-tts-2025-12-15":
            raise CampaignBudgetExceeded("UNPRICED_OPENAI_SPEECH_MODEL")
        if voice not in {"marin", "cedar"} or not 0 < len(text) <= 4096:
            raise CampaignBudgetExceeded("INVALID_OPENAI_SPEECH_INPUT")
        # Speech has no documented max output-token parameter or binary usage
        # receipt. This deliberately over-reserves short learning utterances;
        # unknown/cancelled calls retain the full amount across restarts.
        input_cap = len((text + OPENAI_SPEECH_INSTRUCTIONS).encode("utf-8")) + 512
        reserve = Reservation(input_tokens=input_cap, tts_characters=len(text),
                              usd=max(0.50, len(text) * 0.01))
        check_campaign_admission(self.ledger)
        attempt = self.ledger.reserve("OPENAI_TTS", settings.OPENAI_SPEECH_MODEL, reserve,
                                      metadata={
            "phase": self.phase, "voice": voice, "language": language,
            "speed": speed, "sdkMaxRetries": 0,
            "responseUsageReceipt": "not available on binary speech response",
            "pricing": "input $0.60/M text tokens; output $12/M audio tokens",
            "requestSha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
        receipt_path = self.receipt_dir / f"tts-{attempt:04d}.json"
        report = {"attempt": attempt, "status": "STARTED", "sentOrUnknown": True,
                  "requestSha256": hashlib.sha256(text.encode()).hexdigest()}
        atomic_json(receipt_path, report)
        started = time.monotonic()
        try:
            async with campaign_call_boundary(self.ledger, self.call_seconds):
                response = await self.upstream.synthesize_speech(
                    text=text, voice=voice, language=language, speed=speed)
        except BaseException as exc:
            status = "CANCELLED" if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)) else "FAILED"
            self.ledger.finish(attempt, status=status, elapsed=time.monotonic() - started,
                               failure=type(exc).__name__)
            report.update(status=status, partial=True, errorType=type(exc).__name__)
            atomic_json(receipt_path, report)
            raise
        self.ledger.finish(attempt, status="COMPLETED", elapsed=time.monotonic() - started)
        report.update(status="COMPLETED", partial=False, usageStatus="UNKNOWN_RESERVED",
                      durationSeconds=response.duration_seconds,
                      audioSha256=hashlib.sha256(response.audio_bytes).hexdigest())
        atomic_json(receipt_path, report)
        return response


GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
GEMINI_INPUT_CAP = 8192
GEMINI_AUDIO_OUTPUT_CAP = 16384
GEMINI_SERVICE_ATTEMPTS = 3
GEMINI_NETWORK_ATTEMPTS = 2
GEMINI_INPUT_USD_PER_MILLION = 0.5
GEMINI_AUDIO_USD_PER_MILLION = 10.0


def _call_seconds(value: float) -> float:
    if not 0 < value <= 90:
        raise ValueError("QA audio call timeout must be within (0, 90] seconds")
    return value


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _receipt(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    prompt = _positive_int(getattr(usage, "prompt_token_count", None))
    candidates = _positive_int(getattr(usage, "candidates_token_count", None))
    total = _positive_int(getattr(usage, "total_token_count", None))
    thoughts = getattr(usage, "thoughts_token_count", 0) or 0
    valid = (prompt is not None and candidates is not None and
             isinstance(thoughts, int) and not isinstance(thoughts, bool) and thoughts >= 0)
    audio = None
    if valid and prompt is not None and candidates is not None:
        audio = max(candidates + thoughts, (total - prompt) if total is not None else 0)
    return {"inputTokens": prompt, "audioOutputTokens": audio,
            "totalTokens": total, "usageComplete": valid}


def _tts_amount(input_tokens: int, audio_output_tokens: int, characters: int) -> Reservation:
    return Reservation(
        input_tokens=input_tokens, output_tokens=0, tts_characters=characters,
        usd=(input_tokens * GEMINI_INPUT_USD_PER_MILLION
             + audio_output_tokens * GEMINI_AUDIO_USD_PER_MILLION) / 1_000_000,
    )


class BudgetedSpeechProvider:
    """Wrap the existing GeminiService; preserve its model/SDK/empty-audio retries."""

    def __init__(self, upstream: Any, ledger: CampaignLedger, *, phase: str,
                 call_seconds: float = 90, receipt_dir: Path | None = None) -> None:
        self.upstream, self.ledger, self.phase = upstream, ledger, phase
        self.call_seconds = _call_seconds(call_seconds)
        self.receipt_dir = receipt_dir or ledger.path.parent / "audio-budget-receipts"
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return bool(self.upstream.ready)

    async def warm_up(self) -> None:
        await self.upstream.warm_up()

    async def shutdown(self) -> None:
        await self.upstream.shutdown()

    async def synthesize_speech(self, *, text: str, voice: str, language: str, speed: str):
        from app.core.config import settings

        if settings.GEMINI_MODEL_TTS != GEMINI_TTS_MODEL:
            raise CampaignBudgetExceeded("UNPRICED_TTS_MODEL")
        multiplier = GEMINI_SERVICE_ATTEMPTS * GEMINI_NETWORK_ATTEMPTS
        reservation = _tts_amount(GEMINI_INPUT_CAP * multiplier,
                                  GEMINI_AUDIO_OUTPUT_CAP * multiplier, len(text) * multiplier)
        check_campaign_admission(self.ledger)
        async with self._lock:
            check_campaign_admission(self.ledger)
            # Local client construction is lazy but makes no network request.
            models = self.upstream.client.aio.models
            api_client = getattr(models, "_api_client", None)
            http_options = getattr(api_client, "_http_options", None)
            if http_options is None or not hasattr(http_options, "retry_options"):
                raise CampaignBudgetExceeded("UNINSPECTABLE_GEMINI_RETRY_POLICY")
            if http_options.retry_options is not None:
                raise CampaignBudgetExceeded("UNPRICED_GEMINI_RETRY_POLICY")
            original = models.generate_content
            attempt = self.ledger.reserve("GEMINI_TTS", GEMINI_TTS_MODEL, reservation, metadata={
                "phase": self.phase, "serviceAttemptCap": GEMINI_SERVICE_ATTEMPTS,
                "networkAttemptMultiplier": GEMINI_NETWORK_ATTEMPTS,
                "audioOutputTokenReservation": GEMINI_AUDIO_OUTPUT_CAP * multiplier,
                "audioOutputExcludedFromTextOutputCap": True,
                "ttsCharacterAccounting": "physical-attempt conservative multiplier",
                "requestSha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            })
            receipt_path = self.receipt_dir / f"tts-{attempt:04d}.json"
            report: dict[str, Any] = {"attempt": attempt, "status": "STARTED", "receipts": []}
            atomic_json(receipt_path, report)

            async def capture(*args, **kwargs):
                if len(report["receipts"]) >= GEMINI_SERVICE_ATTEMPTS:
                    raise CampaignBudgetExceeded("GEMINI_SERVICE_ATTEMPT_BOUND_CHANGED")
                entry: dict[str, Any] = {"sequence": len(report["receipts"]) + 1, "status": "STARTED"}
                report["receipts"].append(entry)
                atomic_json(receipt_path, report)
                try:
                    response = await original(*args, **kwargs)
                    entry.update(status="COMPLETED", **_receipt(response))
                    return response
                except BaseException as exc:
                    entry.update(status="INTERRUPTED" if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)) else "FAILED",
                                 errorType=type(exc).__name__)
                    raise
                finally:
                    atomic_json(receipt_path, report)

            started = time.monotonic()
            try:
                with patch.object(models, "generate_content", new=capture):
                    async with campaign_call_boundary(self.ledger, self.call_seconds):
                        response = await self.upstream.synthesize_speech(
                            text=text, voice=voice, language=language, speed=speed)
            except BaseException as exc:
                status = "CANCELLED" if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)) else "FAILED"
                self.ledger.finish(attempt, status=status, elapsed=time.monotonic() - started,
                                   failure=type(exc).__name__)
                report.update(status=status, partial=True, errorType=type(exc).__name__)
                atomic_json(receipt_path, report)
                raise
            receipts = report["receipts"]
            charge = None
            if receipts and all(item.get("usageComplete") for item in receipts):
                charge = _tts_amount(
                    sum(item["inputTokens"] for item in receipts) * GEMINI_NETWORK_ATTEMPTS,
                    sum(item["audioOutputTokens"] for item in receipts) * GEMINI_NETWORK_ATTEMPTS,
                    len(text) * len(receipts) * GEMINI_NETWORK_ATTEMPTS,
                )
            self.ledger.finish(attempt, status="COMPLETED", elapsed=time.monotonic() - started, accounted=charge)
            report.update(status="COMPLETED", partial=False,
                          usageStatus="OBSERVED_CONSERVATIVE" if charge else "UNKNOWN_RESERVED",
                          networkMultiplier=GEMINI_NETWORK_ATTEMPTS)
            atomic_json(receipt_path, report)
            return response


def _wav_seconds(audio: bytes) -> float:
    with wave.open(io.BytesIO(audio), "rb") as reader:
        if reader.getframerate() <= 0:
            raise ValueError("Invalid normalized WAV sample rate")
        return reader.getnframes() / reader.getframerate()


class BudgetedSttProvider:
    """Local STT uses no dollar charge; reserve the existing possible VAD fallback."""

    def __init__(self, upstream: Any, ledger: CampaignLedger, *, phase: str,
                 model: str = "base", call_seconds: float = 90,
                 inference_multiplier: int = 2) -> None:
        self.upstream, self.ledger, self.phase, self.model = upstream, ledger, phase, model
        self.call_seconds = _call_seconds(call_seconds)
        if inference_multiplier not in {1, 2}:
            raise ValueError("Unknown STT provider inference bound")
        self.inference_multiplier = inference_multiplier

    async def transcribe(self, wav_bytes: bytes, *, language: str, phrase_hints=None):
        duration = _wav_seconds(wav_bytes)
        reserve = Reservation(stt_seconds=duration * self.inference_multiplier)
        check_campaign_admission(self.ledger)
        attempt = self.ledger.reserve("LOCAL_STT", self.model, reserve, metadata={
            "phase": self.phase, "audioSeconds": duration,
            "vadFallbackMultiplier": self.inference_multiplier,
            "audioSha256": hashlib.sha256(wav_bytes).hexdigest(), "paidApi": False,
            "nativeThreadCancellation": "Coroutine timeout does not preempt an already running native inference.",
        })
        started = time.monotonic()
        try:
            async with campaign_call_boundary(self.ledger, self.call_seconds):
                response = await self.upstream.transcribe(wav_bytes, language=language, phrase_hints=phrase_hints)
        except BaseException as exc:
            self.ledger.finish(attempt, status="CANCELLED" if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)) else "FAILED",
                               elapsed=time.monotonic() - started, failure=type(exc).__name__)
            raise
        self.ledger.finish(attempt, status="COMPLETED", elapsed=time.monotonic() - started, accounted=reserve)
        return response


def inspect_local_stt_cache() -> dict[str, Any]:
    """Check the shared DI runtime, not standalone Speaking/Listening defaults."""
    from app.features.speech_to_text import FasterWhisperRuntime

    # This constructor resolves settings only: no model load, worker, or network.
    runtime = FasterWhisperRuntime()
    report: dict[str, Any] = {
        "runtimePolicySource": "app/api/dependencies.py:_speech_runtime=FasterWhisperRuntime()",
        "configuredModel": runtime.model_name,
        "modelRevision": runtime.model_revision,
        "modelVersion": runtime.model_version,
        "device": runtime.device, "computeType": runtime.compute_type,
        "cpuThreads": runtime.cpu_threads, "numWorkers": runtime.num_workers,
        "maxConcurrency": runtime.max_concurrency, "queueCapacity": runtime.queue_capacity,
        "warmUpInference": runtime.run_warm_up_inference,
        "ready": False, "downloadRequired": False,
    }
    try:
        from faster_whisper.utils import download_model

        configured = Path(runtime.model_name)
        model_path = configured if configured.is_dir() else Path(download_model(
            runtime.model_name, revision=runtime.model_revision, local_files_only=True,
        ))
        required = ["model.bin", "config.json", "tokenizer.json"]
        missing = [name for name in required if not (model_path / name).is_file()]
        if not list(model_path.glob("vocabulary.*")):
            missing.append("vocabulary.*")
        report.update(modelPath=str(model_path.resolve()), missingFiles=missing,
                      cacheAvailable=not missing, downloadRequired=bool(missing))
    except Exception as exc:
        report.update(cacheAvailable=False, errorType=type(exc).__name__,
                      downloadRequired=type(exc).__name__ in {"LocalEntryNotFoundError", "FileNotFoundError"})
    return report


def make_budgeted_stt_provider(runtime: Any, ledger: CampaignLedger, *, phase: str,
                               mode: str = "speaking", call_seconds: float = 90) -> BudgetedSttProvider:
    """Reuse one prepared shared runtime with each mode's existing provider behavior."""
    from app.features.language_learning.speaking.stt_service import FasterWhisperSpeakingSttProvider
    from app.features.language_learning.listening.stt_provider import FasterWhisperListeningSttProvider

    if mode == "speaking":
        upstream = FasterWhisperSpeakingSttProvider(runtime)
        multiplier = 2
    elif mode == "listening":
        upstream = FasterWhisperListeningSttProvider(runtime)
        multiplier = 1
    else:
        raise ValueError("Unknown STT campaign mode")
    return BudgetedSttProvider(upstream, ledger, phase=phase, model=runtime.model_name,
                              call_seconds=call_seconds, inference_multiplier=multiplier)


async def prepare_local_stt(ledger: CampaignLedger, *, phase: str,
                            report_path: Path, call_seconds: float = 90,
                            mode: str = "speaking"):
    """Return (wrapper or None, runtime or None, safe report). Caller must shutdown.

    Uses identical configured weights/device/compute policy via an existing local
    snapshot. It cannot preempt an already running native model-load thread.
    """
    from app.features.speech_to_text import FasterWhisperRuntime

    timeout = _call_seconds(call_seconds)
    if mode not in {"speaking", "listening"}:
        raise ValueError("Unknown STT campaign mode")
    report = inspect_local_stt_cache()
    atomic_json(report_path, report)
    if not report.get("cacheAvailable"):
        return None, None, report

    class CachedOnlyRuntime(FasterWhisperRuntime):
        def _load_and_warm_model(self):
            from faster_whisper import WhisperModel

            model = WhisperModel(report["modelPath"], local_files_only=True,
                                 device=self.device, compute_type=self.compute_type,
                                 cpu_threads=self.cpu_threads, num_workers=self.num_workers,
                                 revision=self.model_revision)
            if self.run_warm_up_inference:
                import numpy as np

                segments, _ = model.transcribe(np.zeros(4000, dtype=np.float32), language="en",
                                              beam_size=1, vad_filter=False, condition_on_previous_text=False)
                list(segments)
            return model

    runtime = CachedOnlyRuntime()
    warmup_amount = Reservation(stt_seconds=0.25 if runtime.run_warm_up_inference else 0)
    check_campaign_admission(ledger)
    warmup_attempt = ledger.reserve("LOCAL_STT_PREPARE", runtime.model_name, warmup_amount,
                                    metadata={"phase": phase, "paidApi": False,
                                              "downloadAllowed": False})
    started = time.monotonic()
    try:
        async with campaign_call_boundary(ledger, timeout):
            await runtime.warm_up()
    except BaseException as exc:
        ledger.finish(warmup_attempt,
                      status="CANCELLED" if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)) else "FAILED",
                      elapsed=time.monotonic() - started, failure=type(exc).__name__)
        report.update(ready=False, errorType=type(exc).__name__,
                      nativeLoadMayFinishAfterCancellation=True)
        atomic_json(report_path, report)
        await runtime.shutdown(grace_seconds=0.1)
        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
            raise
        return None, None, report
    ledger.finish(warmup_attempt, status="COMPLETED", elapsed=time.monotonic() - started,
                  accounted=warmup_amount)
    report.update(ready=runtime.ready, modelVersion=runtime.model_version)
    atomic_json(report_path, report)
    return make_budgeted_stt_provider(runtime, ledger, phase=phase,
                                      mode=mode, call_seconds=timeout), runtime, report
