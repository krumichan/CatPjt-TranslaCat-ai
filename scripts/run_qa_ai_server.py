"""Isolated authenticated AI server with campaign-only provider instrumentation.

No production file, provider policy or authentication check is overridden.
Only normal language-learning HTTP requests are admitted on the QA port.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def install_qa_stt_lifespan(app: Any, runtime: Any, ledger: Any, report_path: Path) -> None:
    """Warm the original shared learning runtime without unrelated voice services.

    The normal providers also support lazy warmup. Preparing it here accounts for
    that inference and keeps model loading out of the first learner upload.
    """
    from scripts.qa_campaign_budget import Reservation, atomic_json

    original = app.router.lifespan_context

    @asynccontextmanager
    async def learning_lifespan(application: Any):
        if not runtime.ready:
            amount = Reservation(stt_seconds=0.25 if runtime.run_warm_up_inference else 0)
            attempt = ledger.reserve("LOCAL_STT_PREPARE", runtime.model_name, amount,
                                     metadata={"phase": "BE_HTTP_STARTUP", "paidApi": False})
            started = time.monotonic()
            try:
                async with asyncio.timeout(min(90, ledger.remaining_seconds())):
                    await runtime.warm_up()
                if not runtime.ready:
                    raise RuntimeError("QA_SHARED_STT_NOT_READY")
            except BaseException as exc:
                ledger.finish(attempt, status="CANCELLED" if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)) else "FAILED",
                              elapsed=time.monotonic() - started, failure=type(exc).__name__)
                atomic_json(report_path, {"ready": False, "errorType": type(exc).__name__,
                                          "nativeLoadMayFinishAfterCancellation": True})
                await runtime.shutdown()
                raise
            ledger.finish(attempt, status="COMPLETED", elapsed=time.monotonic() - started, accounted=amount)
        atomic_json(report_path, {"ready": runtime.ready, "model": runtime.model_name,
                                  "device": runtime.device, "computeType": runtime.compute_type,
                                  "cpuThreads": runtime.cpu_threads,
                                  "runtimeIdentity": "injected production dependency runtime; no QA model substitution"})
        async with original(application):
            yield

    app.router.lifespan_context = learning_lifespan


def create_qa_app(output: Path, *, environment: Path | None = None) -> Any:
    from scripts.qa_campaign_budget import BudgetedTextProvider, CampaignLedger, atomic_json
    from scripts.qa_campaign_audio_budget import BudgetedOpenAISpeechProvider, BudgetedSttProvider

    environment = environment or output / "integration-environment"
    from scripts.qa_campaign_integration import _owned_manifest
    _owned_manifest(environment)
    secrets = json.loads((environment / "secrets.private.json").read_text(encoding="utf-8"))
    os.environ["SERVER_API_KEY"] = secrets["QA_AI_API_KEY"]
    os.environ["OCR_WARM_UP"] = "false"
    os.environ["AI_VOICE_ENABLED"] = "false"  # learning STT is warmed explicitly below
    audio_temp = output / "ai-process-temp"
    audio_temp.mkdir(exist_ok=True)
    tempfile.tempdir = str(audio_temp)
    ledger = CampaignLedger(output / "budget-ledger.json", output.name, stop_on_caller_cancel=False)
    import app.ai.provider_factory as factory
    from scripts.run_contextual_choice_stabilization_qa import QaCaps, RecordingProvider, _call_summary

    original_text, original_speech = factory.create_text_generation_provider, factory.create_speech_synthesis_provider
    recorder: RecordingProvider | None = None
    diagnostic_path = output / f"be-ai-provider-diagnostics-{os.getpid()}-{time.time_ns()}.json"

    class ManagedRecordingProvider(RecordingProvider):
        @property
        def ready(self) -> bool:
            return bool(self.upstream.ready)

        async def shutdown(self) -> None:
            await self.upstream.shutdown()

    def text_factory() -> Any:
        nonlocal recorder
        if recorder is None:
            recorder = ManagedRecordingProvider(BudgetedTextProvider(original_text(), ledger, phase="BE_HTTP"),
                QaCaps(int(ledger.limits["starts"]), int(ledger.limits["input_tokens"]),
                       int(ledger.limits["output_tokens"]), 90,
                       int(ledger.limits.get("active_seconds", ledger.limits["wall_seconds"]))),
                diagnostic_capture=True,
                stop_on_caller_cancel=False)
            # A server restart must never replace the preceding process's trace.
            recorder.checkpoint = lambda: atomic_json(diagnostic_path, _call_summary(recorder.calls if recorder else []))
        return recorder

    factory.create_text_generation_provider = text_factory
    factory.create_speech_synthesis_provider = lambda: BudgetedOpenAISpeechProvider(original_speech(), ledger, phase="BE_HTTP")
    # dependencies captures the wrapped factories when constructing its services.
    from app.api import dependencies
    from app.main import app
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from starlette.types import ASGIApp, Receive, Scope, Send

    install_qa_stt_lifespan(app, dependencies._speech_runtime, ledger, output / "ai-server-stt-readiness.json")
    if dependencies._speaking_speech_runtime is not dependencies._speech_runtime:
        install_qa_stt_lifespan(app, dependencies._speaking_speech_runtime, ledger, output / "ai-speaking-stt-readiness.json")

    class RejectQaWebSockets:
        def __init__(self, app: ASGIApp) -> None:
            self.app = app

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4403})
                return
            await self.app(scope, receive, send)

    app.add_middleware(RejectQaWebSockets)

    for name in ("_speaking_stt_provider", "_listening_stt_provider"):
        upstream = getattr(dependencies, name)
        wrapped = BudgetedSttProvider(upstream, ledger, phase="BE_HTTP", model=upstream.runtime.model_name,
            inference_multiplier=1 if name == "_listening_stt_provider" else 2)
        # The dependent services already exist; replace only their provider reference.
        if name == "_speaking_stt_provider":
            dependencies._speaking_stt_service.provider = wrapped
        else:
            dependencies._listening_repeat_service.stt_provider = wrapped
    dependencies._level_test_stt_service.provider = BudgetedSttProvider(
        dependencies._level_test_stt_service.provider, ledger, phase="BE_HTTP_LEVEL_TEST",
        model=dependencies._speech_runtime.model_name,
    )

    @app.middleware("http")
    async def qa_scope(request: Request, call_next):
        if request.url.path != "/" and not request.url.path.startswith(("/internal/v1/language-learning/", "/api/v1/language-learning/")):
            return JSONResponse(status_code=403, content={"detail": "QA_FEATURE_SCOPE_ONLY"})
        return await call_next(request)

    from scripts.qa_campaign_integration import PORTS
    atomic_json(output / "ai-server-runtime.json", {
        "pid": os.getpid(), "port": PORTS["ai"], "host": "127.0.0.1", "workers": 1,
        "python": sys.executable, "tempDirectory": str(audio_temp),
        "modelPolicy": "text model policy unchanged; OpenAI-only speech factory; selected Speaking STT unchanged",
        "sttRuntimeModel": dependencies._speech_runtime.model_name,
        "speakingSttRuntimeModel": dependencies._speaking_speech_runtime.model_name,
        "speakingSttBeamSize": dependencies._speaking_stt_provider.beam_size,
        "authentication": "existing X-API-KEY middleware with isolated runtime credential",
        "providerBudget": str(ledger.path), "nonLearningRoutes": "denied",
        "diagnosticArtifact": str(diagnostic_path),
        "cancellationBoundary": "Request cancellation propagates; only QA-owned timeout/cap stops subsequent requests",
        "sourceSha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [ROOT / "app/ai/providers/openai/schema.py",
                         ROOT / "app/ai/model_policy.py",
                         ROOT / "app/core/config.py",
                         ROOT / "app/api/dependencies.py",
                         ROOT / "app/features/speech_to_text/runtime_policy.py",
                         ROOT / "app/features/language_learning/listening/generation_service.py",
                         ROOT / "app/features/language_learning/listening/duration.py",
                         ROOT / "app/features/language_learning/listening/tts_service.py",
                         ROOT / "app/features/language_learning/speaking/stt_service.py",
                         ROOT / "app/features/language_learning/reading_vocabulary/service.py",
                         ROOT / "app/features/language_learning/reading_vocabulary/prompts.py",
                         ROOT / "app/features/language_learning/speaking/conversation_service.py",
                         ROOT / "app/features/language_learning/speaking/coaching_service.py",
                         ROOT / "app/features/language_learning/speaking/evaluation_service.py",
                         ROOT / "app/features/language_learning/speaking/prompts.py",
                         ROOT / "app/schemas/language_learning_speaking/coaching.py",
                         ROOT / "scripts/run_qa_ai_server.py"]},
    })
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment", type=Path, help="Previously owned isolated QA environment; ledger remains in --output")
    args = parser.parse_args()
    import uvicorn
    from scripts.qa_campaign_integration import PORTS
    uvicorn.run(create_qa_app(args.output.resolve(), environment=args.environment.resolve() if args.environment else None),
                host="127.0.0.1", port=PORTS["ai"], workers=1, access_log=False)


if __name__ == "__main__":
    main()
