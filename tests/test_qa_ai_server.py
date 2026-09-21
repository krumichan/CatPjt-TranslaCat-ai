"""Offline QA-server boundary tests in a clean, synthetic-credential process.

The child never reads repository .env files or private campaign credentials and
never starts a socket server. ASGI requests use production middleware/routes.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


_ISOLATED_CHECK = r'''
import asyncio
import json
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch

output = Path(sys.argv[1])
os.environ["SERVER_API_KEY"] = "synthetic-offline-qa-key"
os.environ["OCR_WARM_UP"] = "false"
os.environ["AI_VOICE_ENABLED"] = "false"
os.environ["OPENAI_API_KEY"] = ""
os.environ["GOOGLE_API_KEY"] = ""

# Windows constructs a local socketpair for the loop's wake-up pipe. Create that
# internal primitive before blocking all subsequent socket connections.
loop = asyncio.new_event_loop()

def deny_network(*args, **kwargs):
    raise AssertionError("Offline QA-server test attempted a network connection")

socket.create_connection = deny_network
socket.socket.connect = deny_network

from app.ai.ports import StructuredGenerationResult
import app.ai.provider_factory as factory

class FakeText:
    ready = True
    calls = 0
    shutdown_calls = 0

    async def call_with_metadata(self, type_name, data, schema=None):
        self.calls += 1
        self.last_response = StructuredGenerationResult(
            {"offline": True}, input_tokens=100, output_tokens=40,
            provider="synthetic", model="synthetic-model",
        )
        return self.last_response

    async def shutdown(self):
        self.shutdown_calls += 1

class FakeSpeech:
    provider_name = "openai"
    ready = True
    calls = 0
    shutdown_calls = 0

    async def synthesize_speech(self, **kwargs):
        self.calls += 1
        raise AssertionError("The boundary test must not request audio")

    async def shutdown(self):
        self.shutdown_calls += 1

text, speech = FakeText(), FakeSpeech()
with patch.object(factory, "create_text_generation_provider", return_value=text), patch.object(
    factory, "create_speech_synthesis_provider", return_value=speech
):
    from scripts.run_qa_ai_server import create_qa_app
    with patch("scripts.qa_campaign_integration._owned_manifest", return_value={"owner": "synthetic-offline"}):
        app = create_qa_app(output)
    from app.api import dependencies as dependencies
    from scripts.qa_campaign_budget import BudgetedTextProvider
    from scripts.qa_campaign_audio_budget import BudgetedOpenAISpeechProvider, BudgetedSttProvider
    import httpx

    wrapped_text = dependencies.get_ai_provider()
    wrapped_speech = dependencies.get_speech_synthesis_provider()
    service_names = [
        "_language_learning_writing_service", "_language_learning_reading_vocabulary_service",
        "_listening_generation_service", "_listening_interpretation_service",
        "_listening_summary_service", "_listening_explanation_service",
        "_speaking_conversation_service", "_speaking_assistance_service",
        "_speaking_evaluation_service", "_level_test_service",
    ]
    result = {
        "textServicesAllWrapped": all(getattr(dependencies, name).provider is wrapped_text for name in service_names),
        "textHasBudget": isinstance(wrapped_text.upstream, BudgetedTextProvider),
        "textBudgetUsesFake": wrapped_text.upstream.upstream is text,
        "textFactoryStaysWrapped": factory.create_text_generation_provider() is wrapped_text,
        "speechHasBudget": isinstance(wrapped_speech, BudgetedOpenAISpeechProvider),
        "speechBudgetUsesFake": wrapped_speech.upstream is speech,
        "speechServicesAllWrapped": all(service.provider is wrapped_speech for service in (
            dependencies._listening_tts_service, dependencies._speaking_tts_service)),
        "speechFactoryStaysWrapped": isinstance(factory.create_speech_synthesis_provider(), BudgetedOpenAISpeechProvider),
        "speakingSttHasBudget": isinstance(dependencies._speaking_stt_service.provider, BudgetedSttProvider),
        "listeningSttHasBudget": isinstance(dependencies._listening_repeat_service.stt_provider, BudgetedSttProvider),
        "speakingSttOriginalPreserved": dependencies._speaking_stt_service.provider.upstream is dependencies._speaking_stt_provider,
        "listeningSttOriginalPreserved": dependencies._listening_repeat_service.stt_provider.upstream is dependencies._listening_stt_provider,
        "speakingSttInferenceBound": dependencies._speaking_stt_service.provider.inference_multiplier,
        "listeningSttInferenceBound": dependencies._listening_repeat_service.stt_provider.inference_multiplier,
        "dependencyOverrides": len(app.dependency_overrides),
        "textHasMetadataMethod": callable(getattr(wrapped_text, "call_with_metadata", None)),
    }

    async def inspect():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://offline.invalid") as client:
            path = "/api/v1/language-learning/practice/generate"
            result["noKey"] = (await client.post(path, json={})).status_code
            result["wrongKey"] = (await client.post(path, json={}, headers={"X-API-KEY": "wrong"})).status_code
            result["validKeyReachesSchema"] = (await client.post(path, json={}, headers={"X-API-KEY": "synthetic-offline-qa-key"})).status_code
            result["unrelatedRoutes"] = {}
            for path in ("/docs", "/openapi.json", "/api/v1/translation", "/internal/v1/voice/stream", "/api/v1/language-learning-escape"):
                response = await client.get(path, headers={"X-API-KEY": "synthetic-offline-qa-key"})
                result["unrelatedRoutes"][path] = {"status": response.status_code, "body": response.json()}
            result["healthNoKey"] = (await client.get("/")).status_code
        websocket_messages = []
        inbound = iter([{"type": "websocket.connect"}, {"type": "websocket.disconnect", "code": 1000}])

        async def receive_websocket():
            return next(inbound)

        async def send_websocket(message):
            websocket_messages.append(message)

        await app({
            "type": "websocket", "asgi": {"version": "3.0"}, "scheme": "ws",
            "path": "/internal/v1/voice/streams", "raw_path": b"/internal/v1/voice/streams",
            "query_string": b"", "root_path": "", "subprotocols": [],
            "headers": [(b"x-api-key", b"synthetic-offline-qa-key")],
            "server": ("offline.invalid", 80), "client": ("127.0.0.1", 1234),
        }, receive_websocket, send_websocket)
        result["websocketMessages"] = websocket_messages
        result["providerCallsFromBoundaryRequests"] = text.calls + speech.calls
        if result["textHasMetadataMethod"]:
            response = await wrapped_text.call_with_metadata(
                "LANGUAGE_LEARNING_SPEAKING_CONVERSATION", "synthetic offline fixture", {"type": "object"})
            result["metadataResponse"] = {
                "data": response.data, "provider": response.provider, "model": response.model,
                "inputTokens": response.input_tokens, "outputTokens": response.output_tokens,
            }
            result["originalResponseIdentity"] = response is text.last_response
        result["providerCallsAfterMetadataProbe"] = text.calls
        await wrapped_text.shutdown()
        speech_shutdown = getattr(wrapped_speech, "shutdown", None)
        if speech_shutdown is not None:
            await speech_shutdown()
        result["textShutdown"] = text.shutdown_calls
        result["speechShutdown"] = speech.shutdown_calls

    try:
        loop.run_until_complete(inspect())
    finally:
        loop.close()
    result["ledger"] = json.loads((output / "budget-ledger.json").read_text(encoding="utf-8"))
    runtime_info = json.loads((output / "ai-server-runtime.json").read_text(encoding="utf-8"))
    result["sourceSha256"] = runtime_info["sourceSha256"]
    diagnostic_path = Path(runtime_info["diagnosticArtifact"])
    result["privateTrace"] = {
        "exists": diagnostic_path.is_file(),
        "processOwnedName": diagnostic_path.name.startswith(f"be-ai-provider-diagnostics-{os.getpid()}-"),
        "legacyUnchanged": (output / "be-ai-provider-diagnostics.json").read_text() == "legacy trace sentinel",
    }
    (output / "offline-assertions.json").write_text(json.dumps(result), encoding="utf-8")
'''


@pytest.fixture(scope="module")
def server_boundary_result(tmp_path_factory):
    output = tmp_path_factory.mktemp("qa-server-synthetic")
    secrets_directory = output / "integration-environment"
    secrets_directory.mkdir()
    (secrets_directory / "secrets.private.json").write_text(
        json.dumps({"QA_AI_API_KEY": "synthetic-offline-qa-key"}), encoding="utf-8"
    )
    (output / "be-ai-provider-diagnostics.json").write_text("legacy trace sentinel")
    repository = Path(__file__).resolve().parents[1]
    # Only explicitly non-secret OS bootstrap values are forwarded. In particular,
    # no existing provider credential, OAuth token, or private campaign path enters
    # this process. Empty cwd has no .env to load via Pydantic Settings.
    env = {name: os.environ[name] for name in (
        "SystemRoot", "SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "LOCALAPPDATA",
    ) if name in os.environ}
    env["PYTHONPATH"] = os.pathsep.join([str(repository), *sys.path])
    completed = subprocess.run(
        [sys.executable, "-c", _ISOLATED_CHECK, str(output)], cwd=output,
        env=env, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads((output / "offline-assertions.json").read_text(encoding="utf-8"))


def test_normal_ai_route_keeps_production_api_key_authentication(server_boundary_result) -> None:
    result = server_boundary_result
    assert result["dependencyOverrides"] == 0
    assert result["noKey"] == 401
    assert result["wrongKey"] == 401
    assert result["validKeyReachesSchema"] == 422
    assert result["healthNoKey"] == 200
    assert result["providerCallsFromBoundaryRequests"] == 0


def test_non_language_learning_routes_are_denied_even_with_valid_key(server_boundary_result) -> None:
    for route in server_boundary_result["unrelatedRoutes"].values():
        assert route == {"status": 403, "body": {"detail": "QA_FEATURE_SCOPE_ONLY"}}


def test_unrelated_websocket_is_denied_before_accept_or_provider_start(server_boundary_result) -> None:
    messages = server_boundary_result["websocketMessages"]
    assert len(messages) == 1
    assert messages[0]["type"] == "websocket.close"
    assert messages[0]["code"] == 4403


def test_every_learning_provider_uses_campaign_wrapper_without_replacing_stt_policy(server_boundary_result) -> None:
    for field in (
        "textServicesAllWrapped", "textHasBudget", "textBudgetUsesFake", "textFactoryStaysWrapped",
        "speechHasBudget", "speechBudgetUsesFake", "speechServicesAllWrapped", "speechFactoryStaysWrapped",
        "speakingSttHasBudget", "listeningSttHasBudget", "speakingSttOriginalPreserved", "listeningSttOriginalPreserved",
    ):
        assert server_boundary_result[field], field
    assert server_boundary_result["speakingSttInferenceBound"] == 2
    assert server_boundary_result["listeningSttInferenceBound"] == 1


def test_text_metadata_protocol_is_preserved_and_budgeted(server_boundary_result) -> None:
    result = server_boundary_result
    assert result["textHasMetadataMethod"]
    assert result["originalResponseIdentity"]
    assert result["metadataResponse"] == {
        "data": {"offline": True}, "provider": "synthetic", "model": "synthetic-model",
        "inputTokens": 100, "outputTokens": 40,
    }
    assert result["providerCallsAfterMetadataProbe"] == 1
    calls = result["ledger"]["calls"]
    assert len(calls) == 1
    assert calls[0]["status"] == "COMPLETED"
    assert calls[0]["accounted"]["input_tokens"] == 100
    assert calls[0]["accounted"]["output_tokens"] == 40


def test_qa_wrappers_close_both_upstream_clients(server_boundary_result) -> None:
    assert server_boundary_result["textShutdown"] == 1
    assert server_boundary_result["speechShutdown"] == 1


def test_new_server_trace_keeps_previous_process_diagnostics(server_boundary_result) -> None:
    assert server_boundary_result["privateTrace"] == {
        "exists": True, "processOwnedName": True, "legacyUnchanged": True,
    }


def test_server_provenance_includes_generation_and_evaluation_prompts(server_boundary_result) -> None:
    sources = {key.replace("\\", "/"): value for key, value in server_boundary_result["sourceSha256"].items()}
    for name in (
        "app/ai/model_policy.py",
        "app/features/language_learning/reading_vocabulary/prompts.py",
        "app/features/language_learning/speaking/conversation_service.py",
        "app/features/language_learning/speaking/prompts.py",
    ):
        assert len(sources[name]) == 64


@pytest.mark.asyncio
async def test_qa_startup_prepares_original_shared_stt_and_preserves_lifespan(tmp_path) -> None:
    from scripts.qa_campaign_budget import CampaignLedger
    from scripts.run_qa_ai_server import install_qa_stt_lifespan

    events = []

    class FakeRuntime:
        ready = False
        model_name = "base"
        device = "cpu"
        compute_type = "int8"
        cpu_threads = 2
        run_warm_up_inference = True

        async def warm_up(self):
            events.append("warmup")
            self.ready = True

        async def shutdown(self):
            events.append("shutdown")

    runtime = FakeRuntime()

    @asynccontextmanager
    async def original(app):
        assert runtime.ready
        events.append("original-start")
        yield
        await runtime.shutdown()

    app = SimpleNamespace(router=SimpleNamespace(lifespan_context=original))
    ledger = CampaignLedger(tmp_path / "budget.json", "qa-startup")
    install_qa_stt_lifespan(app, runtime, ledger, tmp_path / "readiness.json")
    async with app.router.lifespan_context(app):
        assert events == ["warmup", "original-start"]
    assert events == ["warmup", "original-start", "shutdown"]
    call, = ledger.snapshot()["calls"]
    assert call["task"] == "LOCAL_STT_PREPARE"
    assert call["status"] == "COMPLETED"
    assert call["accounted"]["stt_seconds"] == 0.25
    assert call["accounted"]["usd"] == 0


@pytest.mark.asyncio
async def test_failed_qa_stt_startup_preserves_failure_and_closes_runtime(tmp_path) -> None:
    from unittest.mock import AsyncMock

    from scripts.qa_campaign_budget import CampaignLedger
    from scripts.run_qa_ai_server import install_qa_stt_lifespan

    runtime = SimpleNamespace(ready=False, model_name="base", run_warm_up_inference=True,
                              warm_up=AsyncMock(side_effect=TimeoutError()), shutdown=AsyncMock())
    app = SimpleNamespace(router=SimpleNamespace(lifespan_context=AsyncMock()))
    ledger = CampaignLedger(tmp_path / "budget.json", "qa-startup")
    install_qa_stt_lifespan(app, runtime, ledger, tmp_path / "readiness.json")
    with pytest.raises(TimeoutError):
        async with app.router.lifespan_context(app):
            pytest.fail("not-ready server must not serve QA requests")
    runtime.shutdown.assert_awaited_once()
    call, = ledger.snapshot()["calls"]
    assert call["status"] == "FAILED"
    assert call["usageStatus"] == "UNKNOWN_RESERVED"
    assert json.loads((tmp_path / "readiness.json").read_text())["errorType"] == "TimeoutError"
