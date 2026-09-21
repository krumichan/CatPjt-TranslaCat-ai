"""Explicit development QA with injected, budgeted providers; never creates clients.

Learner audio is synthetic TTS, not a human pronunciation benchmark. The campaign
uses production services in process: HTTP upload/auth, browser playback, and BE
persistence are separate integration evidence owned by the calling campaign.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import time
import wave
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.ai.ports import SpeechSynthesisProvider, StructuredTextGenerationProvider
from app.features.language_learning.speaking.audio_processor import (
    SpeakingAudioProcessor,
)
from app.features.language_learning.speaking.audio_store import TemporaryTtsAudioStore
from app.features.language_learning.speaking.conversation_service import (
    SpeakingConversationService,
)
from app.features.language_learning.speaking.evaluation_service import (
    SpeakingEvaluationService,
)
from app.features.language_learning.speaking.stt_service import (
    SpeakingSttProvider,
    SpeakingSttService,
)
from app.features.language_learning.speaking.tts_service import SpeakingTtsService
from app.features.language_learning.speaking.turn_service import SpeakingTurnService
from app.schemas.language_learning_speaking import (
    AssistantEvaluationTurn,
    ConversationGenerationRequest,
    ConversationMessage,
    SessionStartRequest,
    SpeakingEvaluationRequest,
    SpeakingEvaluationTurn,
    SpeakingPracticeMode,
)
from scripts.qa_campaign_budget import CampaignBudgetExceeded, atomic_json


SCENARIO_PLAN = {
    "READ_ALOUD": [
        "RIGHT_INTENT",
        "RIGHT_INTENT",
        "RIGHT_INTENT",
        "PARTIAL",
        "RIGHT_INTENT",
        "UNRELATED",
        "RIGHT_INTENT",
        "RIGHT_INTENT",
        "RIGHT_INTENT",
        "PARTIAL",
    ],
    "GUIDED": ["RIGHT_INTENT", "PARTIAL", "UNRELATED", "RIGHT_INTENT", "RIGHT_INTENT"],
    "FREE": ["RIGHT_INTENT", "PARTIAL", "UNRELATED", "RIGHT_INTENT", "RIGHT_INTENT"],
}
CALL_ESTIMATE = {
    "normalTextStarts": 25,
    "normalReferenceTtsStarts": 17,
    "normalSyntheticLearnerTtsStarts": 20,
    "normalSttStarts": 20,
    "normalTotalStarts": 82,
    "serviceRetryUpperBoundStarts": 206,
    "note": "Existing per-stage retry limits only; SDK retries and STT VAD fallback are separate.",
}


def _checkpoint(path: Path, result: dict[str, Any]) -> None:
    atomic_json(path, result)


def _silence_wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 32000)
    return output.getvalue()


def _learner_text(scenario: str, mode: str, script: str, facts: list[str]) -> str:
    if scenario == "PREDECLARED_FOLLOW_UP":
        detail = "。".join(facts) + "。" if mode == "GUIDED" and facts else ""
        return (
            "最後に確認させてください。"
            + detail
            + "この内容で進める前に、担当者と関係者の都合をもう一度確認します。"
            "変更が必要な場合は、理由と代わりの案を早めに共有したいです。"
        )
    if scenario == "UNRELATED":
        return "今日は森に住む動物について話します。私は週末に動物園へ行きました。予定の相談とは関係のない話です。"
    if mode == "READ_ALOUD":
        return (
            script if scenario == "RIGHT_INTENT" else script[: max(1, len(script) // 2)]
        )
    if scenario == "PARTIAL":
        return (
            "予定について相談したいです。詳しい理由や代わりの日程はまだ決めていません。"
        )
    if mode == "GUIDED" and facts:
        return (
            "予定について相談させてください。"
            + "。".join(facts)
            + "。この条件で調整したいです。皆さんの都合も伺ってから決めたいと思います。"
        )
    return "仕事の予定は、担当者の都合と準備に必要な時間を確認してから決めたいです。急に変更が必要になったら、理由を説明して代わりの日程を提案します。一方的に決めず、相手の意見も聞くことが大切だと思います。"


class QaCheckpointFailure(BaseException):
    """QA artifact failure; bypass service retryable provider exception mapping."""


class _Capture:
    def __init__(self, directory: Path, result: dict[str, Any]) -> None:
        self.directory = directory
        self.result = result
        self.path = directory / "speaking-campaign-result.json"
        self.halt_error: BaseException | None = None

    def flush(self) -> None:
        try:
            _checkpoint(self.path, self.result)
        except PermissionError as exc:
            # A checkpoint access failure is not a provider transient. Never let
            # the production stage's retry wrapper start a second billed call.
            failure = QaCheckpointFailure("QA_CHECKPOINT_INACCESSIBLE")
            self.halt_error = failure
            raise failure from exc

    def raise_if_halted(self) -> None:
        if self.halt_error is not None:
            raise self.halt_error

    async def invoke(self, kind: str, request: dict[str, Any], operation):
        self.raise_if_halted()
        event: dict[str, Any] = {
            "sequence": len(self.result["attempts"]) + 1,
            "kind": kind,
            "status": "STARTED",
            "request": request,
        }
        self.result["attempts"].append(event)
        self.flush()
        started = time.monotonic()
        try:
            response = await operation()
            event["status"] = "COMPLETED"
            return response, event
        except BaseException as exc:
            if isinstance(
                exc,
                (
                    CampaignBudgetExceeded,
                    TimeoutError,
                    asyncio.CancelledError,
                    KeyboardInterrupt,
                ),
            ):
                self.halt_error = exc
            event["status"] = (
                "INTERRUPTED"
                if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
                else "FAILED"
            )
            event["errorType"] = type(exc).__name__
            raise
        finally:
            event["latencyMs"] = round((time.monotonic() - started) * 1000, 2)
            self.flush()


class _TextCapture:
    def __init__(self, provider: StructuredTextGenerationProvider, capture: _Capture):
        self.provider, self.capture = provider, capture

    async def call_with_metadata(
        self, type_name: str, data: str, schema: dict | None = None
    ):
        response, event = await self.capture.invoke(
            "TEXT",
            {"type": type_name, "data": data, "schema": schema},
            lambda: self.provider.call_with_metadata(
                type_name=type_name, data=data, schema=schema
            ),
        )
        event["response"] = asdict(response)
        self.capture.flush()
        return response


class _SpeechCapture:
    def __init__(self, provider: SpeechSynthesisProvider, capture: _Capture):
        self.provider, self.capture = provider, capture

    async def synthesize_speech(
        self, *, text: str, voice: str, language: str, speed: str
    ):
        return await self._synthesize(
            text=text,
            voice=voice,
            language=language,
            speed=speed,
            purpose="PRODUCTION_REFERENCE_AUDIO",
        )

    async def synthesize_learner(self, text: str):
        return await self._synthesize(
            text=text,
            voice="marin",
            language="ja",
            speed="NORMAL",
            purpose="QA_SYNTHETIC_LEARNER_AUDIO",
        )

    async def _synthesize(
        self, *, text: str, voice: str, language: str, speed: str, purpose: str
    ):
        request = {"text": text, "voice": voice, "language": language, "speed": speed}
        response, event = await self.capture.invoke(
            "TTS",
            {**request, "purpose": purpose},
            lambda: self.provider.synthesize_speech(**request),
        )
        path = self.capture.directory / f"tts-{event['sequence']:03d}.audio"
        path.write_bytes(response.audio_bytes)
        event["response"] = {
            "path": str(path),
            "sha256": hashlib.sha256(response.audio_bytes).hexdigest(),
            "contentType": response.content_type,
            "durationSeconds": response.duration_seconds,
            "provider": response.provider,
            "model": response.model,
        }
        self.capture.flush()
        return response


class _SttCapture:
    def __init__(self, provider: SpeakingSttProvider, capture: _Capture):
        self.provider, self.capture = provider, capture

    async def transcribe(self, wav_bytes: bytes, *, language: str, phrase_hints=None):
        response, event = await self.capture.invoke(
            "STT",
            {
                "audioSha256": hashlib.sha256(wav_bytes).hexdigest(),
                "language": language,
                "phraseHints": phrase_hints,
            },
            lambda: self.provider.transcribe(
                wav_bytes, language=language, phrase_hints=phrase_hints
            ),
        )
        event["response"] = asdict(response)
        self.capture.flush()
        return response


def _decode_reference(store, processor, assistant) -> dict[str, Any] | None:
    if assistant is None or assistant.audio is None:
        return None
    stored = store.get(assistant.audio.audio_reference)
    if stored is None:
        raise RuntimeError("QA_REFERENCE_AUDIO_NOT_AVAILABLE")
    audio = stored.path.read_bytes()
    decoded = processor.validate_and_normalize(
        audio,
        file_name=stored.path.name,
        content_type=stored.content_type,
        min_seconds=0.1,
        max_seconds=60,
        max_bytes=10 * 1024 * 1024,
    )
    return {
        "path": str(stored.path),
        "sha256": hashlib.sha256(audio).hexdigest(),
        "decodedDurationSeconds": decoded.duration_seconds,
        "boundary": "LOCAL_STORE_DOWNLOAD_EQUIVALENT_NOT_HTTP_OR_BROWSER",
    }


def _evaluation_request(session_id, mode, users, assistants, *, problem=None):
    scope = "SESSION" if problem is None else "READ_ALOUD_PROBLEM"
    identity = session_id if problem is None else f"{session_id}:read-aloud:{problem}"
    return SpeakingEvaluationRequest(
        request_id=f"{identity}:evaluate",
        idempotency_key=f"{identity}:evaluate",
        session_id=identity,
        topic="職場での予定の相談",
        practice_mode=mode,
        evaluation_scope=scope,
        target_level="B1",
        origin_language="ko",
        learning_language="ja",
        user_turns=users,
        assistant_turns=assistants,
        evaluation_policy_version=f"speaking-evaluation-policy:qa:{scope}:{problem}",
    )


async def run_speaking_campaign(
    text_provider: StructuredTextGenerationProvider,
    speech_provider: SpeechSynthesisProvider,
    output_dir: str | Path,
    stt_provider: SpeakingSttProvider | None = None,
    *,
    modes: Sequence[str] | None = None,
    sixth_conversation_turn: bool = False,
) -> dict[str, Any]:
    """Run once, serially; caller MUST inject campaign-budgeted real providers.

    No dependency factory, network client, environment mutation or manual retry.
    Missing STT stops before any billed call. Completed means execution only.
    An optional sixth GUIDED/FREE turn is a predeclared distinct case, never a
    conditional retry or duration padding. Existing 60-second eligibility remains.
    """
    selected = list(SCENARIO_PLAN) if modes is None else list(modes)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(mode not in SCENARIO_PLAN for mode in selected)
    ):
        raise ValueError(
            "QA modes must be a nonempty unique subset of the existing modes"
        )
    if not isinstance(sixth_conversation_turn, bool):
        raise ValueError("sixth_conversation_turn must be an explicit boolean")
    scenario_plan = {mode: list(SCENARIO_PLAN[mode]) for mode in selected}
    if sixth_conversation_turn:
        for mode in ("GUIDED", "FREE"):
            if mode in scenario_plan:
                scenario_plan[mode].append("PREDECLARED_FOLLOW_UP")
    text_starts = sum(
        11 if mode == "READ_ALOUD" else len(items) + 2 + int(sixth_conversation_turn)
        for mode, items in scenario_plan.items()
    )
    reference_starts = sum(
        5 if mode == "READ_ALOUD" else len(items) + 1 + int(sixth_conversation_turn)
        for mode, items in scenario_plan.items()
    )
    learner_starts = sum(len(items) for items in scenario_plan.values())
    call_estimate = {
        **CALL_ESTIMATE,
        "normalTextStarts": text_starts,
        "normalReferenceTtsStarts": reference_starts,
        "normalSyntheticLearnerTtsStarts": learner_starts,
        "normalSttStarts": learner_starts,
        "normalTotalStarts": text_starts + reference_starts + 2 * learner_starts,
        "serviceRetryUpperBoundStarts": 3
        * (text_starts + reference_starts + learner_starts)
        + learner_starts,
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "speaking-campaign-result.json").exists():
        raise ValueError(
            "Campaign artifact already exists; use a distinct case output directory"
        )
    result: dict[str, Any] = {
        "status": "RUNNING",
        "partial": True,
        "error": None,
        "modes": [],
        "attempts": [],
        "source": "SYNTHETIC_TTS_LEARNER_VOICE_NOT_HUMAN_QUALITY_EVIDENCE",
        "scenarioPlan": scenario_plan,
        "sixthConversationTurnPredeclared": sixth_conversation_turn,
        "callEstimate": call_estimate,
        "boundaries": [
            "IN_PROCESS_PRODUCTION_SERVICES",
            "NO_BE_DB_PERSISTENCE",
            "NO_HTTP_UPLOAD_AUTH_TEST",
            "NO_BROWSER_PLAYBACK_TEST",
            "INTENDED_QUALITY_LABELS_ARE_NOT_GOLD",
            "SDK_RETRIES_NOT_COUNTED_HERE",
            "WRAPPER_ATTEMPTS_ARE_NOT_ROOT_ADMITTED_COST_LEDGER",
        ],
    }
    capture = _Capture(directory, result)
    capture.flush()
    if stt_provider is None:
        result.update(
            status="BLOCKED_STT_UNAVAILABLE", error={"type": "STT_NOT_INJECTED"}
        )
        capture.flush()
        return result

    processor = SpeakingAudioProcessor()
    store = TemporaryTtsAudioStore()
    store.base_dir = directory / "reference-audio"
    store.base_dir.mkdir(exist_ok=True)
    text = _TextCapture(text_provider, capture)
    speech = _SpeechCapture(speech_provider, capture)
    turn_service = SpeakingTurnService(
        audio_processor=processor,
        stt_service=SpeakingSttService(_SttCapture(stt_provider, capture)),
        conversation_service=SpeakingConversationService(text),
        tts_service=SpeakingTtsService(speech, store),
    )
    evaluator = SpeakingEvaluationService(text)

    async def run_mode(mode_name: str, scenarios: list[str]) -> None:
        mode = SpeakingPracticeMode(mode_name)
        session_id = f"qa-speaking-{mode_name.lower()}"
        base = {
            "requestId": f"{session_id}:start",
            "idempotencyKey": f"{session_id}:start",
            "sessionId": session_id,
            "originLanguage": "ko",
            "learningLanguage": "ja",
            "topic": "職場での予定の相談",
            "practiceMode": mode.value,
            "conversationStartMode": "AI_FIRST",
            "targetLevel": "B1",
        }
        mode_result: dict[str, Any] = {
            "mode": mode.value,
            "status": "RUNNING",
            "turns": [],
            "problemEvaluations": [],
        }
        result["modes"].append(mode_result)
        started = await turn_service.start_session(
            SessionStartRequest.model_validate(base)
        )
        mode_result["opening"] = started.model_dump(mode="json", by_alias=True)
        mode_result["openingAudio"] = _decode_reference(
            store, processor, started.assistant
        )
        if started.assistant is None or started.assistant.tts_error is not None:
            raise RuntimeError("QA_OPENING_NOT_READY")
        current = started.conversation
        script = started.assistant.text
        history = [ConversationMessage(role="ASSISTANT", text=script)]
        users: list[SpeakingEvaluationTurn] = []
        assistants: list[AssistantEvaluationTurn] = []
        elapsed = 0.0
        for turn_index, scenario in enumerate(scenarios, start=1):
            problem = (
                (turn_index + 1) // 2
                if mode == SpeakingPracticeMode.READ_ALOUD
                else None
            )
            attempt = (turn_index - 1) % 2 + 1 if problem else None
            if mode != SpeakingPracticeMode.READ_ALOUD or attempt == 1:
                assistants.append(
                    AssistantEvaluationTurn(
                        turn_id=f"assistant-{turn_index}",
                        turn_index=turn_index - 1,
                        text=script,
                        script_text=current.script_text if current else None,
                        provided_facts=current.provided_facts if current else [],
                        required_intents=current.required_intents if current else [],
                        response_constraints=current.response_constraints
                        if current
                        else [],
                    )
                )
            learner_text = _learner_text(
                scenario, mode.value, script, current.provided_facts if current else []
            )
            synthetic = await speech.synthesize_learner(learner_text)
            context = ConversationGenerationRequest.model_validate(
                {
                    **base,
                    "requestId": f"{session_id}:turn:{turn_index}",
                    "idempotencyKey": f"{session_id}:turn:{turn_index}",
                    "turnIndex": turn_index,
                    "problemIndex": problem,
                    "attemptIndex": attempt,
                    "readAloudGenerateNextProblem": bool(
                        problem and problem < 5 and attempt == 2
                    ),
                    "conversationHistory": history,
                    "sessionElapsedSeconds": elapsed,
                }
            )
            response = await turn_service.process_turn(
                context=context,
                audio_bytes=synthetic.audio_bytes,
                file_name="synthetic-learner.wav",
                content_type=synthetic.content_type,
            )
            record = {
                "turnIndex": turn_index,
                "problemIndex": problem,
                "scenario": scenario,
                "intendedText": learner_text,
                "request": context.model_dump(mode="json", by_alias=True),
                "response": response.model_dump(mode="json", by_alias=True),
                "referenceAudio": _decode_reference(
                    store, processor, response.assistant
                ),
            }
            mode_result["turns"].append(record)
            capture.flush()
            if response.status != "READY" or response.transcript is None:
                mode_result["status"] = "PARTIAL"
                break
            transcript = response.transcript
            duration = transcript.metadata.audio_duration
            users.append(
                SpeakingEvaluationTurn(
                    turn_id=f"user-{turn_index}",
                    turn_index=turn_index,
                    transcript=transcript.text,
                    stt_confidence=transcript.confidence,
                    duration_seconds=duration,
                    segments=transcript.segments,
                    audio_available=True,
                    audio_quality_signals=transcript.metadata.audio_quality_signals,
                    audio_reference=f"synthetic-{session_id}-{turn_index}",
                    stt_metadata=transcript.metadata,
                    recording_revision=1,
                )
            )
            elapsed += duration
            history.append(ConversationMessage(role="USER", text=transcript.text))
            if problem and attempt == 2:
                evaluation = await evaluator.evaluate(
                    _evaluation_request(
                        session_id, mode, users[-2:], assistants[-1:], problem=problem
                    )
                )
                mode_result["problemEvaluations"].append(
                    evaluation.model_dump(mode="json", by_alias=True)
                )
            if response.assistant:
                script = response.assistant.text
                current = response.conversation
                history.append(ConversationMessage(role="ASSISTANT", text=script))
            if response.conversation and response.conversation.should_end:
                mode_result["stopReason"] = (
                    response.conversation.end_reason or "MODEL_SHOULD_END"
                )
                break

        # Explicit silence probe: it is never persisted in the positive prefix.
        silence_context = ConversationGenerationRequest.model_validate(
            {
                **base,
                "requestId": f"{session_id}:silence",
                "idempotencyKey": f"{session_id}:silence",
                "turnIndex": 1,
                "problemIndex": 1 if mode == SpeakingPracticeMode.READ_ALOUD else None,
                "attemptIndex": 1 if mode == SpeakingPracticeMode.READ_ALOUD else None,
            }
        )
        silence = await turn_service.process_turn(
            context=silence_context,
            audio_bytes=_silence_wav(),
            file_name="silence.wav",
            content_type="audio/wav",
        )
        mode_result["silenceProbe"] = silence.model_dump(mode="json", by_alias=True)
        if users:
            evaluation = await evaluator.evaluate(
                _evaluation_request(session_id, mode, users, assistants)
            )
            mode_result["sessionEvaluation"] = evaluation.model_dump(
                mode="json", by_alias=True
            )
        mode_result["acceptedUserTurns"] = len(users)
        mode_result["totalUserAudioSeconds"] = round(elapsed, 3)
        mode_result["status"] = (
            "COMPLETED" if len(users) == len(scenarios) else "PARTIAL"
        )
        capture.flush()

    try:
        for mode_name, scenarios in scenario_plan.items():
            try:
                await run_mode(mode_name, scenarios)
                # Production services can map provider errors into stage results.
                # Budget/deadline denial still ends the entire QA campaign.
                capture.raise_if_halted()
            except Exception as exc:
                capture.raise_if_halted()
                result["modes"][-1].update(
                    status="FAILED", error={"type": type(exc).__name__}
                )
                capture.flush()
        completed = all(item["status"] == "COMPLETED" for item in result["modes"])
        result.update(
            status="COMPLETED" if completed else "PARTIAL", partial=not completed
        )
    except BaseException as exc:
        terminal = (
            "INTERRUPTED"
            if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
            else "FAILED"
        )
        if result["modes"]:
            result["modes"][-1].update(
                status=terminal, error={"type": type(exc).__name__}
            )
        result.update(status=terminal, partial=True, error={"type": type(exc).__name__})
        capture.flush()
        raise
    capture.flush()
    return result
