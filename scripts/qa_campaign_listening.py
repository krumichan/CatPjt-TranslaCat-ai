"""Offline-orchestrated Listening campaign; callers own providers and paid budgets.

No provider is constructed here. Pass only campaign-budget-wrapped providers.
Artifacts contain synthetic/generated text and audio and must stay private.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
import wave

from app.ai.ports import SpeechSynthesisProvider, StructuredTextGenerationProvider
from app.features.language_learning.listening.audio_store import TemporaryListeningAudioStore
from app.features.language_learning.listening.comprehension_service import ListeningComprehensionService
from app.features.language_learning.listening.dictation_service import ListeningDictationService
from app.features.language_learning.listening.generation_service import ListeningGenerationService
from app.features.language_learning.listening.interpretation_service import ListeningInterpretationService
from app.features.language_learning.listening.summary_service import ListeningSummaryService
from app.features.language_learning.listening.tts_service import ListeningTtsService
from app.features.language_learning.speaking.stt_service import SpeakingSttProvider
from app.schemas.language_learning_listening import (
    ComprehensionEvaluationRequest,
    DictationEvaluationRequest,
    InterpretationEvaluationRequest,
    ListeningItem,
    ListeningSetGenerationRequest,
    ListeningTtsRequest,
    SummaryEvaluationRequest,
)


_MODES = ("DICTATION", "COMPREHENSION", "SUMMARY")
_ITEM_COUNT = 5
_SCENARIO_VERSION = "listening-three-mode-progressive-qa-v1"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _failure(exc: BaseException) -> dict[str, Any]:
    """Never serialize provider exception messages/headers or credentials."""
    code = getattr(exc, "code", None)
    stage = getattr(exc, "stage", None)
    return {
        "type": type(exc).__name__,
        "code": getattr(code, "value", None),
        "stage": getattr(stage, "value", None),
    }


def _history(item: ListeningItem) -> dict[str, Any]:
    metadata = item.diversity_metadata
    return {
        "sourceType": "LISTENING",
        "content": item.source_text,
        "contentHash": item.content_hash,
        "scenarioCategory": metadata.scenario_category.value if metadata else None,
        "communicativeIntent": metadata.communicative_intent.value if metadata else None,
        "taskArchetype": metadata.task_archetype if metadata else None,
        "grammarFocusCodes": metadata.grammar_focus_codes if metadata else [],
        "semanticSummary": metadata.semantic_summary if metadata else None,
        "ageDays": 0,
    }


def _generation_request(
    mode: str,
    order: int,
    prefix: list[ListeningItem],
    previous_modes: list[ListeningItem],
) -> ListeningSetGenerationRequest:
    identifier = f"qa-listening-{mode.lower()}-{order}"
    return ListeningSetGenerationRequest.model_validate({
        "requestId": identifier,
        "idempotencyKey": identifier,
        "userContext": {
            "originLanguage": "ko", "learningLanguage": "ja", "level": "MY_LEVEL",
            "profileFocus": ["LISTENING_RECOGNITION", "MEANING"],
        },
        "setContext": {
            "learningDate": "2026-09-19", "learningMode": mode,
            "topic": {"id": "synthetic-daily", "title": "일상 일정과 이용 안내"},
            "selectedKeywords": [
                {"key": "schedule", "text": "予定", "source": "SYSTEM", "type": "TOPIC"},
                {"key": "confirm", "text": "確認", "source": "CUSTOM", "type": "VOCABULARY"},
            ],
            # BE owns the global slot; AI returns local itemIndex=1.
            "itemCount": 1, "difficulty": "MY_LEVEL",
        },
        "constraints": {
            "audioSecondsMin": 1, "audioSecondsMax": 30,
            "recentContentHashes": [item.content_hash for item in previous_modes + prefix],
            "recentSimilaritySummaries": [],
        },
        "languageComplexity": {"baseComplexityBand": 3, "targetComplexityBand": 3},
        "diversityContext": {
            "currentSession": [_history(item) for item in prefix],
            "sameFeatureRecent": [_history(item) for item in previous_modes],
            "exactContentHashes90d": [item.content_hash for item in previous_modes + prefix],
        },
        "contentDiversityPolicyVersion": "language-learning-diversity",
        "policyVersion": "listening", "modelConfigVersion": "listening-model-config",
    })


def _evaluation_base(identifier: str, variant: str) -> dict[str, Any]:
    return {
        "requestId": f"{identifier}-{variant}",
        "idempotencyKey": f"{identifier}-{variant}",
        "itemId": identifier, "attemptId": f"{identifier}-{variant}",
        "evaluationPurpose": "OFFICIAL", "answerRevealed": False,
        "assistanceUsage": [], "policyVersion": "listening-profile",
        "modelConfigVersion": "listening-model-config",
    }


def _evaluation_cases(mode: str, item: ListeningItem) -> list[tuple[str, str, str]]:
    """Construct known-reference probes, not independent human gold labels."""
    if mode == "COMPREHENSION":
        assert item.correct_option_key is not None
        wrong = next(option.key for option in item.options if option.key != item.correct_option_key)
        return [("COMPREHENSION", "correct", item.correct_option_key),
                ("COMPREHENSION", "wrong", wrong)]
    if mode == "DICTATION":
        return [
            ("DICTATION", "correct", item.source_text),
            ("DICTATION", "partial", item.source_text[:max(1, len(item.source_text) // 2)]),
            ("DICTATION", "wrong", "月の図書館で青いチーズを売っています。"),
            ("INTERPRETATION", "correct", item.reference_meanings[0]),
            ("INTERPRETATION", "partial", item.key_meaning_units[0]),
            ("INTERPRETATION", "wrong", "달의 도서관에서 파란 치즈를 판매한다는 내용입니다."),
        ]
    return [
        ("SUMMARY", "correct", item.source_text),
        ("SUMMARY", "partial", item.summary_key_points[0]),
        ("SUMMARY", "wrong", "月の図書館で青いチーズを売っています。"),
    ]


def _evaluation_request(
    task: str, item: ListeningItem, identifier: str, variant: str, answer: str,
):
    base = _evaluation_base(f"{identifier}-{task.lower()}", variant)
    if task == "DICTATION":
        return DictationEvaluationRequest.model_validate({
            **base, "sourceText": item.source_text, "answer": answer, "learningLanguage": "ja",
        })
    if task == "INTERPRETATION":
        return InterpretationEvaluationRequest.model_validate({
            **base, "sourceText": item.source_text, "answer": answer,
            "learningLanguage": "ja", "originLanguage": "ko",
            "referenceMeanings": item.reference_meanings, "keyMeaningUnits": item.key_meaning_units,
        })
    if task == "COMPREHENSION":
        return ComprehensionEvaluationRequest.model_validate({
            **base, "question": item.question,
            "options": [option.model_dump(mode="json", by_alias=True) for option in item.options],
            "selectedOptionKey": answer, "correctOptionKey": item.correct_option_key,
            "comprehensionFocus": item.comprehension_focus,
            "originLanguage": "ko", "learningLanguage": "ja",
        })
    return SummaryEvaluationRequest.model_validate({
        **base, "sourceText": item.source_text, "summaryKeyPoints": item.summary_key_points,
        "answer": answer, "originLanguage": "ko", "learningLanguage": "ja",
    })


def _audio_check(audio_bytes: bytes, expected_checksum: str, duration_ms: int) -> dict[str, Any]:
    """The production speech provider returns WAV; inspect actual frames, not labels."""
    with wave.open(io.BytesIO(audio_bytes), "rb") as reader:
        frames = reader.getnframes()
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        decoded = reader.readframes(frames)
    measured_duration = frames / sample_rate
    checksum_matches = hashlib.sha256(audio_bytes).hexdigest() == expected_checksum
    complete_frames = len(decoded) == frames * channels * sample_width and frames > 0
    duration_matches = abs(measured_duration * 1000 - duration_ms) <= 50
    return {
        "status": "PASSED" if checksum_matches and complete_frames and duration_matches else "FAILED",
        "checksumMatches": checksum_matches, "completeFrames": complete_frames,
        "durationMatchesMetadata": duration_matches,
        "durationSeconds": measured_duration, "sampleRate": sample_rate,
        "channels": channels, "sampleWidth": sample_width, "frames": frames,
        "withinRequestedEstimatedRange": 8 <= measured_duration <= 20,
        "humanPlaybackReview": "NOT_RUN",
    }


async def run_listening_campaign(
    text_provider: StructuredTextGenerationProvider,
    speech_provider: SpeechSynthesisProvider,
    output_dir: Path | str,
    stt_provider: SpeakingSttProvider | None = None,
    *,
    modes: tuple[str, ...] = _MODES,
) -> dict[str, Any]:
    """Run sequentially with supplied budget-wrapped providers; never close them.

    Normal path: 15 generation + 15 TTS + 30 semantic evaluations = 60 starts.
    Existing services may use their configured retries (at most 180 starts at
    default limits). Optional STT adds 15 reference-roundtrip calls, not human
    pronunciation evidence. The root budget wrapper can stop before any start.
    No harness retries: failure stops this campaign, preserving every prefix.
    """
    if not modes or len(set(modes)) != len(modes) or any(mode not in _MODES for mode in modes):
        raise ValueError("Only unique declared Listening modes are allowed")
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / "listening-campaign.json"
    if artifact.exists():
        raise FileExistsError("Listening campaign output already exists; use a new approved run directory")
    started = time.perf_counter()
    result: dict[str, Any] = {
        "scenarioVersion": _SCENARIO_VERSION, "createdAt": datetime.now(UTC).isoformat(),
        "status": "RUNNING", "partial": True, "error": None,
        "targetItemCount": len(modes) * _ITEM_COUNT,
        "completionScope": "SCHEDULED_EXECUTION_ONLY_NOT_CONTENT_QUALITY_CERTIFICATION",
        "sttStatus": "SCHEDULED_SYNTHETIC_REFERENCE_ONLY" if stt_provider else "NOT_RUN",
        "dataOrigin": "SYNTHETIC_NONPRIVATE_PROFILE_AND_NEW_GENERATED_CONTENT",
        "providerOwnership": "CALLER_SUPPLIED_BUDGET_WRAPPERS; NEVER_CLOSED_BY_SCENARIO",
        "expectedStageCalls": {"generation": len(modes) * 5, "tts": len(modes) * 5,
                               "interpretation": 15 if "DICTATION" in modes else 0,
                               "summary": 15 if "SUMMARY" in modes else 0,
                               "optionalStt": len(modes) * 5 if stt_provider else 0},
        "qualityBoundary": (
            "Reference-derived answer variants are evaluation probes, not independent gold. "
            "Comprehension has no partial-credit answer. Summary full-source copying probes "
            "content coverage, not concise-summary quality. Band metadata is not acoustic "
            "difficulty proof. No DB/API/UI or human playback/recording tested."
        ),
        "modes": [
            {"mode": mode, "status": "NOT_STARTED", "targetItemCount": _ITEM_COUNT,
             "items": [], "partialAnswerApplicable": mode != "COMPREHENSION"}
            for mode in modes
        ],
    }

    def checkpoint() -> None:
        result["totalMs"] = round((time.perf_counter() - started) * 1000, 3)
        result["completedItemCount"] = sum(
            item["status"] == "COMPLETED"
            for mode_result in result["modes"] for item in mode_result["items"]
        )
        _write_json(artifact, result)

    generation = ListeningGenerationService(text_provider)
    evaluators: dict[str, Any] = {
        "DICTATION": ListeningDictationService(),
        "COMPREHENSION": ListeningComprehensionService(),
        "INTERPRETATION": ListeningInterpretationService(text_provider),
        "SUMMARY": ListeningSummaryService(text_provider),
    }
    store = TemporaryListeningAudioStore()
    store.base_dir = directory / "audio"
    store.base_dir.mkdir()
    retained_audio_dir = directory / "captured-audio"
    retained_audio_dir.mkdir()
    tts = ListeningTtsService(speech_provider, store)
    previous_modes: list[ListeningItem] = []
    active: dict[str, Any] | None = None
    mode_result: dict[str, Any] | None = None
    checkpoint()
    try:
        for mode_result in result["modes"]:
            assert mode_result is not None
            mode = mode_result["mode"]
            mode_result["status"] = "RUNNING"
            prefix: list[ListeningItem] = []
            for order in range(1, _ITEM_COUNT + 1):
                identifier = f"qa-listening-{mode.lower()}-{order}"
                active = {"order": order, "status": "RUNNING", "stages": []}
                mode_result["items"].append(active)
                request = _generation_request(mode, order, prefix, previous_modes)
                stage: dict[str, Any] = {
                    "stage": "GENERATION", "status": "STARTED",
                    "request": request.model_dump(mode="json", by_alias=True),
                }
                active["stages"].append(stage)
                checkpoint()
                response = await generation.generate(request)
                stage.update(status="COMPLETED", response=response.model_dump(mode="json", by_alias=True))
                checkpoint()
                if len(response.items) != 1 or response.items[0].item_index != 1:
                    raise ValueError("Progressive Listening generation must return one local itemIndex=1")
                item = response.items[0]
                prefix.append(item)
                tts_request = ListeningTtsRequest.model_validate({
                    "requestId": identifier + "-tts", "idempotencyKey": identifier + "-tts",
                    "itemId": identifier, "sourceText": item.source_text,
                    "contentHash": item.content_hash, "generationVersion": response.generation_version,
                    "learningLanguage": "ja",
                    "voice": {"locale": "ja", "voiceKey": "marin", "version": "openai-speech-v1"},
                })
                stage = {"stage": "TTS", "status": "STARTED",
                         "request": tts_request.model_dump(mode="json", by_alias=True)}
                active["stages"].append(stage)
                checkpoint()
                tts_response = await tts.synthesize(tts_request)
                stage.update(status=tts_response.status,
                             response=tts_response.model_dump(mode="json", by_alias=True))
                checkpoint()
                if tts_response.status != "READY" or tts_response.audio is None:
                    raise RuntimeError("Listening TTS did not produce READY audio")
                audio = store.get(tts_response.audio.audio_reference)
                if audio is None:
                    raise RuntimeError("Listening READY audio missing from store")
                audio_bytes = audio.path.read_bytes()
                # Keep diagnosis evidence beyond the production store's TTL.
                retained_audio = retained_audio_dir / f"{identifier}.wav"
                with retained_audio.open("xb") as handle:
                    handle.write(audio_bytes)
                stage["audioPath"] = str(retained_audio)
                try:
                    stage["audioCheck"] = _audio_check(
                        audio_bytes, tts_response.audio.checksum, tts_response.audio.duration_ms
                    )
                except Exception as exc:
                    stage["audioCheck"] = {"status": "FAILED", "error": _failure(exc)}
                    raise
                checkpoint()
                if stage["audioCheck"]["status"] != "PASSED":
                    raise ValueError("Listening decoded audio integrity check failed")
                if stt_provider is not None:
                    from app.features.language_learning.speaking.audio_processor import SpeakingAudioProcessor

                    normalized = SpeakingAudioProcessor().validate_and_normalize(
                        audio_bytes, file_name="reference.wav", content_type="audio/wav",
                        min_seconds=0.1, max_seconds=60, max_bytes=10 * 1024 * 1024,
                    )
                    stage = {"stage": "SYNTHETIC_REFERENCE_STT", "status": "STARTED"}
                    active["stages"].append(stage)
                    checkpoint()
                    transcription = await stt_provider.transcribe(normalized.wav_bytes, language="ja")
                    stage.update(status="COMPLETED", response=asdict(transcription))
                    checkpoint()
                for task, variant, answer in _evaluation_cases(mode, item):
                    evaluation_request = _evaluation_request(task, item, identifier, variant, answer)
                    stage = {"stage": task, "variant": variant, "status": "STARTED",
                             "request": evaluation_request.model_dump(mode="json", by_alias=True)}
                    active["stages"].append(stage)
                    checkpoint()
                    evaluation = await evaluators[task].evaluate(evaluation_request)
                    stage.update(status="COMPLETED",
                                 response=evaluation.model_dump(mode="json", by_alias=True))
                    checkpoint()
                active["status"] = "COMPLETED"
                checkpoint()
            previous_modes.extend(prefix)
            mode_result["status"] = "COMPLETED"
            checkpoint()
        result.update(status="COMPLETED", partial=False)
    except BaseException as exc:
        interrupted = isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
        result.update(status="INTERRUPTED" if interrupted else "FAILED", error=_failure(exc))
        if mode_result is not None:
            mode_result["status"] = result["status"]
        if active is not None:
            active["status"] = result["status"]
            if active["stages"] and active["stages"][-1]["status"] == "STARTED":
                active["stages"][-1].update(status=result["status"], error=_failure(exc))
        checkpoint()
        if interrupted:
            raise
    checkpoint()
    return result
