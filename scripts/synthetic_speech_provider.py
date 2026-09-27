"""로컬 이행 검사에서만 사용하는 합성 오디오·전사 출력."""
import asyncio
import io
import json
import math
import os
import struct
import threading
import wave
from pathlib import Path

from app.ai.ports import SpeechSynthesisResult
from app.features.speech_to_text.runtime import WhisperRuntimeResult, WhisperRuntimeSegment
from scripts.listening_test_provider import listening_fixture_duration

ROOT = Path(__file__).resolve().parents[1]
_stats_lock = threading.Lock()


async def _speaking_control(stage):
    # 테스트 명시 표식이 있는 동안에만 합성 음성 출력과 호출 수를 제어한다.
    path = ROOT / ".tmp_ktor_m0/speaking-control.json"
    control = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    if not control.get("speechEnabled"):
        return {}
    stats_path = ROOT / ".tmp_ktor_m0/speaking-speech-stats.json"
    with _stats_lock:
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
        stats[stage] = stats.get(stage, 0) + 1
        stats_path.write_text(json.dumps(stats), encoding="utf-8")
    scenario = control.get("scenario") if control.get("stage") in (None, stage) else None
    if scenario == "hold":
        (ROOT / ".tmp_ktor_m0/speaking-provider-held.json").write_text(
            json.dumps({"stage": stage, "held": True}), encoding="utf-8"
        )
        await asyncio.sleep(60)
        raise TimeoutError()
    if scenario == f"{stage.lower()}_failure":
        error = RuntimeError("Synthetic speech unavailable")
        error.status_code = 503
        raise error
    return control


class SyntheticSpeechProvider:
    ready = True

    def __init__(self):
        if os.environ.get("TRANSLACAT_TEST_MODEL_EXECUTION") != "1":
            raise RuntimeError("Synthetic speech requires explicit local test marker")

    async def synthesize_speech(self, *, text, voice, language, speed):
        await _speaking_control("TTS")
        # 실제 HTTP 오디오 저장·재생 경로에 넣을 PCM만 만들며 유료 SDK를 호출하지 않는다.
        buffer = io.BytesIO()
        duration = listening_fixture_duration(text) or 4.0
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            frames = (
                struct.pack("<h", int(3000 * math.sin(index * 0.05)))
                for index in range(int(16000 * duration))
            )
            writer.writeframes(b"".join(frames))
        return SpeechSynthesisResult(
            buffer.getvalue(), "audio/wav", "test-provider", "synthetic-tts", duration,
        )

    async def transcribe(self, audio, **kwargs):
        control = await _speaking_control("STT")
        transcript = "Synthetic transcript"
        if control:
            transcript = (
                "I went to the park yesterday."
                if control.get("practiceMode") == "READ_ALOUD"
                else "Yesterday I went to the park with my friend. We walked around the lake "
                "and talked about our plans for the weekend. After that we found a quiet cafe "
                "and ordered some tea. I enjoyed the fresh air and I would like to go there "
                "again next week."
            )
        return WhisperRuntimeResult(
            transcript, "en", 0.99, 4.0,
            [WhisperRuntimeSegment(0, 4.0, transcript, -0.1, 0.01)],
            "test-provider", "synthetic-stt", "fixture-v1",
        )

    async def has_speech(self, pcm):
        await _speaking_control("EVIDENCE")
        return bool(pcm)
