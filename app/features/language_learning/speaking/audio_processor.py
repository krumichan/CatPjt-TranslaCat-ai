from __future__ import annotations

import io
import math
import wave
from dataclasses import dataclass

import numpy as np

from app.features.language_learning.speaking.errors import SpeakingStageException
from app.features.language_learning.speaking.policy import AUDIO_NORMALIZATION_VERSION
from app.schemas.language_learning_speaking import (
    AudioQualitySignals,
    SpeakingErrorCode,
    SpeakingStage,
)


@dataclass(frozen=True)
class NormalizedAudio:
    wav_bytes: bytes
    duration_seconds: float
    source_format: str
    quality: AudioQualitySignals
    normalization_version: str = AUDIO_NORMALIZATION_VERSION


class SpeakingAudioProcessor:
    TARGET_SAMPLE_RATE = 16_000
    TARGET_CHANNELS = 1
    SILENCE_RMS_THRESHOLD = 0.003

    def validate_and_normalize(
        self,
        audio_bytes: bytes,
        *,
        file_name: str | None,
        content_type: str | None,
        min_seconds: float,
        max_seconds: float,
        max_bytes: int,
    ) -> NormalizedAudio:
        if not audio_bytes:
            raise self._error(SpeakingErrorCode.INVALID_AUDIO, "Audio가 비어 있습니다.")
        if len(audio_bytes) > max_bytes:
            raise self._error(
                SpeakingErrorCode.AUDIO_TOO_LARGE,
                f"Audio 파일은 {max_bytes} bytes 이하여야 합니다.",
            )

        try:
            samples, sample_rate, source_format = self._decode(audio_bytes)
        except SpeakingStageException:
            raise
        except Exception as exc:
            raise self._error(
                SpeakingErrorCode.INVALID_AUDIO,
                "Audio를 해석할 수 없습니다.",
            ) from exc

        if sample_rate <= 0 or samples.size == 0:
            raise self._error(SpeakingErrorCode.INVALID_AUDIO, "유효한 Audio Frame이 없습니다.")

        duration = samples.size / sample_rate
        if duration < min_seconds:
            raise self._error(
                SpeakingErrorCode.AUDIO_TOO_SHORT,
                f"유효 Audio는 최소 {min_seconds:.1f}초 이상이어야 합니다.",
            )
        if duration > max_seconds:
            raise self._error(
                SpeakingErrorCode.AUDIO_TOO_LONG,
                f"Turn Audio는 최대 {max_seconds:.1f}초까지 허용됩니다.",
            )

        normalized = self._resample(samples, sample_rate, self.TARGET_SAMPLE_RATE)
        peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
        rms = float(math.sqrt(float(np.mean(np.square(normalized))))) if normalized.size else 0.0
        silence_ratio = float(np.mean(np.abs(normalized) < 0.002)) if normalized.size else 1.0

        if rms < self.SILENCE_RMS_THRESHOLD:
            raise self._error(
                SpeakingErrorCode.SILENCE_DETECTED,
                "음성 신호를 확인할 수 없습니다.",
            )

        wav_bytes = self._to_wav(normalized, self.TARGET_SAMPLE_RATE)
        return NormalizedAudio(
            wav_bytes=wav_bytes,
            duration_seconds=round(duration, 3),
            source_format=source_format or content_type or file_name or "unknown",
            quality=AudioQualitySignals(
                rms=round(rms, 6),
                peak=round(peak, 6),
                silence_ratio=round(silence_ratio, 6),
                sample_rate=self.TARGET_SAMPLE_RATE,
                channels=self.TARGET_CHANNELS,
            ),
        )

    def _decode(self, data: bytes) -> tuple[np.ndarray, int, str]:
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            return self._decode_wave(data)
        return self._decode_with_av(data)

    def _decode_wave(self, data: bytes) -> tuple[np.ndarray, int, str]:
        try:
            with wave.open(io.BytesIO(data), "rb") as reader:
                channels = reader.getnchannels()
                sample_width = reader.getsampwidth()
                sample_rate = reader.getframerate()
                frames = reader.readframes(reader.getnframes())
        except wave.Error as exc:
            raise self._error(
                SpeakingErrorCode.INVALID_AUDIO,
                "WAV Audio를 해석할 수 없습니다.",
            ) from exc

        if sample_width not in {1, 2, 4}:
            raise self._error(
                SpeakingErrorCode.UNSUPPORTED_AUDIO_FORMAT,
                f"지원하지 않는 WAV sample width입니다: {sample_width}",
            )

        if sample_width == 1:
            raw = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            raw = (raw - 128.0) / 128.0
        elif sample_width == 2:
            raw = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        else:
            raw = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0

        if channels > 1:
            raw = raw.reshape(-1, channels).mean(axis=1)
        return np.clip(raw, -1.0, 1.0), sample_rate, "wav"

    def _decode_with_av(self, data: bytes) -> tuple[np.ndarray, int, str]:
        try:
            import av
        except ImportError as exc:
            raise self._error(
                SpeakingErrorCode.UNSUPPORTED_AUDIO_FORMAT,
                "현재 Runtime에서 해당 Audio 형식을 변환할 수 없습니다.",
            ) from exc

        try:
            container = av.open(io.BytesIO(data), mode="r")
            stream = next((item for item in container.streams if item.type == "audio"), None)
            if stream is None:
                raise ValueError("audio stream not found")
            resampler = av.audio.resampler.AudioResampler(
                format="fltp",
                layout="mono",
                rate=self.TARGET_SAMPLE_RATE,
            )
            chunks: list[np.ndarray] = []
            for frame in container.decode(stream):
                for output in resampler.resample(frame):
                    chunks.append(output.to_ndarray().reshape(-1).astype(np.float32))
            for output in resampler.resample(None):
                chunks.append(output.to_ndarray().reshape(-1).astype(np.float32))
            if not chunks:
                raise ValueError("no decoded frames")
            samples = np.concatenate(chunks)
            return np.clip(samples, -1.0, 1.0), self.TARGET_SAMPLE_RATE, container.format.name
        except SpeakingStageException:
            raise
        except Exception as exc:
            raise self._error(
                SpeakingErrorCode.UNSUPPORTED_AUDIO_FORMAT,
                "지원하지 않거나 손상된 Audio 형식입니다.",
            ) from exc

    @staticmethod
    def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate:
            return samples.astype(np.float32, copy=False)
        source_count = samples.size
        target_count = max(1, int(round(source_count * target_rate / source_rate)))
        source_x = np.linspace(0.0, 1.0, source_count, endpoint=False)
        target_x = np.linspace(0.0, 1.0, target_count, endpoint=False)
        return np.interp(target_x, source_x, samples).astype(np.float32)

    @staticmethod
    def _to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
        pcm = np.clip(samples, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype("<i2")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            writer.writeframes(pcm16.tobytes())
        return buffer.getvalue()

    @staticmethod
    def _error(code: SpeakingErrorCode, message: str) -> SpeakingStageException:
        return SpeakingStageException(
            code=code,
            stage=SpeakingStage.AUDIO_VALIDATION,
            message=message,
            retryable=False,
        )
