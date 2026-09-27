from __future__ import annotations

import io
import math
import wave
from dataclasses import dataclass

import numpy as np


class AudioDecodeError(ValueError):
    pass


@dataclass(frozen=True)
class DecodedAudio:
    wav_bytes: bytes
    duration_seconds: float
    source_format: str
    rms: float
    peak: float
    silence_ratio: float
    sample_rate: int = 16000
    channels: int = 1


class AudioDecoder:
    TARGET_SAMPLE_RATE = 16000

    def decode(self, data: bytes) -> DecodedAudio:
        # 디코딩 한도만 확인한다. 발화 최소 길이·음량·학습 합격 여부는 호출자가 판정한다.
        if not data or len(data) > 10 * 1024 * 1024:
            raise AudioDecodeError("AUDIO_BYTES_INVALID")
        try:
            if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
                samples, rate, source_format = self._wave(data)
            else:
                samples, rate, source_format = self._av(data)
            if rate <= 0 or samples.size == 0:
                raise AudioDecodeError("AUDIO_FRAMES_INVALID")
            duration = samples.size / rate
            if duration > 600:
                raise AudioDecodeError("AUDIO_DECODE_LIMIT_EXCEEDED")

            # 기존 PCM 변환과 원시 품질 측정 수식을 그대로 보존한다.
            normalized = self._resample(samples, rate)
            peak = float(np.max(np.abs(normalized)))
            rms = float(math.sqrt(float(np.mean(np.square(normalized)))))
            silence = float(np.mean(np.abs(normalized) < 0.002))
            return DecodedAudio(
                self._wav(normalized),
                duration,
                source_format,
                rms,
                peak,
                silence,
            )
        except AudioDecodeError:
            raise
        except Exception as exc:
            raise AudioDecodeError("AUDIO_DECODE_FAILED") from exc

    @staticmethod
    def _wave(data: bytes) -> tuple[np.ndarray, int, str]:
        with wave.open(io.BytesIO(data), "rb") as reader:
            channels, width, rate = (
                reader.getnchannels(),
                reader.getsampwidth(),
                reader.getframerate(),
            )
            frame_count = reader.getnframes()
            frames = reader.readframes(frame_count)
        if width not in {1, 2, 4} or channels < 1 or len(frames) != frame_count * channels * width:
            raise AudioDecodeError("AUDIO_FORMAT_INVALID")
        if width == 1:
            raw = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif width == 2:
            raw = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        else:
            raw = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
        if channels > 1:
            raw = raw.reshape(-1, channels).mean(axis=1)
        return np.clip(raw, -1.0, 1.0), rate, "wav"

    def _av(self, data: bytes) -> tuple[np.ndarray, int, str]:
        import av
        from av.audio.frame import AudioFrame
        from av.audio.resampler import AudioResampler

        # 해독 중에도 출력 frame 수를 제한해서 압축 데이터 팽창을 제한한다.
        with av.open(io.BytesIO(data), mode="r") as container:
            stream = next((item for item in container.streams if item.type == "audio"), None)
            if stream is None:
                raise AudioDecodeError("AUDIO_STREAM_MISSING")
            resampler = AudioResampler(
                format="fltp", layout="mono", rate=self.TARGET_SAMPLE_RATE
            )
            chunks: list[np.ndarray] = []
            count = 0
            for frame in container.decode(stream):
                if not isinstance(frame, AudioFrame):
                    raise AudioDecodeError("AUDIO_FRAME_TYPE_INVALID")
                for output in resampler.resample(frame):
                    chunk = output.to_ndarray().reshape(-1).astype(np.float32)
                    count += chunk.size
                    if count > 600 * self.TARGET_SAMPLE_RATE:
                        raise AudioDecodeError("AUDIO_DECODE_LIMIT_EXCEEDED")
                    chunks.append(chunk)
            for output in resampler.resample(None):
                chunks.append(output.to_ndarray().reshape(-1).astype(np.float32))
            if not chunks:
                raise AudioDecodeError("AUDIO_FRAMES_INVALID")
            return (
                np.clip(np.concatenate(chunks), -1.0, 1.0),
                self.TARGET_SAMPLE_RATE,
                container.format.name,
            )

    def _resample(self, samples: np.ndarray, rate: int) -> np.ndarray:
        if rate == self.TARGET_SAMPLE_RATE:
            return samples.astype(np.float32, copy=False)
        target_count = max(1, int(round(samples.size * self.TARGET_SAMPLE_RATE / rate)))
        source_x = np.linspace(0.0, 1.0, samples.size, endpoint=False)
        target_x = np.linspace(0.0, 1.0, target_count, endpoint=False)
        return np.interp(target_x, source_x, samples).astype(np.float32)

    def _wav(self, samples: np.ndarray) -> bytes:
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
        output = io.BytesIO()
        with wave.open(output, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(self.TARGET_SAMPLE_RATE)
            writer.writeframes(pcm16.tobytes())
        return output.getvalue()
