from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ListeningAcousticEvidence:
    clipping_ratio: float
    zero_crossing_rate: float
    noise_score: float
    clipping_score: float


def analyze_normalized_wav(wav_bytes: bytes) -> ListeningAcousticEvidence | None:
    """Return bounded acoustic evidence without retaining or logging raw audio."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
            if reader.getsampwidth() != 2:
                return None
            channels = reader.getnchannels()
            raw = np.frombuffer(
                reader.readframes(reader.getnframes()),
                dtype="<i2",
            ).astype(np.float32)
        if raw.size < 2:
            return None
        if channels > 1:
            raw = raw.reshape(-1, channels).mean(axis=1)
        samples = np.clip(raw / 32768.0, -1.0, 1.0)
        clipping_ratio = float(np.mean(np.abs(samples) >= 0.98))
        zero_crossing_rate = float(np.mean(samples[:-1] * samples[1:] < 0))

        # Sustained, near-random high-frequency crossings are treated as a noise risk.
        noise_risk = max(0.0, min(1.0, (zero_crossing_rate - 0.20) / 0.30))
        noise_score = 1.0 - noise_risk
        clipping_score = 1.0 - max(0.0, min(1.0, clipping_ratio / 0.05))
        return ListeningAcousticEvidence(
            clipping_ratio=round(clipping_ratio, 6),
            zero_crossing_rate=round(zero_crossing_rate, 6),
            noise_score=round(noise_score, 6),
            clipping_score=round(clipping_score, 6),
        )
    except (ValueError, wave.Error):
        return None
