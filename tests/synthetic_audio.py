"""공유 음성 경계 검사용 합성 WAV. 실제 녹음을 읽지 않는다."""

import io
import math
import wave


def make_wav(seconds: float = 1.2, *, silence: bool = False) -> bytes:
    sample_rate = 16000
    count = int(seconds * sample_rate)
    samples = bytearray()
    for index in range(count):
        value = 0 if silence else int(12000 * math.sin(2 * math.pi * 440 * index / sample_rate))
        samples += int(value).to_bytes(2, byteorder="little", signed=True)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples)
    return buffer.getvalue()
