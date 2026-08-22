from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from websockets.asyncio.client import connect


@dataclass(frozen=True)
class RunResult:
    first_partial_ms: float | None
    speech_end_to_completed_ms: float
    ai_total_after_speech_ms: int | None
    stt_version: str | None
    translation_version: str | None


async def run_once(
    *,
    url: str,
    api_key: str,
    pcm_bytes: bytes,
    frame_duration_ms: int,
    endpointing_silence_ms: int,
    source_language_mode: str,
    manual_source_language: str | None,
    target_language: str,
    timeout_seconds: float,
) -> RunResult:
    bytes_per_frame = 16 * frame_duration_ms * 2
    if len(pcm_bytes) % bytes_per_frame != 0:
        raise ValueError(
            "PCM fixture byte length must be an exact multiple of one frame"
        )

    session_id = str(uuid4())
    stream_open = {
        "type": "STREAM_OPEN",
        "requestId": str(uuid4()),
        "sessionId": session_id,
        "channel": "SELF",
        "mode": "MIC",
        "sourceLanguageMode": source_language_mode,
        "manualSourceLanguage": manual_source_language,
        "targetLanguage": target_language,
        "audioFormat": {
            "encoding": "PCM_S16LE",
            "sampleRate": 16000,
            "channels": 1,
            "frameDurationMs": frame_duration_ms,
        },
        "policy": {
            "endpointingSilenceMs": endpointing_silence_ms,
            "minUtteranceDurationMs": 250,
            "maxUtteranceDurationMs": 10000,
            "languageLockConfidence": 0.80,
            "languageSwitchConfidence": 0.85,
            "languageSwitchConsecutiveCount": 3,
        },
    }

    async with connect(
        url,
        additional_headers={"X-API-KEY": api_key},
        max_size=2 * 1024 * 1024,
    ) as websocket:
        await websocket.send(json.dumps(stream_open))
        ready = json.loads(await asyncio.wait_for(websocket.recv(), timeout_seconds))
        if ready.get("type") != "STREAM_READY":
            raise RuntimeError("Voice stream did not become ready")

        first_audio_at = time.perf_counter()
        first_partial_at: float | None = None
        completed_event: dict[str, Any] | None = None

        async def receive_completed() -> None:
            nonlocal first_partial_at, completed_event
            while completed_event is None:
                event = json.loads(await websocket.recv())
                event_type = event.get("type")
                if event_type == "TRANSCRIPT_PARTIAL" and first_partial_at is None:
                    first_partial_at = time.perf_counter()
                elif event_type == "VOICE_PIPELINE_FAILED":
                    error = event.get("error") or {}
                    raise RuntimeError(
                        f"Voice pipeline failed: {error.get('code', 'unknown')}"
                    )
                elif event_type == "VOICE_PIPELINE_COMPLETED":
                    completed_event = event

        receiver = asyncio.create_task(receive_completed())
        frame_seconds = frame_duration_ms / 1000
        for offset in range(0, len(pcm_bytes), bytes_per_frame):
            await websocket.send(pcm_bytes[offset : offset + bytes_per_frame])
            await asyncio.sleep(frame_seconds)

        speech_ended_at = time.perf_counter()
        silence_frame = bytes(bytes_per_frame)
        silence_frames = math.ceil(endpointing_silence_ms / frame_duration_ms)
        for _ in range(silence_frames):
            await websocket.send(silence_frame)
            await asyncio.sleep(frame_seconds)

        await asyncio.wait_for(receiver, timeout=timeout_seconds)
        completed_at = time.perf_counter()
        assert completed_event is not None

        await websocket.send(
            json.dumps({"type": "STREAM_CLOSE", "reason": "BENCHMARK_COMPLETE"})
        )
        while True:
            event = json.loads(
                await asyncio.wait_for(websocket.recv(), timeout_seconds)
            )
            if event.get("type") == "STREAM_CLOSED":
                break

        latency = completed_event.get("latency") or {}
        model = completed_event.get("model") or {}
        return RunResult(
            first_partial_ms=(
                (first_partial_at - first_audio_at) * 1000
                if first_partial_at is not None
                else None
            ),
            speech_end_to_completed_ms=(completed_at - speech_ended_at) * 1000,
            ai_total_after_speech_ms=latency.get("aiTotalAfterSpeechMs"),
            stt_version=model.get("sttVersion"),
            translation_version=model.get("translationVersion"),
        )


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get("VOICE_BENCHMARK_API_KEY", "")
    if not api_key:
        raise ValueError("VOICE_BENCHMARK_API_KEY environment variable is required")
    pcm_bytes = args.pcm_file.read_bytes()
    if not pcm_bytes:
        raise ValueError("PCM fixture is empty")

    results: list[RunResult] = []
    for _ in range(args.iterations):
        batch = await asyncio.gather(
            *(
                run_once(
                    url=args.url,
                    api_key=api_key,
                    pcm_bytes=pcm_bytes,
                    frame_duration_ms=args.frame_duration_ms,
                    endpointing_silence_ms=args.endpointing_silence_ms,
                    source_language_mode=("MANUAL" if args.source_language else "AUTO"),
                    manual_source_language=args.source_language,
                    target_language=args.target_language,
                    timeout_seconds=args.timeout_seconds,
                )
                for _ in range(args.concurrency)
            )
        )
        results.extend(batch)

    partial_values = [
        result.first_partial_ms
        for result in results
        if result.first_partial_ms is not None
    ]
    completion_values = [result.speech_end_to_completed_ms for result in results]
    first = results[0]
    return {
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "cpuCount": os.cpu_count(),
            "providerRegion": args.provider_region,
            "url": args.url,
            "sttVersion": first.stt_version,
            "translationVersion": first.translation_version,
            "frameDurationMs": args.frame_duration_ms,
            "concurrency": args.concurrency,
            "iterations": args.iterations,
        },
        "firstPartialMs": summarize(partial_values),
        "speechEndToCompletedMs": summarize(completion_values),
        "runs": [asdict(result) for result in results],
    }


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "mean": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50": round(percentile(ordered, 0.50), 2),
        "p95": round(percentile(ordered, 0.95), 2),
        "p99": round(percentile(ordered, 0.99), 2),
        "mean": round(statistics.fmean(ordered), 2),
    }


def percentile(ordered_values: list[float], quantile: float) -> float:
    if not ordered_values:
        raise ValueError("At least one value is required")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    if len(ordered_values) == 1:
        return ordered_values[0]
    position = (len(ordered_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered_values[lower_index]
    weight = position - lower_index
    return (
        ordered_values[lower_index] * (1 - weight)
        + ordered_values[upper_index] * weight
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Voice Translation V2 over its internal WebSocket",
    )
    parser.add_argument(
        "pcm_file",
        type=Path,
        help="16kHz mono PCM_S16LE fixture containing speech only",
    )
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8000/internal/v1/voice/streams",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--frame-duration-ms", type=int, default=100)
    parser.add_argument("--endpointing-silence-ms", type=int, default=300)
    parser.add_argument("--source-language", choices=["ko", "ja", "en"])
    parser.add_argument("--target-language", choices=["ko", "ja", "en"], required=True)
    parser.add_argument("--provider-region", default="unknown")
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    args = parser.parse_args()
    if args.iterations < 1 or args.concurrency < 1:
        parser.error("iterations and concurrency must be positive")
    if args.frame_duration_ms not in {20, 100, 200}:
        parser.error("frame-duration-ms must be 20, 100, or 200")
    return args


if __name__ == "__main__":
    print(
        json.dumps(
            asyncio.run(run_benchmark(parse_args())),
            ensure_ascii=False,
            indent=2,
        )
    )
