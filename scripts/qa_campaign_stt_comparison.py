"""Explicit, manifest-fixed local STT comparison. No text/TTS API calls.

Private artifacts retain synthetic audio/transcripts. Run each of the two fixed
configurations once in its own process; reference text is never a model hint.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unicodedata
import wave

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.qa_campaign_budget import CampaignLedger, Reservation, atomic_json


def normalized_text(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", value)
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


def edit_distance(expected: str, observed: str) -> int:
    row = list(range(len(observed) + 1))
    for i, a in enumerate(expected, 1):
        new = [i]
        for j, b in enumerate(observed, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = new
    return row[-1]


def _wave(samples, rate: int = 16000) -> bytes:
    import numpy as np
    stream = io.BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return stream.getvalue()


async def run(output: Path, config_name: str, revision: int = 1,
              baseline_runtime_preflight: Path | None = None) -> None:
    import numpy as np
    import psutil
    from faster_whisper.utils import download_model
    from app.features.language_learning.speaking.audio_processor import SpeakingAudioProcessor
    from app.features.language_learning.speaking.stt_service import FasterWhisperSpeakingSttProvider
    from app.features.speech_to_text import FasterWhisperRuntime
    from scripts.qa_campaign_audio_budget import BudgetedSttProvider

    manifest = json.loads((output / "campaign-manifest.json").read_text(encoding="utf-8"))
    spec = manifest["stt"]
    config = next(x for x in spec["configs"] if x["name"] == config_name)
    cpu_threads = spec["cpuThreads"]
    baseline_provenance = None
    if baseline_runtime_preflight is not None:
        baseline_provenance = json.loads(baseline_runtime_preflight.read_text(encoding="utf-8"))
        if (config_name != "base" or baseline_provenance["configuredModel"] != "base"
                or baseline_provenance["device"] != spec["device"]
                or baseline_provenance["computeType"] != spec["computeType"]):
            raise ValueError("Runtime preflight must describe the existing baseline, not a third candidate")
        cpu_threads = baseline_provenance["cpuThreads"]
    if len(spec["configs"]) != 2 or config_name not in {"base", "small"}:
        raise ValueError("Only the predeclared two configurations are authorized")
    suffix = "" if revision == 1 else f"-r{revision}"
    report_path = output / f"stt-comparison-{config_name}{suffix}.json"
    if report_path.exists():
        raise ValueError("This fixed configuration already has a result; never overwrite or repeat")
    ledger = CampaignLedger(output / "budget-ledger.json", manifest["campaignId"])
    clips = []
    processor = SpeakingAudioProcessor()
    for source in spec["clips"]:
        raw = Path(source["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError("Source audio fingerprint mismatch")
        normalized = processor.validate_and_normalize(
            raw, file_name="qa.wav", content_type="audio/wav",
            min_seconds=0.1, max_seconds=60, max_bytes=15 * 1024 * 1024)
        clips.append(({**source}, normalized.wav_bytes))
    random = np.random.default_rng(570001)
    for label, samples in (("silence-2s", np.zeros(32000)),
                           ("noise-seed570001-2s", random.normal(0, 0.02, 32000))):
        clips.append(({"id":label, "expectedText":"", "language":"ja",
                       "goldStatus":"synthetic no-speech input"}, _wave(samples)))
    source, audio = clips[2]
    with wave.open(io.BytesIO(audio), "rb") as reader:
        samples = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2").astype(float) / 32768
    noise = random.normal(0, np.sqrt(np.mean(samples ** 2)) / (10 ** (15 / 20)), len(samples))
    clips.append(({**source, "id":"old-synthetic-014-noise-15dB", "derived":True}, _wave(samples + noise)))
    report = {"configuration":config, "status":"PREPARING", "partial":True, "clips":[],
              "revision":revision,
              "cpuThreads":cpu_threads, "baselineRuntimePreflight":baseline_provenance,
              "sourceSha256":{name:hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
                              for name in ["app/features/language_learning/speaking/stt_service.py",
                                           "app/features/voice_translation/speech_detector.py", "scripts/qa_campaign_stt_comparison.py"]},
              "manifestSha256":hashlib.sha256((output/'campaign-manifest.json').read_bytes()).hexdigest(),
              "referenceNeverUsedAsHint":True, "clipCount":len(clips),
              "goldLimitation":"TTS input is a proxy, not verified human transcription; no kana/kanji collapsing."}
    atomic_json(report_path, report)
    # Only an explicitly selected local-model download; no new provider account.
    model_path = await asyncio.to_thread(download_model, config_name)
    report["modelSnapshot"] = str(model_path)
    report["modelConfigSha256"] = hashlib.sha256((Path(model_path)/'config.json').read_bytes()).hexdigest()

    class ComparisonRuntime(FasterWhisperRuntime):
        async def transcribe(self, audio, *, options, priority=5):
            return await super().transcribe(audio, options={**options, "beam_size":config["beam"]}, priority=priority)

    runtime = ComparisonRuntime(model_name=str(model_path), model_revision="",
                                device=spec["device"], compute_type=spec["computeType"],
                                cpu_threads=cpu_threads, num_workers=1,
                                max_concurrency=1, run_warm_up_inference=False)
    process = psutil.Process()
    peak = process.memory_info().rss
    async def monitor():
        nonlocal peak
        while True:
            peak = max(peak, process.memory_info().rss)
            await asyncio.sleep(0.1)
    monitoring = asyncio.create_task(monitor())
    started = time.monotonic()
    prep = ledger.reserve("LOCAL_STT_PREPARE", config_name, Reservation(), metadata={"phase":"FIXED_COMPARISON", "paidApi":False})
    try:
        await runtime.warm_up()
        ledger.finish(prep, status="COMPLETED", elapsed=time.monotonic()-started, accounted=Reservation())
        report.update(status="RUNNING", modelVersion=runtime.model_version)
        provider = BudgetedSttProvider(FasterWhisperSpeakingSttProvider(runtime), ledger,
                                      phase="FIXED_COMPARISON", model=config_name)
        for clip, audio in clips:
            with wave.open(io.BytesIO(audio), "rb") as reader:
                duration = reader.getnframes()/reader.getframerate()
            entry = {**clip, "normalizedAudioSha256":hashlib.sha256(audio).hexdigest(),
                     "durationSeconds":duration, "status":"STARTED"}
            report["clips"].append(entry)
            atomic_json(report_path, report)
            began = time.monotonic()
            result = await provider.transcribe(audio, language=clip["language"], phrase_hints=None)
            expected, observed = normalized_text(clip["expectedText"]), normalized_text(result.text)
            errors = edit_distance(expected, observed)
            entry.update(status="COMPLETED", result=asdict(result), latencySeconds=time.monotonic()-began,
                         editDistance=errors, referenceCharacters=len(expected),
                         cer=errors/len(expected) if expected else None,
                         hallucinatedOnNoSpeech=bool(observed) if not expected else None)
            entry["rtf"] = entry["latencySeconds"]/duration
            atomic_json(report_path, report)
        report.update(status="COMPLETED", partial=False)
    except BaseException as exc:
        report.update(status="INTERRUPTED" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "FAILED",
                      errorType=type(exc).__name__)
        if ledger.snapshot()["calls"][prep-1]["status"] == "STARTED":
            ledger.finish(prep, status="FAILED", elapsed=time.monotonic()-started, failure=type(exc).__name__)
        raise
    finally:
        await runtime.shutdown()
        monitoring.cancel()
        await asyncio.gather(monitoring, return_exceptions=True)
        report.update(peakProcessRssMiB=peak/1024**2, elapsedSeconds=time.monotonic()-started)
        atomic_json(report_path, report)
    print(json.dumps({"configuration":config_name,"status":report['status'],"clips":len(report['clips']),
                      "peakProcessRssMiB":report['peakProcessRssMiB']}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, choices=["base", "small"])
    parser.add_argument("--revision", type=int, choices=[1, 2, 3], default=1,
                        help="A documented source fix only, never repeat unchanged code to obtain success")
    parser.add_argument("--baseline-runtime-preflight", type=Path,
                        help="Correct a documented baseline CPU-thread mismatch using its original effective-runtime artifact")
    args = parser.parse_args()
    asyncio.run(run(args.output.resolve(), args.config, args.revision, args.baseline_runtime_preflight))
