"""One bounded OpenAI-only Japanese voice comparison; private WAV artifacts."""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import hashlib
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ai.provider_factory import create_speech_synthesis_provider
from scripts.qa_campaign_audio_budget import BudgetedOpenAISpeechProvider
from scripts.qa_campaign_budget import CampaignLedger, atomic_json


SAMPLES = (
    "明日の会議は午後三時からです。ご参加をお願いいたします。",
    "新宿駅から徒歩五分の会場で、九月二十日に説明会を開きます。",
    "ご不便をおかけしております。復旧の見込みが立ち次第、お知らせいたします。",
)


async def select(campaign: Path, *, label: str = "baseline", sample_limit: int = 6,
                 skip_samples: int = 0) -> dict[str, Any]:
    campaign = campaign.resolve()
    ledger = CampaignLedger(campaign / "budget-ledger.json", campaign.name)
    if not label.isascii() or not label.replace("-", "").isalnum():
        raise ValueError("Invalid local run label")
    if not 0 <= skip_samples < 6 or not 1 <= sample_limit <= 6 - skip_samples:
        raise ValueError("Sample range must stay within the fixed six cases")
    output = campaign / ("voice-select" if label == "baseline" else f"voice-select-{label}")
    if output.exists():
        raise ValueError("Voice selection already started; never repeat same revision")
    output.mkdir(parents=True)
    provider = BudgetedOpenAISpeechProvider(create_speech_synthesis_provider(), ledger,
                                             phase="VOICE_SELECTION")
    report: dict[str, Any] = {
        "startedAt": datetime.now(UTC).isoformat(), "status": "RUNNING", "samples": [],
        "provider": "openai", "model": "gpt-4o-mini-tts-2025-12-15",
        "geminiStarts": 0, "planned": sample_limit, "skippedPreviouslyCompleted": skip_samples,
    }
    atomic_json(output / "results.json", report)
    try:
        selected = 0
        seen = 0
        for voice in ("marin", "cedar"):
            for index, sample in enumerate(SAMPLES, 1):
                seen += 1
                if seen <= skip_samples:
                    continue
                if selected >= sample_limit:
                    break
                selected += 1
                entry: dict[str, Any] = {
                    "voice": voice, "sample": index,
                    "textSha256": hashlib.sha256(sample.encode()).hexdigest(),
                    "characters": len(sample), "status": "STARTED",
                }
                report["samples"].append(entry)
                atomic_json(output / "results.json", report)
                try:
                    result = await provider.synthesize_speech(
                        text=sample, voice=voice, language="ja", speed="NORMAL")
                    path = output / f"{voice}-{index}.wav"
                    path.write_bytes(result.audio_bytes)
                    entry.update(status="COMPLETED", durationSeconds=result.duration_seconds,
                                 audioSha256=hashlib.sha256(result.audio_bytes).hexdigest(),
                                 artifact=str(path), bytes=len(result.audio_bytes))
                except BaseException as exc:
                    entry.update(status="FAILED", errorType=type(exc).__name__,
                                 diagnosticCode=getattr(exc, "diagnostic_code", None),
                                 byteCount=getattr(exc, "byte_count", None),
                                 headerMagic=getattr(exc, "header_magic", None),
                                 httpStatus=getattr(exc, "status_code", None))
                    raise
                finally:
                    atomic_json(output / "results.json", report)
        report["status"] = "COMPLETED"
    except BaseException as exc:
        report["status"] = "INTERRUPTED" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "FAILED"
        report["errorType"] = type(exc).__name__
    finally:
        report["finishedAt"] = datetime.now(UTC).isoformat()
        atomic_json(output / "results.json", report)
        await provider.shutdown()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--sample-limit", type=int, default=6)
    parser.add_argument("--skip-samples", type=int, default=0)
    args = parser.parse_args()
    result = asyncio.run(select(args.campaign, label=args.label,
                                sample_limit=args.sample_limit, skip_samples=args.skip_samples))
    print({"status": result["status"], "completed": sum(
        item["status"] == "COMPLETED" for item in result["samples"]),
        "planned": result["planned"]})


if __name__ == "__main__":
    main()
