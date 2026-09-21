"""One budgeted OpenAI TTS WAV for one distinct, context-aware QA learner reply.

The human-authored Japanese text and WAV remain in the owned private closure
directory. This is a synthesized microphone fixture, not real pronunciation.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.qa_campaign_budget import CampaignLedger, atomic_json
from scripts.qa_campaign_integration import _owned_manifest


async def synthesize(campaign: Path, text_file: Path, wav: Path, report_path: Path) -> dict[str, Any]:
    if wav.exists() or report_path.exists():
        raise ValueError("One-shot fixture already exists")
    _owned_manifest(campaign / "integration-environment")
    closure = campaign / "closure-20260920-v1"
    if (text_file.parent != closure or wav.parent != closure or report_path.parent != closure
            or text_file.stem != wav.stem or report_path.stem != wav.stem):
        raise ValueError("Only matching owned closure text/WAV/receipt stems allowed")
    text = text_file.read_text(encoding="utf-8").strip()
    if not 10 <= len(text) <= 160 or "\n" in text:
        raise ValueError("One context-aware Japanese learner response required")
    from app.ai.provider_factory import create_speech_synthesis_provider
    from scripts.qa_campaign_audio_budget import BudgetedOpenAISpeechProvider

    provider = BudgetedOpenAISpeechProvider(
        create_speech_synthesis_provider(), CampaignLedger(campaign / "budget-ledger.json", campaign.name),
        phase="CLOSURE_BROWSER_SYNTHETIC_LEARNER_VOICE",
    )
    report: dict[str, Any] = {"status": "STARTED", "partial": True,
                              "inputSha256": hashlib.sha256(text.encode()).hexdigest(),
                              "inputCharacters": len(text), "syntheticAudio": True,
                              "outputWav": str(wav)}
    atomic_json(report_path, report)
    try:
        await provider.warm_up()
        response = await provider.synthesize_speech(text=text, voice="marin", language="ja", speed="NORMAL")
        if response.content_type != "audio/wav" or not response.audio_bytes.startswith(b"RIFF"):
            raise ValueError("Provider did not return WAV")
        wav.write_bytes(response.audio_bytes)
        report.update(status="WAV_READY", partial=False, wavSha256=hashlib.sha256(response.audio_bytes).hexdigest(),
                      durationSeconds=response.duration_seconds, bytes=len(response.audio_bytes),
                      provider=response.provider, model=response.model)
    except BaseException as error:
        report.update(status="FAILED", errorType=type(error).__name__)
    finally:
        atomic_json(report_path, report)
        await provider.shutdown()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(synthesize(args.campaign.resolve(), args.text_file.resolve(),
                                    args.wav.resolve(), args.report.resolve()))
    print(json.dumps({key: result.get(key) for key in (
        "status", "errorType", "durationSeconds", "wavSha256", "inputCharacters",
    )}))


if __name__ == "__main__":
    main()
