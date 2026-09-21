"""Explicit opt-in campaign driver; secrets and generated material stay private."""
from __future__ import annotations

import argparse
import asyncio
import copy
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qa_campaign_budget import (  # noqa: E402
    BudgetedTextProvider, CampaignLedger, atomic_json,
)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), *args], text=True, encoding="utf-8")


def prepare(output: Path, case: Path) -> None:
    if (output / "campaign-manifest.json").exists():
        raise ValueError("Existing campaign must resume, never reinitialize")
    repos = {}
    for suffix in ("ai", "be", "fe"):
        repo = ROOT.parent / f"CatPjt-TranslaCat-{suffix}"
        status = git(repo, "status", "--porcelain=v1")
        staged = git(repo, "diff", "--cached", "--binary")
        working = git(repo, "diff", "--binary")
        repos[suffix] = {"path": str(repo), "head": git(repo, "rev-parse", "HEAD").strip(),
                         "status": status, "indexSha256": hashlib.sha256((repo / ".git/index").read_bytes()).hexdigest(),
                         "stagedDiffSha256": hashlib.sha256(staged.encode()).hexdigest(),
                         "workingDiffSha256": hashlib.sha256(working.encode()).hexdigest()}
        # Private snapshot of user changes; contains source only, no ignored .env.
        atomic_json(output / f"baseline-{suffix}.json", {"status": status, "staged": staged, "working": working})
    atomic_json(output / "preflight.json", {"createdAt": datetime.now(UTC).isoformat(), "repositories": repos,
        "python": {"path": sys.executable, "version": sys.version},
        "case": {"path": str(case), "sha256": hashlib.sha256(case.read_bytes()).hexdigest()},
        "isolation": "Pending integration preflight; no existing process or database modified",
        "pricing": {"checkedDate": "2026-09-19", "textReservationUsdPerMillion": [1, 4],
            "reason": "Conservative above current standard model rates; reasoning included in output",
            "sources": ["https://developers.openai.com/api/docs/pricing", "https://developers.openai.com/api/docs/models/gpt-5-mini", "https://developers.openai.com/api/docs/models/gpt-5-nano", "https://ai.google.dev/gemini-api/docs/pricing"]}})
    matrix: list[dict[str, Any]] = [
        {"phase": "fixed-order4", "repetitionsPerRevision": 2, "case": str(case)},
        {"phase": "fixed-plan-remaining", "orders": [4, 5, 6, 7, 8, 9, 10]},
        {"phase": "fresh-vocabulary", "profiles": 3, "itemsEach": 10},
        {"phase": "listening", "modes": ["DICTATION", "COMPREHENSION", "SUMMARY"], "itemsEach": 5},
        {"phase": "speaking", "modes": ["READ_ALOUD", "GUIDED", "FREE"]},
        {"phase": "be-db", "criteria": "normal auth, progressive persistence, evaluation, retry, fencing"},
        {"phase": "browser", "criteria": "normal OAuth; never forge authentication"},
    ]
    for row in matrix:
        row["status"] = "NOT_RUN"
    atomic_json(output / "campaign-manifest.json", {"campaignId": output.name,
        "matrix": matrix, "revisionPolicy": "two fixed case runs/revision; no success-until-repeat",
        "modelsAndEffort": "unchanged production policy", "productionDatabaseWrites": False})
    CampaignLedger(output / "budget-ledger.json", output.name)
    atomic_json(output / "db-manifest.json", {"existingDataModified": False, "qaResources": [], "status": "NOT_STARTED"})


async def order4(output: Path, case_path: Path, repetition: int, revision: str = "r1") -> None:
    from app.ai.provider_factory import create_text_generation_provider
    from scripts.run_contextual_choice_stabilization_qa import (
        QaCaps, load_order_diagnostic_case, run_order_diagnostic_live, source_fingerprint,
    )
    if repetition not in (1, 2):
        raise ValueError("Fixed revision allows exactly two declared repetitions")
    revision_tag = "" if revision == "r1" else f"-{revision}"
    result_path = output / f"order4{revision_tag}-candidate-run{repetition}.json"
    if result_path.exists():
        raise ValueError("Refusing to overwrite or repeat an existing run")
    if revision != "r1":
        revision_path = output / f"fixed-case-{revision}-manifest.json"
        fingerprint = source_fingerprint()
        if revision_path.exists():
            if json.loads(revision_path.read_text(encoding="utf-8"))["sourceSha256"] != fingerprint:
                raise ValueError("A revision's source cannot change between repetitions")
        else:
            if repetition != 1:
                raise ValueError("Declare each revision before its first repetition")
            baseline = json.loads((output / "order4-candidate-run1.json").read_text(encoding="utf-8"))
            if baseline["sourceSha256"] == fingerprint:
                raise ValueError("A new label cannot repeat unchanged production source")
            atomic_json(revision_path, {"revision": revision, "repetitions": 2, "sourceSha256": fingerprint,
                "reason": "Verified repair-scope close-count feedback correction; acceptance unchanged"})
    ledger = CampaignLedger(output / "budget-ledger.json", output.name)
    provider = BudgetedTextProvider(create_text_generation_provider(), ledger, phase=f"order4-{revision}-run{repetition}")
    result = await run_order_diagnostic_live(case_path, load_order_diagnostic_case(case_path), variant="candidate",
        caps=QaCaps(12, 120000, 40000, 90, 600), partial_output=result_path, provider_factory=lambda: provider)
    print(json.dumps({"status": result["status"], "outcomeCategory": result["outcomeCategory"],
        "artifact": str(result_path), "budget": ledger.snapshot()["totalsIncludingReserved"]}))


async def vocabulary(output: Path, case_path: Path, *, remaining: bool) -> None:
    from app.ai.provider_factory import create_text_generation_provider
    from app.features.language_learning.reading_vocabulary.service import ReadingVocabularyGenerationService
    from scripts.run_contextual_choice_stabilization_qa import (
        DEFAULT_FIXTURE, QaCaps, RecordingProvider, _call_summary,
        build_order_diagnostic_request, load_order_diagnostic_case, run_profile,
    )
    name = "fixed-remaining" if remaining else "fresh-vocabulary"
    path = output / (name + ".json")
    if path.exists():
        raise ValueError("Existing run cannot be repeated or overwritten")
    ledger = CampaignLedger(output / "budget-ledger.json", output.name)
    upstream = BudgetedTextProvider(create_text_generation_provider(), ledger, phase=name)
    recorder = RecordingProvider(upstream, QaCaps(600, 2000000, 500000, 90, 21600), diagnostic_capture=True)
    service = ReadingVocabularyGenerationService(recorder)
    state: dict[str, Any] = {"phase": name, "status": "RUNNING", "results": [], "partial": True}

    def checkpoint() -> None:
        atomic_json(path, {**state, "calls": _call_summary(recorder.calls)})

    recorder.checkpoint = checkpoint
    checkpoint()
    try:
        if remaining:
            case = copy.deepcopy(load_order_diagnostic_case(case_path))
            # Reuse the first accepted recorded order4; never pay to regenerate a completed item.
            accepted = None
            for n in (1, 2):
                prior = json.loads((output / f"order4-candidate-run{n}.json").read_text(encoding="utf-8"))
                if prior["status"] == "COMPLETED":
                    accepted = prior["response"]
                    state["reusedOrder4Artifact"] = f"order4-candidate-run{n}.json"
                    break
            if accepted is None:
                state.update(status="BLOCKED", reason="NO_ACCEPTED_ORDER4_NO_EXTRA_RETRY")
                return
            for order in range(4, 11):
                if order != 4:
                    case["targetOrder"] = order
                    response = await service.generate(build_order_diagnostic_request(case))
                    accepted = response.model_dump(mode="json", by_alias=True)
                if accepted is None or len(accepted["questions"]) != 1 or not accepted.get("vocabularyPlan"):
                    raise ValueError("Progressive result omitted single question or plan delta")
                question = {**accepted["questions"][0], "order": order}
                case["acceptedPrefix"].append(question)
                case["requestSnapshot"]["vocabularyPlan"] = accepted["vocabularyPlan"]
                state["results"].append({"order": order, "response": accepted})
                state["acceptedPrefix"] = case["acceptedPrefix"]
                checkpoint()
            state["status"] = "COMPLETED"
        else:
            fixture = json.loads(DEFAULT_FIXTURE.read_text(encoding="utf-8"))
            state["fixture"] = fixture
            for profile in fixture["profiles"]:
                state["results"].append(await run_profile(fixture, profile, service, recorder))
                checkpoint()
            state["status"] = "COMPLETED" if all(row["status"] == "COMPLETED" for row in state["results"]) else "PARTIAL"
        state["partial"] = state["status"] != "COMPLETED"
    except BaseException as exc:
        state.update(status="INTERRUPTED" if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)) else "FAILED", errorType=type(exc).__name__)
    finally:
        checkpoint()
        await upstream.shutdown()
        print(json.dumps({"phase": name, "status": state["status"], "artifact": str(path), "budget": ledger.snapshot()["totalsIncludingReserved"]}))


async def audio_campaign(output: Path, *, speaking: bool) -> None:
    from app.ai.provider_factory import create_speech_synthesis_provider, create_text_generation_provider
    from scripts.qa_campaign_audio_budget import BudgetedOpenAISpeechProvider, prepare_local_stt
    from scripts.qa_campaign_listening import run_listening_campaign
    from scripts.qa_campaign_speaking import run_speaking_campaign
    name = "speaking" if speaking else "listening"
    directory = output / name
    if directory.exists():
        raise ValueError("Existing audio scenario cannot be rerun")
    ledger = CampaignLedger(output / "budget-ledger.json", output.name)
    # Isolate every temporary audio store from the user's running AI process.
    import tempfile
    temporary = output / f"{name}-temp"
    temporary.mkdir()
    tempfile.tempdir = str(temporary)
    text = BudgetedTextProvider(create_text_generation_provider(), ledger, phase=name)
    raw_speech = create_speech_synthesis_provider()
    speech = BudgetedOpenAISpeechProvider(raw_speech, ledger, phase=name)
    stt, runtime, report = await prepare_local_stt(ledger, phase=name, mode=name, report_path=output / f"{name}-stt-preflight.json")
    try:
        if speaking:
            result = await run_speaking_campaign(text, speech, directory, stt_provider=stt)
        else:
            result = {"status": "RUNNING", "modes": []}
            for mode in ("DICTATION", "COMPREHENSION", "SUMMARY"):
                row = await run_listening_campaign(text, speech, directory / mode.lower(), stt_provider=stt, modes=(mode,))
                result["modes"].append(row)
                atomic_json(directory / "listening-all-modes.json", result)
            result["status"] = "COMPLETED" if all(row["status"] == "COMPLETED" for row in result["modes"]) else "PARTIAL"
            atomic_json(directory / "listening-all-modes.json", result)
        print(json.dumps({"phase": name, "status": result["status"], "sttReady": report["ready"],
            "budget": ledger.snapshot()["totalsIncludingReserved"]}))
    finally:
        await text.shutdown()
        await raw_speech.shutdown()
        if runtime is not None:
            await runtime.shutdown()


async def speaking_evaluation_retest(output: Path) -> None:
    from app.ai.provider_factory import create_text_generation_provider
    from scripts.qa_retest_speaking_evaluation import run_captured_speaking_evaluation_retest
    ledger = CampaignLedger(output / "budget-ledger.json", output.name)
    provider = BudgetedTextProvider(create_text_generation_provider(), ledger, phase="speaking-evaluation-enum-retest")
    try:
        result = await run_captured_speaking_evaluation_retest(provider,
            output / "speaking/speaking-campaign-result.json", output / "speaking-evaluation-retest")
        print(json.dumps({"status": result["status"], "budget": ledger.snapshot()["totalsIncludingReserved"]}))
    finally:
        await provider.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "order4", "remaining", "fresh", "listening", "speaking", "speaking-evaluation-retest"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--revision", choices=["r1", "r2"], default="r1")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output.resolve(), args.case.resolve())
    elif args.live:
        if not (args.output / "campaign-manifest.json").exists():
            raise SystemExit("Initialize and freeze the manifest before live calls")
        if args.action == "order4":
            asyncio.run(order4(args.output.resolve(), args.case.resolve(), args.repetition, args.revision))
        elif args.action == "speaking-evaluation-retest":
            asyncio.run(speaking_evaluation_retest(args.output.resolve()))
        elif args.action in ("listening", "speaking"):
            asyncio.run(audio_campaign(args.output.resolve(), speaking=args.action == "speaking"))
        else:
            asyncio.run(vocabulary(args.output.resolve(), args.case.resolve(), remaining=args.action == "remaining"))
    else:
        raise SystemExit("Paid calls require --live and an existing campaign manifest")


if __name__ == "__main__":
    main()
