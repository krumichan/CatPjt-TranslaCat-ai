"""One-shot restart of the manifest-owned QA AI for FREE coaching-v1.

The shared campaign ledger and original environment credentials are retained.
Only the process recorded in ``ai-server-runtime.json`` may be stopped.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import psutil

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The owned campaign was intentionally created with the +2 local port shift.
# Set it before importing the manifest verifier, which materializes PORTS.
os.environ.setdefault("TRANSLACAT_QA_PORT_SHIFT", "2")

from scripts.qa_campaign_budget import CampaignLedger, atomic_json
from scripts.qa_campaign_integration import _owned_manifest


def restart(campaign: Path, *, revision: str = "initial") -> dict[str, object]:
    campaign = campaign.resolve()
    if campaign.name != "openai-speech-campaign-20260920":
        raise ValueError("Exact campaign required")
    environment = campaign / "integration-environment"
    _owned_manifest(environment)
    runtime_path = campaign / "ai-server-runtime.json"
    previous = json.loads(runtime_path.read_text(encoding="utf-8"))
    if previous["port"] != 18084 or previous["providerBudget"] != str(campaign / "budget-ledger.json"):
        raise ValueError("QA port or ledger identity changed")
    process = psutil.Process(int(previous["pid"]))
    command = process.cmdline()
    launcher = Path(__file__).resolve().parent / "run_qa_ai_server.py"
    if (
        not any(part.replace("\\", "/").endswith("scripts/run_qa_ai_server.py") for part in command)
        or str(campaign) not in command
        or str(environment) not in command
        or Path(process.exe()).resolve() != Path(command[0]).resolve()
        or not Path(previous["python"]).is_file()
        or not launcher.is_file()
    ):
        raise ValueError("QA process is not the recorded AI server; refusing shutdown")
    ledger = CampaignLedger(campaign / "budget-ledger.json", campaign.name)
    snapshot = ledger.snapshot()
    if snapshot["stoppedReason"] or any(call["status"] == "STARTED" for call in snapshot["calls"]):
        raise ValueError("Campaign halted or provider call in flight; refusing restart")

    target_dir = campaign / "closure-20260920-v1" / "free-coaching-v1-owned-qa"
    target_dir.mkdir(parents=True, exist_ok=True)
    if revision not in {"initial", "correction-boundary"}:
        raise ValueError("Unknown fixed coaching restart revision")
    suffix = "" if revision == "initial" else f"-{revision}"
    before = target_dir / f"ai-server-runtime-before-restart{suffix}.json"
    result = target_dir / f"ai-restart{suffix}.json"
    coaching_path_suffix = "app/features/language_learning/speaking/coaching_service.py"
    if before.exists() or result.exists():
        if not (before.exists() and result.exists()):
            raise ValueError("Incomplete coaching-v1 restart evidence; refusing a second process change")
        existing = json.loads(result.read_text(encoding="utf-8"))
        new_pid = existing.get("newPid")
        if (
            existing.get("status") != "NEW_OWNED_AI_STARTING"
            or not isinstance(new_pid, int)
            or new_pid not in {process.pid, process.ppid()}
        ):
            raise ValueError("Coaching-v1 AI restart already completed or changed; refusing repeat")
        source_key = next(
            (key for key in previous["sourceSha256"] if key.replace("\\", "/").endswith(coaching_path_suffix)),
            None,
        )
        if source_key is None:
            raise ValueError("Restarted AI runtime lacks coaching source provenance")
        with urllib.request.urlopen("http://127.0.0.1:18084/", timeout=2) as response:
            if response.status != 200:
                raise RuntimeError("Restarted AI health check failed")
        existing.update(
            status="NEW_OWNED_AI_HEALTHY",
            runtimePid=process.pid,
            coachingSourceSha256=previous["sourceSha256"][source_key],
        )
        atomic_json(result, existing)
        return existing
    atomic_json(before, previous)
    process.terminate()
    process.wait(timeout=30)
    evidence: dict[str, object] = {
        "oldPid": process.pid,
        "status": "OLD_OWNED_AI_STOPPED",
        "ledger": str(ledger.path),
    }
    atomic_json(result, evidence)

    environment_values = {**os.environ, "TRANSLACAT_QA_PORT_SHIFT": "2"}
    repository = Path(__file__).resolve().parents[1]
    with (target_dir / "ai.stdout.log").open("ab") as stdout, (target_dir / "ai.stderr.log").open("ab") as stderr:
        child = subprocess.Popen(
            [str(previous["python"]), "-B", str(launcher), "--output", str(campaign),
             "--environment", str(environment)],
            cwd=repository,
            env=environment_values,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    evidence.update(status="NEW_OWNED_AI_STARTING", newPid=child.pid)
    atomic_json(result, evidence)
    for _ in range(120):
        if child.poll() is not None:
            raise RuntimeError("Restarted QA AI exited; inspect private stderr")
        try:
            with urllib.request.urlopen("http://127.0.0.1:18084/", timeout=2) as response:
                if response.status == 200:
                    current = json.loads(runtime_path.read_text(encoding="utf-8"))
                    if current["pid"] != child.pid and psutil.Process(current["pid"]).ppid() != child.pid:
                        raise ValueError("AI runtime manifest was not refreshed")
                    coaching_path = next(
                        key for key in current["sourceSha256"]
                        if key.replace("\\", "/").endswith(coaching_path_suffix)
                    )
                    evidence.update(
                        status="NEW_OWNED_AI_HEALTHY",
                        runtimePid=current["pid"],
                        coachingSourceSha256=current["sourceSha256"][coaching_path],
                    )
                    atomic_json(result, evidence)
                    return evidence
        except OSError:
            time.sleep(1)
    raise TimeoutError("Restarted owned QA AI did not become healthy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--revision", choices=("initial", "correction-boundary"), default="initial")
    args = parser.parse_args()
    print(json.dumps(restart(args.campaign, revision=args.revision), ensure_ascii=False))
