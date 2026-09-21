"""Restart only the manifest-owned QA AI server against current checkout/ledger."""
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

from scripts.qa_campaign_budget import CampaignLedger, atomic_json
from scripts.qa_campaign_integration import _owned_manifest


def restart(campaign: Path) -> dict[str, object]:
    if campaign.name != "openai-speech-campaign-20260920":
        raise ValueError("Exact campaign required")
    _owned_manifest(campaign / "integration-environment")
    runtime_path = campaign / "ai-server-runtime.json"
    previous = json.loads(runtime_path.read_text(encoding="utf-8"))
    pid = previous["pid"]
    if previous["port"] != 18084 or previous["providerBudget"] != str(campaign / "budget-ledger.json"):
        raise ValueError("QA port or ledger identity changed")
    process = psutil.Process(pid)
    command = process.cmdline()
    script = str((Path(__file__).resolve().parent / "run_qa_ai_server.py"))
    if (not any(part.replace("\\", "/").endswith("scripts/run_qa_ai_server.py") for part in command)
            or str(campaign) not in command or str(campaign / "integration-environment") not in command
            # The venv's sys.executable is a launcher; Win32 resolves the
            # process image to its underlying pinned QA Python binary.
            or Path(process.exe()).resolve() != Path(command[0]).resolve()
            or not Path(previous["python"]).is_file()
            or not Path(script).is_file()):
        raise ValueError("QA process is not the recorded AI server; refusing shutdown")
    ledger = CampaignLedger(campaign / "budget-ledger.json", campaign.name)
    if any(call["status"] == "STARTED" for call in ledger.snapshot()["calls"]):
        raise ValueError("In-flight budgeted provider call; refusing shutdown")
    closure = campaign / "closure-20260920-v1"
    snapshot = closure / "ai-server-runtime-before-free-v3-restart.json"
    if snapshot.exists():
        raise ValueError("Previous QA restart evidence exists; refusing repeat")
    atomic_json(snapshot, previous)
    process.terminate()
    process.wait(timeout=30)
    evidence = {"oldPid": pid, "status": "OLD_OWNED_AI_STOPPED", "ledger": str(ledger.path)}
    target = closure / "ai-restart-free-v3.json"
    if target.exists():
        raise ValueError("Previous restart result exists; refusing repeat")
    atomic_json(target, evidence)
    environment = {**os.environ, "TRANSLACAT_QA_PORT_SHIFT": "2"}
    repository = Path(__file__).resolve().parents[1]
    with (closure / "ai-free-v3.stdout.log").open("ab") as stdout, (closure / "ai-free-v3.stderr.log").open("ab") as stderr:
        child = subprocess.Popen(
            [str(previous["python"]), "-B", script, "--output", str(campaign),
             "--environment", str(campaign / "integration-environment")],
            cwd=repository, env=environment, stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW,
        )
    evidence.update(status="NEW_OWNED_AI_STARTING", newPid=child.pid)
    atomic_json(target, evidence)
    for _ in range(90):
        if child.poll() is not None:
            raise RuntimeError("Restarted QA AI exited; inspect private stderr")
        try:
            with urllib.request.urlopen("http://127.0.0.1:18084/", timeout=2) as response:
                if response.status == 200:
                    current = json.loads(runtime_path.read_text(encoding="utf-8"))
                    if current["pid"] != child.pid and psutil.Process(current["pid"]).ppid() != child.pid:
                        raise ValueError("AI runtime manifest was not refreshed")
                    evidence.update(status="NEW_OWNED_AI_HEALTHY", runtimePid=current["pid"], currentPromptSha256=current["sourceSha256"][
                        "app/features/language_learning/speaking/prompts.py"])
                    atomic_json(target, evidence)
                    return evidence
        except OSError:
            time.sleep(1)
    raise TimeoutError("Restarted owned QA AI did not become healthy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restart(args.campaign.resolve()), ensure_ascii=False))
