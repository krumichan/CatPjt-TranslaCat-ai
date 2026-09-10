"""Failure-injection tests for deploy-release.sh; all Docker/Git/HTTP calls are fake.
Run: python scripts/check_deploy_release.py
No server connections, images, containers or credentials are used.
"""
from __future__ import annotations
import base64
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deploy-release.sh"
NAME = re.search(r'^CONTAINER="([^"]+)"', SCRIPT.read_text(encoding="utf-8"), re.MULTILINE).group(1)
SHA = "1" * 40
FAKE = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
state_path = Path(os.environ["PROBE_STATE"])
state = json.loads(state_path.read_text())
fault = os.environ["PROBE_FAULT"]
container = os.environ["PROBE_CONTAINER"]
sha = os.environ["PROBE_SHA"]
with open(os.environ["PROBE_CALLS"], "a") as out:
    out.write(json.dumps([name, *args]) + "\n")
def end(code=0, output=""):
    state_path.write_text(json.dumps(state))
    if output: print(output)
    sys.exit(code)
if name == "git": end(0, "0" * 40 if fault == "checkout" else sha)
if name == "sleep": end()
if name == "curl": end(0, "503" if fault in ("health", "rollback_start") else "200")
if args[:2] == ["network", "inspect"]: end(1)
if args[:2] == ["network", "create"]: end(1 if fault == "network" else 0, "fake-network")
if args[0] == "build": end(7 if fault == "build" else 0)
if args[:2] == ["container", "inspect"]: end(0 if args[-1] in state else 1)
if args[0] == "inspect":
    item = state.get(args[-1])
    if item is None: end(1)
    fmt = args[2]
    if "Running" in fmt: end(0, str(item["running"]).lower())
    if "Health" in fmt: end(0, "unhealthy" if fault in ("health", "rollback_start") else "healthy")
    end(0, "0" * 40 if fault == "revision" else item["sha"])
if args[0] == "stop":
    state[args[-1]]["running"] = False
    end(9 if fault == "stop" else 0)
if args[0] == "rename":
    old, new = args[1:]
    if fault == "rename" and old == container: end(10)
    if old not in state or new in state: end(1)
    state[new] = state.pop(old)
    end()
if args[0] == "run":
    if fault == "run": end(11)
    target = args[args.index("--name") + 1]
    state[target] = {"running": True, "sha": sha, "env": "new-env"}
    end(0, "new-container-id")
if args[0] == "start":
    if fault == "rollback_start": end(12)
    state[args[-1]]["running"] = True
    end()
if args[0] == "rm":
    state.pop(args[-1], None)
    end()
if args[0] == "logs": end(0, "simulated log")
end(99, "unsupported probe call: " + repr(args))
'''

def main() -> None:
    if shutil.which("bash") is None:
        raise SystemExit("Bash is required (Linux/macOS or Git Bash/WSL). No live deployment is performed.")
    results = []
    cases = [(name, True, True) for name in ("checkout", "environment", "network", "build", "stop", "rename", "run", "revision", "health", "success")]
    cases += [("health", True, False), ("success", False, False), ("health", False, False), ("rollback_start", True, True)]
    for fault, exists, running in cases:
        with tempfile.TemporaryDirectory(prefix="translacat-release-probe-") as directory:
            root = Path(directory)
            bins = root / "bin"
            bins.mkdir()
            original = {NAME: {"running": running, "sha": "old-sha", "env": "old-env"}} if exists else {}
            (root / "state.json").write_text(json.dumps(original))
            (root / "calls.jsonl").write_text("")
            for command in ("git", "docker", "curl", "sleep"):
                executable = bins / command
                executable.write_text(FAKE.replace("#!/usr/bin/env python3", "#!" + sys.executable + " -S"))
                executable.chmod(0o755)
            environment = os.environ | {
                "PATH": str(bins) + os.pathsep + os.environ["PATH"],
                "PROBE_STATE": str(root / "state.json"), "PROBE_CALLS": str(root / "calls.jsonl"),
                "PROBE_FAULT": fault, "PROBE_CONTAINER": NAME, "PROBE_SHA": SHA,
                "DEPLOY_ENV_B64": "%%%bad-base64" if fault == "environment" else base64.b64encode(b"FAKE=probe\n").decode(),
                "DEPLOY_HEALTH_ATTEMPTS": "2", "DEPLOY_HEALTH_DELAY_SECONDS": "0",
            }
            process = subprocess.run(["bash", str(SCRIPT), SHA], cwd=root, env=environment,
                                     capture_output=True, text=True, timeout=20)
            after = json.loads((root / "state.json").read_text())
            calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
            assert not list(root.glob(".deploy-env.*")), (fault, "temporary environment leaked")
            if fault == "success":
                assert process.returncode == 0, process.stderr
                assert after[NAME]["sha"] == SHA and after[NAME]["running"]
                if exists:
                    backups = [item for key, item in after.items() if key != NAME]
                    assert len(backups) == 1 and backups[0]["env"] == "old-env" and not backups[0]["running"]
            else:
                assert process.returncode != 0, (fault, "failure reported success")
                if fault == "rollback_start":
                    assert "CRITICAL" in process.stderr and after[NAME]["env"] == "old-env"
                else:
                    assert after == original, (fault, original, after, process.stderr)
                if fault in ("checkout", "environment", "network", "build"):
                    assert not any(call[:2] == ["docker", "stop"] for call in calls), (fault, calls)
            results.append({"fault": fault, "previous_exists": exists, "previous_running": running,
                            "exit": process.returncode, "assertions": "PASS"})
            print(f"PASS {fault}: previous={exists}/{running}, exit={process.returncode}")
    print(f"{len(results)}/{len(results)} failure-injection scenarios passed. All external commands were fake.")

if __name__ == "__main__":
    main()
