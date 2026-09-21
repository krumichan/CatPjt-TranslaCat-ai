"""Resume only the recorded post-stop/pre-start retest QA BE checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

import psutil

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.qa_campaign_budget import atomic_json
from scripts.qa_free_owned_schema import RETEST_SCHEMA


def resume(campaign: Path, be_repository: Path) -> dict[str, object]:
    if campaign.name != "openai-speech-campaign-20260920":
        raise ValueError("Exact campaign required")
    original = json.loads((campaign / "integration-environment" / "manifest.json").read_text(encoding="utf-8"))
    prior = json.loads((campaign / "closure-20260920-v1" / "free-v3-owned-qa" / "manifest.json").read_text(encoding="utf-8"))
    fresh = campaign / "closure-20260920-v1" / "free-v3-retest-owned-qa"
    manifest_path = fresh / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stopped_pid = prior["beProcess"]["pid"]
    if (original["owner"] != "translacat-isolated-qa" or original["campaign"] != campaign.name
            or manifest["status"] != "ORIGINAL_QA_BE_STOPPED_NEW_NOT_STARTED"
            or manifest["stoppedOwnedPid"] != stopped_pid or manifest["database"] != RETEST_SCHEMA
            or "beProcess" in manifest or psutil.pid_exists(stopped_pid)):
        raise ValueError("Not the exact owned post-stop checkpoint")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 18083)) == 0:
            raise ValueError("QA BE port already occupied; refusing to start another")
    jar = (be_repository / "build" / "libs" / "spring-boot-translacat-0.0.1-SNAPSHOT.jar").resolve()
    if not jar.is_file() or hashlib.sha256(jar.read_bytes()).hexdigest() != prior["beProcess"]["jarSha256"]:
        raise ValueError("Previously tested BE jar changed")
    java = Path(original["beProcess"]["command"][0]).resolve()
    if not java.is_file():
        raise ValueError("Owned QA Java executable absent")
    private = json.loads((fresh / "secrets.private.json").read_text(encoding="utf-8"))
    platform_keys = {"SYSTEMROOT", "WINDIR", "PATH", "JAVA_HOME", "TEMP", "TMP", "USERPROFILE",
                     "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA", "COMSPEC", "PROGRAMDATA",
                     "USERDOMAIN", "USERNAME"}
    environment = {key: value for key, value in os.environ.items() if key.upper() in platform_keys}
    environment.update(private)
    command = [str(java), "-jar", str(jar), "--spring.profiles.active=qa",
               f"--spring.config.location=classpath:/application.properties,file:{(fresh / 'application-qa.properties').as_posix()}"]
    with (fresh / "be.stdout.log").open("ab") as stdout, (fresh / "be.stderr.log").open("ab") as stderr:
        child = subprocess.Popen(command, cwd=be_repository, env=environment,
                                 stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
    manifest.update(status="BE_STARTING", beProcess={"pid": child.pid, "jarSha256": prior["beProcess"]["jarSha256"],
                                                      "profile": "qa", "database": RETEST_SCHEMA})
    atomic_json(manifest_path, manifest)
    for _ in range(90):
        if child.poll() is not None:
            raise RuntimeError("Retest QA BE exited; inspect private stderr")
        try:
            with urllib.request.urlopen("http://127.0.0.1:18083/api/v1/health", timeout=2) as response:
                if response.status == 200:
                    manifest["status"] = "BE_HEALTHY_NORMAL_LOGIN_REQUIRED"
                    atomic_json(manifest_path, manifest)
                    return {"status": manifest["status"], "database": RETEST_SCHEMA,
                            "pid": child.pid, "stoppedOwnedPid": stopped_pid}
        except OSError:
            time.sleep(1)
    raise TimeoutError("Retest QA BE did not become healthy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--be-repository", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(resume(args.campaign.resolve(), args.be_repository.resolve())))
