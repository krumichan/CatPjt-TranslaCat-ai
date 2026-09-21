"""Replace only the recorded owned QA BE with a fresh, isolated FREE schema BE.

The original schema, Redis DB 0, audio files, and session9 remain untouched.
This does not create an auth token or call an AI provider.
"""
from __future__ import annotations

import argparse
import hashlib
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

from scripts.qa_campaign_budget import atomic_json
from scripts.qa_free_owned_schema import (
    COACHING_RETEST_SCHEMA,
    COACHING_SCHEMA,
    NEW_SCHEMA,
    RETEST_SCHEMA,
)


def launch(
    campaign: Path,
    be_repository: Path,
    *,
    revision_retest: bool = False,
    coaching_v1: bool = False,
    coaching_v1_retest: bool = False,
) -> dict[str, object]:
    campaign = campaign.resolve()
    original = campaign / "integration-environment"
    selected = sum((revision_retest, coaching_v1, coaching_v1_retest))
    if selected > 1:
        raise ValueError("Choose exactly one isolated schema target")
    directory = (
        "free-coaching-v1-retest-owned-qa"
        if coaching_v1_retest
        else "free-coaching-v1-owned-qa"
        if coaching_v1
        else "free-v3-retest-owned-qa"
        if revision_retest
        else "free-v3-owned-qa"
    )
    fresh = campaign / "closure-20260920-v1" / directory
    owner = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    manifest_path = fresh / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = (
        COACHING_RETEST_SCHEMA
        if coaching_v1_retest
        else COACHING_SCHEMA
        if coaching_v1
        else RETEST_SCHEMA
        if revision_retest
        else NEW_SCHEMA
    )
    prior_directory = (
        campaign / "closure-20260920-v1" / "free-coaching-v1-owned-qa"
        if coaching_v1_retest
        else campaign / "closure-20260920-v1" / "free-v3-retest-owned-qa"
        if coaching_v1
        else campaign / "closure-20260920-v1" / "free-v3-owned-qa"
        if revision_retest
        else original
    )
    prior_manifest = json.loads((prior_directory / "manifest.json").read_text(encoding="utf-8"))
    if (owner["owner"] != "translacat-isolated-qa" or owner["campaign"] != campaign.name
            or manifest["status"] != "READY_FOR_FRESH_BE_AND_NORMAL_GOOGLE_LOGIN"
            or manifest["database"] != schema or "beProcess" in manifest
            or ((revision_retest or coaching_v1 or coaching_v1_retest)
                and prior_manifest.get("status") != "BE_HEALTHY_NORMAL_LOGIN_REQUIRED")
            or prior_manifest["owner"] != owner["owner"]):
        raise ValueError("Exact owned QA runtime boundary required")
    prior = prior_manifest["beProcess"]
    process = psutil.Process(int(prior["pid"]))
    command = process.cmdline()
    expected_jar = str((be_repository.resolve() / "build" / "libs" / "spring-boot-translacat-0.0.1-SNAPSHOT.jar").resolve())
    if (len(command) < 5 or expected_jar not in command
            or not any("spring.profiles.active=qa" in part for part in command)
            or not any((prior_directory / "application-qa.properties").as_posix() in part for part in command)
            or Path(process.exe()).resolve() != Path(prior.get("command", command)[0]).resolve()):
        raise ValueError("Port owner is not the recorded QA BE; refusing shutdown")
    original_schema = (
        prior_manifest["database"]
        if revision_retest or coaching_v1 or coaching_v1_retest
        else owner["names"]["database"]
    )
    old_private = json.loads((prior_directory / "secrets.private.json").read_text(encoding="utf-8"))
    environment = {**os.environ, "MYSQL_PWD": old_private["QA_DB_PASSWORD"]}
    docker = Path("C:/Program Files/Docker/Docker/resources/bin/docker.exe")
    query = subprocess.run(
        [str(docker), "exec", "--env", "MYSQL_PWD", owner["names"]["mysqlContainer"],
         "mysql", "--batch", "--skip-column-names", "-uqa_app", original_schema,
         "-e", "SELECT COUNT(*) FROM language_learning_practice_set WHERE generation_status IN ('PENDING','GENERATING')"],
        env=environment, capture_output=True, text=True, check=False, timeout=30,
    )
    if query.returncode or query.stdout.strip() != "0":
        raise ValueError("Original QA has active generation or cannot be audited; refusing shutdown")
    if not Path(expected_jar).is_file():
        raise ValueError("Current BE bootJar missing")
    jar_sha = hashlib.sha256(Path(expected_jar).read_bytes()).hexdigest()
    java = Path(prior.get("command", command)[0]).resolve()
    if not java.is_file():
        raise ValueError("Owned QA Java executable absent")
    process.terminate()
    process.wait(timeout=30)
    manifest.update(status="ORIGINAL_QA_BE_STOPPED_NEW_NOT_STARTED", stoppedOwnedPid=process.pid)
    atomic_json(manifest_path, manifest)

    private = json.loads((fresh / "secrets.private.json").read_text(encoding="utf-8"))
    platform_keys = {"SYSTEMROOT", "WINDIR", "PATH", "JAVA_HOME", "TEMP", "TMP", "USERPROFILE",
                     "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA", "COMSPEC", "PROGRAMDATA",
                     "USERDOMAIN", "USERNAME"}
    runtime = {key: value for key, value in os.environ.items() if key.upper() in platform_keys}
    runtime.update(private)
    new_command = [str(java), "-jar", expected_jar, "--spring.profiles.active=qa",
                   f"--spring.config.location=classpath:/application.properties,file:{(fresh / 'application-qa.properties').as_posix()}"]
    with (fresh / "be.stdout.log").open("ab") as output, (fresh / "be.stderr.log").open("ab") as error:
        new = subprocess.Popen(new_command, cwd=be_repository.resolve(), env=runtime,
                               stdin=subprocess.DEVNULL, stdout=output, stderr=error,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    manifest.update(status="BE_STARTING", beProcess={"pid": new.pid, "jarSha256": jar_sha,
                                                        "profile": "qa", "database": schema})
    atomic_json(manifest_path, manifest)
    url = "http://127.0.0.1:18083/api/v1/health"
    for _ in range(90):
        if new.poll() is not None:
            raise RuntimeError("Fresh QA BE exited; inspect private stderr")
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=2) as response:
                if response.status == 200:
                    manifest["status"] = "BE_HEALTHY_NORMAL_LOGIN_REQUIRED"
                    atomic_json(manifest_path, manifest)
                    return {key: manifest[key] for key in ("status", "database", "beProcess", "stoppedOwnedPid")}
        except (OSError, ValueError):
            time.sleep(1)
    raise TimeoutError("Fresh owned QA BE health not ready; inspect private logs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--be-repository", type=Path, required=True)
    parser.add_argument("--revision-retest", action="store_true")
    parser.add_argument("--coaching-v1", action="store_true")
    parser.add_argument("--coaching-v1-retest", action="store_true")
    args = parser.parse_args()
    print(json.dumps(launch(
        args.campaign,
        args.be_repository,
        revision_retest=args.revision_retest,
        coaching_v1=args.coaching_v1,
        coaching_v1_retest=args.coaching_v1_retest,
    ), ensure_ascii=False))
