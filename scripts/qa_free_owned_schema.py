"""One-shot additional schema in the already owned QA MySQL, never the original DB.

No user rows are copied. Secrets are read from the owned private directory and
written only to a new private QA runtime file; stdout contains no credentials.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.qa_campaign_budget import atomic_json
from scripts.qa_campaign_integration import DOCKER, _verify_local_docker


NEW_SCHEMA = "translacat_qa_openai_speech_campaign_20260920_free_v3"
RETEST_SCHEMA = "translacat_qa_openai_speech_campaign_20260920_free_v3_retest"
COACHING_SCHEMA = "translacat_qa_openai_speech_campaign_20260920_coaching_v1"
COACHING_RETEST_SCHEMA = "translacat_qa_openai_speech_campaign_20260920_coaching_v1_retest"


def prepare(
    campaign: Path,
    *,
    revision_retest: bool = False,
    coaching_v1: bool = False,
    coaching_v1_retest: bool = False,
) -> dict[str, object]:
    campaign = campaign.resolve()
    selected = sum((revision_retest, coaching_v1, coaching_v1_retest))
    if selected > 1:
        raise ValueError("Choose exactly one isolated schema target")
    schema = (
        COACHING_RETEST_SCHEMA
        if coaching_v1_retest
        else COACHING_SCHEMA
        if coaching_v1
        else RETEST_SCHEMA
        if revision_retest
        else NEW_SCHEMA
    )
    if campaign.name != "openai-speech-campaign-20260920":
        raise ValueError("Existing campaign and ledger required")
    original = campaign / "integration-environment"
    manifest = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["owner"] != "translacat-isolated-qa" or manifest["campaign"] != campaign.name
            or manifest["names"]["database"] == schema or manifest["ports"]["mysql"] != 13318):
        raise ValueError("Original QA owner/server boundary mismatch")
    directory = (
        "free-coaching-v1-retest-owned-qa"
        if coaching_v1_retest
        else "free-coaching-v1-owned-qa"
        if coaching_v1
        else "free-v3-retest-owned-qa"
        if revision_retest
        else "free-v3-owned-qa"
    )
    redis_database = 4 if coaching_v1_retest else 3 if coaching_v1 else 2 if revision_retest else 1
    target = campaign / "closure-20260920-v1" / directory
    if target.exists():
        raise ValueError("New QA directory already exists; never overwrite or reseed")
    _verify_local_docker()
    container = manifest["names"]["mysqlContainer"]
    expected_id = next(item["id"] for item in manifest["createdResources"]
                       if item["type"] == "container" and item["name"] == container)
    actual = subprocess.run([str(DOCKER), "inspect", container, "--format", "{{.Id}}"],
                            capture_output=True, text=True, check=True, timeout=20).stdout.strip()
    if actual != expected_id:
        raise ValueError("Owned MySQL container identity changed")
    private = json.loads((original / "secrets.private.json").read_text(encoding="utf-8"))
    environment = {**os.environ, "MYSQL_PWD": private["QA_DB_ROOT_PASSWORD"]}

    def query(sql: str) -> str:
        completed = subprocess.run(
            [str(DOCKER), "exec", "--env", "MYSQL_PWD", container,
             "mysql", "--batch", "--skip-column-names", "-uroot", "-e", sql],
            env=environment, capture_output=True, text=True, timeout=30, check=False,
        )
        if completed.returncode:
            raise RuntimeError("Owned QA MySQL command failed; output suppressed")
        return completed.stdout.strip()

    version = query("SELECT @@version")
    existing = query("SHOW DATABASES").splitlines()
    if manifest["names"]["database"] not in existing or schema in existing:
        raise ValueError("Original QA database absent or new database already exists")
    target.mkdir(parents=True)
    evidence: dict[str, object] = {
        "owner": "translacat-isolated-qa", "campaign": campaign.name,
        "database": schema, "originalDatabase": manifest["names"]["database"],
        "mysqlContainerId": expected_id, "mysqlVersion": version,
        "redisDatabase": redis_database,
        "source": "new schema, no copied users/sets/session9",
        "status": "PREPARED_NOT_CREATED",
    }
    atomic_json(target / "manifest.json", evidence)
    query(f"CREATE DATABASE `{schema}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    evidence["status"] = "SCHEMA_CREATED_NOT_CONFIGURED"
    atomic_json(target / "manifest.json", evidence)
    query(f"GRANT ALL PRIVILEGES ON `{schema}`.* TO 'qa_app'@'%'")
    query("FLUSH PRIVILEGES")
    if schema not in query("SHOW DATABASES").splitlines():
        raise RuntimeError("New QA schema not visible")

    source = (original / "application-qa.properties").read_text(encoding="utf-8")
    if source.count(manifest["names"]["database"]) != 1:
        raise ValueError("Old database URL must be unique")
    source = source.replace(manifest["names"]["database"], schema)
    old_storage = str(original / "storage").replace("\\", "/")
    if source.count(old_storage) != 1:
        raise ValueError("Old QA storage root must be unique")
    source = source.replace(old_storage, str(target / "storage").replace("\\", "/"))
    source += f"\nspring.data.redis.database={redis_database}\n"
    (target / "storage").mkdir()
    (target / "application-qa.properties").write_text(source, encoding="utf-8")
    new_private = {key: private[key] for key in (
        "QA_DB_PASSWORD", "QA_AI_API_KEY", "QA_GOOGLE_CLIENT_ID")}
    new_private["QA_JWT_SECRET"] = base64.b64encode(secrets.token_bytes(64)).decode("ascii")
    atomic_json(target / "secrets.private.json", new_private)
    evidence["status"] = "READY_FOR_FRESH_BE_AND_NORMAL_GOOGLE_LOGIN"
    evidence["storage"] = str(target / "storage")
    atomic_json(target / "manifest.json", evidence)
    return {key: evidence[key] for key in (
        "owner", "campaign", "database", "originalDatabase", "mysqlContainerId", "redisDatabase", "status")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--revision-retest", action="store_true", help="Create only the second owned QA schema")
    parser.add_argument("--coaching-v1", action="store_true", help="Create the first coaching-v1 owned QA schema")
    parser.add_argument("--coaching-v1-retest", action="store_true", help="Create the second coaching-v1 owned QA schema")
    args = parser.parse_args()
    print(json.dumps(prepare(
        args.campaign,
        revision_retest=args.revision_retest,
        coaching_v1=args.coaching_v1,
        coaching_v1_retest=args.coaching_v1_retest,
    ), ensure_ascii=False))
