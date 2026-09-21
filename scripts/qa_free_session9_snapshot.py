"""Private, read-only export of the exact owned FREE evaluation request.

The learner transcript and conversation remain in a campaign-owned diagnostic
artifact; stdout contains only hashes and structural counts. No provider call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.qa_campaign_budget import atomic_json
from scripts.qa_campaign_integration import _owned_manifest, _qa_mysql_query


def snapshot(campaign: Path, output: Path, evidence_output: Path) -> dict[str, object]:
    environment = campaign / "integration-environment"
    manifest = _owned_manifest(environment)
    if manifest["campaign"] != "openai-speech-campaign-20260920":
        raise ValueError("Not the owned follow-up QA campaign")
    private = json.loads((environment / "secrets.private.json").read_text(encoding="utf-8"))
    owner = _qa_mysql_query(
        manifest,
        private,
        "SELECT COUNT(*) FROM language_learning_speaking_session s "
        "JOIN user u ON u.id=s.user_id WHERE s.id=9 AND s.user_id=3 "
        "AND u.social_type='GOOGLE' AND u.public_id='TC-GA5T-4LRB' "
        "AND s.practice_mode='FREE';",
    )
    if owner != "1":
        raise ValueError("Session 9 ownership/mode mismatch")
    rows = _qa_mysql_query(
        manifest,
        private,
        "SELECT j.problem_index,j.status,j.manual_retry_count,HEX(j.request_json) "
        "FROM language_learning_speaking_evaluation_job j "
        "WHERE j.session_id=9 ORDER BY j.problem_index;",
    ).splitlines()
    if len(rows) != 1:
        raise ValueError("Expected one durable whole-session evaluation request")
    problem, status, retries, encoded = rows[0].split("\t", 3)
    if problem != "0" or status != "SUCCEEDED":
        raise ValueError("Unexpected evaluation job identity/status")
    raw = bytes.fromhex(encoded)
    request = json.loads(raw)
    if request.get("sessionId") != "9" or request.get("practiceMode") != "FREE":
        raise ValueError("Stored request identity mismatch")
    report: dict[str, object] = {
        "source": "owned isolated QA DB evaluation_job.request_json; no provider call",
        "sessionId": 9,
        "jobStatus": status,
        "manualRetryCount": int(retries),
        "requestSha256": hashlib.sha256(raw).hexdigest(),
        "request": request,
    }
    if output.exists() or evidence_output.exists():
        raise ValueError("Refusing to overwrite private fixed evidence")
    evidence_rows = _qa_mysql_query(
        manifest,
        private,
        "SELECT m.metric_type,m.state,m.score,m.confidence,HEX(m.summary),"
        "HEX(m.evidence_json),HEX(m.not_evaluable_reason) "
        "FROM language_learning_speaking_evaluation_metric m "
        "JOIN language_learning_speaking_evaluation e ON e.id=m.evaluation_id "
        "WHERE e.session_id=9 ORDER BY m.metric_type;",
    ).splitlines()
    def decode(value: str) -> str | None:
        return None if value == "NULL" else bytes.fromhex(value).decode("utf-8")

    metrics = []
    for row in evidence_rows:
        kind, state, score, confidence, summary, items, reason = row.split("\t", 6)
        metrics.append({
            "type": kind, "state": state, "score": None if score == "NULL" else float(score),
            "confidence": float(confidence), "summary": decode(summary),
            "evidence": json.loads(decode(items) or "[]"), "notEvaluableReason": decode(reason),
        })
    if len(metrics) != 8:
        raise ValueError("Expected the complete eight-metric stored response")
    atomic_json(output, report)
    atomic_json(evidence_output, {
        "source": "owned isolated QA DB evaluation_metric rows; provider raw response unavailable",
        "sessionId": 9, "metrics": metrics,
    })
    return {
        "sessionId": 9,
        "requestSha256": report["requestSha256"],
        "jobStatus": status,
        "manualRetryCount": int(retries),
        "userTurns": len(request.get("userTurns", [])),
        "assistantTurns": len(request.get("assistantTurns", [])),
        "output": str(output),
        "evidenceOutput": str(evidence_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot(
        args.campaign.resolve(), args.output.resolve(), args.evidence_output.resolve()
    )))


if __name__ == "__main__":
    main()
