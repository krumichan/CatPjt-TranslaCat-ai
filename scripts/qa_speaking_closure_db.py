"""Read-only, owner-fenced Speaking DB evidence for normal browser sessions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.qa_campaign_budget import atomic_json
from scripts.qa_campaign_integration import _owned_manifest, _qa_mysql_query


def snapshot(campaign: Path, session_id: int, mode: str) -> dict:
    if type(session_id) is not int or session_id <= 0 or mode not in {"READ_ALOUD", "FREE"}:
        raise ValueError("Exact positive owned Speaking session and mode required")
    environment = campaign / "integration-environment"
    manifest = _owned_manifest(environment)
    private = json.loads((environment / "secrets.private.json").read_text(encoding="utf-8"))
    owner = ("s.id=" + str(session_id) + " AND s.user_id=3 AND u.id=s.user_id "
             "AND u.social_type='GOOGLE' AND u.public_id='TC-GA5T-4LRB'")
    raw = _qa_mysql_query(manifest, private,
        "SELECT s.id,s.practice_mode,s.status,s.evaluation_status,s.completed_turns,s.total_duration_seconds "
        "FROM language_learning_speaking_session s JOIN user u ON " + owner + ";")
    fields = raw.split("\t")
    if len(fields) != 6 or int(fields[0]) != session_id or fields[1] != mode:
        raise ValueError("Speaking session/Google owner mismatch")
    turns_raw = _qa_mysql_query(manifest, private,
        "SELECT t.id,t.turn_index,t.status,t.recording_revision,t.duration_seconds,"
        "SHA2(t.transcript,256),t.failed_stage FROM language_learning_speaking_turn t "
        "JOIN language_learning_speaking_session s ON s.id=t.session_id JOIN user u ON "
        + owner + " WHERE t.session_id=" + str(session_id) + " ORDER BY t.turn_index;")
    turns = [dict(zip(("id", "index", "status", "revision", "durationSeconds", "transcriptSha256", "failedStage"),
                      row.split("\t"), strict=True)) for row in turns_raw.splitlines() if row]
    evaluations_raw = _qa_mysql_query(manifest, private,
        "SELECT e.problem_index,e.status,e.attempt_count,e.overall_score "
        "FROM language_learning_speaking_read_aloud_problem_evaluation e "
        "JOIN language_learning_speaking_session s ON s.id=e.session_id JOIN user u ON "
        + owner + " WHERE e.session_id=" + str(session_id) + " ORDER BY e.problem_index;") if mode == "READ_ALOUD" else ""
    evaluations = [dict(zip(("problemIndex", "status", "attemptCount", "overallScore"),
                            row.split("\t"), strict=True)) for row in evaluations_raw.splitlines() if row]
    whole = _qa_mysql_query(manifest, private,
        "SELECT e.status,e.overall_score,e.evaluation_confidence,e.eligibility_json FROM language_learning_speaking_evaluation e "
        "JOIN language_learning_speaking_session s ON s.id=e.session_id JOIN user u ON "
        + owner + " WHERE e.session_id=" + str(session_id) + " ORDER BY e.id;")
    whole_rows = [line.split("\t", 3) for line in whole.splitlines() if line]
    whole_summary = [{"status": row[0], "score": row[1], "confidence": row[2],
                      "eligibility": json.loads(row[3])} for row in whole_rows]
    metric_raw = _qa_mysql_query(manifest, private,
        "SELECT m.metric_type,m.state,m.score,m.confidence,m.not_evaluable_reason "
        "FROM language_learning_speaking_evaluation_metric m "
        "JOIN language_learning_speaking_evaluation e ON e.id=m.evaluation_id "
        "JOIN language_learning_speaking_session s ON s.id=e.session_id JOIN user u ON "
        + owner + " WHERE e.session_id=" + str(session_id) + " ORDER BY m.metric_type;")
    metrics = [dict(zip(("type", "state", "score", "confidence", "notEvaluableReason"),
                        row.split("\t"), strict=True)) for row in metric_raw.splitlines() if row]
    report = {"sessionId": session_id, "mode": mode, "status": fields[2],
              "evaluationStatus": fields[3], "completedTurns": int(fields[4]),
              "totalDurationSeconds": fields[5], "turns": turns,
              "perProblemEvaluations": evaluations,
              "wholeSessionEvaluations": whole_summary, "metricSummaries": metrics,
              "source": "owned isolated MySQL, read-only; transcript values hashed; user3 normal Google auth"}
    atomic_json(campaign / "closure-20260920-v1" / f"speaking-session{session_id}-db.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--mode", choices=("READ_ALOUD", "FREE"), required=True)
    args = parser.parse_args()
    report = snapshot(args.campaign.resolve(), args.session, args.mode)
    print(json.dumps({"sessionId": report["sessionId"], "status": report["status"],
                      "evaluationStatus": report["evaluationStatus"], "turns": len(report["turns"]),
                      "perProblemEvaluations": report["perProblemEvaluations"],
                      "wholeSessionEvaluations": report["wholeSessionEvaluations"]}))


if __name__ == "__main__":
    main()
