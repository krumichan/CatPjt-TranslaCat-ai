"""Private, read-only review export for an exact owned QA Reading set.

The file contains unreleased answers and passage text; never put it in INFO logs
or a public repository. This script starts no provider or generation work.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import qa_campaign_integration as qa


def decode(encoded: str) -> str | None:
    return None if encoded == "NULL" else bytes.fromhex(encoded).decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--set-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    environment = args.environment.resolve()
    output = args.output.resolve()
    if args.set_id <= 0 or output.parent != environment.parent or output.exists():
        parser.error("New private output inside owned campaign and positive set ID required")
    manifest = qa._owned_manifest(environment)
    if manifest["campaign"] != "openai-speech-campaign-20260920":
        raise ValueError("Wrong QA campaign")
    private = json.loads((environment / "secrets.private.json").read_text(encoding="utf-8"))
    def query(sql: str) -> str:
        return qa._qa_mysql_query(manifest, private, sql)

    row = query(
        "SELECT s.generation_status,HEX(s.generation_request_json) "
        "FROM language_learning_practice_set s JOIN user u ON u.id=s.user_id "
        f"WHERE s.id={args.set_id} AND u.id=3 AND u.social_type='GOOGLE' AND s.domain='READING';"
    ).split("\t")
    if len(row) != 2 or row[0] != "READY":
        raise ValueError("Exact READY owned QA Reading set required")
    request = json.loads(decode(row[1]) or "null")
    rows = query(
        "SELECT q.order_no,HEX(q.passage_text),HEX(q.prompt),HEX(q.options_json),"
        "HEX(q.correct_answer_json),HEX(q.evidence_text),HEX(q.explanation_learning),HEX(q.explanation_origin) "
        "FROM language_learning_practice_question q "
        f"WHERE q.practice_set_id={args.set_id} ORDER BY q.order_no;"
    )
    questions = []
    for line in rows.splitlines():
        fields = line.split("\t")
        if len(fields) != 8:
            raise ValueError("Unexpected question review row")
        questions.append({
            "order": int(fields[0]), "passageText": decode(fields[1]),
            "prompt": decode(fields[2]), "options": json.loads(decode(fields[3]) or "null"),
            "correctAnswer": json.loads(decode(fields[4]) or "null"),
            "evidenceText": decode(fields[5]), "explanationLearning": decode(fields[6]),
            "explanationOrigin": decode(fields[7]),
        })
    if [item["order"] for item in questions] != [1, 2, 3, 4, 5]:
        raise ValueError("Not a complete five-question Reading set")
    qa._write_json(output, {"setId": args.set_id, "privateDiagnostic": True,
                            "planBundles": request.get("readingBundles"), "questions": questions,
                            "providerCallsStarted": 0})
    print(json.dumps({"setId": args.set_id, "questions": len(questions),
                      "privateArtifact": str(output)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
