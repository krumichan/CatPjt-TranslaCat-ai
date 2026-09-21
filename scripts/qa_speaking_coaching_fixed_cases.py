"""Run the six pre-fixed FREE coaching-v1 cases once against the owned QA AI.

The artifact contains synthetic transcripts and provider responses, so it is
written only below the private campaign directory.  This script never changes
production data and never retries a case at the harness level.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.qa_campaign_budget import CampaignLedger, atomic_json


def _turn(index: int, transcript: str, *, confidence: float = 0.94, excluded: bool = False) -> dict:
    return {
        "turnId": f"learner-{index}",
        "turnIndex": index,
        "transcript": transcript,
        "sttConfidence": confidence,
        "durationSeconds": 7.0,
        "segments": [],
        "audioReference": f"synthetic-fixed-{index}",
        "audioAvailable": False,
        "excludedFromEvaluation": excluded,
        "assistanceUsage": [],
        "recordingRevision": 1,
    }


FIXED_CASES = [
    {
        "id": "normal-weekend",
        "topic": "주말 경험",
        "question": "週末は何をして過ごしましたか。",
        "transcript": "週末は家族と公園へ行きました。天気が良かったので、一緒に散歩して楽しかったです。",
        "class": "SUFFICIENT_NORMAL",
    },
    {
        "id": "normal-work",
        "topic": "업무 계획",
        "question": "今週、仕事で取り組みたいことは何ですか。",
        "transcript": "今週は報告書を早めに完成させたいです。そのために、毎朝予定を確認してから作業を始めます。",
        "class": "SUFFICIENT_NORMAL",
    },
    {
        "id": "understandable-past-error",
        "topic": "영화 경험",
        "question": "昨日はどんな映画を見ましたか。",
        "transcript": "昨日、友達と映画を見ますた。話は少し難しいでしたが、最後まで面白かったです。",
        "class": "UNDERSTANDABLE_ERROR",
    },
    {
        "id": "understandable-ability-error",
        "topic": "마감 계획",
        "question": "忙しい仕事をどうやって終える予定ですか。",
        "transcript": "仕事が忙しいけど、順番を決めたら明日まで終わるできると思います。",
        "class": "UNDERSTANDABLE_ERROR",
    },
    {
        "id": "insufficient-excluded",
        "topic": "여행",
        "question": "旅行について教えてください。",
        "transcript": "来月、京都へ行きたいです。",
        "class": "INSUFFICIENT_EVIDENCE",
        "excluded": True,
    },
    {
        "id": "insufficient-low-confidence",
        "topic": "취미",
        "question": "趣味について教えてください。",
        "transcript": "えっと、たぶん、わかりません。",
        "class": "INSUFFICIENT_EVIDENCE",
        "confidence": 0.2,
    },
]


def _request(case: dict) -> dict:
    turn = _turn(
        1,
        case["transcript"],
        confidence=float(case.get("confidence", 0.94)),
        excluded=bool(case.get("excluded", False)),
    )
    source = json.dumps({"case": case["id"], "turns": [turn]}, ensure_ascii=False, sort_keys=True)
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return {
        "requestId": f"coaching-fixed-{case['id']}",
        "idempotencyKey": f"coaching-fixed-{case['id']}-{source_hash}",
        "sessionId": f"fixed-{case['id']}",
        "topic": case["topic"],
        "practiceMode": "FREE",
        "evaluationScope": "SESSION",
        "originLanguage": "ko",
        "learningLanguage": "ja",
        "userTurns": [turn],
        "assistantTurns": [
            {"turnId": "assistant-0", "turnIndex": 0, "text": case["question"]},
            {"turnId": "assistant-trailing", "turnIndex": 1, "text": "もう少し教えてください。"},
        ],
        "evaluationPolicyVersion": "speaking-evaluation-policy-v2",
        "resultKind": "SESSION_COACHING",
        "resultPolicyVersion": "free-session-coaching-v1",
        "sourceSnapshotHash": source_hash,
    }


def run(campaign: Path, output: Path, *, base_url: str) -> dict:
    campaign = campaign.resolve()
    output = output.resolve()
    if campaign.name != "openai-speech-campaign-20260920" or campaign not in output.parents:
        raise ValueError("Output must remain inside the authorized private campaign")
    if output.exists():
        raise ValueError("Fixed-case output already exists; never repeat or overwrite")
    ledger = CampaignLedger(campaign / "budget-ledger.json", campaign.name)
    before = ledger.snapshot()
    if before["stoppedReason"] or any(call["status"] == "STARTED" for call in before["calls"]):
        raise ValueError("Campaign halted or has an in-flight call")
    secrets = json.loads((campaign / "integration-environment" / "secrets.private.json").read_text(encoding="utf-8"))
    results = []
    for case in FIXED_CASES:
        payload = _request(case)
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/v1/language-learning/speaking/coach",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-API-KEY": secrets["QA_AI_API_KEY"]},
            method="POST",
        )
        entry = {"case": case, "request": payload, "startedAt": datetime.now(UTC).isoformat()}
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                entry.update(httpStatus=response.status, response=json.loads(response.read().decode("utf-8")))
        except urllib.error.HTTPError as exc:
            entry.update(httpStatus=exc.code, response=json.loads(exc.read().decode("utf-8")))
        except BaseException as exc:
            entry.update(error={"type": type(exc).__name__, "message": str(exc)})
            results.append(entry)
            atomic_json(output, {
                "status": "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                "partial": True,
                "createdAt": datetime.now(UTC).isoformat(),
                "cases": results,
            })
            raise
        results.append(entry)
        atomic_json(output, {
            "status": "RUNNING",
            "partial": True,
            "createdAt": datetime.now(UTC).isoformat(),
            "cases": results,
        })
    after = ledger.snapshot()
    artifact = {
        "status": "COMPLETED",
        "partial": False,
        "createdAt": datetime.now(UTC).isoformat(),
        "caseContract": "six cases fixed before the final-revision provider run; one HTTP execution per case",
        "providerStartsBefore": before["totalsIncludingReserved"]["starts"],
        "providerStartsAfter": after["totalsIncludingReserved"]["starts"],
        "cases": results,
    }
    atomic_json(output, artifact)
    return {
        "status": artifact["status"],
        "cases": len(results),
        "httpSucceeded": sum(item.get("httpStatus") == 200 for item in results),
        "providerStartsDelta": artifact["providerStartsAfter"] - artifact["providerStartsBefore"],
        "output": str(output),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18084")
    args = parser.parse_args()
    print(json.dumps(run(args.campaign, args.output, base_url=args.base_url), ensure_ascii=False))
