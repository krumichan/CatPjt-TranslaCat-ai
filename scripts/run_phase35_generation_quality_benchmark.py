from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.features.language_learning.quality import (  # noqa: E402
    DiversityCandidate,
    DiversityValidator,
)
from app.schemas.language_learning_quality import DiversityMetadata  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Phase 3.5 generated 20-question Level Test sessions for same-session "
            "duplicate/structural/background-knowledge violations."
        )
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--required-sessions", type=int, default=20)
    parser.add_argument("--required-items", type=int, default=20)
    args = parser.parse_args()

    raw = json.loads(args.dataset.read_text(encoding="utf-8"))
    sessions = raw["sessions"] if isinstance(raw, dict) else raw
    violations: list[dict[str, object]] = []

    for session in sessions:
        session_id = str(session.get("sessionId", "unknown"))
        items = session.get("items", [])
        if len(items) != args.required_items:
            violations.append(
                {
                    "sessionId": session_id,
                    "reason": "ITEM_COUNT",
                    "actual": len(items),
                    "expected": args.required_items,
                }
            )
            continue

        accepted: list[DiversityCandidate] = []
        validator = DiversityValidator()
        for index, raw_item in enumerate(items, start=1):
            metadata = DiversityMetadata.model_validate(raw_item["diversityMetadata"])
            content = str(raw_item["content"])
            decision = validator.validate(DiversityCandidate(content, metadata), accepted)
            if not decision.accepted:
                violations.append(
                    {
                        "sessionId": session_id,
                        "item": index,
                        "reason": decision.reason,
                    }
                )
                continue
            if raw_item.get("humanBackgroundKnowledgeViolation") is True:
                violations.append(
                    {
                        "sessionId": session_id,
                        "item": index,
                        "reason": "HUMAN_BACKGROUND_KNOWLEDGE",
                    }
                )
            accepted.append(DiversityCandidate(content, decision.metadata))

    passed = len(sessions) >= args.required_sessions and not violations
    result = {
        "sessionCount": len(sessions),
        "requiredSessions": args.required_sessions,
        "requiredItemsPerSession": args.required_items,
        "violationCount": len(violations),
        "violations": violations,
        "passed": passed,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
