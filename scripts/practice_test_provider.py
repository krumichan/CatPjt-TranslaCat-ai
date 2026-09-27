"""명시적 합성 HTTP 서버에서만 사용하는 Reading 모델 출력 fixture."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from app.ai.ports import StructuredGenerationResult

_TEST_ROOT = Path(__file__).resolve().parents[1] / ".tmp_ktor_m0"


async def execute_practice_fixture(kwargs: dict[str, Any]) -> StructuredGenerationResult | None:
    if not str(kwargs.get("schema_name_value", "")).startswith("practice_"):
        return None
    if os.environ.get("TRANSLACAT_TEST_MODEL_EXECUTION") != "1":
        raise RuntimeError("Practice fixture requires explicit local test marker")

    # 실제 LL이 전송한 프롬프트·schema 경계를 통과하며 고정된 모델 출력만 반환한다.
    root = Path(__file__).resolve().parents[2] / "CatPjt-TranslaCat-ll/src/main/resources/practice"
    stages = ("passage", "generation", "verification", "repair", "explanation")
    stage = next(
        (
            name
            for name in stages
            if kwargs.get("instructions")
            == (root / f"{name}-system.txt").read_text(encoding="utf-8")
        ),
        None,
    )
    if stage is None:
        raise ValueError("Practice instruction contract mismatch")
    messages = kwargs.get("messages")
    if not isinstance(messages, list) or len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("Practice message contract mismatch")
    text = messages[0]["content"]
    payload = json.loads(
        text.split("<practice-data>\n", 1)[1].split("\n</practice-data>", 1)[0]
        if "<practice-data>\n" in text
        else text.split("\n\n", 1)[1]
    )

    # 명시적 합성 서버의 격리 제어 파일은 모델 응답만 고정하며 업무 상태는 변경하지 않는다.
    control_path = _TEST_ROOT / "practice-control.json"
    control = json.loads(control_path.read_text(encoding="utf-8")) if control_path.exists() else {}
    stats_path = _TEST_ROOT / "practice-stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    stats[stage] = stats.get(stage, 0) + 1
    _TEST_ROOT.mkdir(exist_ok=True)
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    if (
        control.get("mode") == payload.get("mode")
        and control.get("failure") == "HOLD_P2_PASSAGE"
        and stage == "passage"
        and payload.get("passageId") == "p2"
    ):
        # OS 강제 종료 검사는 실제 HTTP 호출이 진행 중인 동안 합성 출력만 지연한다.
        (_TEST_ROOT / "practice-hold.json").write_text(
            '{"stage":"passage","held":true}', encoding="utf-8"
        )
        await asyncio.sleep(45)

    if (
        control.get("mode") == payload.get("mode")
        and control.get("failure") == "P2_VERIFIER_BINDING"
        and stage == "verification"
        and all(question.get("passageId") == "p2" for question in payload["questions"])
    ):
        return StructuredGenerationResult(
            {"verdicts": []}, 1, 1, "test-provider", "synthetic-reading-invalid-binding"
        )

    if stage == "passage":
        passage = (
            "Mina put a raincoat by the door. The sky grew dark.\n\n"
            "She moved the picnic indoors. Everyone brought a warm drink."
        )
        clues = [
            "Mina put a raincoat by the door.",
            "The sky grew dark.",
            "She moved the picnic indoors.",
        ]
        plans = [
            {
                **{
                    key: slot[key]
                    for key in ("globalOrder", "skillTag", "difficulty", "complexityBand")
                },
                "clueQuote": clues[index],
                "questionFocus": f"Synthetic focus {slot['globalOrder']}",
                "unstatedInference": f"Mina expected a storm at time {slot['globalOrder']}.",
            }
            for index, slot in enumerate(payload["plannedQuestionSlots"])
        ]
        output = {
            "passageId": payload["passageId"],
            "passageText": passage,
            "questionPlans": plans,
            "inferenceClueQuote": clues[0],
            "unstatedInference": "Mina expected a storm.",
            "inferencePlans": [
                {
                    "questionOrder": plan["globalOrder"],
                    "clueQuote": plan["clueQuote"],
                    "unstatedInference": plan["unstatedInference"],
                }
                for plan in plans
            ],
        }
    elif stage == "generation":
        output = {
            "questions": [
                {
                    "order": slot["order"],
                    "questionType": "SINGLE_CHOICE",
                    "difficulty": slot["difficulty"],
                    "complexityBand": slot["complexityBand"],
                    "skillTag": slot["skillTag"],
                    "passageId": slot["passageId"],
                    "passageText": slot["passageText"],
                    "prompt": f"Which claim matches judgment {slot['order']}?",
                    "options": [{"key": key, "text": f"Synthetic option {key}"} for key in "ABCD"],
                    "correctAnswer": ["A"],
                    "evidenceText": "The sky grew dark.",
                    "explanationLearning": "The passage contains the relevant evidence.",
                    "targetExpression": None,
                    "canonicalKey": None,
                    "reviewTarget": False,
                    "vocabularyCandidates": ["raincoat"],
                    "inferenceClueQuote": slot["currentQuestionPlan"]["clueQuote"],
                    "unstatedInference": slot["currentQuestionPlan"]["unstatedInference"],
                }
                for slot in payload["candidateSlots"]
            ]
        }
    elif stage == "verification":
        output = {
            "verdicts": [
                {
                    "order": question["order"],
                    "bestAnswerKey": "A",
                    "ambiguous": False,
                    "supported": True,
                    "reason": "Synthetic assessment",
                    "modeFit": True,
                    "answerLeakage": False,
                    "contextDependent": True,
                    "distractorsPlausible": True,
                    "stemPresuppositionsSupported": True,
                    "distinctReadingTask": True,
                    "readingOperation": "DISCOURSE_STRUCTURE"
                    if question["skillTag"] == "STRUCTURE"
                    else "INFERENCE",
                    "stemEvidenceSpanIds": [f"{question['passageId']}:s1"],
                    "boundedStructureScope": True,
                }
                for question in payload["questions"]
            ]
        }
    elif stage == "explanation":
        output = {
            "explanations": [
                {
                    "order": question["order"],
                    "text": "합성 지문의 단서를 바탕으로 정답을 확인합니다.",
                }
                for question in payload["questions"]
            ]
        }
    else:
        output = {"wrongOptions": payload["wrongOptions"]}
    return StructuredGenerationResult(output, 1, 1, "test-provider", "synthetic-reading")
