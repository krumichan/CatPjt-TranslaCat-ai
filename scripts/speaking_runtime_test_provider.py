"""실제 Core→LL Speaking 검사의 모델 출력만 제공하는 명시적 합성 Provider."""

import asyncio
import json
import os
from pathlib import Path

from app.ai.ports import StructuredGenerationResult

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "LANGUAGE_LEARNING_SPEAKING_CONVERSATION": "CONVERSATION",
    "LANGUAGE_LEARNING_SPEAKING_ASSISTANCE": "ASSISTANCE",
    "LANGUAGE_LEARNING_SPEAKING_EVALUATION": "EVALUATION",
    "LANGUAGE_LEARNING_SPEAKING_SESSION_COACHING": "COACHING",
}


async def execute_speaking_runtime_fixture(kwargs):
    stage = TASKS.get(kwargs.get("schema_name_value"))
    if stage is None:
        return None
    if os.environ.get("TRANSLACAT_TEST_MODEL_EXECUTION") != "1":
        raise RuntimeError("Synthetic provider requires explicit test marker")
    prompt = json.loads(kwargs["messages"][0]["content"].rsplit("\n", 1)[-1])
    if (
        not str(prompt.get("topic", "")).startswith("TRANSLACAT_SYNTHETIC_SPEAKING_")
        or prompt.get("originLanguage") != "ko"
        or prompt.get("learningLanguage") != "en"
    ):
        return None

    # 합성 제어는 Provider 응답·지연에만 적용한다. 세션·lease·Growth 상태는 LL이 처리한다.
    control_path = ROOT / ".tmp_ktor_m0/speaking-control.json"
    control = (
        json.loads(control_path.read_text(encoding="utf-8-sig")) if control_path.exists() else {}
    )
    scenario = control.get("scenario", "normal")
    if control.get("stage") not in (None, stage):
        scenario = "normal"
    if scenario == "hold":
        (ROOT / ".tmp_ktor_m0/speaking-provider-held.json").write_text(
            json.dumps({"stage": stage, "held": True}), encoding="utf-8"
        )
        await asyncio.sleep(60)
        raise TimeoutError()
    if scenario == "conversation_failure" and stage == "CONVERSATION":
        error = RuntimeError("Synthetic conversation unavailable")
        error.status_code = 503
        raise error
    if scenario == "evaluation_failure" and stage in {"EVALUATION", "COACHING"}:
        error = RuntimeError("Synthetic evaluation unavailable")
        error.status_code = 503
        raise error
    if scenario == "invalid_schema":
        return StructuredGenerationResult({"invalid": True}, 7, 2, "synthetic", "fixed")

    # 출력 내용은 고정하고 증거 식별자·인용만 제출된 합성 요청과 결합한다.
    if stage == "CONVERSATION":
        mode = prompt.get("practiceMode", "FREE")
        text = (
            "I went to the park yesterday."
            if mode == "READ_ALOUD"
            else "What did you do yesterday?"
        )
        output = {
            "assistantText": text,
            "intent": "SYNTHETIC_PRACTICE",
            "difficulty": "A2",
            "resolvedTopic": prompt["topic"],
            "shouldEnd": False,
        }
        if mode == "READ_ALOUD":
            output["scriptText"] = text
        if mode == "GUIDED":
            output.update(
                providedFacts=["You visited a park yesterday."],
                requiredIntents=["Describe yesterday's activity."],
                responseConstraints=["Use one complete sentence."],
            )
    elif stage == "ASSISTANCE":
        output = {"type": prompt["assistanceType"], "content": "합성 도움말"}
    elif stage == "EVALUATION":
        capabilities = prompt["evaluationCapabilities"]
        ids = capabilities["textEvidenceTurnIds"]
        if not ids:
            raise ValueError("Synthetic evaluation requires usable evidence")
        output = {
            "evaluationConfidence": 0.9,
            "metrics": [
                {
                    "type": metric,
                    "state": "EVALUATED",
                    "score": 80.0,
                    "confidence": 0.9,
                    "summary": "합성 평가 결과",
                    "evidence": [{"turnId": ids[0], "message": "합성 발화 근거"}],
                }
                for metric in capabilities["modelAssessableMetrics"]
            ],
        }
        if "GRAMMAR" in capabilities["modelAssessableMetrics"] and len(ids) >= 2:
            output["profileSignals"] = [
                {
                    "metricType": "GRAMMAR",
                    "direction": "STRENGTH",
                    "confidence": 0.9,
                    "evidenceTurnIds": ids[:2],
                    "patternKey": "synthetic-past-tense",
                    "recommendedFocus": "합성 과거형 연습",
                }
            ]
    else:
        turns = prompt["eligibleLearnerTurns"]
        if not turns:
            raise ValueError("Synthetic coaching requires usable evidence")
        turn = turns[0]
        output = {
            "contentStatus": "GROUNDED",
            "items": [
                {
                    "observationId": "synthetic-observation",
                    "kind": "OBSERVATION",
                    "turnId": turn["turnId"],
                    "sourceExcerpt": turn["transcript"][:80],
                    "message": "합성 대화에서 과거 활동을 설명했습니다.",
                }
            ],
        }
        if scenario == "wrong_snapshot":
            output["sourceSnapshotHash"] = "forbidden-model-owned-snapshot"
    return StructuredGenerationResult(output, 7, 2, "synthetic", "fixed")
