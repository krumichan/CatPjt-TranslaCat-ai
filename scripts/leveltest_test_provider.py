"""명시적 로컬 서버 전용 Level Test 합성 모델 출력. 업무 상태는 변경하지 않는다."""

from __future__ import annotations

import copy
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.ai.ports import StructuredGenerationResult
from scripts.leveltest_runtime_provider import execute_leveltest_runtime


@lru_cache(maxsize=4)
def _fixtures(name: str) -> list[dict[str, Any]]:
    root = Path(__file__).resolve().parents[2] / "CatPjt-TranslaCat-ll/src/test/resources/contracts"
    return json.loads((root / f"leveltest-{name}-python-golden.json").read_text(encoding="utf-8"))


def execute_leveltest_fixture(kwargs: dict[str, Any]) -> StructuredGenerationResult | None:
    schema = kwargs.get("schema_name_value")
    generation_schemas = {"LevelTestQuestionGenerationPayload", "LevelTestVocabContextDesignPayload", "LevelTestVocabContextRepairPayload",
                          "LevelTestChoiceSemanticVerificationPayload", "LevelTestTaskSufficiencyVerificationPayload"}
    if schema not in generation_schemas | {"LANGUAGE_LEARNING_WRITING_EVALUATION", "ListeningINTERPRETATION", "LevelTestSpeakingEvaluationPayload"}:
        return None
    if os.environ.get("TRANSLACAT_TEST_MODEL_EXECUTION") != "1":
        raise RuntimeError("Level Test fixture requires explicit local test marker")
    messages = kwargs.get("messages")
    if not isinstance(messages, list) or len(messages) != 1 or messages[0].get("role") != "user":
        return None
    prompt = messages[0].get("content")
    if not isinstance(prompt, str):
        return None

    # 생성 전체 업무는 LL에서 실행하며 서버는 검증된 합성 정상 경로의 모델 출력만 제공한다.
    if schema in generation_schemas:
        kinds = {"LevelTestQuestionGenerationPayload": "GENERATION", "LevelTestVocabContextDesignPayload": "VOCAB_CONTEXT_DESIGN",
                 "LevelTestVocabContextRepairPayload": "VOCAB_CONTEXT_REPAIR", "LevelTestChoiceSemanticVerificationPayload": "CHOICE_VERIFICATION",
                 "LevelTestTaskSufficiencyVerificationPayload": "TASK_SUFFICIENCY_VERIFICATION"}
        for case in _fixtures("generation-execution"):
            if not case["name"].endswith("_normal"):
                continue
            for call in case["calls"]:
                if call["type"] == "LANGUAGE_LEARNING_LEVEL_TEST_" + kinds[schema] and call["prompt"] == prompt:
                    return StructuredGenerationResult(copy.deepcopy(call["output"]), 7, 4, "test-provider", "synthetic-level-generation")
        return execute_leveltest_runtime(kwargs)

    # 텍스트 평가 입력은 Python 원본에서 추출한 합성 프롬프트와 완전히 같은 경우만 받는다.
    if schema != "LevelTestSpeakingEvaluationPayload":
        name = "writing" if schema == "LANGUAGE_LEARNING_WRITING_EVALUATION" else "listening"
        fixture = next((item for item in _fixtures(name) if item["prompt"] == prompt), None)
        if fixture is None:
            return execute_leveltest_runtime(kwargs)
    else:
        # 실제 기술 STT의 품질 필드는 달라질 수 있지만 합성 세션·문항·원문만 허용한다.
        prefix = "Evaluate this single Level Test speaking response. Do not calculate a cross-item/domain score.\n\n"
        if not prompt.startswith(prefix):
            return None
        payload = json.loads(prompt[len(prefix):])
        fixture = next((item for item in _fixtures("speaking")
            if all(payload.get(key) == value for key, value in item["request"].items())), None)
        if fixture is None or payload.get("transcript", {}).get("text") != "Synthetic transcript":
            return execute_leveltest_runtime(kwargs)
        if payload["transcript"].get("metadata", {}).get("provider") != "test-provider":
            return None
        assets = Path(__file__).resolve().parents[2] / "CatPjt-TranslaCat-ll/src/main/resources/leveltest"
        if kwargs.get("instructions") != (assets / "speaking-evaluation-system-prompt.txt").read_text(encoding="utf-8"):
            raise ValueError("Level Test speaking instruction contract mismatch")

    # 같은 원본 fixture의 모델 출력만 반환하고 채택·점수·성장 처리는 실제 LL이 수행한다.
    return StructuredGenerationResult(copy.deepcopy(fixture["providerOutput"]), 17, 11, "test-provider", "synthetic-leveltest")
