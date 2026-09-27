"""명시적 로컬 세션 검증용 정적 모델 출력만 반환한다. 상태·판정·저장은 LL을 통과한다."""

from __future__ import annotations

import copy
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.ai.ports import StructuredGenerationResult


@lru_cache(maxsize=1)
def runtime_candidates() -> list[dict[str, Any]]:
    return json.loads(
        (
            Path(__file__).resolve().parents[1] / "tests/fixtures/leveltest_runtime_models.json"
        ).read_text(encoding="utf-8")
    )


@lru_cache(maxsize=3)
def _evaluation(name: str) -> list[dict[str, Any]]:
    path = (
        Path(__file__).resolve().parents[2]
        / f"CatPjt-TranslaCat-ll/src/test/resources/contracts/leveltest-{name}-python-golden.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def execute_leveltest_runtime(kwargs: dict[str, Any]) -> StructuredGenerationResult | None:
    if os.environ.get("TRANSLACAT_TEST_MODEL_EXECUTION") != "1":
        raise RuntimeError("Level Test runtime fixture requires explicit local test marker")
    messages = kwargs.get("messages")
    if not isinstance(messages, list) or len(messages) != 1 or messages[0].get("role") != "user":
        return None
    prompt = messages[0].get("content")
    if not isinstance(prompt, str):
        return None
    schema = kwargs.get("schema_name_value")
    output = None
    model = "synthetic-level-generation"
    input_tokens, output_tokens = 7, 4
    candidates = runtime_candidates()

    # 입력은 고정 한국어→영어 합성 시나리오로 한정하고 외부 일반 요청을 fixture에 매칭하지 않는다.
    try:
        if schema == "LevelTestQuestionGenerationPayload":
            request = json.loads(prompt.rsplit("\n", 1)[1])
            if (
                not str(request.get("requestId", "")).startswith("lt:")
                or request.get("originLanguage") != "ko"
                or request.get("learningLanguage") != "en"
            ):
                return None
            fixture = next(
                (item for item in candidates if item["number"] == request.get("questionNumber")),
                None,
            )
            if fixture is None or fixture["candidate"]["itemType"] != request.get("itemType"):
                return None
            candidate = copy.deepcopy(fixture["candidate"])
            candidate["complexityBand"] = request["targetComplexityBand"]
            if request.get("preferredScenarioCategories"):
                candidate["diversityMetadata"]["scenarioCategory"] = request[
                    "preferredScenarioCategories"
                ][0]
            output = {"candidates": [candidate, copy.deepcopy(candidate)]}
        elif schema == "LevelTestVocabContextDesignPayload":
            request = json.loads(prompt.split("\n\n", 1)[1])
            if request.get("originLanguage") != "ko" or request.get("learningLanguage") != "en":
                return None
            scenario = next(iter(request.get("preferredScenarioCategories", [])), "WORK")
            output = {
                "designs": [
                    {
                        "designId": key,
                        "targetExpression": target,
                        "targetMeaning": "Synthetic meaning",
                        "semanticConstraint": "Synthetic contrast",
                        "scenarioCategory": scenario,
                        "communicativeIntent": "DESCRIBE",
                    }
                    for key, target in [("A", "first"), ("B", "second")]
                ]
            }
        elif schema in {
            "LevelTestChoiceSemanticVerificationPayload",
            "LevelTestTaskSufficiencyVerificationPayload",
        }:
            request = json.loads(prompt.split("\n\n", 1)[1])
            if not any(
                item["candidate"]["promptText"] == request.get("promptText") for item in candidates
            ):
                return None
            if schema == "LevelTestChoiceSemanticVerificationPayload":
                output = {
                    "verifiable": True,
                    "plausibleOptionKeys": ["A"],
                    "nearEquivalentOptionKeys": [],
                    "bestOptionKey": "A",
                    "bestOptionAdvantage": "CLEAR",
                }
            else:
                output = {
                    "sufficient": True,
                    "requiresExternalKnowledge": False,
                    "requiresProblemSolving": False,
                    "providedFactsSufficient": True,
                    "communicativeGoalsClear": True,
                    "instructionAndTaskRolesSeparated": True,
                }
        elif schema == "LANGUAGE_LEARNING_WRITING_EVALUATION":
            request = json.loads(
                prompt.split("<evaluation-data>\n", 1)[1].split("\n</evaluation-data>", 1)[0]
            )
            if (
                request.get("context") != "LEVEL_TEST"
                or request.get("userAnswer") != "Synthetic answer"
            ):
                return None
            if not any(
                item["candidate"]["promptText"] == request.get("originSentence")
                for item in candidates
            ):
                return None
            fixture = next(
                (
                    item
                    for item in _evaluation("writing")
                    if item["itemType"] == request.get("taskType")
                ),
                None,
            )
            output = fixture["providerOutput"] if fixture else None
            model, input_tokens, output_tokens = "synthetic-leveltest", 17, 11
        elif schema == "ListeningINTERPRETATION":
            request = json.loads(prompt.split("\n\n", 1)[1])
            if request.get("answer") != "Synthetic answer" or not any(
                item["candidate"]["referencePayload"].get("sourceText") == request.get("sourceText")
                for item in candidates
            ):
                return None
            fixture = next(
                item
                for item in _evaluation("listening")
                if item["request"]["itemType"] == "LISTENING_INTERPRETATION"
            )
            output = copy.deepcopy(fixture["providerOutput"])
            # 고정 출력의 근거 문자열은 같은 정적 runtime fixture에 있는 두 의미 단위를 사용한다.
            original = fixture["request"]["keyMeaningUnits"]
            runtime = next(item for item in candidates if item["number"] == 14)["candidate"][
                "referencePayload"
            ]["keyMeaningUnits"]
            encoded = json.dumps(output, ensure_ascii=False)
            for left, right in zip(original, runtime, strict=True):
                encoded = encoded.replace(left, right)
            output = json.loads(encoded)
            model, input_tokens, output_tokens = "synthetic-leveltest", 17, 11
        elif schema == "LevelTestSpeakingEvaluationPayload":
            request = json.loads(prompt.split("\n\n", 1)[1])
            if (
                request.get("transcript", {}).get("text") != "Synthetic transcript"
                or request.get("transcript", {}).get("metadata", {}).get("provider")
                != "test-provider"
            ):
                return None
            if not any(
                item["candidate"]["promptText"] == request.get("promptText") for item in candidates
            ):
                return None
            fixture = next(
                (
                    item
                    for item in _evaluation("speaking")
                    if item["itemType"] == request.get("itemType")
                ),
                None,
            )
            output = fixture["providerOutput"] if fixture else None
            model, input_tokens, output_tokens = "synthetic-leveltest", 17, 11
    except (KeyError, IndexError, ValueError, TypeError):
        return None

    return (
        None
        if output is None
        else StructuredGenerationResult(
            copy.deepcopy(output), input_tokens, output_tokens, "test-provider", model
        )
    )
