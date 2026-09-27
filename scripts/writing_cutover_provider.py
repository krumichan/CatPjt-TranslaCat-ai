"""실제 로컬 계정의 Writing 통합 검사용 고정 모델 출력."""
import asyncio
import json
from functools import lru_cache
from pathlib import Path

from app.ai.ports import StructuredGenerationResult
from app.ai.providers.openai.response import OpenAIProviderResponseError


@lru_cache(maxsize=1)
def _fixtures():
    path = Path(__file__).resolve().parents[1] / "tests/fixtures/writing_cutover_provider.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _payload(kwargs, tag):
    messages = kwargs.get("messages")
    if not isinstance(messages, list) or len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("Synthetic writing message required")
    return json.loads(messages[0]["content"].split(f"<{tag}>\n", 1)[1].split(f"\n</{tag}>", 1)[0])


async def execute_writing_fixture(kwargs):
    name = kwargs.get("schema_name_value")
    if name not in {"writing_candidate_batch", "writing_task_review",
                    "LANGUAGE_LEARNING_WRITING_EVALUATION"}:
        return None
    fixtures = _fixtures()
    candidates = fixtures["candidates"]
    if name == "writing_candidate_batch":
        payload = _payload(kwargs, "learning-data")
        control_root = Path(__file__).resolve().parents[1] / ".tmp_ktor_m0"
        control_path = control_root / "writing-control.json"
        control = (json.loads(control_path.read_text(encoding="utf-8"))
                   if control_path.exists() else {})
        if control.get("writingType") == payload.get("writingType"):
            if (control.get("failure") == "REFUSE_REGENERATION"
                    and payload.get("requestId", "").startswith("daily-regen-")):
                raise OpenAIProviderResponseError(
                    "Synthetic refusal", reason_code="REFUSAL", retryable=False,
                )
            if (control.get("failure") == "HOLD_AFTER_FIRST"
                    and len(payload["diversityContext"]["currentSession"]) == 1):
                # 실제 모델 HTTP가 진행 중일 때 OS 재시작 검사에 필요한 합성 출력만 지연한다.
                (control_root / "writing-hold.json").write_text('{"held":true}', encoding="utf-8")
                await asyncio.sleep(45)

        # 실패 원본 대조는 후속 기존 시도에서도 같은 합법적 거부를 고정한다.
        if (payload.get("requestId") in {
                "synthetic-regenerate-refusal", "synthetic-generation-refusal",
                "synthetic-source-preservation-fail"}
                or payload.get("snapshotId") == "synthetic-regeneration-refusal-seed"):
            raise OpenAIProviderResponseError(
                "Synthetic refusal", reason_code="REFUSAL", retryable=False,
            )
        if not payload.get("requestId", "").startswith("daily-generate-"):
            return None
        if (payload["originLanguage"] != "ko" or payload["learningLanguage"] != "en"
                or payload.get("selectedKeywords")):
            raise ValueError("Synthetic unkeyed Korean-English writing context required")

        # 사용한 고정 출력은 반복하지 않는다. 실제 다양성·난이도 검사는 LL 업무 경로가 수행한다.
        seen = {item["content"] for item in payload["diversityContext"]["currentSession"]}
        choices = [item for item in candidates
                   if item["writingType"] == payload["writingType"]
                   and item["band"] == payload["generationPlan"]["targetBand"]
                   and item["draft"]["originText"] not in seen]
        if not choices:
            raise ValueError("No unused synthetic writing output")
        output = {"items": [choices[0]["draft"]]}
    elif name == "writing_task_review":
        payload = _payload(kwargs, "writing-review-data")
        text = next(item["text"] for item in payload["segments"] if item["id"] == "O1")
        fixture = next((item for item in candidates if item["draft"]["originText"] == text), None)
        if fixture is None:
            return None

        # 검증 응답의 band는 고정 fixture에 기록된 값이며 LL의 목표 band를 읽어 덮어쓰지 않는다.
        output = {
            "candidateId": payload["candidateId"], "contentHash": payload["contentHash"],
            "confidence": None, "verdict": "PASS", "observedWritingType": fixture["writingType"],
            "difficultyStatus": "ASSESSED", "estimatedBand": fixture["band"],
            "alternativeBand": None, "difficultyConfidence": None,
            "difficultyEvidenceSegmentIds": ["O1"], "issues": [], "productionDemandChecks": None,
            "checks": [{"criterion": name, "status": "PASS", "evidenceSegmentIds": evidence}
                       for name, evidence in (("ORIGIN_LANGUAGE", ["O1"]),
                                              ("TASK_VALIDITY", ["O1"]),
                                              ("ANSWER_LEAK", ["O1", "N1"]),
                                              ("NATURALNESS", ["O1"]), ("NOTE_QUALITY", ["N1"]))],
        }
    else:
        payload = _payload(kwargs, "evaluation-data")
        source_texts = {item["draft"]["originText"] for item in candidates}
        if (payload.get("userAnswer") != "Synthetic answer"
                or payload.get("originSentence") not in source_texts):
            return None
        output = fixtures["evaluation"]
    return StructuredGenerationResult(output, 7, 2, "test-provider", "synthetic-writing")
