"""합성 Writing 검토도 최신 유형 관측과 실제 과제 근거를 제공한다."""

import json
from pathlib import Path

import pytest

from scripts.writing_cutover_provider import execute_writing_fixture

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures/writing_cutover_provider.json").read_text(encoding="utf-8")
)["candidates"]


def review_request(fixture):
    draft = fixture["draft"]
    segments = [{"id": "O1", "text": draft["originText"]}]
    for field, prefix in (
        ("providedFacts", "F"),
        ("requiredIntents", "I"),
        ("responseConstraints", "C"),
    ):
        segments.extend(
            {"id": f"{prefix}{i + 1}", "text": text} for i, text in enumerate(draft[field])
        )
    segments.append({"id": "N1", "text": draft["focusReason"]})
    payload = {
        "candidateId": "synthetic-current-review",
        "contentHash": "a" * 64,
        "writingType": fixture["writingType"],
        "segments": segments,
    }
    return {
        "schema_name_value": "writing_task_review",
        "messages": [
            {
                "role": "user",
                "content": "<writing-review-data>\n"
                + json.dumps(payload)
                + "\n</writing-review-data>",
            }
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("fixture", FIXTURES)
async def test_review_has_type_specific_evidence_without_changing_fixed_band(fixture):
    # 준비 / 실행: 기존 고정 과제의 실제 필드로 요청한다.
    result = await execute_writing_fixture(review_request(fixture))

    # 검증: 목표 난이도를 복사하지 않고 해당 fixture의 관측을 반환한다.
    assert result.data["estimatedBand"] == fixture["band"]
    writing_type = fixture["writingType"]
    if writing_type == "FREE":
        observation = result.data["contentChoiceObservation"]
        assert observation["status"] == "SUBSTANTIVE_CHOICE"
        assert observation["choiceDescription"]
        assert observation["evidenceSegmentIds"] == ["O1"]
    elif writing_type == "GUIDED":
        observation = result.data["productionObservation"]
        assert observation["minimumRequiredMeaning"]
        assert observation["necessaryRelations"]
        assert observation["evidenceSegmentIds"] == ["O1", "F1", "I1"]
    else:
        assert "contentChoiceObservation" not in result.data
        assert "productionObservation" not in result.data
