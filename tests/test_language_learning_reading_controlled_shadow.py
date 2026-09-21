from __future__ import annotations

import asyncio
import json
import re
from collections import Counter

import pytest
from pydantic import ValidationError

from app.ai.model_policy import AiModelTier, get_task_model_policy
from app.ai.ports import StructuredGenerationResult
from app.ai.prompt_registry import PROMPT_MAP
from app.ai.providers.openai.schema import build_openai_text_config
from app.core.config import Settings, settings
from app.features.language_learning.reading_vocabulary.prompts import (
    PRACTICE_VERIFICATION_SYSTEM_PROMPT,
    READING_STRUCTURE_MODE_FIT_CLARIFICATION,
)
from app.features.language_learning.reading_vocabulary.reading_difficulty_shadow import (
    READING_DIFFICULTY_SHADOW_SCHEMA,
    READING_DIFFICULTY_SHADOW_SYSTEM_PROMPT,
    READING_DIFFICULTY_SHADOW_TYPE_NAME,
    reading_difficulty_shadow_sample_bucket,
    should_sample_reading_difficulty_shadow,
)
from app.features.language_learning.reading_vocabulary.reading_difficulty_recipe import (
    READING_DIFFICULTY_RECIPE_VERSION,
    READING_DIFFICULTY_SHADOW_RUBRIC_VERSION,
    READING_PASSAGE_BLUEPRINT_VERSION,
)
from app.features.language_learning.reading_vocabulary.service import (
    ReadingVocabularyGenerationService,
    _PRACTICE_VERIFICATION_SCHEMA,
)
from scripts.run_reading_quality_parity_benchmark import (
    QUALITY_ONLY_SCHEMA,
    QUALITY_ONLY_SYSTEM_PROMPT,
)
from tests import test_language_learning_reading_vocabulary as practice_fixtures


@pytest.fixture(autouse=True)
def historical_shadow_fake_uses_luna(monkeypatch):
    """Keep pre-selection provider call-count characterization on its original route."""
    monkeypatch.setattr(settings, "AI_READING_GENERATION_MODEL", "LUNA")


class ControlledShadowProvider(practice_fixtures.PipelineProvider):
    def __init__(self, *, shadow_behavior: str = "valid", **kwargs):
        super().__init__(**kwargs)
        self.shadow_behavior = shadow_behavior
        self.shadow_payloads: list[dict] = []
        self.shadow_schemas: list[dict | None] = []
        self.calls_before_shadow: Counter | None = None
        self.shadow_started = asyncio.Event()
        self.shadow_cleaned_up = asyncio.Event()

    async def call_with_metadata(self, type_name, data, schema=None):
        if type_name != READING_DIFFICULTY_SHADOW_TYPE_NAME:
            raise AssertionError(f"unexpected metadata task: {type_name}")
        self.calls_before_shadow = Counter(self.calls)
        self.calls[type_name] += 1
        self.shadow_schemas.append(schema)
        payload = _shadow_data(data)
        self.shadow_payloads.append(payload)
        self.shadow_started.set()

        if self.shadow_behavior in {"timeout", "cancel"}:
            try:
                await asyncio.Future()
            finally:
                self.shadow_cleaned_up.set()
        if self.shadow_behavior == "provider_error":
            raise RuntimeError("synthetic shadow provider failure")
        if self.shadow_behavior == "malformed":
            data = {"passageAssessments": "not-an-array", "questionAssessments": []}
        else:
            data = self._shadow_response(payload)
            if self.shadow_behavior == "missing_passage":
                data["passageAssessments"] = data["passageAssessments"][:-1]
            elif self.shadow_behavior == "duplicate_passage":
                data["passageAssessments"].append(data["passageAssessments"][0])
            elif self.shadow_behavior == "missing_question":
                data["questionAssessments"] = data["questionAssessments"][:-1]
            elif self.shadow_behavior == "duplicate_question":
                data["questionAssessments"].append(data["questionAssessments"][0])

        return StructuredGenerationResult(
            data=data,
            input_tokens=321,
            output_tokens=123,
            provider="test-provider",
            model="test-mini",
        )

    async def call_with_image(
        self,
        type_name,
        prompt,
        image_bytes,
        mime_type,
        schema=None,
    ):
        raise AssertionError("Reading controlled shadow must not call an image provider")

    @staticmethod
    def _shadow_response(payload: dict) -> dict:
        return {
            "passageAssessments": [
                {
                    "passageId": passage["passageId"],
                    **_assessment(passage["passageSegments"][0]["id"]),
                }
                for passage in payload["passages"]
            ],
            "questionAssessments": [
                {
                    "order": question["order"],
                    **_assessment(question["prompt"]["segmentId"]),
                }
                for question in payload["questions"]
            ],
        }


def _assessment(evidence_id: str) -> dict:
    return {
        "difficultyStatus": "ASSESSED",
        "observedBand": 3,
        "alternativeBand": None,
        "issueCodes": ["DIRECT_EVIDENCE_DEMAND"],
        "evidenceSegmentIds": [evidence_id],
        "difficultyConfidence": 0.82,
    }


def _shadow_data(prompt: str) -> dict:
    match = re.search(
        r"<reading-difficulty-data>\n(.+?)\n</reading-difficulty-data>",
        prompt,
        re.S,
    )
    if match is None:
        raise AssertionError("shadow prompt payload missing")
    return json.loads(match.group(1))


def _service(
    provider: ControlledShadowProvider,
    *,
    enabled: bool = True,
    sample_percent: float = 100.0,
    timeout_seconds: float = 1.0,
) -> ReadingVocabularyGenerationService:
    return ReadingVocabularyGenerationService(
        provider,
        difficulty_shadow_enabled=enabled,
        difficulty_shadow_sample_percent=sample_percent,
        difficulty_shadow_timeout_seconds=timeout_seconds,
    )


def _baseline_call_count() -> Counter:
    service = ReadingVocabularyGenerationService
    return Counter(
        {
            service.PASSAGE_TYPE_NAME: 2,
            service.TYPE_NAME: 3,
            service.VERIFICATION_TYPE_NAME: 1,
            service.ORIGIN_EXPLANATION_TYPE_NAME: 2,
        }
    )


def test_deterministic_sampling_is_stable_and_honors_boundaries():
    request_id = "stable-reading-request"
    assert reading_difficulty_shadow_sample_bucket(
        request_id
    ) == reading_difficulty_shadow_sample_bucket(request_id)
    assert not any(
        should_sample_reading_difficulty_shadow(f"request-{index}", 0)
        for index in range(100)
    )
    assert all(
        should_sample_reading_difficulty_shadow(f"request-{index}", 100)
        for index in range(100)
    )
    assert should_sample_reading_difficulty_shadow(
        request_id, 17.25
    ) == should_sample_reading_difficulty_shadow(request_id, 17.25)


@pytest.mark.parametrize("invalid", [-0.01, 100.01, float("nan"), float("inf"), True])
def test_sampling_rejects_invalid_percentage(invalid):
    with pytest.raises(ValueError):
        should_sample_reading_difficulty_shadow("request", invalid)


def test_shadow_settings_are_default_off_and_validate_bounds():
    fields = Settings.model_fields
    assert fields["AI_READING_DIFFICULTY_SHADOW_ENABLED"].default is False
    assert fields["AI_READING_DIFFICULTY_SHADOW_SAMPLE_PERCENT"].default == 0
    assert fields["AI_READING_DIFFICULTY_SHADOW_TIMEOUT_SECONDS"].default == 12
    with pytest.raises(ValidationError):
        Settings(AI_READING_DIFFICULTY_SHADOW_SAMPLE_PERCENT=100.01)


@pytest.mark.asyncio
async def test_default_off_preserves_reading_call_count():
    provider = ControlledShadowProvider()
    response = await ReadingVocabularyGenerationService(provider).generate(
        practice_fixtures._request()
    )

    assert len(response.questions) == 5
    assert provider.calls == _baseline_call_count()
    assert provider.calls[READING_DIFFICULTY_SHADOW_TYPE_NAME] == 0


@pytest.mark.asyncio
async def test_enabled_but_unselected_request_preserves_reading_call_count():
    provider = ControlledShadowProvider()
    response = await _service(provider, sample_percent=0).generate(
        practice_fixtures._request()
    )

    assert len(response.questions) == 5
    assert provider.calls == _baseline_call_count()


@pytest.mark.asyncio
async def test_selected_shadow_adds_exactly_one_call_without_changing_response(caplog):
    baseline_provider = ControlledShadowProvider()
    baseline = await ReadingVocabularyGenerationService(baseline_provider).generate(
        practice_fixtures._request()
    )
    selected_provider = ControlledShadowProvider()
    with caplog.at_level("INFO"):
        selected = await _service(selected_provider).generate(practice_fixtures._request())

    expected = _baseline_call_count()
    expected[READING_DIFFICULTY_SHADOW_TYPE_NAME] = 1
    assert selected_provider.calls == expected
    assert selected_provider.calls_before_shadow == _baseline_call_count()
    assert selected.model_dump() == baseline.model_dump()
    assert "provider=test-provider model=test-mini" in caplog.text
    assert "input_tokens=321 output_tokens=123" in caplog.text
    assert "sample_percentage=100.0 cohort=selected" in caplog.text
    assert f"recipe_version={READING_DIFFICULTY_RECIPE_VERSION}" in caplog.text
    assert f"passage_recipe_version={READING_PASSAGE_BLUEPRINT_VERSION}" in caplog.text
    assert (
        f"measurement_rubric_version={READING_DIFFICULTY_SHADOW_RUBRIC_VERSION}"
        in caplog.text
    )
    assert "scope=passage" in caplog.text
    assert "scope=question" in caplog.text
    assert "今日は会社で会議があります" not in caplog.text
    assert "本文の内容" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("behavior", ["timeout", "provider_error", "malformed"])
async def test_shadow_runtime_and_schema_failures_are_fail_open_without_quality_retry(
    behavior,
    caplog,
):
    provider = ControlledShadowProvider(shadow_behavior=behavior)
    timeout = 0.01 if behavior == "timeout" else 1.0
    with caplog.at_level("WARNING"):
        response = await _service(provider, timeout_seconds=timeout).generate(
            practice_fixtures._request()
        )

    assert len(response.questions) == 5
    assert provider.calls[ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME] == 1
    assert provider.calls[READING_DIFFICULTY_SHADOW_TYPE_NAME] == 1
    assert provider.candidate_slot_calls == [[1, 2], [3, 4], [5]]
    assert "Reading difficulty shadow failed open" in caplog.text
    if behavior == "timeout":
        assert provider.shadow_cleaned_up.is_set()


@pytest.mark.asyncio
async def test_shadow_cancellation_propagates_and_cleans_up_provider_coroutine():
    provider = ControlledShadowProvider(shadow_behavior="cancel")
    task = asyncio.create_task(_service(provider).generate(practice_fixtures._request()))
    await provider.shadow_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.shadow_cleaned_up.is_set()
    assert provider.calls[ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME] == 1
    assert provider.calls[READING_DIFFICULTY_SHADOW_TYPE_NAME] == 1


@pytest.mark.asyncio
async def test_shadow_payload_is_blind_unique_and_uses_stable_segment_ids():
    provider = ControlledShadowProvider()
    await _service(provider).generate(practice_fixtures._request())

    payload = provider.shadow_payloads[0]
    serialized = json.dumps(payload, ensure_ascii=False)
    assert [passage["passageId"] for passage in payload["passages"]] == ["p1", "p2"]
    assert serialized.count('"passageText"') == 2
    assert [item["band"] for item in payload["readingDifficultyRubric"]["passageBands"]] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert payload["readingDifficultyRubric"]["version"] == (
        READING_DIFFICULTY_SHADOW_RUBRIC_VERSION
    )
    forbidden_keys = {
        "requestId",
        "complexityBand",
        "difficulty",
        "difficultyRecipe",
        "expectedAnswerKey",
        "correctAnswer",
    }
    assert forbidden_keys.isdisjoint(_all_keys(payload))
    assert all(
        question["prompt"]["segmentId"] == f"question:{question['order']}:prompt"
        for question in payload["questions"]
    )
    assert "CURRENT" not in serialized
    assert "EASIER" not in serialized
    assert "CHALLENGE" not in serialized


def _all_keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_all_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value)) if value else set()
    return set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "behavior",
    ["missing_passage", "duplicate_passage", "missing_question", "duplicate_question"],
)
async def test_missing_or_duplicate_shadow_coverage_fails_open(behavior, caplog):
    provider = ControlledShadowProvider(shadow_behavior=behavior)
    with caplog.at_level("WARNING"):
        response = await _service(provider).generate(practice_fixtures._request())

    assert len(response.questions) == 5
    assert provider.calls[ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME] == 1
    assert provider.calls[READING_DIFFICULTY_SHADOW_TYPE_NAME] == 1
    assert "coverage mismatch" not in caplog.text
    assert "Reading difficulty shadow failed open" in caplog.text


@pytest.mark.asyncio
async def test_vocabulary_never_calls_reading_difficulty_shadow():
    provider = ControlledShadowProvider()
    response = await _service(provider).generate(practice_fixtures._vocab_request())

    assert len(response.questions) == 10
    assert provider.calls[READING_DIFFICULTY_SHADOW_TYPE_NAME] == 0


def test_primary_quality_contract_matches_pre_calibration_snapshot():
    expected = QUALITY_ONLY_SYSTEM_PROMPT.replace(
        "- answerLeakage=true",
        f"{READING_STRUCTURE_MODE_FIT_CLARIFICATION}\n- answerLeakage=true",
    )
    assert PRACTICE_VERIFICATION_SYSTEM_PROMPT == expected
    assert _PRACTICE_VERIFICATION_SCHEMA == QUALITY_ONLY_SCHEMA
    assert "readingDifficultyRubric" not in PRACTICE_VERIFICATION_SYSTEM_PROMPT
    assert "passageDifficultyAssessments" not in _PRACTICE_VERIFICATION_SCHEMA[
        "properties"
    ]


def test_shadow_task_has_separate_mini_policy_prompt_and_closed_required_schema():
    policy = get_task_model_policy(READING_DIFFICULTY_SHADOW_TYPE_NAME)
    assert policy.tier == AiModelTier.MINI
    assert PROMPT_MAP[READING_DIFFICULTY_SHADOW_TYPE_NAME] == (
        READING_DIFFICULTY_SHADOW_SYSTEM_PROMPT
    )
    assert READING_DIFFICULTY_SHADOW_SCHEMA["additionalProperties"] is False
    assert set(READING_DIFFICULTY_SHADOW_SCHEMA["required"]) == {
        "passageAssessments",
        "questionAssessments",
    }
    for collection in READING_DIFFICULTY_SHADOW_SCHEMA["properties"].values():
        item = collection["items"]
        assert item["additionalProperties"] is False
        assert set(item["required"]) == set(item["properties"])
    openai_config = build_openai_text_config(
        type_name=READING_DIFFICULTY_SHADOW_TYPE_NAME,
        schema=READING_DIFFICULTY_SHADOW_SCHEMA,
        verbosity="low",
    )
    assert openai_config["format"]["strict"] is True
