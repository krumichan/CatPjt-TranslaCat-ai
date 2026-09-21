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
from app.features.language_learning.reading_vocabulary.service import (
    ReadingVocabularyGenerationService,
    _PRACTICE_VERIFICATION_SCHEMA,
)
from app.features.language_learning.reading_vocabulary.vocabulary_difficulty_recipe import (
    MEANING_RELATION_DIFFICULTY_RECIPE_VERSION,
    VOCABULARY_DIFFICULTY_RECIPE_VERSION,
    VOCABULARY_DIFFICULTY_SHADOW_RUBRIC_VERSION,
)
from app.features.language_learning.reading_vocabulary.vocabulary_difficulty_shadow import (
    VOCABULARY_DIFFICULTY_SHADOW_SCHEMA,
    VOCABULARY_DIFFICULTY_SHADOW_SYSTEM_PROMPT,
    VOCABULARY_DIFFICULTY_SHADOW_TYPE_NAME,
    should_sample_vocabulary_difficulty_shadow,
    vocabulary_difficulty_shadow_sample_bucket,
)
from tests import test_language_learning_reading_vocabulary as practice_fixtures


@pytest.fixture(autouse=True)
def historical_shared_pipeline_fake_uses_luna(monkeypatch):
    """The Reading negative-control fixture uses the pre-selection task route."""
    monkeypatch.setattr(settings, "AI_READING_GENERATION_MODEL", "LUNA")


def _shadow_data(prompt: str) -> dict:
    match = re.search(
        r"<vocabulary-difficulty-data>\n(.+?)\n</vocabulary-difficulty-data>",
        prompt,
        re.S,
    )
    assert match
    return json.loads(match.group(1))


class VocabularyControlledShadowProvider(practice_fixtures.PipelineProvider):
    def __init__(self, *, shadow_behavior: str = "valid", **kwargs):
        super().__init__(**kwargs)
        self.shadow_behavior = shadow_behavior
        self.shadow_payloads: list[dict] = []
        self.shadow_schemas: list[dict | None] = []
        self.calls_before_shadow: Counter | None = None
        self.shadow_started = asyncio.Event()
        self.shadow_cleaned_up = asyncio.Event()

    async def call_with_metadata(self, type_name, data, schema=None):
        if type_name != VOCABULARY_DIFFICULTY_SHADOW_TYPE_NAME:
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
            response: dict = {"questionAssessments": "not-an-array"}
        else:
            response = {
                "questionAssessments": [
                    {
                        "order": question["order"],
                        "difficultyStatus": "ASSESSED",
                        "observedBand": 3,
                        "alternativeBand": None,
                        "issueCodes": ["VISIBLE_CUE_INTEGRATION"],
                        "evidenceSegmentIds": [question["prompt"]["segmentId"]],
                        "difficultyConfidence": 0.81,
                    }
                    for question in payload["questions"]
                ]
            }
            if self.shadow_behavior == "missing_question":
                response["questionAssessments"] = response["questionAssessments"][:-1]
            if self.shadow_behavior == "unsafe_issue_code":
                response["questionAssessments"][0]["issueCodes"] = [
                    payload["questions"][0]["prompt"]["text"]
                ]
        return StructuredGenerationResult(
            data=response,
            input_tokens=432,
            output_tokens=210,
            provider="test-provider",
            model="test-mini",
        )


def _service(
    provider,
    *,
    enabled: bool = True,
    sample_percent: float = 100.0,
    timeout_seconds: float = 1.0,
) -> ReadingVocabularyGenerationService:
    return ReadingVocabularyGenerationService(
        provider,
        vocabulary_difficulty_shadow_enabled=enabled,
        vocabulary_difficulty_shadow_sample_percent=sample_percent,
        vocabulary_difficulty_shadow_timeout_seconds=timeout_seconds,
    )


def _baseline_call_count(mode: str) -> Counter:
    service = ReadingVocabularyGenerationService
    if mode == "MEANING_RELATION":
        return Counter(
            {
                service.TYPE_NAME: 5,
                service.VERIFICATION_TYPE_NAME: 1,
                service.ORIGIN_EXPLANATION_TYPE_NAME: 3,
            }
        )
    if mode == "USAGE_DISTINCTION":
        return Counter(
            {
                service.TYPE_NAME: 10,
                service.PRESCREEN_TYPE_NAME: 10,
                service.VERIFICATION_TYPE_NAME: 10,
                service.ORIGIN_EXPLANATION_TYPE_NAME: 3,
            }
        )
    return Counter(
        {
            service.TYPE_NAME: 10,
            service.VERIFICATION_TYPE_NAME: 6,
            service.ORIGIN_EXPLANATION_TYPE_NAME: 3,
        }
    )


def test_vocabulary_sampling_is_deterministic_and_validates_bounds():
    request_id = "stable-vocabulary-request"
    assert vocabulary_difficulty_shadow_sample_bucket(
        request_id
    ) == vocabulary_difficulty_shadow_sample_bucket(request_id)
    assert not should_sample_vocabulary_difficulty_shadow(request_id, 0)
    assert should_sample_vocabulary_difficulty_shadow(request_id, 100)
    assert should_sample_vocabulary_difficulty_shadow(
        request_id, 17.25
    ) == should_sample_vocabulary_difficulty_shadow(request_id, 17.25)
    for invalid in (-0.01, 100.01, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            should_sample_vocabulary_difficulty_shadow(request_id, invalid)


def test_vocabulary_shadow_settings_are_default_off_and_validate_bounds():
    fields = Settings.model_fields
    assert fields["AI_VOCABULARY_DIFFICULTY_SHADOW_ENABLED"].default is False
    assert fields["AI_VOCABULARY_DIFFICULTY_SHADOW_SAMPLE_PERCENT"].default == 0
    assert fields["AI_VOCABULARY_DIFFICULTY_SHADOW_TIMEOUT_SECONDS"].default == 12
    with pytest.raises(ValidationError):
        Settings(AI_VOCABULARY_DIFFICULTY_SHADOW_SAMPLE_PERCENT=100.01)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    ["MEANING_RELATION", "USAGE_DISTINCTION", "COMPOSITION"],
)
async def test_default_off_preserves_each_vocabulary_mode_call_count(mode):
    provider = VocabularyControlledShadowProvider()
    response = await ReadingVocabularyGenerationService(provider).generate(
        practice_fixtures._vocab_request(mode=mode)
    )

    assert len(response.questions) == 10
    assert provider.calls == _baseline_call_count(mode)
    assert provider.calls[VOCABULARY_DIFFICULTY_SHADOW_TYPE_NAME] == 0


@pytest.mark.asyncio
async def test_enabled_but_unselected_adds_no_call():
    provider = VocabularyControlledShadowProvider()
    response = await _service(provider, sample_percent=0).generate(
        practice_fixtures._vocab_request(mode="MEANING_RELATION")
    )

    assert len(response.questions) == 10
    assert provider.calls == _baseline_call_count("MEANING_RELATION")


@pytest.mark.asyncio
async def test_selected_shadow_adds_exactly_one_call_and_changes_no_response(caplog):
    baseline_provider = VocabularyControlledShadowProvider()
    request = practice_fixtures._vocab_request(mode="USAGE_DISTINCTION")
    baseline = await ReadingVocabularyGenerationService(baseline_provider).generate(request)
    selected_provider = VocabularyControlledShadowProvider()
    with caplog.at_level("INFO"):
        selected = await _service(selected_provider).generate(request)

    expected = _baseline_call_count("USAGE_DISTINCTION")
    expected[VOCABULARY_DIFFICULTY_SHADOW_TYPE_NAME] = 1
    assert selected_provider.calls == expected
    assert selected_provider.calls_before_shadow == _baseline_call_count(
        "USAGE_DISTINCTION"
    )
    assert selected.model_dump() == baseline.model_dump()
    assert f"recipe_version={VOCABULARY_DIFFICULTY_RECIPE_VERSION}" in caplog.text
    assert (
        f"measurement_rubric_version={VOCABULARY_DIFFICULTY_SHADOW_RUBRIC_VERSION}"
        in caplog.text
    )
    assert "provider=test-provider model=test-mini" in caplog.text
    assert "input_tokens=432 output_tokens=210" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    ["MEANING_RELATION", "USAGE_DISTINCTION", "COMPOSITION"],
)
async def test_shadow_payload_is_blind_and_contains_only_learner_visible_content(mode):
    provider = VocabularyControlledShadowProvider()
    await _service(provider).generate(practice_fixtures._vocab_request(mode=mode))
    payload = provider.shadow_payloads[0]

    assert payload["mode"] == mode
    assert payload["vocabularyDifficultyRubric"]["version"] == (
        VOCABULARY_DIFFICULTY_SHADOW_RUBRIC_VERSION
    )
    forbidden = {
        "requestId",
        "requestedBand",
        "complexityBand",
        "difficulty",
        "correctAnswer",
        "targetExpression",
        "canonicalKey",
        "reviewTarget",
    }

    def walk(value):
        if isinstance(value, dict):
            assert forbidden.isdisjoint(value)
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(payload)
    assert all("subtype" in question for question in payload["questions"])
    assert all(
        "usageIntent" in question
        for question in payload["questions"]
    ) is (mode == "USAGE_DISTINCTION")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "behavior",
    [
        "provider_error",
        "malformed",
        "missing_question",
        "unsafe_issue_code",
        "timeout",
    ],
)
async def test_shadow_failure_is_fail_open_and_timeout_cleans_up(behavior, caplog):
    provider = VocabularyControlledShadowProvider(shadow_behavior=behavior)
    with caplog.at_level("WARNING"):
        response = await _service(provider, timeout_seconds=0.01).generate(
            practice_fixtures._vocab_request(mode="MEANING_RELATION")
        )

    assert len(response.questions) == 10
    assert provider.calls[VOCABULARY_DIFFICULTY_SHADOW_TYPE_NAME] == 1
    assert "Vocabulary difficulty shadow failed open" in caplog.text
    assert "最も適切なものはどれですか" not in caplog.text
    if behavior == "timeout":
        assert provider.shadow_cleaned_up.is_set()


@pytest.mark.asyncio
async def test_shadow_cancellation_propagates_and_cleans_up():
    provider = VocabularyControlledShadowProvider(shadow_behavior="cancel")
    task = asyncio.create_task(
        _service(provider).generate(
            practice_fixtures._vocab_request(mode="MEANING_RELATION")
        )
    )
    await provider.shadow_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.shadow_cleaned_up.is_set()


@pytest.mark.asyncio
async def test_shadow_telemetry_does_not_log_raw_learner_content(caplog):
    provider = VocabularyControlledShadowProvider()
    with caplog.at_level("INFO"):
        await _service(provider).generate(
            practice_fixtures._vocab_request(mode="MEANING_RELATION")
        )

    assert "最も適切なものはどれですか" not in caplog.text
    assert "最も適切な内容" not in caplog.text
    assert "mode=MEANING_RELATION" in caplog.text
    assert "subtype=DIRECT_MEANING" in caplog.text
    assert "requested_band=3" in caplog.text
    assert "sample_percentage=100.0 cohort=selected" in caplog.text
    assert f"recipe_version={MEANING_RELATION_DIFFICULTY_RECIPE_VERSION}" in caplog.text
    assert (
        f"measurement_rubric_version={VOCABULARY_DIFFICULTY_SHADOW_RUBRIC_VERSION}"
        in caplog.text
    )


@pytest.mark.asyncio
async def test_reading_never_calls_vocabulary_difficulty_shadow():
    provider = VocabularyControlledShadowProvider()
    response = await _service(provider).generate(practice_fixtures._request())

    assert len(response.questions) == 5
    assert provider.calls[VOCABULARY_DIFFICULTY_SHADOW_TYPE_NAME] == 0


def test_shadow_has_separate_mini_policy_prompt_and_strict_schema():
    policy = get_task_model_policy(VOCABULARY_DIFFICULTY_SHADOW_TYPE_NAME)
    assert policy.tier == AiModelTier.MINI
    assert PROMPT_MAP[VOCABULARY_DIFFICULTY_SHADOW_TYPE_NAME] == (
        VOCABULARY_DIFFICULTY_SHADOW_SYSTEM_PROMPT
    )
    assert VOCABULARY_DIFFICULTY_SHADOW_SCHEMA["additionalProperties"] is False
    assert VOCABULARY_DIFFICULTY_SHADOW_SCHEMA["required"] == [
        "questionAssessments"
    ]
    item = VOCABULARY_DIFFICULTY_SHADOW_SCHEMA["properties"][
        "questionAssessments"
    ]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == set(item["properties"])
    openai_config = build_openai_text_config(
        type_name=VOCABULARY_DIFFICULTY_SHADOW_TYPE_NAME,
        schema=VOCABULARY_DIFFICULTY_SHADOW_SCHEMA,
        verbosity="low",
    )
    assert openai_config["format"]["strict"] is True


def test_primary_quality_schema_does_not_gain_difficulty_fields():
    verdict = _PRACTICE_VERIFICATION_SCHEMA["properties"]["verdicts"]["items"]
    assert set(verdict["properties"]) == {
        "order",
        "bestAnswerKey",
        "ambiguous",
        "supported",
        "reason",
        "modeFit",
        "answerLeakage",
        "contextDependent",
        "distractorsPlausible",
    }
