from __future__ import annotations

import ast
import asyncio
import copy
from collections import Counter
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.features.language_learning.difficulty import contracts as difficulty_contracts
from app.features.language_learning.listening.errors import ListeningStageException
from app.features.language_learning.listening.generation_service import ListeningGenerationService
from app.features.language_learning.listening.policy import DIFFICULTY_DURATION_RANGES
from app.features.language_learning.quality import resolve_listening_complexity_band
from app.features.language_learning.quality.diversity import character_ngram_cosine
from app.features.language_learning.reading_vocabulary.service import (
    ReadingVocabularyGenerationService,
)
from app.features.language_learning.speaking.conversation_service import (
    SpeakingConversationService,
)
from app.schemas.language_learning_level_test import LevelTestQuestionGenerationRequest
from app.schemas.language_learning_quality import LanguageComplexityContext
from tests import test_language_learning_current as level_fixtures
from tests import test_language_learning_listening as listening_fixtures
from tests import test_language_learning_reading_vocabulary as practice_fixtures
from tests import test_language_learning_speaking as speaking_fixtures


def _near_history(content: str) -> str:
    """Build fixed-threshold history that strict diversity rejects and relaxed accepts."""
    for changed in range(1, min(len(content), 200)):
        history = ("過" * changed) + content[changed:]
        similarity = character_ngram_cosine(content, history)
        if 0.88 <= similarity < 0.91:
            return history
    suffix_alphabet = "追加履歴比較対象文章差分検証用文字列"
    for added in range(1, min(500, 4000 - len(content))):
        suffix = "".join(
            suffix_alphabet[index % len(suffix_alphabet)]
            for index in range(added)
        )
        history = content + suffix
        similarity = character_ngram_cosine(content, history)
        if 0.88 <= similarity < 0.91:
            return history
    raise AssertionError("could not construct a strict-only similarity history fixture")


def test_common_difficulty_remains_isolated_from_all_migrating_services_and_dtos():
    root = Path(difficulty_contracts.__file__).parent
    forbidden_import_prefixes = (
        "app.features.language_learning.listening",
        "app.features.language_learning.reading_vocabulary",
        "app.features.language_learning.speaking",
        "app.features.language_learning.level_test",
        "app.schemas.language_learning_listening",
        "app.schemas.language_learning_practice",
        "app.schemas.language_learning_speaking",
        "app.schemas.language_learning_level_test",
    )
    imports: list[str] = []
    source = ""
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        source += text
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)

    assert not [
        name
        for name in imports
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in forbidden_import_prefixes
        )
    ]
    forbidden_literals = {
        "CONTENT_DIVERSITY_EXHAUSTED",
        "REPAIR_READING_AMBIGUITY",
        "REPAIR_AMBIGUITY",
        "CONVERSATION_GENERATION_FAILED",
        "QUESTION_CONTENT_INVALID",
        "BEST_ANSWER",
        "UNIQUE_ANSWER",
    }
    assert not {literal for literal in forbidden_literals if literal in source}


@pytest.mark.parametrize(
    ("difficulty", "base_band", "expected_band", "duration"),
    [
        ("EASY", 1, 1, (5.0, 12.0)),
        ("MY_LEVEL", 3, 3, (8.0, 20.0)),
        ("CHALLENGE", 5, 5, (15.0, 30.0)),
    ],
)
def test_listening_target_band_clamp_and_duration_profile_are_current_baseline(
    difficulty: str,
    base_band: int,
    expected_band: int,
    duration: tuple[float, float],
):
    context = LanguageComplexityContext(baseComplexityBand=base_band)
    assert resolve_listening_complexity_band(difficulty, context) == expected_band
    assert DIFFICULTY_DURATION_RANGES[difficulty] == duration


def test_listening_explicit_target_band_overrides_difficulty_mapping():
    context = LanguageComplexityContext(
        baseComplexityBand=1,
        targetComplexityBand=4,
    )
    assert resolve_listening_complexity_band("EASY", context) == 4


@pytest.mark.asyncio
async def test_listening_normal_generation_is_one_luna_stage_and_no_semantic_stage():
    provider = listening_fixtures.FakeStructuredProvider(
        [listening_fixtures.generation_payload()]
    )
    response = await ListeningGenerationService(
        provider,
        automatic_retries=0,
    ).generate(listening_fixtures.generation_request())

    assert len(response.items) == 2
    assert [call[0] for call in provider.calls] == [
        "LANGUAGE_LEARNING_LISTENING_GENERATION"
    ]


@pytest.mark.asyncio
async def test_listening_fixed_three_rounds_ignore_automatic_retry_zero(caplog):
    invalid = {"items": [{}]}
    provider = listening_fixtures.FakeStructuredProvider(
        [copy.deepcopy(invalid) for _ in range(3)]
    )
    service = ListeningGenerationService(provider, automatic_retries=0)

    with caplog.at_level("INFO"), pytest.raises(ListeningStageException) as raised:
        await service.generate(listening_fixtures.generation_request())

    assert raised.value.code.value == "INVALID_RESPONSE_SCHEMA"
    assert len(provider.calls) == 3
    assert "request_id=generation-1" in caplog.text
    assert "attempt=3/3" in caplog.text


class _BlockingStructuredProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.cleaned = 0
        self.started = asyncio.Event()

    async def call_with_metadata(self, type_name, data, schema=None):
        self.calls += 1
        self.started.set()
        try:
            await asyncio.Future()
        finally:
            self.cleaned += 1


@pytest.mark.asyncio
async def test_listening_timeout_retries_three_times_and_cleans_provider_coroutines():
    provider = _BlockingStructuredProvider()
    service = ListeningGenerationService(
        provider,
        timeout_seconds=0.01,
        automatic_retries=0,
    )

    with pytest.raises(ListeningStageException) as raised:
        await service.generate(listening_fixtures.generation_request())

    assert raised.value.code.value == "PROVIDER_TIMEOUT"
    assert provider.calls == 3
    assert provider.cleaned == 3


@pytest.mark.asyncio
async def test_listening_cancellation_propagates_and_cleans_provider_coroutine():
    provider = _BlockingStructuredProvider()
    service = ListeningGenerationService(
        provider,
        timeout_seconds=10,
        automatic_retries=0,
    )
    task = asyncio.create_task(
        service.generate(
            listening_fixtures.generation_request(
                requestId="listening-cancel",
                idempotencyKey="listening-cancel-idem",
            )
        )
    )
    await provider.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.calls == 1
    assert provider.cleaned == 1


@pytest.mark.asyncio
async def test_listening_relaxed_history_fallback_keeps_three_round_baseline():
    payload = listening_fixtures.generation_payload()
    payload["items"] = [payload["items"][0]]
    payload["items"][0]["sourceText"] = (
        "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみ"
    )
    history = _near_history(payload["items"][0]["sourceText"])
    request = listening_fixtures.generation_request()
    request = request.model_copy(
        deep=True,
        update={
            "request_id": "listening-relaxed",
            "idempotency_key": "listening-relaxed-idem",
            "set_context": request.set_context.model_copy(update={"item_count": 1}),
            "constraints": request.constraints.model_copy(
                update={"recent_similarity_summaries": [history]}
            ),
        },
    )
    provider = listening_fixtures.FakeStructuredProvider(
        [copy.deepcopy(payload) for _ in range(3)]
    )

    response = await ListeningGenerationService(
        provider,
        automatic_retries=0,
    ).generate(request)

    assert len(provider.calls) == 3
    assert response.diversity_summary.fallback_used is True
    assert response.diversity_summary.rejected_similarity == 3


@pytest.mark.asyncio
async def test_reading_normal_five_question_provider_snapshot_and_target_binding():
    provider = practice_fixtures.PipelineProvider()
    response = await ReadingVocabularyGenerationService(provider).generate(
        practice_fixtures._request()
    )

    service = ReadingVocabularyGenerationService
    assert provider.calls == Counter(
        {
            service.PASSAGE_TYPE_NAME: 2,
            service.TYPE_NAME: 3,
            service.VERIFICATION_TYPE_NAME: 1,
            service.ORIGIN_EXPLANATION_TYPE_NAME: 2,
        }
    )
    assert [item.complexity_band for item in response.questions] == [3, 2, 3, 4, 3]
    assert [item.passage_id for item in response.questions] == ["p1", "p1", "p1", "p2", "p2"]
    assert len(provider.verification_payloads) == 1
    assert len(provider.verification_payloads[0]["questions"]) == 5


@pytest.mark.parametrize(
    ("mode", "expected", "verification_batch_sizes"),
    [
        (
            "MEANING_RELATION",
            {"generator": 5, "prescreen": 0, "mini": 1, "explanation": 3},
            [10],
        ),
        (
            "USAGE_DISTINCTION",
            {"generator": 10, "prescreen": 10, "mini": 10, "explanation": 3},
            [1] * 10,
        ),
        (
            "COMPOSITION",
            {"generator": 10, "prescreen": 0, "mini": 6, "explanation": 3},
            [1] * 6,
        ),
    ],
)
@pytest.mark.asyncio
async def test_vocabulary_mode_provider_snapshots_are_independent(
    mode: str,
    expected: dict[str, int],
    verification_batch_sizes: list[int],
):
    provider = practice_fixtures.PipelineProvider()
    response = await ReadingVocabularyGenerationService(provider).generate(
        practice_fixtures._vocab_request(mode=mode)
    )
    service = ReadingVocabularyGenerationService

    assert len(response.questions) == 10
    assert provider.calls[service.TYPE_NAME] == expected["generator"]
    assert provider.calls[service.PRESCREEN_TYPE_NAME] == expected["prescreen"]
    assert provider.calls[service.VERIFICATION_TYPE_NAME] == expected["mini"]
    assert provider.calls[service.ORIGIN_EXPLANATION_TYPE_NAME] == expected["explanation"]
    assert [len(payload["questions"]) for payload in provider.verification_payloads] == (
        verification_batch_sizes
    )
    assert [item.complexity_band for item in response.questions] == [
        3,
        2,
        3,
        4,
        3,
        2,
        3,
        4,
        3,
        3,
    ]
    if mode == "COMPOSITION":
        assert [
            item.order
            for item in response.questions
            if item.question_type.value == "ORDERING"
        ] == [1, 3, 6, 8]


class _FlakyPracticeVerifier(practice_fixtures.PipelineProvider):
    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    async def call(self, type_name, data, schema=None):
        if (
            type_name == ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME
            and self.failures > 0
        ):
            self.calls[type_name] += 1
            self.failures -= 1
            raise RuntimeError("temporary verifier failure")
        return await super().call(type_name, data, schema)


@pytest.mark.asyncio
async def test_reading_semantic_provider_retry_is_three_calls_for_one_batch(caplog):
    provider = _FlakyPracticeVerifier(failures=2)

    with caplog.at_level("WARNING"):
        response = await ReadingVocabularyGenerationService(provider).generate(
            practice_fixtures._request()
        )

    assert len(response.questions) == 5
    assert provider.calls[ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME] == 3
    assert "request_id=r1 attempt=2/3" in caplog.text
    assert provider.candidate_slot_calls == [[1, 2], [3, 4], [5]]


class _BlockingPracticeProvider(practice_fixtures.PipelineProvider):
    def __init__(self, blocked_type: str) -> None:
        super().__init__()
        self.blocked_type = blocked_type
        self.started = asyncio.Event()
        self.cleaned = 0

    async def call(self, type_name, data, schema=None):
        if type_name == self.blocked_type:
            self.calls[type_name] += 1
            self.started.set()
            try:
                await asyncio.Future()
            finally:
                self.cleaned += 1
        return await super().call(type_name, data, schema)


@pytest.mark.asyncio
async def test_reading_passage_timeout_retries_three_times_and_cleans_each_call():
    provider = _BlockingPracticeProvider(
        ReadingVocabularyGenerationService.PASSAGE_TYPE_NAME
    )
    service = ReadingVocabularyGenerationService(provider, timeout_seconds=0.01)

    with pytest.raises(HTTPException) as raised:
        await service.generate(practice_fixtures._request())

    assert raised.value.status_code == 502
    assert raised.value.detail["code"] == "AI_GENERATION_FAILED"
    assert raised.value.detail["cause"] == "ValueError"
    assert provider.calls[ReadingVocabularyGenerationService.PASSAGE_TYPE_NAME] == 3
    assert provider.cleaned == 3


@pytest.mark.asyncio
async def test_reading_semantic_cancellation_propagates_and_cleans_provider():
    provider = _BlockingPracticeProvider(
        ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME
    )
    service = ReadingVocabularyGenerationService(provider, timeout_seconds=10)
    task = asyncio.create_task(service.generate(practice_fixtures._request()))
    await provider.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.cleaned == 1
    assert provider.calls[ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME] == 1


@pytest.mark.asyncio
async def test_vocabulary_semantic_cancellation_propagates_and_cleans_provider():
    provider = _BlockingPracticeProvider(
        ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME
    )
    service = ReadingVocabularyGenerationService(provider, timeout_seconds=10)
    task = asyncio.create_task(
        service.generate(practice_fixtures._vocab_request(mode="MEANING_RELATION"))
    )
    await provider.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.cleaned == 1
    assert provider.calls[ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME] == 1


@pytest.mark.asyncio
async def test_speaking_generation_prompt_and_normal_provider_snapshot():
    provider = speaking_fixtures.FakeStructuredProvider(
        results=[speaking_fixtures.conversation_payload()]
    )
    service = SpeakingConversationService(
        provider,
        timeout_seconds=1,
        automatic_retries=0,
    )

    await service.generate(
        speaking_fixtures.conversation_request(
            targetLevel="B1",
            practiceMode="FREE",
        )
    )

    assert [call[0] for call in provider.calls] == [
        "LANGUAGE_LEARNING_SPEAKING_CONVERSATION"
    ]
    assert '"targetLevel":"B1"' in provider.calls[0][1]
    assert '"practiceMode":"FREE"' in provider.calls[0][1]


@pytest.mark.asyncio
async def test_speaking_automatic_retry_max_is_three_but_manual_retry_adds_no_stage_call():
    invalid = {"assistantText": "missing required fields"}
    automatic_provider = speaking_fixtures.FakeStructuredProvider(
        results=[invalid, invalid, speaking_fixtures.conversation_payload()]
    )
    automatic_service = SpeakingConversationService(
        automatic_provider,
        timeout_seconds=1,
        automatic_retries=2,
    )
    await automatic_service.generate(
        speaking_fixtures.conversation_request(
            idempotencyKey="speaking-auto-three",
            sessionPolicySnapshot={"automaticRetryLimitPerStage": 2},
        )
    )
    assert len(automatic_provider.calls) == 3

    manual_provider = speaking_fixtures.FakeStructuredProvider(
        results=[speaking_fixtures.conversation_payload()]
    )
    manual_service = SpeakingConversationService(
        manual_provider,
        timeout_seconds=1,
        automatic_retries=0,
    )
    await manual_service.generate(
        speaking_fixtures.conversation_request(
            idempotencyKey="speaking-manual-one",
            manualRetryAttempt=1,
        )
    )
    assert len(manual_provider.calls) == 1


@pytest.mark.asyncio
async def test_speaking_generation_cancellation_propagates_and_cleans_provider():
    provider = _BlockingStructuredProvider()
    service = SpeakingConversationService(
        provider,
        timeout_seconds=10,
        automatic_retries=2,
    )
    task = asyncio.create_task(
        service.generate(
            speaking_fixtures.conversation_request(
                idempotencyKey="speaking-cancel",
            )
        )
    )
    await provider.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.calls == 1
    assert provider.cleaned == 1


def _level_service(provider):
    return level_fixtures.CurrentLevelTestTest()._service(provider)


def test_level_test_prompt_requests_two_candidates_but_runtime_accepts_one():
    payload = level_fixtures.level_generation_payload()
    payload["candidates"] = [payload["candidates"][1]]
    provider = level_fixtures.QueueProvider(structured=[payload])
    request = level_fixtures.level_question_request().model_copy(
        deep=True,
        update={
            "request_id": "level-single-candidate",
            "idempotency_key": "level-single-candidate-idem",
            "diversity_context": level_fixtures.DiversityContext(),
        },
    )

    response = asyncio.run(_level_service(provider).generate_question(request))

    assert response.prompt_text == payload["candidates"][0]["promptText"]
    assert len(provider.calls) == 1
    assert "Generate two candidate questions" in provider.calls[0][1]
    candidate_schema = provider.calls[0][2]["properties"]["candidates"]
    assert candidate_schema["minItems"] == 2
    assert candidate_schema["maxItems"] == 2


def test_level_test_content_rejection_uses_exactly_three_generation_rounds():
    payload = level_fixtures.level_generation_payload()
    for candidate in payload["candidates"]:
        candidate["complexityBand"] = 3
    provider = level_fixtures.QueueProvider(
        structured=[copy.deepcopy(payload) for _ in range(3)]
    )

    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            _level_service(provider).generate_question(
                level_fixtures.level_question_request()
            )
        )

    assert raised.value.status_code == 422
    assert [call[0] for call in provider.calls] == [
        "LANGUAGE_LEARNING_LEVEL_TEST_GENERATION"
    ] * 3


@pytest.mark.asyncio
async def test_level_test_timeout_retries_three_rounds_and_cleans_provider_calls():
    provider = _BlockingStructuredProvider()
    service = _level_service(provider)
    service.generation_timeout_seconds = 0.01

    with pytest.raises(HTTPException) as raised:
        await service.generate_question(level_fixtures.level_question_request())

    assert raised.value.status_code == 504
    assert provider.calls == 3
    assert provider.cleaned == 3


@pytest.mark.asyncio
async def test_level_test_cancellation_propagates_and_cleans_provider():
    provider = _BlockingStructuredProvider()
    service = _level_service(provider)
    service.generation_timeout_seconds = 10
    task = asyncio.create_task(
        service.generate_question(
            level_fixtures.level_question_request().model_copy(
                update={
                    "request_id": "level-cancel",
                    "idempotency_key": "level-cancel-idem",
                }
            )
        )
    )
    await provider.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.calls == 1
    assert provider.cleaned == 1


def test_level_test_relaxed_diversity_fallback_keeps_three_round_baseline():
    payload = level_fixtures.level_generation_payload()
    payload["candidates"] = [payload["candidates"][1]]
    service = _level_service(level_fixtures.QueueProvider())
    normalized, _ = level_fixtures.LevelTestGenerationNormalizer.normalize(
        copy.deepcopy(payload),
        origin_language="ko",
        learning_language="ja",
    )
    candidate = level_fixtures.LevelTestQuestionGenerationPayload.model_validate(
        normalized
    ).candidates[0]
    history = _near_history(service._diversity_content(candidate))
    request_data = level_fixtures.level_question_request().model_dump(
        mode="json",
        by_alias=True,
    )
    request_data.update(
        requestId="level-relaxed",
        idempotencyKey="level-relaxed-idem",
        diversityContext={
            "sameFeatureRecent": [
                {"sourceType": "LEVEL_TEST", "content": history}
            ]
        },
    )
    request = LevelTestQuestionGenerationRequest.model_validate(request_data)
    provider = level_fixtures.QueueProvider(
        structured=[copy.deepcopy(payload) for _ in range(3)]
    )

    response = asyncio.run(_level_service(provider).generate_question(request))

    assert len(provider.calls) == 3
    assert response.diversity_summary.fallback_used is True
    assert response.diversity_summary.rejected_similarity == 3
