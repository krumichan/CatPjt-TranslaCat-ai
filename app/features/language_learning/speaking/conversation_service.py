from __future__ import annotations

import asyncio
import time

from pydantic import ValidationError

from app.ai.ports import StructuredTextGenerationProvider
from app.core.config import settings
from app.features.language_learning.speaking.errors import SpeakingStageException
from app.features.language_learning.speaking.generation_difficulty_adapter import (
    build_speaking_generation_difficulty_spec,
    project_speaking_generation_validation,
)
from app.features.language_learning.speaking.idempotency import InMemoryIdempotencyStore
from app.features.language_learning.speaking.policy import (
    SPEAKING_CONVERSATION_PROMPT_VERSION,
    resolve_assistance_level,
)
from app.features.language_learning.speaking.prompts import build_conversation_prompt
from app.features.language_learning.speaking.provider_error import map_provider_exception
from app.features.language_learning.speaking.retry import run_with_stage_retry
from app.schemas.language_learning_speaking import (
    ConversationGenerationRequest,
    ConversationGenerationResponse,
    ConversationPayload,
    ConversationResult,
    SpeakingErrorCode,
    SpeakingStage,
    SpeakingUsage,
    StageUsage,
)


class SpeakingConversationService:
    TYPE_NAME = "LANGUAGE_LEARNING_SPEAKING_CONVERSATION"

    def __init__(
        self,
        provider: StructuredTextGenerationProvider,
        *,
        timeout_seconds: float | None = None,
        automatic_retries: int | None = None,
        idempotency_store: InMemoryIdempotencyStore[ConversationGenerationResponse]
        | None = None,
    ) -> None:
        self.provider = provider
        self.timeout_seconds = (
            timeout_seconds or settings.AI_SPEAKING_CONVERSATION_TIMEOUT_SECONDS
        )
        self.automatic_retries = (
            settings.AI_SPEAKING_AUTOMATIC_RETRY_LIMIT
            if automatic_retries is None
            else automatic_retries
        )
        self.idempotency_store = idempotency_store or InMemoryIdempotencyStore()

    async def generate(
        self,
        request: ConversationGenerationRequest,
    ) -> ConversationGenerationResponse:
        if request.manual_retry_attempt > settings.AI_SPEAKING_MANUAL_RETRY_LIMIT:
            raise SpeakingStageException(
                code=SpeakingErrorCode.MANUAL_RETRY_LIMIT_EXCEEDED,
                stage=SpeakingStage.CONVERSATION,
                message="대화 생성 수동 재시도 가능 횟수를 초과했습니다.",
                retryable=False,
            )

        response, _ = await self.idempotency_store.execute(
            request.idempotency_key,
            lambda: self._generate_once(request),
        )
        return response.model_copy(deep=True)

    async def _generate_once(
        self,
        request: ConversationGenerationRequest,
    ) -> ConversationGenerationResponse:
        prompt = build_conversation_prompt(request)
        schema = ConversationPayload.model_json_schema()
        difficulty_spec = build_speaking_generation_difficulty_spec(request)
        started = time.perf_counter()

        async def operation():
            try:
                result = await asyncio.wait_for(
                    self.provider.call_with_metadata(
                        type_name=self.TYPE_NAME,
                        data=prompt,
                        schema=schema,
                    ),
                    timeout=self.timeout_seconds,
                )
                if not isinstance(result.data, dict):
                    raise ValueError("structured conversation response must be object")
                payload = ConversationPayload.model_validate(result.data)
                if request.correction_mode.value == "COACHING" and any(
                    correction.improvement_link is None
                    for correction in payload.coaching_corrections
                ):
                    raise ValueError(
                        "COACHING correction에는 improvementLink가 필요합니다."
                    )
                validation = project_speaking_generation_validation(
                    difficulty_spec,
                    lambda: self._validate_practice_mode_payload(request, payload),
                )
                if not validation.passed:
                    raise ValueError(validation.primary_issue)
                return result, payload
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise SpeakingStageException(
                    code=SpeakingErrorCode.PROVIDER_TIMEOUT,
                    stage=SpeakingStage.CONVERSATION,
                    message="대화 생성 Provider 응답 시간이 초과되었습니다.",
                    retryable=True,
                ) from exc
            except (ValidationError, ValueError) as exc:
                raise SpeakingStageException(
                    code=SpeakingErrorCode.INVALID_RESPONSE_SCHEMA,
                    stage=SpeakingStage.CONVERSATION,
                    message="대화 생성 응답 Schema가 유효하지 않습니다.",
                    retryable=True,
                ) from exc
            except SpeakingStageException:
                raise
            except Exception as exc:
                raise map_provider_exception(
                    exc,
                    stage=SpeakingStage.CONVERSATION,
                    fallback_code=SpeakingErrorCode.CONVERSATION_GENERATION_FAILED,
                    fallback_message="AI 대화 응답 생성에 실패했습니다.",
                ) from exc

        max_retries = min(
            self.automatic_retries,
            request.session_policy_snapshot.automatic_retry_limit_per_stage,
        )
        result, payload = await run_with_stage_retry(
            operation,
            max_retries=max_retries,
        )
        should_end, end_reason = self._resolve_end_policy(request, payload)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return ConversationGenerationResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            turn_index=request.turn_index,
            assistant_text=payload.assistant_text,
            conversation=ConversationResult(
                intent=payload.intent,
                resolved_topic=payload.resolved_topic,
                script_text=payload.script_text,
                provided_facts=payload.provided_facts,
                required_intents=payload.required_intents,
                response_constraints=payload.response_constraints,
                difficulty=payload.difficulty,
                should_end=should_end,
                end_reason=end_reason,
                hint=payload.hint,
                coaching_corrections=payload.coaching_corrections,
                session_summary=payload.session_summary,
                assistance_level=resolve_assistance_level(request.assistance_usage),
            ),
            usage=SpeakingUsage(
                conversation=StageUsage(
                    latency_ms=elapsed_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    provider=result.provider,
                    model=result.model,
                    prompt_version=SPEAKING_CONVERSATION_PROMPT_VERSION,
                )
            ),
        )

    @staticmethod
    def _validate_practice_mode_payload(request, payload: ConversationPayload) -> None:
        if (
            request.is_initial_turn
            and request.category == "KEYWORDS"
            and (not payload.resolved_topic or not payload.resolved_topic.strip())
        ):
            raise ValueError("KEYWORDS 초기 Turn에는 resolvedTopic이 필요합니다.")

        mode = request.practice_mode.value
        if mode == "READ_ALOUD":
            if not payload.script_text or payload.script_text.strip() != payload.assistant_text.strip():
                raise ValueError("READ_ALOUD는 assistantText와 동일한 scriptText가 필요합니다.")
            if payload.provided_facts or payload.required_intents or payload.response_constraints:
                raise ValueError("READ_ALOUD에는 structured guidance를 둘 수 없습니다.")
            return
        if mode == "GUIDED":
            if payload.script_text is not None:
                raise ValueError("GUIDED에는 scriptText를 둘 수 없습니다.")
            if not payload.provided_facts or not payload.required_intents or not payload.response_constraints:
                raise ValueError("GUIDED에는 사실/전달 내용/답변 조건이 모두 필요합니다.")
            return
        if mode == "FREE":
            if payload.script_text is not None:
                raise ValueError("FREE에는 scriptText를 둘 수 없습니다.")
            return
        raise ValueError("지원하지 않는 Speaking practiceMode입니다.")

    @staticmethod
    def _resolve_end_policy(
        request: ConversationGenerationRequest,
        payload: ConversationPayload,
    ) -> tuple[bool, str | None]:
        if payload.should_end:
            return True, payload.end_reason

        policy = request.session_policy_snapshot
        if request.turn_index >= policy.max_turns:
            return True, "MAX_TURNS"

        if request.session_elapsed_seconds >= policy.max_session_minutes * 60:
            return True, "MAX_SESSION_DURATION"

        return False, payload.end_reason
