from __future__ import annotations

import time

from app.features.language_learning.speaking.audio_processor import SpeakingAudioProcessor
from app.features.language_learning.speaking.conversation_service import SpeakingConversationService
from app.features.language_learning.speaking.errors import SpeakingStageException
from app.features.language_learning.speaking.idempotency import InMemoryIdempotencyStore
from app.features.language_learning.speaking.stt_service import SpeakingSttService
from app.features.language_learning.speaking.tts_service import SpeakingTtsService
from app.schemas.language_learning_speaking import (
    AssistantTurn,
    ConversationGenerationRequest,
    ConversationResult,
    ConversationStartMode,
    SessionStartRequest,
    SessionStartResponse,
    SpeakingError,
    SpeakingStage,
    SpeakingUsage,
    SttRequestContext,
    SttResponse,
    TtsRequest,
    TurnProcessResponse,
)


class SpeakingTurnService:
    def __init__(
        self,
        *,
        audio_processor: SpeakingAudioProcessor,
        stt_service: SpeakingSttService,
        conversation_service: SpeakingConversationService,
        tts_service: SpeakingTtsService,
        idempotency_store: InMemoryIdempotencyStore[TurnProcessResponse] | None = None,
    ) -> None:
        self.audio_processor = audio_processor
        self.stt_service = stt_service
        self.conversation_service = conversation_service
        self.tts_service = tts_service
        self.idempotency_store = idempotency_store or InMemoryIdempotencyStore()

    async def start_session(self, request: SessionStartRequest) -> SessionStartResponse:
        resolved = self._resolve_start_mode(request)
        if resolved == ConversationStartMode.USER_FIRST:
            return SessionStartResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                resolved_start_mode=resolved,
                assistant=None,
                conversation=None,
                usage=SpeakingUsage(),
            )

        conversation_payload = request.model_dump()
        conversation_payload.update(
            conversation_start_mode=resolved,
            is_initial_turn=True,
            transcript=None,
        )
        conversation = await self.conversation_service.generate(
            ConversationGenerationRequest.model_validate(conversation_payload)
        )
        assistant = await self._assistant_with_tts(
            request_id=request.request_id,
            idempotency_key=request.idempotency_key,
            session_id=request.session_id,
            learning_language=request.learning_language,
            voice=request.voice,
            playback_speed=request.playback_speed,
            automatic_retry_limit=(
                request.session_policy_snapshot.automatic_retry_limit_per_stage
            ),
            text=conversation.assistant_text,
        )
        usage = self._merge_usage(conversation.usage, assistant[1])
        return SessionStartResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            resolved_start_mode=resolved,
            assistant=assistant[0],
            conversation=conversation.conversation,
            usage=usage,
        )

    async def process_turn(
        self,
        *,
        context: ConversationGenerationRequest,
        audio_bytes: bytes,
        file_name: str | None,
        content_type: str | None,
    ) -> TurnProcessResponse:
        response, replayed = await self.idempotency_store.execute(
            context.idempotency_key,
            lambda: self._process_turn_once(
                context=context,
                audio_bytes=audio_bytes,
                file_name=file_name,
                content_type=content_type,
            ),
        )
        result = response.model_copy(deep=True)
        result.idempotent_replay = replayed
        return result

    async def _process_turn_once(
        self,
        *,
        context: ConversationGenerationRequest,
        audio_bytes: bytes,
        file_name: str | None,
        content_type: str | None,
    ) -> TurnProcessResponse:
        usage = SpeakingUsage()
        transcript = None
        try:
            stt = await self.transcribe_audio(
                context=SttRequestContext(
                    request_id=context.request_id,
                    idempotency_key=context.idempotency_key,
                    session_id=context.session_id,
                    turn_index=context.turn_index,
                    learning_language=context.learning_language,
                    phrase_hints=[
                        keyword.text for keyword in context.selected_keywords
                    ],
                    audio_reference=context.audio_reference,
                    audio_format=context.audio_format,
                    duration_seconds=context.duration_seconds,
                    session_policy_snapshot=context.session_policy_snapshot,
                    manual_retry_attempt=context.manual_retry_attempt,
                ),
                audio_bytes=audio_bytes,
                file_name=file_name,
                content_type=content_type,
            )
            transcript = stt.transcript
            usage = self._merge_usage(usage, stt.usage)
        except SpeakingStageException as exc:
            return self._partial_failure(
                context=context,
                stage=exc.stage,
                error=exc.to_schema(),
                transcript=None,
                usage=usage,
            )

        try:
            conversation_request = context.model_copy(update={"transcript": transcript})
            conversation = await self.conversation_service.generate(
                conversation_request
            )
            usage = self._merge_usage(usage, conversation.usage)
        except SpeakingStageException as exc:
            return self._partial_failure(
                context=context,
                stage=exc.stage,
                error=exc.to_schema(),
                transcript=transcript,
                usage=usage,
            )

        assistant, tts_usage = await self._assistant_with_tts(
            request_id=context.request_id,
            idempotency_key=context.idempotency_key,
            session_id=context.session_id,
            learning_language=context.learning_language,
            voice=context.voice,
            playback_speed=context.playback_speed,
            automatic_retry_limit=(
                context.session_policy_snapshot.automatic_retry_limit_per_stage
            ),
            text=conversation.assistant_text,
        )
        usage = self._merge_usage(usage, tts_usage)
        return TurnProcessResponse(
            request_id=context.request_id,
            session_id=context.session_id,
            turn_index=context.turn_index,
            status="READY" if assistant.tts_error is None else "PARTIAL_FAILURE",
            transcript=transcript,
            assistant=assistant,
            conversation=conversation.conversation,
            failed_stage=(
                SpeakingStage.TTS.value if assistant.tts_error is not None else None
            ),
            error=assistant.tts_error,
            usage=usage,
            internal_metadata={
                "conversationPromptVersion": (
                    usage.conversation.prompt_version
                    if usage.conversation
                    else None
                ),
            },
        )

    async def transcribe_audio(
        self,
        *,
        context: SttRequestContext,
        audio_bytes: bytes,
        file_name: str | None,
        content_type: str | None,
    ) -> SttResponse:
        policy = context.session_policy_snapshot
        normalized = self.audio_processor.validate_and_normalize(
            audio_bytes,
            file_name=file_name,
            content_type=content_type,
            min_seconds=policy.min_valid_audio_seconds,
            max_seconds=policy.max_turn_audio_seconds,
            max_bytes=policy.max_audio_file_bytes,
        )
        return await self.stt_service.transcribe(
            request_id=context.request_id,
            session_id=context.session_id,
            turn_index=context.turn_index,
            learning_language=context.learning_language,
            normalized_audio=normalized,
            phrase_hints=context.phrase_hints,
            idempotency_key=context.idempotency_key,
            automatic_retry_limit=policy.automatic_retry_limit_per_stage,
            manual_retry_attempt=context.manual_retry_attempt,
        )

    async def _assistant_with_tts(
        self,
        *,
        request_id: str,
        idempotency_key: str,
        session_id: str,
        learning_language: str,
        voice: str,
        playback_speed: str,
        automatic_retry_limit: int,
        text: str,
    ) -> tuple[AssistantTurn, SpeakingUsage]:
        try:
            tts = await self.tts_service.synthesize(
                TtsRequest(
                    request_id=request_id,
                    idempotency_key=f"{idempotency_key}:tts",
                    session_id=session_id,
                    text=text,
                    learning_language=learning_language,
                    voice=voice,
                    playback_speed=playback_speed,
                    automatic_retry_limit=automatic_retry_limit,
                )
            )
            return (
                AssistantTurn(
                    text=text,
                    voice=voice,
                    audio=tts.audio,
                ),
                tts.usage,
            )
        except SpeakingStageException as exc:
            return (
                AssistantTurn(
                    text=text,
                    voice=voice,
                    audio=None,
                    tts_error=exc.to_schema(),
                ),
                SpeakingUsage(),
            )

    @staticmethod
    def _resolve_start_mode(request: SessionStartRequest) -> ConversationStartMode:
        if request.conversation_start_mode != ConversationStartMode.TOPIC_RECOMMENDED:
            return request.conversation_start_mode
        assert request.topic_recommended_start_mode is not None
        if request.topic_recommended_start_mode == ConversationStartMode.TOPIC_RECOMMENDED:
            return ConversationStartMode.AI_FIRST
        return request.topic_recommended_start_mode

    @staticmethod
    def _partial_failure(
        *,
        context: ConversationGenerationRequest,
        stage: SpeakingStage,
        error: SpeakingError,
        transcript,
        usage: SpeakingUsage,
    ) -> TurnProcessResponse:
        return TurnProcessResponse(
            request_id=context.request_id,
            session_id=context.session_id,
            turn_index=context.turn_index,
            status="PARTIAL_FAILURE",
            transcript=transcript,
            failed_stage=stage.value,
            error=error,
            usage=usage,
        )

    @staticmethod
    def _merge_usage(left: SpeakingUsage, right: SpeakingUsage) -> SpeakingUsage:
        return SpeakingUsage(
            stt=right.stt or left.stt,
            conversation=right.conversation or left.conversation,
            tts=right.tts or left.tts,
            evaluation=right.evaluation or left.evaluation,
        )
