from app.ai.provider_factory import (
    create_speech_synthesis_provider,
    create_text_generation_provider,
)
from app.ai.providers.openai.client import OpenAIService
from app.core.config import settings
from app.features.receipt.service import ReceiptAnalysisService
from app.features.speech_to_text import FasterWhisperRuntime
from app.features.speech_to_text.runtime_policy import speaking_runtime
from app.features.translation.service import TranslationService
from app.features.voice_translation.hosted_stt import OpenAIWhisperVoiceSttProvider
from app.features.voice_translation.preview_translation import OpenAIVoicePreviewTranslationProvider
from app.features.voice_translation.realtime_preview import OpenAILivePreviewProvider
from app.features.voice_translation.speech_detector import SileroSpeechEvidenceGuard
from app.features.voice_translation.stream import VoiceStreamApplicationService
from app.features.voice_translation.stt import FasterWhisperVoiceSttProvider, VoiceSttProvider
from app.features.voice_translation.transcribe_stt import OpenAITranscribeVoiceSttProvider
from app.features.voice_translation.translation import VoiceTranslationService
from app.services.ocr_service import OCRService
from app.services.stt_service import STTService

_ai_provider = create_text_generation_provider()
_model_execution_provider = OpenAIService()
_speech_provider = create_speech_synthesis_provider()


def get_speech_execution_provider():
    return _speech_provider


_speech_runtime = FasterWhisperRuntime()
_speaking_speech_runtime = speaking_runtime(_speech_runtime, settings)
_stt_service = STTService(runtime=_speech_runtime)
_ocr_service = OCRService()

_translation_service = TranslationService(
    provider=_ai_provider,
)

_receipt_analysis_service = ReceiptAnalysisService(
    ocr_service=_ocr_service,
    ai_provider=_ai_provider,
)

# 외부 음성 인식은 명시적 설정일 때만 선택하며 로컬 오류에 따라 자동 전환하지 않는다.
_voice_stt_provider: VoiceSttProvider
if settings.AI_VOICE_STT_PROVIDER == "openai-transcribe":
    _voice_stt_provider = OpenAITranscribeVoiceSttProvider()
elif settings.AI_VOICE_STT_PROVIDER == "openai":
    _voice_stt_provider = OpenAIWhisperVoiceSttProvider()
else:
    _voice_stt_provider = FasterWhisperVoiceSttProvider(_speech_runtime)
_voice_speech_evidence_guard = SileroSpeechEvidenceGuard()
_voice_translation_service = VoiceTranslationService(_ai_provider)
_voice_stream_service = VoiceStreamApplicationService(
    stt_provider=_voice_stt_provider,
    translation_service=_voice_translation_service,
    speech_evidence_guard=_voice_speech_evidence_guard,
    preview_provider=(
        OpenAILivePreviewProvider(settings.OPENAI_API_KEY)
        if settings.AI_VOICE_REALTIME_PREVIEW_ENABLED else None
    ),
    preview_language_provider=(
        OpenAITranscribeVoiceSttProvider()
        if settings.AI_VOICE_REALTIME_PREVIEW_ENABLED else None
    ),
    preview_translation_service=(
        VoiceTranslationService(
            OpenAIVoicePreviewTranslationProvider(_model_execution_provider),
            max_retries=0,
            max_concurrency=2,
            timeout_seconds=3,
            total_timeout_seconds=3,
        )
        if settings.AI_VOICE_REALTIME_PREVIEW_ENABLED else None
    ),
)


def get_ai_provider():
    return _ai_provider


def get_model_execution_provider() -> OpenAIService:
    return _model_execution_provider


def get_speech_synthesis_provider():
    return _speech_provider


def get_speech_runtime() -> FasterWhisperRuntime:
    return _speech_runtime


def get_speaking_speech_runtime() -> FasterWhisperRuntime:
    return _speaking_speech_runtime


def get_translation_service() -> TranslationService:
    return _translation_service


def get_stt_service() -> STTService:
    return _stt_service


def get_ocr_service() -> OCRService:
    return _ocr_service


def get_receipt_analysis_service() -> ReceiptAnalysisService:
    return _receipt_analysis_service


def get_voice_translation_service() -> VoiceTranslationService:
    return _voice_translation_service


def get_voice_stt_provider() -> VoiceSttProvider:
    return _voice_stt_provider


def get_voice_speech_evidence_guard() -> SileroSpeechEvidenceGuard:
    return _voice_speech_evidence_guard


def get_voice_stream_service() -> VoiceStreamApplicationService:
    return _voice_stream_service
