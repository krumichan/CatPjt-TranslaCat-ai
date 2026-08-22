from app.ai.provider_factory import create_text_generation_provider
from app.core.config import settings
from app.features.chat_ai_reply.service import ChatAiReplyService
from app.features.chat_translation.service import ChatTranslationService
from app.features.language_learning.speaking.assistance_service import (
    SpeakingAssistanceService,
)
from app.features.language_learning.speaking.audio_processor import (
    SpeakingAudioProcessor,
)
from app.features.language_learning.speaking.audio_store import TemporaryTtsAudioStore
from app.features.language_learning.speaking.conversation_service import (
    SpeakingConversationService,
)
from app.features.language_learning.speaking.evaluation_service import (
    SpeakingEvaluationService,
)
from app.features.language_learning.speaking.stt_service import (
    FasterWhisperSpeakingSttProvider,
    SpeakingSttService,
)
from app.features.language_learning.speaking.tts_service import SpeakingTtsService
from app.features.language_learning.speaking.turn_service import SpeakingTurnService
from app.features.language_learning.writing.service import (
    LanguageLearningWritingService,
)
from app.features.receipt.service import ReceiptAnalysisService
from app.features.speech_to_text import FasterWhisperRuntime
from app.features.translation.service import TranslationService
from app.features.voice_translation.stt import FasterWhisperVoiceSttProvider
from app.features.voice_translation.speech_detector import SileroSpeechEvidenceGuard
from app.features.voice_translation.stream import VoiceStreamApplicationService
from app.features.voice_translation.translation import VoiceTranslationService
from app.services.ocr_service import OCRService
from app.services.stt_service import STTService

_ai_provider = create_text_generation_provider()

_speech_runtime = FasterWhisperRuntime()
_stt_service = STTService(runtime=_speech_runtime)
_ocr_service = OCRService()

_translation_service = TranslationService(
    provider=_ai_provider,
)

_chat_translation_service = ChatTranslationService(
    provider=_ai_provider,
)

_chat_ai_reply_service = ChatAiReplyService(
    provider=_ai_provider,
)

_language_learning_writing_service = LanguageLearningWritingService(
    provider=_ai_provider,
)

_speaking_audio_processor = SpeakingAudioProcessor()
_speaking_audio_store = TemporaryTtsAudioStore(
    ttl_seconds=settings.AI_SPEAKING_TTS_AUDIO_TTL_SECONDS
)
_speaking_stt_provider = FasterWhisperSpeakingSttProvider(runtime=_speech_runtime)
_speaking_stt_service = SpeakingSttService(_speaking_stt_provider)
_speaking_conversation_service = SpeakingConversationService(_ai_provider)
_speaking_assistance_service = SpeakingAssistanceService(_ai_provider)
_speaking_tts_service = SpeakingTtsService(
    provider=_ai_provider,
    audio_store=_speaking_audio_store,
)
_speaking_evaluation_service = SpeakingEvaluationService(_ai_provider)
_speaking_turn_service = SpeakingTurnService(
    audio_processor=_speaking_audio_processor,
    stt_service=_speaking_stt_service,
    conversation_service=_speaking_conversation_service,
    tts_service=_speaking_tts_service,
)

_receipt_analysis_service = ReceiptAnalysisService(
    ocr_service=_ocr_service,
    ai_provider=_ai_provider,
)

_voice_stt_provider = FasterWhisperVoiceSttProvider(_speech_runtime)
_voice_speech_evidence_guard = SileroSpeechEvidenceGuard()
_voice_translation_service = VoiceTranslationService(_ai_provider)
_voice_stream_service = VoiceStreamApplicationService(
    stt_provider=_voice_stt_provider,
    translation_service=_voice_translation_service,
    speech_evidence_guard=_voice_speech_evidence_guard,
)


def get_ai_provider():
    return _ai_provider


def get_speech_runtime() -> FasterWhisperRuntime:
    return _speech_runtime


def get_translation_service() -> TranslationService:
    return _translation_service


def get_chat_translation_service() -> ChatTranslationService:
    return _chat_translation_service


def get_chat_ai_reply_service() -> ChatAiReplyService:
    return _chat_ai_reply_service


def get_language_learning_writing_service() -> LanguageLearningWritingService:
    return _language_learning_writing_service


def get_language_learning_speaking_turn_service() -> SpeakingTurnService:
    return _speaking_turn_service


def get_language_learning_speaking_conversation_service() -> (
    SpeakingConversationService
):
    return _speaking_conversation_service


def get_language_learning_speaking_assistance_service() -> SpeakingAssistanceService:
    return _speaking_assistance_service


def get_language_learning_speaking_tts_service() -> SpeakingTtsService:
    return _speaking_tts_service


def get_language_learning_speaking_evaluation_service() -> SpeakingEvaluationService:
    return _speaking_evaluation_service


def get_language_learning_speaking_audio_store() -> TemporaryTtsAudioStore:
    return _speaking_audio_store


def get_stt_service() -> STTService:
    return _stt_service


def get_ocr_service() -> OCRService:
    return _ocr_service


def get_receipt_analysis_service() -> ReceiptAnalysisService:
    return _receipt_analysis_service


def get_voice_translation_service() -> VoiceTranslationService:
    return _voice_translation_service


def get_voice_speech_evidence_guard() -> SileroSpeechEvidenceGuard:
    return _voice_speech_evidence_guard


def get_voice_stream_service() -> VoiceStreamApplicationService:
    return _voice_stream_service
