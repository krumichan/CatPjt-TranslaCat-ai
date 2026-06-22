from app.ai.provider_factory import create_text_generation_provider
from app.features.chat_translation.service import ChatTranslationService
from app.features.receipt.service import ReceiptAnalysisService
from app.features.translation.service import TranslationService
from app.services.ocr_service import OCRService
from app.services.stt_service import STTService

_ai_provider = create_text_generation_provider()
_stt_service = STTService()
_ocr_service = OCRService()
_translation_service = TranslationService(provider=_ai_provider)
_chat_translation_service = ChatTranslationService(provider=_ai_provider)
_receipt_analysis_service = ReceiptAnalysisService(
    ocr_service=_ocr_service,
    gemini_service=_ai_provider,
)


def get_ai_provider():
    return _ai_provider


def get_translation_service() -> TranslationService:
    return _translation_service


def get_chat_translation_service() -> ChatTranslationService:
    return _chat_translation_service


def get_stt_service() -> STTService:
    return _stt_service


def get_ocr_service() -> OCRService:
    return _ocr_service


def get_receipt_analysis_service() -> ReceiptAnalysisService:
    return _receipt_analysis_service


# Backward-compatible dependency name.
def get_gemini_service():
    return _ai_provider
