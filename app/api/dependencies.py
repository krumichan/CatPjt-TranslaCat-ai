from app.services.gemini_service import GeminiService
from app.services.ocr_service import OCRService
from app.services.receipt_analysis_service import ReceiptAnalysisService
from app.services.stt_service import STTService

# 전역 변수로 유일 인스턴스 유지.
_gemini_service = GeminiService()
_stt_service = STTService()
_ocr_service = OCRService()
_receipt_analysis_service = ReceiptAnalysisService(
    ocr_service=_ocr_service,
    gemini_service=_gemini_service,
)


def get_gemini_service() -> GeminiService:
    """GeminiService 인스턴스를 반환하는 의존성 주입 함수"""
    return _gemini_service


def get_stt_service() -> STTService:
    """STTService 인스턴스를 반환하는 의존성 주입 함수"""
    return _stt_service


def get_ocr_service() -> OCRService:
    """OCRService 인스턴스를 반환하는 의존성 주입 함수"""
    return _ocr_service


def get_receipt_analysis_service() -> ReceiptAnalysisService:
    """ReceiptAnalysisService 인스턴스를 반환하는 의존성 주입 함수"""
    return _receipt_analysis_service