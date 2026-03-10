from app.services.gemini_service import GeminiService
from app.services.stt_service import STTService

# 전역 변수로 유일 인스턴스 유지.
_gemini_service = GeminiService()
_stt_service = STTService()

def get_gemini_service() -> GeminiService:
    """GeminiService 인스턴스를 반환하는 의존성 주입 함수"""
    return _gemini_service

def get_stt_service() -> STTService:
    """STTService 인스턴스를 반환하는 의존성 주입 함수"""
    return _stt_service