import os
import shutil
import tempfile
import uuid

from faster_whisper import WhisperModel
import logging

logger = logging.getLogger(__name__)

class STTService:
    """
    faster-whisper 라이브러리를 사용하여 오디오 파일을 텍스트로 변환하는 서비스 클래스.
    
    이 서비스는 OpenAI의 Whisper 모델을 CTranslate2를 통해 최적화하여 구현되었으며,
    CPU 환경에서도 빠른 속도로 음성 인식을 수행할 수 있도록 설정되어 있습니다.
    """

    def __init__(self):
        """
        STT 서비스를 초기화하고 Whisper 모델을 메모리에 로드합니다.

        설정 상세:
            - model_size: "base" (속도와 정확도의 균형을 맞춘 기본 모델)
            - device: "cpu" (GPU가 없는 환경을 위해 CPU 사용)
            - compute_type: "int8" (8비트 양자화를 통해 메모리 사용량 절감 및 속도 향상)
        """
        self.model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=1, num_workers=1)
    
    async def transcribe_file(self, upload_file) -> str:
        """
        업로드된 파일 객체를 받아서 임시 저장 후 텍스트로 변환합니다.
        파일 저장부터 삭제까지의 라이프사이클을 관리합니다.
        """
        temp_file_path = self._save_temp_file(upload_file)
        
        try:
            # 기존에 작성한 변환 로직 호출
            result_text = await self._do_transcribe(temp_file_path)
            return result_text
        finally:
            self._cleanup_temp_file(temp_file_path)
    
    def _save_temp_file(self, upload_file) -> str:
        """파일을 임시 디렉토리에 저장"""
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{upload_file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return file_path

    async def _do_transcribe(self, audio_path: str, lang: str = "ja") -> str:
        """
        오디오 파일을 분석하여 텍스트로 변환(Transcription)합니다.

        Args:
            audio_path (str): 분석할 오디오 파일의 로컬 경로.
            lang (str, optional): 음성 언어 코드. 기본값은 "ja" (일본어).

        Returns:
            str: 인식된 텍스트 전체 문자열. 공백이 제거된 상태로 반환됩니다.

        Raises:
            Exception: 오디오 파일 읽기 실패나 모델 분석 중 오류 발생 시 예외를 발생시킵니다.
        """
        try:
            segments, _ = self.model.transcribe(audio_path, beam_size=1, language=lang, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
            # 인식 정보 로그 출력 (필요 시 주석 해제)
            # logger.info(f"언어 감지: {info.language} (확률: {info.language_probability:.2f})")
            # logger.info(f"오디오 길이: {info.duration:.2f}초")
            return "".join([segment.text for segment in segments]).strip()
        except Exception as e:
            logger.error(f"STT 분석 실패: {e}")
            raise e

    def _cleanup_temp_file(self, path: str):
        """임시 파일 삭제"""
        if os.path.exists(path):
            os.remove(path)