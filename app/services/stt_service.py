import logging
import os
import shutil
import tempfile
import uuid

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class STTService:
    def __init__(self) -> None:
        self.model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
            cpu_threads=1,
            num_workers=1,
        )

    async def transcribe_file(self, upload_file) -> str:
        temp_file_path = self._save_temp_file(upload_file)

        try:
            return await self._do_transcribe(temp_file_path)
        finally:
            self._cleanup_temp_file(temp_file_path)

    def _save_temp_file(self, upload_file) -> str:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{upload_file.filename}")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

        return file_path

    async def _do_transcribe(self, audio_path: str, lang: str = "ja") -> str:
        try:
            segments, _ = self.model.transcribe(
                audio_path,
                beam_size=1,
                language=lang,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )

            return "".join(segment.text for segment in segments).strip()
        except Exception as exc:
            logger.error("STT 분석 실패: %s", exc)
            raise

    def _cleanup_temp_file(self, path: str) -> None:
        if os.path.exists(path):
            os.remove(path)
