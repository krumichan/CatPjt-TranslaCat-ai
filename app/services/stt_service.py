import io
import logging

from app.core.config import settings
from app.features.speech_to_text import FasterWhisperRuntime, InferencePriority

logger = logging.getLogger(__name__)


class EmptyAudioFile(ValueError):
    pass


class AudioFileTooLarge(ValueError):
    pass


class STTService:
    """Compatibility adapter for the legacy multipart STT endpoint."""

    def __init__(self, runtime: FasterWhisperRuntime | None = None) -> None:
        self.runtime = runtime or FasterWhisperRuntime()

    async def transcribe_file(self, upload_file) -> str:
        audio_bytes = await upload_file.read(settings.AI_STT_MAX_AUDIO_FILE_BYTES + 1)
        if not audio_bytes:
            raise EmptyAudioFile("업로드된 Audio File이 비어 있습니다.")
        if len(audio_bytes) > settings.AI_STT_MAX_AUDIO_FILE_BYTES:
            raise AudioFileTooLarge("업로드된 Audio File이 허용 크기를 초과했습니다.")
        if not self.runtime.ready:
            await self.runtime.warm_up()
        try:
            result = await self.runtime.transcribe(
                io.BytesIO(audio_bytes),
                options={
                    "beam_size": 1,
                    "language": None,
                    "vad_filter": True,
                    "vad_parameters": {"min_silence_duration_ms": 500},
                    "condition_on_previous_text": False,
                },
                priority=InferencePriority.STANDARD,
            )
            return result.text
        except Exception:
            logger.error("Legacy STT processing failed")
            raise
