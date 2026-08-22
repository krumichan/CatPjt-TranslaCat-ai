from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Server
    SERVER_API_KEY: str = ""

    # AI Provider
    AI_TEXT_PROVIDER: str = "gemini"
    AI_STT_MAX_AUDIO_FILE_BYTES: int = Field(
        default=25 * 1024 * 1024,
        ge=1024,
    )

    # Google / Gemini
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"

    # AI Chat Member Reply
    AI_CHAT_CONTEXT_DEFAULT_MAX_MESSAGES: int = 30
    AI_CHAT_CONTEXT_HARD_MAX_MESSAGES: int = 100
    AI_CHAT_CONTEXT_DEFAULT_MAX_CHARACTERS: int = 12_000
    AI_CHAT_CONTEXT_HARD_MAX_CHARACTERS: int = 50_000
    AI_CHAT_REPLY_MAX_CHARACTERS: int = 800
    AI_CHAT_REPLY_HARD_MAX_CHARACTERS: int = 4_000
    AI_CHAT_REPLY_TIMEOUT_SECONDS: float = 20.0

    # Language Learning / Adaptive Writing
    AI_LANGUAGE_LEARNING_HARD_MAX_SENTENCE_COUNT: int = 100
    AI_LANGUAGE_LEARNING_HARD_MAX_SELECTED_KEYWORDS: int = 20
    AI_LANGUAGE_LEARNING_GENERATION_TIMEOUT_SECONDS: float = 30.0
    AI_LANGUAGE_LEARNING_EVALUATION_TIMEOUT_SECONDS: float = 30.0
    AI_LANGUAGE_LEARNING_LEVEL_TEST_TIMEOUT_SECONDS: float = 30.0
    AI_LANGUAGE_LEARNING_GENERATION_MAX_RETRIES: int = 3
    AI_LANGUAGE_LEARNING_EVALUATION_MAX_RETRIES: int = 1

    # Language Learning / AI Speaking
    AI_SPEAKING_STT_MODEL_NAME: str = "tiny"
    AI_SPEAKING_STT_DEVICE: str = "cpu"
    AI_SPEAKING_STT_COMPUTE_TYPE: str = "int8"
    AI_SPEAKING_STT_TIMEOUT_SECONDS: float = 30.0
    AI_SPEAKING_CONVERSATION_TIMEOUT_SECONDS: float = 30.0
    AI_SPEAKING_TTS_TIMEOUT_SECONDS: float = 30.0
    AI_SPEAKING_EVALUATION_TIMEOUT_SECONDS: float = 60.0
    AI_SPEAKING_AUTOMATIC_RETRY_LIMIT: int = 2
    AI_SPEAKING_MANUAL_RETRY_LIMIT: int = 1
    AI_SPEAKING_MIN_VALID_AUDIO_SECONDS: float = 1.0
    AI_SPEAKING_MAX_TURN_AUDIO_SECONDS: float = 60.0
    AI_SPEAKING_MAX_AUDIO_FILE_BYTES: int = 10 * 1024 * 1024
    AI_SPEAKING_STT_LOW_CONFIDENCE_THRESHOLD: float = 0.55
    AI_SPEAKING_EVALUATION_CONFIDENCE_THRESHOLD: float = 0.70
    AI_SPEAKING_TTS_AUDIO_TTL_SECONDS: int = 3600
    GEMINI_TTS_MODEL_NAME: str = "gemini-2.5-flash-preview-tts"

    # Voice Translation V2 / Internal Streaming Pipeline
    AI_VOICE_ENABLED: bool = True
    AI_VOICE_STT_MODEL_NAME: str = "base"
    AI_VOICE_STT_MODEL_REVISION: str | None = None
    AI_VOICE_STT_DEVICE: str = "cpu"
    AI_VOICE_STT_COMPUTE_TYPE: str = "int8"
    AI_VOICE_STT_CPU_THREADS: int = Field(default=2, ge=1, le=64)
    AI_VOICE_STT_NUM_WORKERS: int = Field(default=1, ge=1, le=16)
    AI_VOICE_STT_MAX_CONCURRENCY: int = Field(default=1, ge=1, le=16)
    AI_VOICE_STT_QUEUE_CAPACITY: int = Field(default=8, ge=2, le=1024)
    AI_VOICE_MAX_ACTIVE_STREAMS: int = Field(default=8, ge=1, le=1024)
    AI_VOICE_STT_PARTIAL_TIMEOUT_SECONDS: float = Field(default=3.0, gt=0)
    AI_VOICE_STT_FINAL_TIMEOUT_SECONDS: float = Field(default=8.0, gt=0)
    AI_VOICE_STT_RUN_WARM_UP_INFERENCE: bool = True
    AI_VOICE_PARTIAL_INTERVAL_MS: int = Field(default=600, ge=400, le=800)
    AI_VOICE_MIN_FRAME_DURATION_MS: int = Field(default=20, ge=20, le=200)
    AI_VOICE_MAX_FRAME_DURATION_MS: int = Field(default=200, ge=20, le=200)
    AI_VOICE_MAX_BUFFERED_AUDIO_MS: int = Field(default=3000, ge=500, le=30_000)
    AI_VOICE_VAD_RMS_THRESHOLD: float = Field(default=0.012, ge=0, le=1)
    AI_VOICE_VAD_SILERO_GUARD_ENABLED: bool = True
    AI_VOICE_VAD_SILERO_THRESHOLD: float = Field(default=0.50, ge=0, le=1)
    AI_VOICE_VAD_SILERO_MIN_SPEECH_MS: int = Field(default=100, ge=32, le=250)
    AI_VOICE_VAD_SILERO_TIMEOUT_SECONDS: float = Field(default=1.0, gt=0)
    AI_VOICE_VAD_START_EVIDENCE_MS: int = Field(default=40, ge=20, le=250)
    AI_VOICE_VAD_PRE_ROLL_MS: int = Field(default=100, ge=0, le=500)
    AI_VOICE_VAD_POST_ROLL_MS: int = Field(default=100, ge=0, le=500)
    AI_VOICE_FORCE_SPLIT_OVERLAP_MS: int = Field(default=100, ge=0, le=500)
    AI_VOICE_STT_NO_SPEECH_PROBABILITY_THRESHOLD: float = Field(
        default=0.80,
        ge=0,
        le=1,
    )
    AI_VOICE_LANGUAGE_MIN_CONFIDENCE: float = Field(default=0.50, ge=0, le=1)
    AI_VOICE_TRANSLATION_MODEL_NAME: str = "gemini-2.5-flash"
    AI_VOICE_TRANSLATION_TIMEOUT_SECONDS: float = Field(default=3.0, gt=0)
    AI_VOICE_TRANSLATION_MAX_RETRIES: int = Field(default=1, ge=0, le=3)
    AI_VOICE_TRANSLATION_MAX_CONCURRENCY: int = Field(default=4, ge=1, le=64)
    AI_VOICE_TRANSLATION_MAX_OUTPUT_TOKENS: int = Field(
        default=1024,
        ge=64,
        le=4096,
    )
    AI_VOICE_TRANSLATION_IDEMPOTENCY_TTL_SECONDS: int = Field(
        default=600,
        ge=1,
    )
    AI_VOICE_TRANSLATION_IDEMPOTENCY_MAX_ENTRIES: int = Field(
        default=1000,
        ge=1,
    )
    AI_VOICE_PROMPT_VERSION: str = "voice-v2"
    AI_VOICE_VAD_VERSION: str = "energy-silero-v1"
    AI_VOICE_SCHEMA_VERSION: str = "voice-stream-v2"
    AI_VOICE_BACKPRESSURE_RETRY_AFTER_MS: int = Field(default=100, ge=0)
    AI_VOICE_STREAM_OPEN_TIMEOUT_SECONDS: float = Field(default=5.0, gt=0)
    AI_VOICE_SHUTDOWN_GRACE_SECONDS: float = Field(default=5.0, gt=0)

    # Logging
    LOG_LEVEL: str = "INFO"
    APP_LOG_LEVEL: str = "DEBUG"
    THIRD_PARTY_LOG_LEVEL: str = "WARNING"
    UVICORN_LOG_LEVEL: str = "INFO"

    # OCR
    OCR_LANGUAGE: str = "japan"
    OCR_VERSION: str = "PP-OCRv3"
    OCR_WARM_UP: bool = True
    OCR_MAX_IMAGE_WIDTH: int = 900
    OCR_MAX_IMAGE_HEIGHT: int = 1400
    OCR_MAX_IMAGE_PIXELS: int = 2_000_000
    OCR_IMAGE_QUALITY: int = 85
    OCR_MAX_FILE_SIZE: int = 5 * 1024 * 1024
    OCR_ENABLE_MKLDNN: bool = True
    OCR_CPU_THREADS: int = 2
    OCR_TEXT_RECOGNITION_BATCH_SIZE: int = 6
    OCR_TEXT_DET_LIMIT_SIDE_LEN: int = 960
    OCR_TEXT_DET_LIMIT_TYPE: str = "max"
    RECEIPT_ANALYSIS_MODE: str = "OCR_WITH_AI"
    GEMINI_VISION_CONFIDENCE_THRESHOLD: float = 0.75

    OCR_ALLOWED_CONTENT_TYPES: set[str] = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }
    OCR_ALLOWED_EXTENSIONS: set[str] = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    # Paddle
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_voice_frame_range(self) -> "Settings":
        if self.AI_VOICE_MIN_FRAME_DURATION_MS > self.AI_VOICE_MAX_FRAME_DURATION_MS:
            raise ValueError(
                "AI_VOICE_MIN_FRAME_DURATION_MS must not exceed the maximum"
            )
        return self


settings = Settings()
