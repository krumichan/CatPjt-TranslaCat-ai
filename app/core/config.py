from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Server
    SERVER_API_KEY: str = ""

    # AI Provider
    AI_TEXT_PROVIDER: str = "gemini"

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


settings = Settings()
