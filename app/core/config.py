from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Server
    SERVER_API_KEY: str = ""

    # Google / Gemini
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"

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