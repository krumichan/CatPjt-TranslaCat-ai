import logging
import os
import warnings
from logging.config import dictConfig

from app.core.config import settings


def setup_logging() -> None:
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("FLAGS_minloglevel", "2")
    os.environ.setdefault(
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK",
        str(settings.PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK),
    )

    warnings.filterwarnings("ignore", message="No ccache found.*")
    warnings.filterwarnings(
        "ignore",
        message="urllib3 .* or chardet .* doesn't match a supported version.*",
    )

    third_party_level = settings.THIRD_PARTY_LOG_LEVEL.upper()

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
        },
        "root": {
            "level": settings.LOG_LEVEL.upper(),
            "handlers": ["console"],
        },
        "loggers": {
            "app": {
                "level": settings.APP_LOG_LEVEL.upper(),
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn": {"level": settings.UVICORN_LOG_LEVEL.upper()},
            "uvicorn.error": {"level": settings.UVICORN_LOG_LEVEL.upper()},
            "uvicorn.access": {"level": settings.UVICORN_LOG_LEVEL.upper()},
            "PIL": {"level": third_party_level},
            "PIL.TiffImagePlugin": {"level": third_party_level},
            "python_multipart": {"level": third_party_level},
            "multipart": {"level": third_party_level},
            "filelock": {"level": third_party_level},
            "httpcore": {"level": third_party_level},
            "httpx": {"level": third_party_level},
            "urllib3": {"level": third_party_level},
            "google_genai": {"level": third_party_level},
            "google_genai.models": {"level": third_party_level},
            "paddle": {"level": third_party_level},
            "paddlex": {"level": third_party_level},
        },
    }

    dictConfig(logging_config)
