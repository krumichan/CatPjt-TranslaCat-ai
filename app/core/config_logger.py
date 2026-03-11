import logging
from logging.config import dictConfig

def setup_logging():
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
            "level": "DEBUG",  # 여기서 전체 기본 레벨 설정
            "handlers": ["console"],
        },
        "loggers": {
            "app": {  # 특정 모듈(예: app 하위)만 레벨 조정 가능
                "level": "DEBUG",
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn.error": {"level": "INFO"}, # 유비콘 에러는 INFO만
            "uvicorn.access": {"level": "INFO"}, # 접근 로그는 INFO만
        },
    }
    dictConfig(logging_config)