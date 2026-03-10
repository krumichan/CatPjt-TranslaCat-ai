from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    SERVER_API_KEY: str = ""
    
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = ""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore" # .env에 다른 설정이 있어도 무시하도록 설정
    )

settings = Settings()