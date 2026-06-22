from typing import List

from pydantic import BaseModel, Field


class SingleTranslationRequest(BaseModel):
    text: str = Field(..., description="번역할 단일 문장")
    type: str = Field(..., description="번역 타입 (novel, chat 등)")


class BatchTranslationRequest(BaseModel):
    texts: List[str] = Field(..., description="번역할 문장들의 리스트")
    type: str = Field(..., description="번역 타입 (novel, chat 등)")
