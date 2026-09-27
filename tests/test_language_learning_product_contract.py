"""Practice의 공개 업무 경로는 LL Gateway 전환 뒤 Python에서 제거된다."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.ai.prompt_registry import get_prompt_rule
from app.api.v1 import api_router


def test_replaced_practice_endpoint_is_not_registered():
    # 준비: 실제 공개 router를 사용하되 모델 실행이나 서비스 lifespan은 시작하지 않는다.
    app = FastAPI()
    app.include_router(api_router)

    # 실행
    with TestClient(app) as client:
        response = client.post("/language-learning/practice/generate", json={})

    # 검증: 신규·기존 Vocabulary 구분도 LL이 소유하며 Python 업무 경로는 없다.
    assert response.status_code == 404


def test_practice_business_prompts_are_not_registered_in_python():
    # 준비
    tasks = (
        "LANGUAGE_LEARNING_READING_VOCABULARY_GENERATION",
        "LANGUAGE_LEARNING_READING_PASSAGE_GENERATION",
        "LANGUAGE_LEARNING_READING_VOCABULARY_VERIFICATION",
        "LANGUAGE_LEARNING_READING_VOCABULARY_ORIGIN_EXPLANATION",
    )

    # 실행 및 검증: Kotlin이 prompt/schema를 명시한 범용 실행만 사용한다.
    assert all(get_prompt_rule(task) is None for task in tasks)
