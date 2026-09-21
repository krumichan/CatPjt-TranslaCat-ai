"""Public entry policy; legacy generation fixtures remain usable offline."""
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_language_learning_reading_vocabulary_service
from app.api.v1.language_learning_practice import generate_practice, router
from app.schemas.language_learning_practice import PracticeGenerationRequest


def _request(domain: str, *, plan_only: bool = False) -> PracticeGenerationRequest:
    return PracticeGenerationRequest.model_validate({
        "requestId": "product-contract", "domain": domain,
        "mode": "CONTEXTUAL_CHOICE" if domain == "VOCABULARY" else "COMPREHENSION",
        "originLanguage": "ko", "learningLanguage": "ja",
        "questionCount": 10 if domain == "VOCABULARY" else 5,
        "complexityBand": 3, "easierCount": 2 if domain == "VOCABULARY" else 1,
        "currentCount": 6 if domain == "VOCABULARY" else 3,
        "challengeCount": 2 if domain == "VOCABULARY" else 1,
        "generationDate": "2026-09-19", "vocabularyPlanOnly": plan_only,
    })


@pytest.mark.parametrize("plan_only", [False, True])
def test_retired_vocabulary_api_never_enters_generation(plan_only):
    service = AsyncMock()
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_language_learning_reading_vocabulary_service] = lambda: service
    with TestClient(app) as client:
        response = client.post("/language-learning/practice/generate", json=_request(
            "VOCABULARY", plan_only=plan_only,
        ).model_dump(mode="json", by_alias=True))
    assert response.status_code == 410
    assert response.json()["detail"] == {
        "code": "DAILY_VOCABULARY_RETIRED", "policyVersion": "reading-first-v1",
    }
    service.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_reading_entry_delegates_unchanged_request():
    request = _request("READING")
    service = AsyncMock()
    result = await generate_practice(request, service)
    assert result is service.generate.return_value
    service.generate.assert_awaited_once_with(request)
