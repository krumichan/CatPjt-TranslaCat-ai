"""Exercise the actual OpenAI adapter with responses.create doubled, no SDK/network."""
from __future__ import annotations
import asyncio
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from app.features.language_learning.writing.verification import MINI_QUALITY_TASK, TaskReview
from app.ai.providers.openai.response import OpenAIProviderResponseError


def adapter(monkeypatch, response):
    sdk = ModuleType("openai")
    class NoClient:
        def __init__(self, **kwargs):
            raise AssertionError("A real SDK client must not be constructed")
    sdk.AsyncOpenAI = NoClient
    monkeypatch.setitem(sys.modules, "openai", sdk)
    path = Path(__file__).parents[1] / "app/ai/providers/openai/client.py"
    spec = importlib.util.spec_from_file_location("_writing_openai_adapter_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    service = module.OpenAIService()
    calls = []
    async def create(**kwargs):
        calls.append(kwargs)
        return response
    service._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    return service, calls


def test_adapter_uses_strict_review_schema_and_reports_actual_usage(monkeypatch):
    response = SimpleNamespace(status="completed", output_text=json.dumps({"probe": True}), output=[{"content": None}],
                               model="served-model", usage=SimpleNamespace(input_tokens=12, output_tokens=8))
    service, calls = adapter(monkeypatch, response)
    result = asyncio.run(service.call_with_metadata(MINI_QUALITY_TASK, "PRIVATE_PROMPT", TaskReview.model_json_schema(by_alias=True)))
    assert result.data == {"probe": True}
    assert result.model == "served-model" and result.provider == "openai"
    assert result.input_tokens == 12 and result.output_tokens == 8
    assert len(calls) == 1 and calls[0]["text"]["format"]["strict"] is True
    assert calls[0]["store"] is False


@pytest.mark.parametrize("response,code", [
    (SimpleNamespace(status="incomplete", incomplete_details={"reason": "max_output_tokens"}), "OUTPUT_TOKEN_LIMIT"),
    (SimpleNamespace(status="completed", output_text="PRIVATE_BROKEN_JSON", output=[]), "JSON_INVALID"),
    (SimpleNamespace(status="completed", output_text="", output=[{"content": [{"type": "refusal", "refusal": "PRIVATE_REFUSAL"}]}]), "REFUSAL"),
])
def test_adapter_propagates_safe_failure_codes_without_raw_response_logging(monkeypatch, response, code, caplog):
    service, calls = adapter(monkeypatch, response)
    with caplog.at_level(logging.INFO), pytest.raises(OpenAIProviderResponseError) as exc:
        asyncio.run(service.call_with_metadata(MINI_QUALITY_TASK, "PRIVATE_PROMPT", TaskReview.model_json_schema(by_alias=True)))
    assert exc.value.reason_code == code
    assert len(calls) == 1 and "PRIVATE_" not in caplog.text
