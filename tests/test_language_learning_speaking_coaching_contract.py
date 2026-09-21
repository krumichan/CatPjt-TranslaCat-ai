"""Evidence-v2 retires an unused mandatory link, preserving public compatibility."""
from __future__ import annotations

from copy import deepcopy

import pytest

from app.ai.providers.openai.schema import build_openai_text_config
from app.features.language_learning.speaking.conversation_service import (
    SpeakingConversationService, _conversation_response_schema,
)
from app.schemas.language_learning_speaking import ConversationPayload
from tests.test_language_learning_speaking import (
    FakeStructuredProvider, conversation_payload, conversation_request,
)


def _correction_payload() -> dict:
    payload = conversation_payload()
    payload["coachingCorrections"] = [{
        "original": "Synthetic original", "improved": "Synthetic improved",
        "explanation": "Synthetic explanation", "improvementLink": None,
    }]
    return payload


def test_coaching_wire_preserves_optional_link_without_inventing_resource_semantics() -> None:
    public_before = ConversationPayload.model_json_schema()
    request = conversation_request(correctionMode="COACHING")
    schema = _conversation_response_schema(request)
    assert schema == public_before
    wire = build_openai_text_config(type_name=SpeakingConversationService.TYPE_NAME,
                                    schema=schema, verbosity="low")["format"]
    assert wire["strict"] is False
    link = wire["schema"]["$defs"]["CoachingCorrection"]["properties"]["improvementLink"]
    assert {"type": "null"} in link["anyOf"]
    assert "improvementLink" not in wire["schema"]["$defs"]["CoachingCorrection"]["required"]
    assert ConversationPayload.model_json_schema() == public_before


def test_conversation_schema_keeps_original_optional_nullable_link() -> None:
    schema = _conversation_response_schema(conversation_request(correctionMode="CONVERSATION"))
    assert schema == ConversationPayload.model_json_schema()
    wire = build_openai_text_config(type_name=SpeakingConversationService.TYPE_NAME,
                                    schema=schema, verbosity="low")["format"]["schema"]
    correction = wire["$defs"]["CoachingCorrection"]
    assert "improvementLink" not in correction["required"]
    assert {"type": "null"} in correction["properties"]["improvementLink"]["anyOf"]


@pytest.mark.asyncio
@pytest.mark.parametrize("omitted", [False, True])
async def test_coaching_null_or_missing_unsupported_link_needs_no_fabrication(omitted: bool) -> None:
    payload = _correction_payload()
    if omitted:
        payload["coachingCorrections"][0].pop("improvementLink")
    original = deepcopy(payload)
    assert ConversationPayload.model_validate(payload).coaching_corrections[0].improvement_link is None
    provider = FakeStructuredProvider(results=[payload])
    service = SpeakingConversationService(provider, timeout_seconds=1, automatic_retries=2)
    result = await service.generate(conversation_request(correctionMode="COACHING"))
    assert result.conversation.coaching_corrections[0].improvement_link is None
    assert len(provider.calls) == 1
    assert payload == original


@pytest.mark.asyncio
async def test_coaching_valid_string_is_passed_through_in_one_call_without_synthesis() -> None:
    payload = _correction_payload()
    payload["coachingCorrections"][0]["improvementLink"] = "provider-owned existing string"
    original = deepcopy(payload)
    provider = FakeStructuredProvider(results=[payload])
    result = await SpeakingConversationService(provider, timeout_seconds=1).generate(
        conversation_request(correctionMode="COACHING"))
    assert result.conversation.coaching_corrections[0].improvement_link == "provider-owned existing string"
    assert len(provider.calls) == 1
    assert payload == original


@pytest.mark.asyncio
async def test_conversation_mode_still_accepts_nullable_link_without_extra_calls() -> None:
    provider = FakeStructuredProvider(results=[_correction_payload()])
    result = await SpeakingConversationService(provider, timeout_seconds=1).generate(
        conversation_request(correctionMode="CONVERSATION"))
    assert result.conversation.coaching_corrections[0].improvement_link is None
    assert len(provider.calls) == 1
