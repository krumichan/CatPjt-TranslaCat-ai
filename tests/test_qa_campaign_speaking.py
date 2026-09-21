from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.ai.ports import SpeechSynthesisResult, StructuredGenerationResult
from app.features.language_learning.speaking.stt_service import SttProviderResult, SttProviderSegment
from scripts.qa_campaign_speaking import SCENARIO_PLAN, run_speaking_campaign
from scripts.qa_campaign_speaking import QaCheckpointFailure, _Capture
from scripts.qa_campaign_budget import CampaignBudgetExceeded
from tests.test_language_learning_speaking import evaluation_payload, make_wav


class CampaignTextProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_with_metadata(self, type_name, data, schema=None):
        request = json.loads(data.split("\n\n")[-1])
        self.calls.append((type_name, request))
        if type_name.endswith("EVALUATION"):
            payload = evaluation_payload()
            turn_id = request["userTurns"][0]["turnId"]
            for metric in payload["metrics"]:
                if request["practiceMode"] == "READ_ALOUD" and metric["type"] != "MEANING":
                    metric.update(state="NOT_EVALUABLE", score=None, evidence=[], notEvaluableReason="Unsupported")
                for evidence in metric["evidence"]:
                    evidence["turnId"] = turn_id
            if request["practiceMode"] == "READ_ALOUD":
                payload["profileSignals"] = []
            for key in ("profileSignals", "recommendedExpressions", "pronunciationPractice"):
                for item in payload[key]:
                    item["evidenceTurnIds"] = [turn["turnId"] for turn in request["userTurns"][:2]]
        else:
            text = f"第{len(self.calls)}問です。仕事の予定を相談しましょう。"
            mode = request["practiceMode"]
            payload = {
                "assistantText": text, "intent": "DAILY_CHAT", "difficulty": "B1",
                "scriptText": text if mode == "READ_ALOUD" else None,
                "providedFacts": ["会議は水曜日です"] if mode == "GUIDED" else [],
                "requiredIntents": ["都合を伝える"] if mode == "GUIDED" else [],
                "responseConstraints": ["理由を説明する"] if mode == "GUIDED" else [],
            }
        return StructuredGenerationResult(payload, input_tokens=10, output_tokens=10, provider="fake", model="fake")


class CampaignSpeechProvider:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.audio = make_wav(12)

    async def synthesize_speech(self, **kwargs):
        self.calls.append(kwargs)
        return SpeechSynthesisResult(self.audio, "audio/wav", "fake", "fake", 12)


class CampaignSttProvider:
    calls = 0

    async def transcribe(self, wav_bytes, *, language, phrase_hints=None):
        self.calls += 1
        return SttProviderResult(
            "会議の予定を相談したいです。", "ja", 0.99,
            [SttProviderSegment(0, 12, "会議の予定を相談したいです。", -0.01)],
            "fake", "fake",
        )


@pytest.mark.asyncio
async def test_checkpoint_access_denial_after_provider_completion_cannot_be_retried_as_stt(tmp_path, monkeypatch):
    from scripts import qa_campaign_speaking as qa

    original = qa._checkpoint
    writes = 0
    starts = 0

    def failure_after_provider(path, value):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise PermissionError(5, "QA artifact locked")
        original(path, value)

    async def provider_operation():
        nonlocal starts
        starts += 1
        return "transcript"

    monkeypatch.setattr(qa, "_checkpoint", failure_after_provider)
    capture = _Capture(tmp_path, {"attempts": []})
    with pytest.raises(QaCheckpointFailure, match="QA_CHECKPOINT_INACCESSIBLE"):
        await capture.invoke("STT", {"audioSha256": "fake"}, provider_operation)
    assert starts == 1
    assert capture.result["attempts"][0]["status"] == "COMPLETED"
    # The previous complete checkpoint remains readable until the update can
    # be written; an access denial cannot pretend the completed stage is saved.
    assert json.loads(capture.path.read_text(encoding="utf-8"))["attempts"][0]["status"] == "STARTED"


@pytest.mark.asyncio
async def test_missing_stt_blocks_before_any_billed_call(tmp_path: Path) -> None:
    text, speech = CampaignTextProvider(), CampaignSpeechProvider()
    result = await run_speaking_campaign(text, speech, tmp_path)
    assert result["status"] == "BLOCKED_STT_UNAVAILABLE"
    assert result["partial"] is True
    assert text.calls == []
    assert speech.calls == []
    artifact = json.loads((tmp_path / "speaking-campaign-result.json").read_text(encoding="utf-8"))
    assert artifact == result


@pytest.mark.asyncio
async def test_campaign_uses_production_flows_without_new_stages(tmp_path: Path) -> None:
    text, speech, stt = CampaignTextProvider(), CampaignSpeechProvider(), CampaignSttProvider()
    result = await run_speaking_campaign(text, speech, tmp_path, stt)

    assert result["status"] == "COMPLETED"
    assert result["partial"] is False
    assert result["error"] is None
    assert "NOT_HUMAN" in result["source"]
    assert len(text.calls) == 25
    # READ_ALOUD may synthesize the same reference once more for the second
    # attempt; that is a real provider start, not an extra learner turn.
    reference_starts = sum(
        event["request"].get("purpose") == "PRODUCTION_REFERENCE_AUDIO"
        for event in result["attempts"]
    )
    assert reference_starts in {17, 18}
    assert len(speech.calls) == 20 + reference_starts
    assert stt.calls == 20
    assert len(result["attempts"]) == 25 + len(speech.calls) + stt.calls
    assert len(result["attempts"]) <= result["callEstimate"]["serviceRetryUpperBoundStarts"]
    assert sum(event["request"].get("purpose") == "QA_SYNTHETIC_LEARNER_AUDIO"
               for event in result["attempts"]) == 20
    assert all(event["status"] == "COMPLETED" for event in result["attempts"])
    assert [mode["acceptedUserTurns"] for mode in result["modes"]] == [10, 5, 5]
    repeat = result["modes"][0]
    assert len(repeat["problemEvaluations"]) == 5
    assert [turn["problemIndex"] for turn in repeat["turns"]] == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert [turn["request"]["readAloudGenerateNextProblem"] for turn in repeat["turns"]] == [False, True, False, True, False, True, False, True, False, False]
    for mode in result["modes"]:
        assert mode["silenceProbe"]["error"]["code"] == "SILENCE_DETECTED"
        assert mode["sessionEvaluation"]["status"] == "EVALUATED"
        assert mode["openingAudio"]["decodedDurationSeconds"] == 12
        assert [turn["scenario"] for turn in mode["turns"]] == SCENARIO_PLAN[mode["mode"]]
    assert "qualityPass" not in result


@pytest.mark.asyncio
async def test_cancelled_campaign_preserves_partial_stage_and_starts_no_followup(tmp_path: Path) -> None:
    class CancelledText(CampaignTextProvider):
        async def call_with_metadata(self, type_name, data, schema=None):
            raise asyncio.CancelledError()

    speech = CampaignSpeechProvider()
    with pytest.raises(asyncio.CancelledError):
        await run_speaking_campaign(CancelledText(), speech, tmp_path, CampaignSttProvider())
    artifact = json.loads((tmp_path / "speaking-campaign-result.json").read_text(encoding="utf-8"))
    assert artifact["status"] == "INTERRUPTED"
    assert artifact["error"]["type"] == "CancelledError"
    assert artifact["partial"] is True
    assert len(artifact["attempts"]) == 1
    assert artifact["attempts"][0]["status"] == "INTERRUPTED"
    assert speech.calls == []


@pytest.mark.asyncio
async def test_actual_short_audio_keeps_insufficient_evidence_instead_of_padding(tmp_path: Path) -> None:
    class ShortStt(CampaignSttProvider):
        async def transcribe(self, wav_bytes, *, language, phrase_hints=None):
            result = await super().transcribe(wav_bytes, language=language, phrase_hints=phrase_hints)
            return SttProviderResult(result.text, result.language, result.language_probability,
                                     [SttProviderSegment(0, 1.2, result.text, -0.01)], "fake", "fake")

    text, speech = CampaignTextProvider(), CampaignSpeechProvider()
    speech.audio = make_wav(1.2)
    result = await run_speaking_campaign(text, speech, tmp_path, ShortStt())
    for mode in result["modes"][1:]:
        assert mode["totalUserAudioSeconds"] == 6
        assert mode["sessionEvaluation"]["status"] == "INSUFFICIENT_EVIDENCE"
        assert "VALID_SPEECH_SECONDS" in mode["sessionEvaluation"]["eligibility"]["missingRequirements"]
    assert len(text.calls) == 23


@pytest.mark.asyncio
async def test_one_mode_failure_does_not_skip_other_predeclared_modes(tmp_path: Path) -> None:
    class FailedReadAloud(CampaignTextProvider):
        async def call_with_metadata(self, type_name, data, schema=None):
            request = json.loads(data.split("\n\n")[-1])
            if request["practiceMode"] == "READ_ALOUD":
                self.calls.append((type_name, request))
                raise RuntimeError("offline injected mode failure")
            return await super().call_with_metadata(type_name, data, schema)

    text = FailedReadAloud()
    result = await run_speaking_campaign(text, CampaignSpeechProvider(), tmp_path, CampaignSttProvider())
    assert [item["mode"] for item in result["modes"]] == ["READ_ALOUD", "GUIDED", "FREE"]
    assert [item["status"] for item in result["modes"]] == ["FAILED", "COMPLETED", "COMPLETED"]
    assert result["status"] == "PARTIAL"
    assert result["partial"] is True
    # Production's existing initial attempt + two stage retries remain unchanged;
    # the QA driver adds no new attempt after that mode finishes with failure.
    assert sum(request["practiceMode"] == "READ_ALOUD" for _, request in text.calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [CampaignBudgetExceeded, TimeoutError])
async def test_budget_or_deadline_stops_all_modes_even_if_service_maps_error(tmp_path: Path, error) -> None:
    class HaltedText(CampaignTextProvider):
        async def call_with_metadata(self, type_name, data, schema=None):
            raise error("offline stop")

    speech = CampaignSpeechProvider()
    with pytest.raises(error):
        await run_speaking_campaign(HaltedText(), speech, tmp_path, CampaignSttProvider())
    artifact = json.loads((tmp_path / "speaking-campaign-result.json").read_text(encoding="utf-8"))
    assert artifact["status"] == "FAILED"
    assert artifact["error"]["type"] == error.__name__
    assert len(artifact["modes"]) == 1
    assert len(artifact["attempts"]) == 1
    assert speech.calls == []


@pytest.mark.asyncio
async def test_read_aloud_only_is_a_distinct_preserved_campaign(tmp_path: Path) -> None:
    text, speech, stt = CampaignTextProvider(), CampaignSpeechProvider(), CampaignSttProvider()
    result = await run_speaking_campaign(text, speech, tmp_path, stt, modes=["READ_ALOUD"])
    assert result["status"] == "COMPLETED"
    assert [mode["mode"] for mode in result["modes"]] == ["READ_ALOUD"]
    assert list(result["scenarioPlan"]) == ["READ_ALOUD"]
    assert result["callEstimate"]["normalTotalStarts"] == len(result["attempts"]) == 36
    assert len(text.calls) == 11
    assert len(speech.calls) == 15
    assert stt.calls == 10
    artifact = (tmp_path / "speaking-campaign-result.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        await run_speaking_campaign(text, speech, tmp_path, stt, modes=["READ_ALOUD"])
    assert (tmp_path / "speaking-campaign-result.json").read_bytes() == artifact
    assert len(text.calls) == 11


@pytest.mark.asyncio
async def test_sixth_conversation_turn_is_predeclared_not_an_evidence_override(tmp_path: Path) -> None:
    text, speech, stt = CampaignTextProvider(), CampaignSpeechProvider(), CampaignSttProvider()
    result = await run_speaking_campaign(text, speech, tmp_path, stt,
                                         modes=["GUIDED"], sixth_conversation_turn=True)
    mode = result["modes"][0]
    assert result["sixthConversationTurnPredeclared"] is True
    assert result["scenarioPlan"]["GUIDED"][-1] == "PREDECLARED_FOLLOW_UP"
    assert mode["acceptedUserTurns"] == 6
    assert mode["totalUserAudioSeconds"] == 72
    assert mode["turns"][-1]["intendedText"].startswith("最後に確認させてください。")
    assert mode["sessionEvaluation"]["eligibility"]["requiredSpeechSeconds"] == 60
    # The optional last assistant may finish without audio, or a provider
    # generation stage may be consumed before TTS. Reserve both stage maxima.
    assert result["callEstimate"]["normalTotalStarts"] == 29
    assert len(result["attempts"]) <= 29
    assert len(text.calls) in {8, 9}
    assert len(speech.calls) in {13, 14}
    assert stt.calls == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("modes", [[], ["READ_ALOUD", "READ_ALOUD"], ["UNSUPPORTED"]])
async def test_invalid_mode_selection_never_starts_provider(tmp_path: Path, modes) -> None:
    text, speech = CampaignTextProvider(), CampaignSpeechProvider()
    with pytest.raises(ValueError, match="unique subset"):
        await run_speaking_campaign(text, speech, tmp_path, CampaignSttProvider(), modes=modes)
    assert text.calls == speech.calls == []
