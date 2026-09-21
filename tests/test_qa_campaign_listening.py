import asyncio
import io
import json
from pathlib import Path
import wave

import pytest

from app.ai.ports import SpeechSynthesisResult, StructuredGenerationResult
from scripts.qa_campaign_listening import run_listening_campaign
from tests.test_language_learning_listening import generation_payload, interpretation_payload, summary_payload


_TEXTS = [
    "雨が降りそうなので、出かける前に窓を閉めてください。",
    "新しい図書館の利用時間は、平日と休日で異なります。",
    "駅の券売機が故障しているため、切符は窓口で購入できます。",
    "料理教室に参加する人は、エプロンを持ってきてください。",
    "荷物を受け取れなかった場合は、配達日時を指定し直せます。",
    "市役所の駐車場は工事中ですので、公共交通機関をご利用ください。",
    "明日の研修では、新しい予約システムの使い方を説明します。",
    "商品に傷があれば、レシートと一緒に店へお持ちください。",
    "運動会は暑さを避けるため、午前中に全ての競技を行います。",
    "自転車を借りるには、受付で身分証明書を提示してください。",
    "病院の予約を変更する際は、前日までに電話でご連絡ください。",
    "この映画には字幕が付いていますが、吹き替え音声はありません。",
    "ごみの収集日は祝日でも変わりません。朝八時までに出しましょう。",
    "ホテルの朝食会場は二階です。食券を忘れずにお持ちください。",
    "交流会の参加費には飲み物代が含まれますが、食事代は別料金です。",
]


class TextProvider:
    def __init__(self):
        self.calls = []
        self.generation_count = 0

    async def call_with_metadata(self, type_name, data, schema=None):
        self.calls.append((type_name, data, schema))
        payload = json.loads(data.split("\n\n")[-1])
        if type_name.endswith("GENERATION"):
            index = self.generation_count
            self.generation_count += 1
            item = generation_payload()["items"][0]
            item["sourceText"] = _TEXTS[index]
            item["diversityMetadata"]["taskArchetype"] = f"SYNTHETIC_{index}"
            mode = payload["setContext"]["learningMode"]
            if mode == "COMPREHENSION":
                item.update({
                    "question": "説明の内容に合っているものはどれですか。",
                    "options": [{"key": key, "text": f"選択肢{key}"} for key in "ABCD"],
                    "correctOptionKey": "B", "comprehensionFocus": "DETAIL",
                })
            if mode == "SUMMARY":
                item["summaryKeyPoints"] = ["重要な案内", "必要な手続き"]
            return StructuredGenerationResult(data={"items": [item]}, input_tokens=10, output_tokens=20)
        if type_name.endswith("INTERPRETATION"):
            response = interpretation_payload()
            response["deliveredMeaningUnits"] = payload["keyMeaningUnits"]
            response["omittedMeaningUnits"] = []
            return StructuredGenerationResult(data=response, input_tokens=10, output_tokens=20)
        assert type_name.endswith("SUMMARY_EVALUATION")
        return StructuredGenerationResult(data=summary_payload(), input_tokens=10, output_tokens=20)


class SpeechProvider:
    def __init__(self, valid=True):
        self.calls = []
        self.valid = valid

    async def synthesize_speech(self, **kwargs):
        self.calls.append(kwargs)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(24000)
            writer.writeframes(b"\x01\x10" * (24000 * 10))
        return SpeechSynthesisResult(
            audio_bytes=buffer.getvalue() if self.valid else b"invalid-audio",
            content_type="audio/wav", provider="fake", model="fake-tts", duration_seconds=10,
        )


@pytest.mark.asyncio
async def test_campaign_runs_real_services_for_all_modes_and_preserves_progressive_context(tmp_path):
    text, speech = TextProvider(), SpeechProvider()
    result = await run_listening_campaign(text, speech, tmp_path / "success")

    assert result["status"] == "COMPLETED"
    assert result["partial"] is False
    assert len(text.calls) == 45
    assert len(speech.calls) == 15
    assert all(call["speed"] == "NORMAL" for call in speech.calls)
    for mode_index, mode in enumerate(result["modes"]):
        assert mode["status"] == "COMPLETED"
        assert len(mode["items"]) == 5
        for order, item in enumerate(mode["items"], 1):
            assert item["status"] == "COMPLETED"
            generation = item["stages"][0]
            request = generation["request"]
            assert request["setContext"]["itemCount"] == 1
            assert len(request["diversityContext"]["currentSession"]) == order - 1
            assert len(request["diversityContext"]["sameFeatureRecent"]) == mode_index * 5
            assert generation["response"]["items"][0]["itemIndex"] == 1
            audio = item["stages"][1]
            assert audio["audioCheck"]["status"] == "PASSED"
            assert Path(audio["audioPath"]).is_relative_to(tmp_path)
            if mode["mode"] == "DICTATION":
                correct = next(stage for stage in item["stages"]
                               if stage["stage"] == "DICTATION" and stage["variant"] == "correct")
                assert correct["response"]["overall"]["score"] == 100
            if mode["mode"] == "COMPREHENSION":
                assert not mode["partialAnswerApplicable"]
                assert [stage["response"]["overall"]["score"] for stage in item["stages"][2:]] == [100, 0]
    persisted = json.loads((tmp_path / "success" / "listening-campaign.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "COMPLETED"
    with pytest.raises(FileExistsError):
        await run_listening_campaign(text, speech, tmp_path / "success")
    assert len(text.calls) == 45


@pytest.mark.asyncio
async def test_audio_decode_failure_keeps_generation_and_stops_followup_calls(tmp_path):
    text, speech = TextProvider(), SpeechProvider(valid=False)
    result = await run_listening_campaign(text, speech, tmp_path / "decode-failure")

    assert result["status"] == "FAILED"
    assert result["partial"] is True
    assert len(text.calls) == len(speech.calls) == 1
    assert result["modes"][0]["items"][0]["stages"][0]["response"]["items"]
    persisted = json.loads((tmp_path / "decode-failure" / "listening-campaign.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "FAILED"


@pytest.mark.asyncio
async def test_cancel_is_checkpointed_and_propagated_without_hidden_continuation(tmp_path):
    class CancelledProvider(TextProvider):
        async def call_with_metadata(self, type_name, data, schema=None):
            self.calls.append((type_name, data, schema))
            raise asyncio.CancelledError()

    text, speech = CancelledProvider(), SpeechProvider()
    with pytest.raises(asyncio.CancelledError):
        await run_listening_campaign(text, speech, tmp_path / "cancelled")
    assert len(text.calls) == 1
    assert speech.calls == []
    result = json.loads((tmp_path / "cancelled" / "listening-campaign.json").read_text(encoding="utf-8"))
    assert result["status"] == "INTERRUPTED"
    assert result["error"]["type"] == "CancelledError"
    assert result["modes"][0]["items"][0]["stages"][0]["status"] == "INTERRUPTED"
