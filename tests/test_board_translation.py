import pytest

from app.ai.prompt_registry import get_prompt_rule
from app.features.translation.service import TranslationService


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("locale", "language"),
    [("ko", "Korean"), ("ja", "Japanese"), ("en", "English")],
)
async def test_board_title_and_body_follow_target_language(locale, language):
    # 준비: 운영 서비스는 호출하지 않고 프롬프트와 요청 형태를 확인한다.
    class Provider:
        async def call(self, *, type_name, data, schema):
            assert language in (get_prompt_rule(type_name) or "")
            assert '"タイトル"' in data
            assert schema == {"type": "ARRAY", "items": {"type": "STRING"}}
            return ["Translated title", "Translated body"]

    # 실행
    result = await TranslationService(Provider()).translate_batch(
        ["タイトル", "本文"], f"BOARD_{locale}"
    )

    # 검증
    assert result == ["Translated title", "Translated body"]
