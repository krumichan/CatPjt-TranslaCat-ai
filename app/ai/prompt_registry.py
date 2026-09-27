from app.features.receipt.prompts import RECEIPT_ANALYSIS_PROMPT
from app.features.translation.prompts import TRANSLATION_PROMPT_MAP
from app.features.voice_translation.prompts import VOICE_TRANSLATION_SYSTEM_PROMPT

PROMPT_MAP = {
    **TRANSLATION_PROMPT_MAP,
    "RECEIPT_ANALYSIS": RECEIPT_ANALYSIS_PROMPT,
    "VOICE_TRANSLATION": VOICE_TRANSLATION_SYSTEM_PROMPT,
}


def get_prompt_rule(type_name: str) -> str | None:
    return PROMPT_MAP.get(type_name)
