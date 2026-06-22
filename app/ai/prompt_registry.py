from app.features.chat_translation.prompts import CHAT_MESSAGE_TRANSLATION_PROMPT
from app.features.receipt.prompts import RECEIPT_ANALYSIS_PROMPT
from app.features.translation.prompts import TRANSLATION_PROMPT_MAP

PROMPT_MAP = {
    **TRANSLATION_PROMPT_MAP,
    "RECEIPT_ANALYSIS": RECEIPT_ANALYSIS_PROMPT,
    "CHAT_MESSAGE_TRANSLATION": CHAT_MESSAGE_TRANSLATION_PROMPT,
}


def get_prompt_rule(type_name: str) -> str | None:
    return PROMPT_MAP.get(type_name)
