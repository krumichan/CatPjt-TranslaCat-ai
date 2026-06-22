# Backward-compatible prompt exports.
# New code should import from app.features.*.prompts or app.ai.prompt_registry.

from app.ai.prompt_registry import PROMPT_MAP
from app.features.chat_translation.prompts import build_chat_translation_prompt
from app.features.receipt.prompts import (
    build_receipt_text_analysis_prompt,
    build_receipt_vision_prompt,
)
