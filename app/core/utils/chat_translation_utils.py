import json


def normalize_chat_translation_result(result: str) -> str:
    """채팅 번역 결과를 순수 문자열로 정리한다."""
    cleaned = result.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    cleaned = cleaned.strip().strip("\"")

    try:
        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            for key in ["translated_text", "translatedText", "text"]:
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        if isinstance(parsed, str) and parsed.strip():
            return parsed.strip()

    except json.JSONDecodeError:
        pass

    return cleaned