from __future__ import annotations


_TASK_SHELLS = {
    "ko": {
        "MEANING": (
            '이 문맥에서 "{target_expression}"이 나타내는 의미로 가장 적절한 것은 '
            "무엇입니까?"
        ),
        "SYNONYM": (
            '이 문맥에서 "{target_expression}"의 의미를 가장 잘 유지하는 표현은 '
            "무엇입니까?"
        ),
        "ANTONYM": (
            '이 문맥에서 "{target_expression}"과 같은 관점에서 반대 의미를 나타내는 '
            "표현은 무엇입니까?"
        ),
        "DISTINCTION": (
            '이 문맥에서 "{target_expression}"과 의미와 쓰임을 구별할 때 가장 적절한 '
            "표현은 무엇입니까?"
        ),
    },
    "ja": {
        "MEANING": (
            "この文脈で「{target_expression}」が表す意味として、最も適切なものは"
            "どれですか。"
        ),
        "SYNONYM": (
            "この文脈で「{target_expression}」の意味を最もよく保つ表現はどれですか。"
        ),
        "ANTONYM": (
            "この文脈で「{target_expression}」と同じ観点から、反対の意味を表すものは"
            "どれですか。"
        ),
        "DISTINCTION": (
            "この文脈で「{target_expression}」と意味・使われ方を区別するとき、最も"
            "適切なものはどれですか。"
        ),
    },
    "en": {
        "MEANING": (
            'In this context, which option best expresses the meaning of "{target_expression}"?'
        ),
        "SYNONYM": (
            'In this context, which expression best preserves the meaning of '
            '"{target_expression}"?'
        ),
        "ANTONYM": (
            'In this context, which expression has the opposite meaning to '
            '"{target_expression}" along the same dimension?'
        ),
        "DISTINCTION": (
            'In this context, which option best distinguishes its meaning and use from '
            '"{target_expression}"?'
        ),
    },
}


def render_meaning_relation_prompt(
    *,
    learning_language: str,
    skill_tag: str,
    target_expression: str,
    meaning_context: str,
) -> str:
    """Combine model-authored semantic content with an application-owned task shell."""

    context = meaning_context.strip()
    target = target_expression.strip()
    if not context:
        raise ValueError("MEANING_RELATION B3+ requires non-empty meaningContext")
    if not target:
        raise ValueError("MEANING_RELATION task shell requires targetExpression")

    language = learning_language.strip().lower().split("-", 1)[0].split("_", 1)[0]
    try:
        shell = _TASK_SHELLS[language][skill_tag]
    except KeyError as exc:
        raise ValueError(
            "MEANING_RELATION server task shell does not support learningLanguage/skillTag"
        ) from exc
    return f"{context}\n\n{shell.format(target_expression=target)}"
