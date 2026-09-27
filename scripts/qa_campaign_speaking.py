"""외부 BE Speaking 검사용 고정 합성 발화 자료. Python 학습 업무를 실행하지 않는다."""


def _learner_text(scenario: str, mode: str, script: str, facts: list[str]) -> str:
    if scenario == "PREDECLARED_FOLLOW_UP":
        detail = "。".join(facts) + "。" if mode == "GUIDED" and facts else ""
        return (
            "最後に確認させてください。"
            + detail
            + "この内容で進める前に、担当者と関係者の都合をもう一度確認します。"
            "変更が必要な場合は、理由と代わりの案を早めに共有したいです。"
        )
    if scenario == "UNRELATED":
        return (
            "今日は森に住む動物について話します。私は週末に動物園へ行きました。"
            "予定の相談とは関係のない話です。"
        )
    if mode == "READ_ALOUD":
        return script if scenario == "RIGHT_INTENT" else script[: max(1, len(script) // 2)]
    if scenario == "PARTIAL":
        return "予定について相談したいです。詳しい理由や代わりの日程はまだ決めていません。"
    if mode == "GUIDED" and facts:
        return (
            "予定について相談させてください。"
            + "。".join(facts)
            + "。この条件で調整したいです。皆さんの都合も伺ってから決めたいと思います。"
        )
    return (
        "仕事の予定は、担当者の都合と準備に必要な時間を確認してから決めたいです。"
        "急に変更が必要になったら、理由を説明して代わりの日程を提案します。"
        "一方的に決めず、相手の意見も聞くことが大切だと思います。"
    )
