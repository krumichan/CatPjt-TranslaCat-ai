from app.ai.model_policy import _TASK_POLICIES
from app.ai.prompt_registry import get_prompt_rule


def test_retired_chat_business_symbols_are_absent_from_production_source():
    # 준비: 인증 경계의 chat_auth는 유지하되 퇴역 업무 식별자는 소스 전체에서 검사한다.
    from pathlib import Path

    app_root = Path(__file__).resolve().parents[1] / "app"
    retired_symbols = (
        "AI_CHAT_REPLY",
        "CHAT_MESSAGE_TRANSLATION",
        "/api/v1/chat/",
        "translate_chat_message",
        "build_chat_ai_reply_prompt",
        "normalize_chat_translation_result",
    )

    # 실행: 삭제된 파일뿐 아니라 이름만 바꿔 남긴 정책도 찾는다.
    remaining = {
        str(path.relative_to(app_root)): symbol
        for path in app_root.rglob("*.py")
        for symbol in retired_symbols
        if symbol in path.read_text(encoding="utf-8")
    }

    # 검증: AI 운영 소스에는 Chat 업무 recipe가 남지 않는다.
    assert remaining == {}


def test_chat_business_tasks_and_routes_are_owned_by_chat_service():
    # 준비
    from unittest.mock import patch

    with patch("app.core.config_logger.setup_logging"):
        from app.main import app

    # 실행
    paths = {route.path for route in app.routes if hasattr(route, "path")}

    # 검증 — 일반 실행 route는 남고 Python Chat 업무 recipe는 등록되지 않는다.
    assert "/internal/v1/model/execute" in paths
    assert not any(path.startswith("/api/v1/chat/") for path in paths)
    for task in ("AI_CHAT_REPLY", "CHAT_MESSAGE_TRANSLATION"):
        assert get_prompt_rule(task) is None
        assert task not in _TASK_POLICIES


def test_reading_vocabulary_pipeline_prompts_are_owned_by_ktor():
    # 준비: 실제 전환이 끝난 Practice task는 Python prompt registry에서 제거한다.
    for type_name in [
        "LANGUAGE_LEARNING_READING_VOCABULARY_GENERATION",
        "LANGUAGE_LEARNING_READING_PASSAGE_GENERATION",
        "LANGUAGE_LEARNING_READING_VOCABULARY_USAGE_PRESCREEN",
        "LANGUAGE_LEARNING_READING_VOCABULARY_VERIFICATION",
        "LANGUAGE_LEARNING_READING_VOCABULARY_ORIGIN_EXPLANATION",
        "LANGUAGE_LEARNING_READING_VOCABULARY_ORIGIN_EXPLANATION_FALLBACK",
    ]:
        # 실행 및 검증
        assert get_prompt_rule(type_name) is None


def test_writing_pipeline_prompts_and_model_policy_are_owned_by_ktor():
    # 준비: 제거된 Python 업무 이름이 공통 실행기의 정책으로 남지 않아야 한다.
    tasks = [
        "LANGUAGE_LEARNING_DAILY_WRITING_GENERATION",
        "LANGUAGE_LEARNING_WRITING_EVALUATION",
        "LANGUAGE_LEARNING_WRITING_SOURCE_LOCALIZATION",
        "LANGUAGE_LEARNING_WRITING_DIFFICULTY_PRESCREEN",
        "LANGUAGE_LEARNING_WRITING_TASK_VERIFICATION",
        "LANGUAGE_LEARNING_WRITING_DIFFICULTY_VERIFICATION",
        "LANGUAGE_LEARNING_WRITING_NOTE_VERIFICATION",
        "LANGUAGE_LEARNING_WRITING_NOTE_LOCALIZATION",
        "LANGUAGE_LEARNING_LEVEL_TEST_QUESTION",
    ]

    # 실행 및 검증: taskName은 진단 용도로만 전달되며 업무 분기를 되살리지 않는다.
    for task in tasks:
        assert get_prompt_rule(task) is None
        assert task not in _TASK_POLICIES


def test_retired_writing_http_routes_are_not_registered():
    # 준비: 합성 서버의 차단 middleware 없이 실제 앱 등록 경로를 확인한다.
    from unittest.mock import patch

    with patch("app.core.config_logger.setup_logging"):
        from app.main import app

    # 실행
    paths = {route.path for route in app.routes if hasattr(route, "path")}

    # 검증: 범용 실행 API는 남고 이전 Writing 업무 endpoint는 제거되어야 한다.
    assert "/internal/v1/model/execute" in paths
    assert not any(path.startswith("/api/v1/language-learning/writing/") for path in paths)
    assert "/api/v1/language-learning/level-test/evaluate/text" not in paths


def test_level_test_pipeline_is_owned_by_ktor():
    # 준비: 새 범용 실행기는 호출자의 prompt/schema만 실행하며 퇴역 업무 분기를 갖지 않는다.
    from unittest.mock import patch

    with patch("app.core.config_logger.setup_logging"):
        from app.main import app
    tasks = [
        "GENERATION",
        "VOCAB_CONTEXT_DESIGN",
        "VOCAB_CONTEXT_REPAIR",
        "CHOICE_VERIFICATION",
        "TASK_SUFFICIENCY_VERIFICATION",
        "SPEAKING_EVALUATION",
    ]

    # 실행
    paths = {route.path for route in app.routes if hasattr(route, "path")}

    # 검증: 생성·음성 평가를 포함한 기존 업무 endpoint가 실제 앱에서 제거됐다.
    assert not any(path.startswith("/api/v1/language-learning/level-test/") for path in paths)
    assert "/internal/v1/model/execute" in paths
    for suffix in tasks:
        name = f"LANGUAGE_LEARNING_LEVEL_TEST_{suffix}"
        assert get_prompt_rule(name) is None
        assert name not in _TASK_POLICIES


def test_listening_pipeline_is_owned_by_ktor():
    # 준비: 앱 등록과 공통 실행 정책 양쪽에 기존 Listening 업무가 남지 않아야 한다.
    from unittest.mock import patch

    with patch("app.core.config_logger.setup_logging"):
        from app.main import app
    tasks = ["GENERATION", "INTERPRETATION", "SUMMARY_EVALUATION", "EXPLANATION"]

    # 실행
    paths = {route.path for route in app.routes if hasattr(route, "path")}

    # 검증: LL 소유 업무 endpoint는 제거하고 범용 기술 실행 API를 유지한다.
    assert not any(path.startswith("/api/v1/language-learning/listening/") for path in paths)
    assert "/internal/v1/model/execute" in paths
    assert "/internal/v1/speech/synthesize" in paths
    assert "/internal/v1/speech/transcribe" in paths
    for suffix in tasks:
        name = f"LANGUAGE_LEARNING_LISTENING_{suffix}"
        assert get_prompt_rule(name) is None
        assert name not in _TASK_POLICIES


def test_speaking_pipeline_is_owned_by_ktor():
    # 준비: 실제 앱 등록과 공통 업무 정책을 확인한다.
    from unittest.mock import patch

    with patch("app.core.config_logger.setup_logging"):
        from app.main import app
    tasks = ["CONVERSATION", "ASSISTANCE", "EVALUATION", "SESSION_COACHING"]

    # 실행
    paths = {route.path for route in app.routes if hasattr(route, "path")}

    # 검증: Speaking 업무는 사라지고 범용 모델·음성 경계는 유지한다.
    assert not any(path.startswith("/api/v1/language-learning/speaking/") for path in paths)
    assert "/internal/v1/model/execute" in paths
    assert "/internal/v1/speech/transcribe" in paths
    for suffix in tasks:
        name = f"LANGUAGE_LEARNING_SPEAKING_{suffix}"
        assert get_prompt_rule(name) is None
        assert name not in _TASK_POLICIES
