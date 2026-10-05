import pytest

from scripts.local_smoke_test_provider import execute_local_smoke_fixture
from scripts.synthetic_paths import synthetic_runtime_directory


def _chat_translation_request(source):
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are a chat message translator.\n\n"
                    "Target language code: ja\nSource language code: auto\n\n"
                    f"<message>\n{source}\n</message>"
                ),
            }
        ],
        "schema_name_value": None,
    }


@pytest.mark.parametrize(
    "source",
    [
        "Fresh-first-1791028994156",
        "Fresh-second-1791028994156",
        "Fresh-unread-1791028994156",
        "Local synthetic reply",
    ],
)
def test_chat_translation_accepts_exact_owned_fixture_sources(monkeypatch, source):
    # 준비: 실제 CHAT 번역의 단일 user message와 원문 태그 계약을 사용한다.
    monkeypatch.setenv("TRANSLACAT_TEST_MODEL_EXECUTION", "1")

    # 실행
    result = execute_local_smoke_fixture(_chat_translation_request(source))

    # 검증
    assert result is not None
    assert result.data == "Local synthetic translation"
    assert result.provider == "test-provider"


@pytest.mark.parametrize(
    "source",
    [
        "ordinary user input",
        "Fresh-first-",
        "Fresh-first-not-a-timestamp",
        "Fresh-other-1791028994156",
        "Fresh-first-1791028994156 extra text",
        "prefix Fresh-first-1791028994156",
        "Fresh-first-1791028994156\nordinary user input",
        "Local synthetic reply from a real user",
    ],
)
def test_chat_translation_does_not_accept_unowned_or_partial_sources(monkeypatch, source):
    # 준비
    monkeypatch.setenv("TRANSLACAT_TEST_MODEL_EXECUTION", "1")

    # 실행 및 검증: 허용된 원문과 완전히 일치하지 않으면 기존 unknown 실패 경로로 보낸다.
    assert execute_local_smoke_fixture(_chat_translation_request(source)) is None


def test_owned_translation_requires_test_mode_and_does_not_enable_structured_reply(monkeypatch):
    # 준비
    request = _chat_translation_request("Local synthetic reply")
    monkeypatch.delenv("TRANSLACAT_TEST_MODEL_EXECUTION", raising=False)

    # 실행 및 검증
    with pytest.raises(RuntimeError):
        execute_local_smoke_fixture(request)
    monkeypatch.setenv("TRANSLACAT_TEST_MODEL_EXECUTION", "1")
    assert execute_local_smoke_fixture(request | {"schema_name_value": "AI_CHAT_REPLY"}) is None


@pytest.mark.parametrize("schema", [None, "AI_CHAT_REPLY"])
def test_explicit_smoke_returns_only_synthetic_output(monkeypatch, schema):
    # 준비
    monkeypatch.setenv("TRANSLACAT_TEST_MODEL_EXECUTION", "1")
    request = {
        "messages": [{"role": "user", "content": "TRANSLACAT_LOCAL_SMOKE_20261003"}],
        "schema_name_value": schema,
    }

    # 실행
    result = execute_local_smoke_fixture(request)

    # 검증
    assert result is not None
    assert result.provider == "test-provider"
    if schema:
        assert result.data["shouldRespond"] is True
    else:
        assert result.data == "Local synthetic translation"


def test_smoke_requires_explicit_marker_and_rejects_unsupported_schema(monkeypatch):
    # 준비
    monkeypatch.delenv("TRANSLACAT_TEST_MODEL_EXECUTION", raising=False)
    request = {"messages": [{"role": "user", "content": "TRANSLACAT_LOCAL_SMOKE"}]}

    # 실행 및 검증: 활성화되지 않은 실행이나 일반 입력에 정상 결과를 만들어 주지 않는다.
    with pytest.raises(RuntimeError):
        execute_local_smoke_fixture(request)
    assert execute_local_smoke_fixture({"messages": [{"content": "ordinary user input"}]}) is None
    monkeypatch.setenv("TRANSLACAT_TEST_MODEL_EXECUTION", "1")
    with pytest.raises(ValueError):
        execute_local_smoke_fixture(request | {"schema_name_value": "unknown"})


def test_synthetic_artifacts_use_selected_absolute_directory(monkeypatch, tmp_path):
    # 준비
    monkeypatch.setenv("TRANSLACAT_TEST_RUNTIME_DIRECTORY", str(tmp_path))

    # 실행 및 검증
    assert synthetic_runtime_directory() == tmp_path
    monkeypatch.setenv("TRANSLACAT_TEST_RUNTIME_DIRECTORY", "relative-path")
    with pytest.raises(ValueError):
        synthetic_runtime_directory()
