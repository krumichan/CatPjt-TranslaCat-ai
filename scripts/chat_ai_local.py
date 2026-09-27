import argparse
import io
import json
import os
import secrets
import sys
from pathlib import Path

from dotenv import dotenv_values
from dotenv.parser import parse_stream


class LocalConfigurationError(Exception):
    pass


def _read_configuration(path: Path) -> dict[str, str | None]:
    # 파싱 오류에 원문을 포함하지 않으며 중복 이름의 조용한 마지막 값 우선을 막는다.
    raw = path.read_bytes()
    if len(raw) > 1024 * 1024 or raw.startswith(b"\xef\xbb\xbf"):
        raise LocalConfigurationError(
            "AI environment file must be UTF-8 without BOM, at most 1 MiB"
        )
    text = raw.decode("utf-8", errors="strict")
    seen: set[str] = set()
    for binding in parse_stream(io.StringIO(text)):
        if binding.error:
            raise LocalConfigurationError("AI environment file contains an invalid setting")
        if binding.key is None:
            continue
        name = binding.key.upper()
        if name in seen:
            raise LocalConfigurationError("AI environment file contains a duplicate setting")
        seen.add(name)
    return {name.upper(): value for name, value in dotenv_values(stream=io.StringIO(text)).items()}


def prepare_environment(
    environment_file: Path,
    api_key_file: Path,
    environment: str,
    disable_warmup: bool,
) -> dict[str, object]:
    # 기존 발급자/Provider 설정은 같은 파일에서 재사용하며 Development 방향키만 연결한다.
    if environment != "Development":
        raise LocalConfigurationError("This launcher accepts Development only")
    for path in (environment_file, api_key_file):
        if not path.is_absolute() or not path.is_file():
            raise LocalConfigurationError("AI local launch requires existing absolute file paths")
    supplied = _read_configuration(environment_file)
    raw_key = api_key_file.read_bytes()
    if not 32 <= len(raw_key) <= 4096 or any(
        value <= 32 or value >= 127 or value in (34, 39) for value in raw_key
    ):
        raise LocalConfigurationError(
            "CHAT API key must be 32..4096 ASCII bytes without whitespace or BOM"
        )
    key = raw_key.decode("ascii")
    current = {name.upper(): value for name, value in os.environ.items()}
    expected = {
        "CHAT_AUTH_MODE": "dedicated",
        "CHAT_AUTH_ENVIRONMENT": environment,
        "CHAT_SERVER_API_KEY": key,
        "AI_SETTINGS_ENV_FILE": str(environment_file),
    }

    # 양측에 이미 다른 값이 있으면 자동 회전하지 않는다. 미지원 _FILE도 조용히 무시하지 않는다.
    for source in (supplied, current):
        for name, value in expected.items():
            existing = source.get(name)
            if existing and not secrets.compare_digest(existing.encode(), value.encode()):
                raise LocalConfigurationError(
                    f"AI setting conflicts with the selected source: {name}"
                )
        if any(
            name.endswith("_FILE") and name != "AI_SETTINGS_ENV_FILE"
            for name in source
            if name.startswith(("CHAT_", "SERVER_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"))
        ):
            raise LocalConfigurationError(
                "AI does not support the supplied secret _FILE environment name; use -ApiKeyFile"
            )
    for name in ("SERVER_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        if supplied.get(name) and current.get(name) and supplied[name] != current[name]:
            raise LocalConfigurationError(
                f"AI secret has conflicting environment and file sources: {name}"
            )
    for source in (supplied, current):
        global_key = source.get("SERVER_API_KEY")
        if global_key and secrets.compare_digest(global_key.encode(), raw_key):
            raise LocalConfigurationError("CHAT API key must be independent from SERVER_API_KEY")

    # 이 process에만 적용한다. -DisableWarmup은 비용/모델 다운로드 없는 기동을 위한 명시 선택이다.
    os.environ.update(expected)
    if disable_warmup:
        os.environ["AI_VOICE_ENABLED"] = "false"
        os.environ["OCR_WARM_UP"] = "false"
    return {
        "environment": environment,
        "chatAuthMode": "dedicated",
        "environmentFile": str(environment_file),
        "chatApiKey": "APPLIED",
        "warmupDisabled": disable_warmup,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start the actual AI service with a dedicated local Chat credential"
    )
    parser.add_argument("--environment", choices=["Development"], required=True)
    parser.add_argument("--environment-file", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--disable-warmup", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    try:
        if not 1024 <= args.port <= 65535:
            raise LocalConfigurationError("AI local port must be 1024..65535")
        result = prepare_environment(
            args.environment_file, args.api_key_file, args.environment, args.disable_warmup
        )
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from app.core.chat_auth import validate_chat_auth_configuration
        from app.core.config import settings

        validate_chat_auth_configuration(settings)
        result["pythonSettings"] = "VERIFIED"
        result["globalCredential"] = "REUSED" if settings.SERVER_API_KEY else "NEEDS_INPUT"
        result["providerCredential"] = (
            "REUSED"
            if (
                settings.OPENAI_API_KEY
                if settings.AI_TEXT_PROVIDER == "openai"
                else settings.GOOGLE_API_KEY
            )
            else "NEEDS_INPUT"
        )
        print(json.dumps(result), flush=True)
    except LocalConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2
    except Exception:
        # Pydantic의 원래 ValidationError에는 입력 값이 포함될 수 있어 그대로 출력하지 않는다.
        print(
            "AI local configuration failed validation; secret values were suppressed",
            file=sys.stderr,
        )
        return 2

    if not args.validate_only:
        import uvicorn

        uvicorn.run("app.main:app", host="127.0.0.1", port=args.port, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
