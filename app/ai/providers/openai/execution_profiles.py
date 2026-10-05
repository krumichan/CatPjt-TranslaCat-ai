"""서버가 허용한 범용 실행 옵션. 업무용 번역 정책이나 공통 tier는 변경하지 않는다."""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ExecutionProfile:
    model: str
    effort: str
    version: str = "2026-10-04.1"
    service_tier: str = "default"
    verbosity: str = "low"
    max_output_tokens: int = 32768


EXECUTION_PROFILES = MappingProxyType(
    {
        "luna-5.6-none-v1": ExecutionProfile("gpt-5.6-luna", "none"),
        "luna-5.6-low-v1": ExecutionProfile("gpt-5.6-luna", "low"),
        "luna-6-none-v1": ExecutionProfile("gpt-6-luna", "none"),
        "luna-6-low-v1": ExecutionProfile("gpt-6-luna", "low"),
        "luna-6-medium-v1": ExecutionProfile("gpt-6-luna", "medium"),
        "sol-6.1-low-v1": ExecutionProfile("gpt-6.1-sol", "low"),
    }
)
