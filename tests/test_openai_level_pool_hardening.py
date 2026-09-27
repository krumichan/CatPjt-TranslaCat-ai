import asyncio
import time

from app.ai.provider_pool import AiProviderPool
from app.ai.providers.openai.schema import normalize_openai_response_schema


def test_openai_schema_turns_small_patterns_into_enums():
    normalized = normalize_openai_response_schema(
        {
            "type": "object",
            "properties": {
                "generationPlanId": {
                    "anyOf": [
                        {"type": "string", "pattern": "^[AB]$"},
                        {"type": "null"},
                    ]
                }
            },
        }
    )

    branch = normalized["properties"]["generationPlanId"]["anyOf"][0]
    assert branch == {"type": "string", "enum": ["A", "B"]}


def test_pool_waits_for_short_cooldown_instead_of_immediate_503():
    class Provider:
        ready = True

        async def call(self, type_name: str, data: str, schema=None):
            return "ok"

    async def run():
        pool = AiProviderPool()
        pool.dock(name="openai", provider=Provider(), max_concurrency=1, cooldown_seconds=0.01)
        pool._slots[0].activate_cooldown(0.01)
        started = time.monotonic()
        assert await pool.call(type_name="TEST", data="hello") == "ok"
        assert time.monotonic() - started >= 0.005

    asyncio.run(run())
