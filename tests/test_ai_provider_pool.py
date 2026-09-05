import asyncio

from app.ai.provider_pool import AiProviderPool


class _TransientError(RuntimeError):
    status_code = 429


class _FakeProvider:
    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.calls = 0
        self.ready = True

    async def call(self, type_name: str, data: str, schema=None):
        self.calls += 1
        if self.fail:
            raise _TransientError("busy")
        return {"provider": self.name}

    def model_name_for(self, type_name: str) -> str:
        return f"{self.name}-model"



def test_transient_failure_falls_through_to_next_docked_provider():
    async def run():
        first = _FakeProvider("first", fail=True)
        second = _FakeProvider("second")
        pool = AiProviderPool()
        pool.dock(name="first", provider=first, max_concurrency=1, cooldown_seconds=10)
        pool.dock(name="second", provider=second, max_concurrency=1, cooldown_seconds=10)

        result = await pool.call(type_name="TEST", data="hello")

        assert result == {"provider": "second"}
        assert first.calls == 1
        assert second.calls == 1

    asyncio.run(run())


def test_non_transient_failure_is_not_hidden_by_fallback():
    class _BadRequestProvider(_FakeProvider):
        async def call(self, type_name: str, data: str, schema=None):
            self.calls += 1
            raise ValueError("bad schema")

    async def run():
        first = _BadRequestProvider("first")
        second = _FakeProvider("second")
        pool = AiProviderPool()
        pool.dock(name="first", provider=first, max_concurrency=1, cooldown_seconds=10)
        pool.dock(name="second", provider=second, max_concurrency=1, cooldown_seconds=10)

        try:
            await pool.call(type_name="TEST", data="hello")
        except ValueError as exc:
            assert str(exc) == "bad schema"
        else:
            raise AssertionError("ValueError was expected")
        assert second.calls == 0

    asyncio.run(run())
