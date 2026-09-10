"""Generate paired BE/AI regression fixtures using the actual AI service and an in-memory provider.

Run from the AI root: python -m scripts.export_speaking_release_contract [optional BE test resource directory]
No production provider or credentials are used. Review fixture changes before committing them.
"""
import asyncio
import json
import sys
from pathlib import Path
from tests.test_speaking_release_contract import contract_cases, make_case


async def main():
    destinations = [Path("tests/fixtures/speaking-release")]
    if len(sys.argv) > 1:
        destinations.append(Path(sys.argv[1]))
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
    for name, request, payload in contract_cases():
        response, _ = await make_case(request, payload)
        body = response.model_dump(by_alias=True, mode="json")
        if body["usage"].get("evaluation"):
            body["usage"]["evaluation"]["latencyMs"] = 0
        data = json.dumps({"request": request.model_dump(by_alias=True, mode="json"), "response": body},
                          ensure_ascii=False, indent=2) + "\n"
        for destination in destinations:
            (destination / f"{name}.json").write_text(data, encoding="utf-8")
        print(name, body["status"], len(body["metrics"]))


if __name__ == "__main__":
    asyncio.run(main())
