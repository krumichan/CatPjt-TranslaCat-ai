from __future__ import annotations

import ast
from pathlib import Path


def test_python_runtime_has_no_language_learning_business_sources_or_imports():
    # 준비: 실행 서비스 소스만 검사하며 원본 보존 자료와 합성 Provider는 제외한다.
    app_root = Path(__file__).parents[1] / "app"
    forbidden_prefixes = (
        "app.features.language_learning",
        "app.schemas.language_learning",
        "app.api.v1.language_learning",
    )
    business_sources = [
        path.relative_to(app_root).as_posix()
        for path in (app_root / "features" / "language_learning").rglob("*.py")
        if path.name != "__init__.py"
    ]

    # 실행: 직접 import와 상위 패키지에서 가져오는 import를 모두 검사한다.
    business_imports: list[str] = []
    for path in app_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
                names.extend(f"{node.module}.{alias.name}" for alias in node.names)
            else:
                continue
            if any(name.startswith(forbidden_prefixes) for name in names):
                business_imports.append(f"{path.relative_to(app_root)}:{node.lineno}")

    # 검증: 모델 실행 경계 안에 프롬프트·판정·진행 업무의 역참조가 없어야 한다.
    assert not business_sources, business_sources
    assert not business_imports, business_imports
