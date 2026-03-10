from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI

def set_custom_openapi(app: FastAPI):
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version="1.0.0",
        description="Project Cat AI Server API 명세서 - 서버 간 통신 보안 적용됨",
        routes=app.routes,
    )

    # 보안 스키마 정의 (Authorize 버튼용)
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-KEY",
            "description": "서버 간 통신용 API 키를 입력하세요."
        }
    }
    
    # 모든 API에 보안 설정 적용
    openapi_schema["security"] = [{"ApiKeyAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema