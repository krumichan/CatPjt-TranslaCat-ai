"""카메라 내부 API가 원본과 무관하게 반환하는 한정된 처리 오류."""


class CameraServiceError(Exception):
    """처리 단계와 HTTP 상태에 맞는 고정 오류 코드를 전달한다."""

    def __init__(self, status_code: int, code: str) -> None:
        """원문이나 이미지가 포함되지 않은 고정 진단 코드를 보관한다."""
        super().__init__(code)
        self.status_code = status_code
        self.code = code
