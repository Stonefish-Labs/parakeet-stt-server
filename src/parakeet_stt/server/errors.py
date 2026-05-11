from __future__ import annotations


class RequestError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.param = param
        self.code = code
        super().__init__(message)

    def payload(self) -> dict:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


def invalid_request(message: str, *, param: str | None = None, code: str | None = None) -> RequestError:
    return RequestError(400, message, param=param, code=code)

