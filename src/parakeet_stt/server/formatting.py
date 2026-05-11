from __future__ import annotations

from fastapi.responses import JSONResponse, PlainTextResponse, Response


def format_transcription_response(text: str, response_format: str, headers: dict[str, str] | None = None) -> Response:
    if response_format == "json":
        return JSONResponse({"text": text}, headers=headers)
    if response_format == "text":
        return PlainTextResponse(text, headers=headers)
    raise ValueError(f"Unsupported response_format: {response_format}")

