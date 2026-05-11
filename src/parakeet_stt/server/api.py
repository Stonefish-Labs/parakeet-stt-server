from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from parakeet_stt import __version__
from parakeet_stt.server.engine import ParakeetEngine
from parakeet_stt.server.errors import RequestError
from parakeet_stt.server.formatting import format_transcription_response
from parakeet_stt.server.schemas import (
    SUPPORTED_RESPONSE_FORMATS,
    TranscriptionRequest,
    model_infos,
    normalize_response_format,
)
from parakeet_stt.server.settings import Settings


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("parakeet-stt")


def create_app(settings: Settings | None = None, engine: ParakeetEngine | None = None) -> FastAPI:
    app_settings = settings or Settings.from_env()
    app_engine = engine or ParakeetEngine(app_settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if app_settings.warmup_on_load:
            await app_engine.load()
        yield

    app = FastAPI(
        title="Parakeet OpenAI-Compatible STT",
        description="An OpenAI-compatible speech-to-text API for NVIDIA Parakeet.",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestError)
    async def request_error_handler(_request: Any, exc: RequestError):
        return JSONResponse(status_code=exc.status_code, content=exc.payload())

    def require_api_key(authorization: str | None = Header(default=None)) -> None:
        if not app_settings.api_key:
            return
        expected = f"Bearer {app_settings.api_key}"
        if authorization != expected:
            raise RequestError(
                401,
                "Missing or invalid bearer token.",
                error_type="authentication_error",
                code="unauthorized",
            )

    @app.get("/")
    async def root() -> dict:
        return {
            "name": "Parakeet OpenAI-Compatible STT",
            "version": __version__,
            "endpoints": ["/health", "/v1/models", "/v1/audio/transcriptions"],
        }

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "backend": {
                "name": "parakeet-stt",
                "ready": app_engine.model is not None,
                "model": app_settings.model_id,
                "max_concurrent": app_settings.max_concurrent,
                "response_formats": sorted(SUPPORTED_RESPONSE_FORMATS),
            },
            "limits": {
                "max_file_bytes": app_settings.max_file_bytes,
                "temp_dir": str(app_settings.temp_dir),
            },
            "device": app_engine.device_info(),
        }

    @app.get("/v1/models")
    async def list_models(_auth: None = Depends(require_api_key)) -> dict:
        return {"object": "list", "data": [model.model_dump() for model in model_infos(app_settings.model_id)]}

    @app.get("/v1/models/{model_id}")
    async def get_model(model_id: str, _auth: None = Depends(require_api_key)) -> dict:
        for model in model_infos(app_settings.model_id):
            if model.id == model_id:
                return model.model_dump()
        raise RequestError(
            404,
            f"Model '{model_id}' not found.",
            param="model",
            code="model_not_found",
        )

    @app.post("/v1/audio/transcriptions")
    async def create_transcription(
        file: UploadFile = File(...),
        model: str | None = Form(default=None),
        response_format: str | None = Form(default="json"),
        language: str | None = Form(default=None),
        prompt: str | None = Form(default=None),
        temperature: float | None = Form(default=None),
        _auth: None = Depends(require_api_key),
    ):
        normalized_format = normalize_response_format(response_format)
        request = TranscriptionRequest(
            model=model or app_settings.model_id,
            response_format=normalized_format,
            language=language,
            prompt=prompt,
            temperature=temperature,
        )
        result = await app_engine.transcribe_upload(file, request)
        headers = {
            "Cache-Control": "no-cache",
            "X-Parakeet-Model": str(result.metadata["backend_model"]),
            "X-Parakeet-Requested-Model": str(result.metadata["model"]),
            "X-Parakeet-Elapsed": str(result.metadata["elapsed_seconds"]),
            "X-Parakeet-Bytes": str(result.metadata["bytes"]),
        }
        return format_transcription_response(result.text, normalized_format, headers=headers)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = Settings.from_env()
    uvicorn.run("parakeet_stt.server.api:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()

