from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from parakeet_stt.server.errors import invalid_request


DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
MODEL_ALIASES = {
    DEFAULT_MODEL_ID,
    "parakeet-tdt-0.6b-v3",
    "parakeet",
}
SUPPORTED_RESPONSE_FORMATS = {"json", "text"}
KNOWN_UNSUPPORTED_RESPONSE_FORMATS = {"verbose_json", "srt", "vtt", "diarized_json"}


@dataclass(frozen=True)
class TranscriptionRequest:
    model: str
    response_format: str
    language: str | None = None
    prompt: str | None = None
    temperature: float | None = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 1778452800
    owned_by: str = "parakeet-stt"


def accepted_model_ids(configured_model_id: str) -> set[str]:
    return MODEL_ALIASES | {configured_model_id}


def model_infos(configured_model_id: str) -> list[ModelInfo]:
    return [ModelInfo(id=model_id) for model_id in sorted(accepted_model_ids(configured_model_id))]


def is_supported_model(model_id: str, configured_model_id: str) -> bool:
    return model_id in accepted_model_ids(configured_model_id)


def normalize_response_format(value: str | None) -> str:
    response_format = (value or "json").strip().lower()
    if response_format in SUPPORTED_RESPONSE_FORMATS:
        return response_format
    if response_format in KNOWN_UNSUPPORTED_RESPONSE_FORMATS:
        raise invalid_request(
            f"response_format '{response_format}' is not supported by this Parakeet STT v1 server. "
            "Supported formats: json, text.",
            param="response_format",
            code="unsupported_response_format",
        )
    raise invalid_request(
        f"Unsupported response_format: {response_format}. Supported formats: json, text.",
        param="response_format",
        code="unsupported_response_format",
    )

