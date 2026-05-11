from __future__ import annotations

import io
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from parakeet_stt.server.api import create_app
from parakeet_stt.server.engine import ParakeetEngine
from parakeet_stt.server.settings import Settings


def make_settings(tmp_path: Path, api_key: str = "", warmup_on_load: bool = False) -> Settings:
    return Settings(
        host="127.0.0.1",
        port=5092,
        cors_origins=["*"],
        api_key=api_key,
        model_id="nvidia/parakeet-tdt-0.6b-v3",
        device=None,
        max_concurrent=1,
        warmup_on_load=warmup_on_load,
        max_file_bytes=1024 * 1024,
        temp_dir=tmp_path,
    )


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)
    return buffer.getvalue()


class FakeModel:
    def __init__(self) -> None:
        self.paths_seen: list[str] = []

    def transcribe(self, paths: list[str]) -> list[Any]:
        self.paths_seen.extend(paths)
        assert all(Path(path).exists() for path in paths)
        return [SimpleNamespace(text="hello from parakeet")]


def test_models_are_listed_without_auth_when_key_is_empty(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    engine = ParakeetEngine(settings)
    client = TestClient(create_app(settings, engine))

    response = client.get("/v1/models")

    assert response.status_code == 200
    ids = {item["id"] for item in response.json()["data"]}
    assert "nvidia/parakeet-tdt-0.6b-v3" in ids
    assert "parakeet" in ids


def test_auth_rejects_missing_bearer_token_when_key_is_set(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, api_key="secret")
    engine = ParakeetEngine(settings)
    client = TestClient(create_app(settings, engine))

    response = client.get("/v1/models")

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"


def test_auth_accepts_matching_bearer_token(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, api_key="secret")
    engine = ParakeetEngine(settings)
    client = TestClient(create_app(settings, engine))

    response = client.get("/v1/models", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200


def test_transcription_returns_json_and_cleans_temp_file(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    engine = ParakeetEngine(settings)
    engine.model = FakeModel()
    engine.device = "cpu"
    client = TestClient(create_app(settings, engine))

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("speech.wav", wav_bytes(), "audio/wav")},
        data={"model": "nvidia/parakeet-tdt-0.6b-v3", "response_format": "json"},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "hello from parakeet"}
    assert list(tmp_path.iterdir()) == []


def test_transcription_returns_plain_text(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    engine = ParakeetEngine(settings)
    engine.model = FakeModel()
    engine.device = "cpu"
    client = TestClient(create_app(settings, engine))

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("speech.wav", wav_bytes(), "audio/wav")},
        data={"model": "parakeet", "response_format": "text"},
    )

    assert response.status_code == 200
    assert response.text == "hello from parakeet"
    assert response.headers["content-type"].startswith("text/plain")


def test_unsupported_response_format_is_openai_shaped_error(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    engine = ParakeetEngine(settings)
    client = TestClient(create_app(settings, engine))

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("speech.wav", wav_bytes(), "audio/wav")},
        data={"model": "nvidia/parakeet-tdt-0.6b-v3", "response_format": "verbose_json"},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["param"] == "response_format"
    assert payload["error"]["code"] == "unsupported_response_format"


def test_unknown_model_is_rejected(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    engine = ParakeetEngine(settings)
    engine.model = FakeModel()
    client = TestClient(create_app(settings, engine))

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("speech.wav", wav_bytes(), "audio/wav")},
        data={"model": "not-a-model", "response_format": "json"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "model_not_found"
