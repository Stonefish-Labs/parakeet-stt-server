from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from parakeet_stt.server.errors import RequestError, invalid_request
from parakeet_stt.server.schemas import TranscriptionRequest, is_supported_model, normalize_response_format
from parakeet_stt.server.settings import Settings


logger = logging.getLogger("parakeet-stt")


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    metadata: dict[str, Any]


class ParakeetEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = None
        self.device = None
        self._load_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(settings.max_concurrent)
        self._warmed_up = False
        self.settings.temp_dir.mkdir(parents=True, exist_ok=True)

    async def load(self) -> None:
        if self.model is not None:
            return
        async with self._load_lock:
            if self.model is not None:
                return
            self.model, self.device = await asyncio.to_thread(self._load_model)
            logger.info("Loaded Parakeet model=%s device=%s", self.settings.model_id, self.device)
            if self.settings.warmup_on_load and not self._warmed_up:
                await asyncio.to_thread(self._warmup)
                self._warmed_up = True

    def _load_model(self):
        torch, nemo_asr = require_nemo()
        selected_device = pick_device(torch, self.settings.device)
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.settings.model_id)
        if hasattr(model, "to"):
            model.to(selected_device)
        if hasattr(model, "eval"):
            model.eval()
        return model, selected_device

    def _warmup(self) -> None:
        if self.model is None:
            return
        path = self._warmup_audio_path()
        try:
            logger.info("Running discarded warmup transcription")
            self._transcribe_path(path)
        except Exception as exc:
            logger.warning("Warmup transcription failed; continuing without warmup: %s", exc)
        finally:
            path.unlink(missing_ok=True)

    def _warmup_audio_path(self) -> Path:
        import numpy as np
        import soundfile as sf

        sample_rate = 16000
        duration_seconds = 0.75
        samples = int(sample_rate * duration_seconds)
        audio = np.zeros(samples, dtype=np.float32)
        fd, raw_path = tempfile.mkstemp(prefix="parakeet-warmup-", suffix=".wav", dir=self.settings.temp_dir)
        os.close(fd)
        path = Path(raw_path)
        sf.write(path, audio, sample_rate)
        return path

    async def transcribe_upload(self, upload: UploadFile, request: TranscriptionRequest) -> TranscriptionResult:
        if not is_supported_model(request.model, self.settings.model_id):
            raise invalid_request(
                f"Unsupported model: {request.model}",
                param="model",
                code="model_not_found",
            )
        response_format = normalize_response_format(request.response_format)
        path = self._new_upload_path(upload)
        bytes_written = 0
        started = time.perf_counter()
        try:
            bytes_written = await self._save_upload(upload, path)
            await self.load()
            async with self._semaphore:
                text = await asyncio.to_thread(self._transcribe_path, path)
        finally:
            path.unlink(missing_ok=True)
            try:
                await upload.close()
            except Exception:
                pass

        elapsed = time.perf_counter() - started
        return TranscriptionResult(
            text=text,
            metadata={
                "model": request.model,
                "backend_model": self.settings.model_id,
                "response_format": response_format,
                "language": request.language,
                "prompt_provided": bool(request.prompt),
                "bytes": bytes_written,
                "elapsed_seconds": round(elapsed, 3),
            },
        )

    def _new_upload_path(self, upload: UploadFile) -> Path:
        fd, raw_path = tempfile.mkstemp(prefix="parakeet-upload-", suffix=upload_suffix(upload), dir=self.settings.temp_dir)
        os.close(fd)
        return Path(raw_path)

    async def _save_upload(self, upload: UploadFile, path: Path) -> int:
        total = 0
        with path.open("wb") as output:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > self.settings.max_file_bytes:
                    raise RequestError(
                        413,
                        f"Audio file is too large. Max size is {self.settings.max_file_bytes} bytes.",
                        param="file",
                        code="file_too_large",
                    )
                output.write(chunk)
        if total == 0:
            raise invalid_request("Uploaded audio file is empty.", param="file", code="empty_file")
        return total

    def _transcribe_path(self, path: Path) -> str:
        if self.model is None:
            raise RuntimeError("Parakeet model is not loaded")
        output = self.model.transcribe([str(path)])
        return extract_text(output)

    def device_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "device": self.device or self.settings.device or "auto",
            "gpu_available": False,
            "gpu_name": None,
            "vram_total": None,
        }
        try:
            import torch

            if torch.cuda.is_available():
                index = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(index)
                info.update(
                    {
                        "gpu_available": True,
                        "gpu_name": torch.cuda.get_device_name(index),
                        "vram_total": f"{props.total_memory / 1024**3:.2f} GB",
                    }
                )
        except Exception as exc:
            info["error"] = str(exc)
        return info


def require_nemo():
    try:
        import torch
        import nemo.collections.asr as nemo_asr
    except ImportError as exc:
        raise RuntimeError(
            "NeMo ASR dependencies are not installed. Install with: "
            "uv sync --extra server --extra nemo"
        ) from exc
    return torch, nemo_asr


def pick_device(torch: Any, requested_device: str | None) -> str:
    if requested_device:
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {requested_device}, but CUDA is not available")
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def upload_suffix(upload: UploadFile) -> str:
    if upload.filename:
        suffix = Path(upload.filename).suffix.lower()
        if suffix:
            return suffix
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    return {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/flac": ".flac",
        "audio/mp4": ".m4a",
        "audio/ogg": ".ogg",
        "audio/webm": ".webm",
    }.get(content_type, ".wav")


def extract_text(output: Any) -> str:
    while isinstance(output, tuple) and output:
        output = output[0]
    if isinstance(output, list | tuple):
        if not output:
            return ""
        first = output[0]
    else:
        first = output

    if isinstance(first, str):
        return first.strip()
    if isinstance(first, dict):
        value = first.get("text") or first.get("pred_text") or first.get("transcript")
        return str(value or "").strip()
    value = getattr(first, "text", None)
    if value is not None:
        return str(value).strip()
    return str(first).strip()

