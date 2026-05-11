from __future__ import annotations

from types import SimpleNamespace

from parakeet_stt.server.engine import extract_text, upload_suffix


def test_extract_text_handles_common_nemo_outputs() -> None:
    assert extract_text([" hello "]) == "hello"
    assert extract_text([SimpleNamespace(text="world")]) == "world"
    assert extract_text(([{"text": "nested"}], "ignored")) == "nested"
    assert extract_text([]) == ""


def test_upload_suffix_prefers_filename() -> None:
    upload = SimpleNamespace(filename="clip.webm", content_type="audio/wav")

    assert upload_suffix(upload) == ".webm"


def test_upload_suffix_falls_back_to_content_type() -> None:
    upload = SimpleNamespace(filename="", content_type="audio/flac")

    assert upload_suffix(upload) == ".flac"

