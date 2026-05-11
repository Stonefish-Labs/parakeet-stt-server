from __future__ import annotations

import argparse
from pathlib import Path

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the Parakeet STT OpenAI endpoint.")
    parser.add_argument("audio", type=Path, help="Audio file to transcribe.")
    parser.add_argument("--url", default="http://127.0.0.1:5092/v1", help="OpenAI base URL.")
    parser.add_argument("--api-key", default="not-needed", help="Bearer token value.")
    parser.add_argument("--format", default="json", choices=["json", "text"], help="response_format value.")
    args = parser.parse_args()

    client = OpenAI(base_url=args.url, api_key=args.api_key)
    with args.audio.open("rb") as audio:
        result = client.audio.transcriptions.create(
            model="nvidia/parakeet-tdt-0.6b-v3",
            file=audio,
            response_format=args.format,
        )
    print(result if isinstance(result, str) else result.text)


if __name__ == "__main__":
    main()
