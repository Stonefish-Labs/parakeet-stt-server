# Parakeet STT Server

OpenAI-compatible speech-to-text server for NVIDIA Parakeet v3.

This is self-hosted inference. The `nvidia/parakeet-tdt-0.6b-v3` value is the
model ID used to download weights into the local Hugging Face/NeMo cache; the
server runs transcription inside your container on your local GPU.

The service is intended for Hermes or other OpenAI SDK clients that call:

```text
POST /v1/audio/transcriptions
```

Default URL:

```text
http://127.0.0.1:5092/v1
```

## Local Development

```powershell
git clone https://github.com/Stonefish-Labs/parakeet-stt-server.git
cd parakeet-stt-server
uv sync --group dev --extra server
uv run --extra server pytest
```

Install NeMo only when you are ready to run the model locally:

```powershell
uv sync --group dev --extra server --extra nemo
uv run --extra server --extra nemo parakeet-stt-server
```

## Docker

```powershell
Copy-Item .env.parakeet-stt.example .env.parakeet-stt
docker compose --env-file .env.parakeet-stt -f docker-compose.parakeet-stt.yml up --build -d
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:5092/health
```

## OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:5092/v1",
    api_key="not-needed",
)

with open("speech.wav", "rb") as audio:
    result = client.audio.transcriptions.create(
        model="nvidia/parakeet-tdt-0.6b-v3",
        file=audio,
        response_format="json",
    )

print(result.text)
```

## Hermes

```yaml
stt:
  enabled: true
  provider: openai
  openai:
    model: nvidia/parakeet-tdt-0.6b-v3
```

```bash
STT_OPENAI_BASE_URL=http://<host-or-tailnet-ip>:5092/v1
VOICE_TOOLS_OPENAI_KEY=not-needed
```

If `PARAKEET_STT_API_KEY` is set on the server, use that same value for
`VOICE_TOOLS_OPENAI_KEY`.

More details:

- `docs/OPENAI_STT_SERVER.md`
- `docs/DOCKER.md`
