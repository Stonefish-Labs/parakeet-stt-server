# Parakeet OpenAI-Compatible STT Server

This service exposes NVIDIA Parakeet v3 through the OpenAI transcription API
shape used by Hermes.

Inference is local. The model name is a download identifier for the Parakeet
weights, not a remote NVIDIA service.

## Endpoints

- `GET /health`
- `GET /v1/models`
- `GET /v1/models/{model_id}`
- `POST /v1/audio/transcriptions`

## Transcription Request

The transcription endpoint accepts multipart form data:

```text
file=<audio file>
model=nvidia/parakeet-tdt-0.6b-v3
response_format=json
```

Supported `response_format` values:

- `json`: returns `{ "text": "..." }`
- `text`: returns `text/plain`

The server accepts but does not use `language`, `prompt`, and `temperature` in
v1. Parakeet v3 automatically detects language for its supported language set.

Unsupported OpenAI/Whisper formats such as `verbose_json`, `srt`, `vtt`, and
`diarized_json` return a `400` with an OpenAI-shaped error body.

## OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:5092/v1",
    api_key="not-needed",
)

with open("speech.wav", "rb") as audio:
    transcript = client.audio.transcriptions.create(
        model="nvidia/parakeet-tdt-0.6b-v3",
        file=audio,
        response_format="json",
    )

print(transcript.text)
```

Plain text response:

```python
with open("speech.wav", "rb") as audio:
    text = client.audio.transcriptions.create(
        model="nvidia/parakeet-tdt-0.6b-v3",
        file=audio,
        response_format="text",
    )
print(text)
```

## Auth

If `PARAKEET_STT_API_KEY` is empty, any bearer token is accepted. If it is set,
clients must use that exact value:

```python
client = OpenAI(
    base_url="http://127.0.0.1:5092/v1",
    api_key="<PARAKEET_STT_API_KEY>",
)
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
