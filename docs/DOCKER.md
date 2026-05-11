# Parakeet STT Docker Service

Docker is the preferred way to run the Parakeet STT server. The image owns the
Python, CUDA, PyTorch, and NeMo dependency stack. Hugging Face and NeMo caches
are stored in named Docker volumes.

The container does not call a hosted NVIDIA inference endpoint. It downloads
the Parakeet weights once, caches them locally, and runs `model.transcribe(...)`
on the Docker GPU runtime.

## Prerequisites

- Docker Desktop with Linux containers.
- NVIDIA Container Toolkit / Docker GPU support.
- Optional: a reachable LAN, VPN, or tailnet address if remote clients need access.

Quick GPU check:

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

## Start

```powershell
Copy-Item .env.parakeet-stt.example .env.parakeet-stt
docker compose --env-file .env.parakeet-stt -f docker-compose.parakeet-stt.yml up --build -d
```

Default URL:

```text
http://127.0.0.1:5092/v1
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:5092/health
```

## Logs

```powershell
docker compose --env-file .env.parakeet-stt -f docker-compose.parakeet-stt.yml logs -f parakeet-stt
```

## Stop

```powershell
docker compose --env-file .env.parakeet-stt -f docker-compose.parakeet-stt.yml down
```

## Configuration

`.env.parakeet-stt` controls the host bind, port, model, and optional bearer
token:

```text
PARAKEET_STT_BIND_ADDR=127.0.0.1
PARAKEET_STT_HOST_PORT=5092
PARAKEET_STT_MODEL=nvidia/parakeet-tdt-0.6b-v3
PARAKEET_STT_DEVICE=auto
PARAKEET_STT_MAX_CONCURRENT=1
PARAKEET_STT_WARMUP_ON_LOAD=true
PARAKEET_STT_MAX_FILE_MB=100
PARAKEET_STT_API_KEY=
```

If `PARAKEET_STT_API_KEY` is non-empty, OpenAI clients should use that value as
their API key.

To expose the service to another machine, set `PARAKEET_STT_BIND_ADDR` to a
specific LAN/VPN address or `0.0.0.0`, then point clients at that reachable host.

## Notes

- First startup downloads Parakeet and NeMo artifacts. The named cache volumes
  keep those downloads for future restarts.
- `PARAKEET_STT_MAX_CONCURRENT=1` is the default for predictable GPU latency.
- The default host port is `5092`.
