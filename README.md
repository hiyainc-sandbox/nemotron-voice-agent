# Voice Agent With NVIDIA Open Models

[![Demo Video](https://img.youtube.com/vi/8Fkz2PC54BI/maxresdefault.jpg)](https://www.youtube.com/watch?v=8Fkz2PC54BI)

This repo is sample code for building voice agents with NVIDIA open source models:
  - Nemotron 3.0 ASR and Nemotron 3.5 ASR (streaming speech-to-text)
  - Nemotron 3 Nano LLM

For speech, the bot uses on-device [Pocket TTS](https://pypi.org/project/pocket-tts/) (Kyutai).

Run locally on an NVIDIA DGX Spark or RTX 5090. Or deploy to the cloud.

Accompanying blog posts:
- [Nemotron Speech ASR Open Source Model Launch Post](https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents)
- [More About Voice Agent Architectures and This Agent's Design]()

## Two things live here

1. **The voice agent** — Nemotron ASR + Nemotron 3 Nano LLM + on-device Pocket TTS, run
   locally (DGX Spark / RTX 5090) or in the cloud.
2. **A production streaming-ASR serving runtime** — a from-scratch native C++ `ws_server` for the
   Nemotron Speech ASR model that you **build, run on an RTX 5090, and deploy as an L40S cluster**.
   It is a byte-exact drop-in for the Python ASR server with much higher per-GPU stream density. See
   **[Production streaming-ASR serving](#production-streaming-asr-serving-rtx-5090--l40s-cluster)** below.

## Repository map

| path | what |
|---|---|
| `src/nemotron_speech/` | Python ASR + TTS servers (`server.py`) |
| `pipecat_bots/` | The voice-agent bot (`bot.py`) + its STT / TTS / LLM Pipecat services |
| `scripts/` | local container management (`nemotron.sh`) and test clients |
| `Dockerfile.unified` | the all-in-one local container (ASR + TTS + LLM), built from source for Blackwell |
| `runtime/` | the native C++ `ws_server` ASR serving runtime — build, run, and artifact regeneration |
| `deploy/` | L40S cluster deploy: runbook, systemd unit, HAProxy generator, drain/smoke tooling |
| `ec2-bench/` | minimal EC2 GPU provisioning helpers used by the deploy runbook |
| `tests/` | ASR/TTS tests, the Python↔C++ byte-exact compatibility oracle, and a WebSocket smoke client |
| `docs/` | architecture and latency explainers |

## Quick start - Run everything locally (DGX Spark or RTX 5090)

### 1. Build the Unified Container

```bash
docker build -f Dockerfile.unified -t nemotron-unified:cuda13 .
```

Build time: 2-3 hours (builds PyTorch, NeMo, vLLM, llama.cpp from source for CUDA 13.1 / Blackwell).

### 2. Start the Container

```bash
# Start with default Q8 model (auto-detected from HuggingFace cache)
./scripts/nemotron.sh start

# Or specify a model explicitly
./scripts/nemotron.sh start --model ~/.cache/huggingface/hub/models--unsloth--Nemotron-3-Nano-30B-A3B-GGUF/snapshots/.../Q8_0.gguf

# Start with vLLM instead of llama.cpp (requires ~72GB VRAM)
./scripts/nemotron.sh start --mode vllm
```

### 3. Start the on-device Pocket TTS server

The voice bot speaks with [**Pocket TTS**](https://pypi.org/project/pocket-tts/) (Kyutai's on-device
TTS, a published package). Start it in its own terminal and leave it running on port 8001 (it downloads
the model on first run):

```bash
# runs the published package in an ephemeral env (or: pip install pocket-tts && pocket-tts serve ...)
uvx pocket-tts serve --port 8001 --language english
```

### 4. Run the Voice Bot

`pipecat_bots/bot.py` reads its **ASR and LLM endpoints from required env vars** — point them at the
servers from step 2 (and Pocket TTS from step 3):

```bash
NVIDIA_ASR_URL=ws://localhost:8080 \
NVIDIA_LLM_URL=http://localhost:8000/v1 \
POCKET_TTS_URL=http://localhost:8001 \
  uv run pipecat_bots/bot.py
```

Open the URL it prints (default `http://localhost:7860`) in your browser.

## Quick start - Deploy the bot to Pipecat Cloud

The bot connects to your ASR, LLM, and TTS endpoints over the network, so it can run anywhere —
including [Pipecat Cloud](https://docs.pipecat.ai/deployment/pipecat-cloud/introduction). You bring
the model endpoints (e.g. the L40S ASR cluster below); this section deploys only the **bot**.

> [!NOTE]
> Sign up for a [Pipecat Cloud](https://docs.pipecat.ai/deployment/pipecat-cloud/introduction) account [here](https://pipecat.daily.co/)

#### 1. Login to your Pipecat Cloud account using the CLI

```bash
# Install Pipecat Cloud package
uv sync --group bot

# Login
pipecat cloud auth login
```

#### 2. Create a new secret set with the necessary API keys

```bash
pipecat cloud secrets set gdx-spark-bot-secrets \
  NVIDIA_ASR_URL=wss:// \
  NVIDIA_LLM_URL=https:// \
  POCKET_TTS_URL=https://
```

_Alternatively, create your secret set from a `.env` file:_

```bash
pipecat cloud secrets set gdx-spark-bot-secrets --file .env
```

#### 3. Create image pull secret

Image pull secrets are used to authenticate with private Docker registries when deploying agents. [See docs](https://docs.pipecat.ai/deployment/pipecat-cloud/fundamentals/secrets#image-pull-secrets).

```bash
pipecat cloud secrets image-pull-secret gdx-spark-bot-pull-secret https://index.docker.io/v1/
```

___Optional: Create a PCC deploy toml___:

To speed up deployment you can create a `pcc-deploy.toml` in the project root. This file is read by the Pipecat CLI to pre-fill command arguments:

```bash
agent_name = "gdx-spark-bot"
image = "your-docker-repository/gdx-spark-bot:latest"
secret_set = "gdx-spark-bot-secrets"
image_credentials = "gdx-spark-bot-pull-secret"
agent_profile = "agent-1x"

[scaling]
	min_agents = 1
```

#### 4. Build and push Docker image

```bash
docker build -f Dockerfile.bot -t gdx-spark-bot:latest .

# Optional: tag image
docker tag gdx-spark-bot:latest your-docker-repository/gdx-spark-bot:latest

# Push to image repository e.g. Docker Hub
docker push your-docker-repository/gdx-spark-bot:latest
```

#### 5. Deploy

Run `deploy` command:

```bash
pipecat cloud deploy

# ...or if not using pcc-deploy.toml

pipecat cloud deploy gdx-spark-bot your-docker-repository/gdx-spark-bot:latest \
--credentials gdx-spark-bot-pull-secret \
--secrets gdx-spark-bot-secrets \
--profile agent-1x
```

#### 6. Start bot using CLI

Create a public access key for Pipecat Cloud. Set this is a the default key when prompted:

```bash
pipecat cloud organizations keys create
```

Start an active session with your deployed bot:

```bash
pipecat cloud agent start gdx-spark-bot --use-daily
```

[See docs](https://docs.pipecat.ai/deployment/pipecat-cloud/fundamentals/active-sessions) for REST and Python usage.


## The Voice Bot

`pipecat_bots/bot.py` is the single voice agent: **Nemotron streaming STT → Nemotron LLM
(OpenAI-compatible) → on-device Pocket TTS**, with on-device **Smart Turn v3** endpointing and a
SmallWebRTC transport. The ASR and LLM endpoints are **required env vars**:

| Variable | Required | Default | Description |
|----------|:--:|---------|-------------|
| `NVIDIA_ASR_URL` | ✅ | — | Nemotron streaming ASR WebSocket endpoint (e.g. `ws://localhost:8080`) |
| `NVIDIA_LLM_URL` | ✅ | — | OpenAI-compatible LLM endpoint (e.g. `http://localhost:8000/v1`) |
| `NVIDIA_LLM_MODEL` | | `nvidia/nemotron-3-nano` | Model name your LLM server serves (per its `/v1/models`) |
| `NVIDIA_LLM_API_KEY` | | `EMPTY` | API key (local vLLM / llama.cpp ignore it) |
| `NEMOTRON_ENABLE_THINKING` | | `false` | Enable LLM reasoning (keep off for voice unless the server runs a reasoning parser) |
| `POCKET_TTS_URL` | | `http://127.0.0.1:8001` | On-device Pocket TTS server |
| `POCKET_TTS_VOICE` | | `alba` | Pocket TTS voice |

### Transport

SmallWebRTC is the default (opens a local browser client): `uv run pipecat_bots/bot.py -t webrtc`.
Other Pipecat transports (Daily, Twilio) can be added to `transport_params` in `bot.py`.

### Services in `pipecat_bots/`

| Service | File | Description |
|---------|------|-------------|
| `NVidiaWebSocketSTTService` | `nvidia_stt.py` | Nemotron streaming ASR over WebSocket; finalizes on the VAD stop for Smart Turn |
| `VLLMOpenAILLMService` | `nemotron_llm.py` | OpenAI-compatible LLM client (TTFB measured to the first spoken token) |
| `PocketTTSService` | `pocket_tts.py` | On-device Pocket TTS (HTTP streaming) |

## Local Container Management

Use `./scripts/nemotron.sh` to manage the container:

```bash
# Start the container
./scripts/nemotron.sh start [OPTIONS]
  --mode MODE         LLM mode: llamacpp-q8 (default), llamacpp-q4, vllm
  --model PATH        Path to model file
  --no-asr            Disable ASR service
  --no-llm            Disable LLM service
  -f, --foreground    Run in foreground (default: detached)

# Stop the container
./scripts/nemotron.sh stop

# Restart the container
./scripts/nemotron.sh restart [OPTIONS]

# Check status
./scripts/nemotron.sh status

# View logs
./scripts/nemotron.sh logs          # All logs interleaved
./scripts/nemotron.sh logs asr      # ASR logs only
./scripts/nemotron.sh logs llm      # LLM logs only

# Open shell in container
./scripts/nemotron.sh shell

# Show help
./scripts/nemotron.sh help
```

### Service Endpoints

| Service | Port | Protocol | Health Check |
|---------|------|----------|--------------|
| ASR | 8080 | WebSocket | `http://localhost:8080/health` |
| LLM | 8000 | HTTP | `http://localhost:8000/health` |

> TTS is **on-device Pocket TTS**, run separately (`uvx pocket-tts serve --port 8001`), not part of
> the container. The bot reaches it via `POCKET_TTS_URL`.

## Building the Container

```bash
# Build the unified container (2-3 hours)
docker build -f Dockerfile.unified -t nemotron-unified:cuda13 .
```

The build compiles from source for CUDA 13.1 / Blackwell (sm_121):
- PyTorch (with NVRTC support)
- torchaudio
- NeMo ASR/TTS
- vLLM
- llama.cpp

## Model Requirements

| Model | Source | Size | Used With |
|-------|--------|------|-----------|
| Nemotron Speech ASR (English) | HuggingFace `nvidia/nemotron-speech-streaming-en-0.6b` (auto-downloaded) | ~2.4GB | All configurations |
| Nemotron Speech ASR (Multilingual) | HuggingFace `nvidia/NVIDIA-Nemotron-3.5-ASR-Streaming-Multilingual-0.6b` (auto-downloaded) | ~2.4GB | Optional — dual-model language routing (see below) |
| Nemotron-3-Nano Q8 | HuggingFace `unsloth/Nemotron-3-Nano-30B-A3B-GGUF` | ~32GB | llama.cpp on DGX Spark |
| Nemotron-3-Nano Q4 | HuggingFace `unsloth/Nemotron-3-Nano-30B-A3B-GGUF` | ~16GB | llama.cpp on RTX 5090 |
| Nemotron-3-Nano BF16 | HuggingFace `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | ~72GB | vLLM (cloud/multi-GPU) |
| Pocket TTS | published `pocket-tts` package (downloads its model on first `serve`) | ~small | On-device TTS, run separately |

Download LLM models (ASR is auto-downloaded on first run; Pocket TTS downloads its model on first `serve`):

```bash
# GGUF quantized models (Q8 and Q4 variants for llama.cpp)
huggingface-cli download unsloth/Nemotron-3-Nano-30B-A3B-GGUF

# BF16 full precision (for vLLM)
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

### Multilingual (dual-model serving)

To serve **both** the English specialist and the multilingual checkpoint behind one endpoint —
with each model on the full high-throughput batched path — use the **router + 2 backends** recipe in
**[`deploy/dual-model-router/`](deploy/dual-model-router/README.md)**:

```bash
EN_PY=/path/to/standard-nemo-venv/bin/python \
ML_PY=/path/to/ea-nemo-venv/bin/python \
ML_MODEL=/path/to/nemotron-asr-streaming-multilingual-0.6b.nemo \
  deploy/dual-model-router/run_dual_model.sh
```

A thin router (`ws://host:8080`) routes each connection by `?language=` to an English backend
(standard NeMo, rc1) or a multilingual backend (the **EA NeMo build**, rc3, prompted). The two
checkpoints need **different, incompatible NeMo runtimes**, so each backend runs in its own venv;
the router keeps English on its validated runtime (byte-identical) while both get the batched
scheduler. The Pipecat service (`pipecat_bots/nvidia_stt.py`) already connects to one URL and sends a
`language`, so just point it at the router. See the deploy README for the runtime requirements and
measured perf (router tax ~0; multilingual at near-parity with English).

> **Advanced / not recommended:** `server.py` also has an in-process `--multilingual-model`
> (`NEMOTRON_MULTILINGUAL_MODEL`) flag that loads both checkpoints in one process. It runs on the
> **serial path only** and forces English onto the EA runtime (a different model class — no longer the
> validated English path). Prefer the router above.

## Production streaming-ASR serving (RTX 5090 → L40S cluster)

The voice-agent ASR service above is the Python server in `src/nemotron_speech/`. For high-density
production serving there is a **from-scratch native C++ runtime** in `runtime/` — `ws_server` — that
is a **byte-exact drop-in** for the Python ASR WebSocket server (the compat oracle in
`tests/server_compat/run_compat.py` verifies 8/8 parity) and serves many more concurrent streams per
GPU. The path is **clone → regenerate artifacts → build → run on a 5090 → deploy an L40S cluster**:

1. **Step 0 — model & artifacts.** The runtime loads compiled AOTI/TorchScript artifacts that are not
   committed (large + GPU-arch-specific). Regenerate them from the public checkpoint
   `nvidia/nemotron-speech-streaming-en-0.6b` following **[`runtime/ARTIFACTS.md`](runtime/ARTIFACTS.md)**.
   No private buckets or credentials are involved.

2. **Build + run on an RTX 5090 (sm_120).** Build `ws_server` in the container and run it on the host:
   **[`runtime/README.md`](runtime/README.md)**. Verify byte-exact parity with the Python server via the
   compat oracle.

3. **Deploy a cluster on L40S (sm_89).** Provision g6e/L40S boxes, recompile the AOTI artifacts for
   sm_89, install the systemd unit, and front the fleet with HAProxy:
   **[`deploy/RUNBOOK.md`](deploy/RUNBOOK.md)** (rationale and sizing in
   [`deploy/DEPLOYMENT.md`](deploy/DEPLOYMENT.md)). The sm_89 recompile + a density sweep are encoded in
   **[`runtime/run_l40s_density.README.md`](runtime/run_l40s_density.README.md)**, and the minimal EC2
   provisioning helpers are in [`ec2-bench/`](ec2-bench/README.md).

Because the AOTI artifacts are GPU-architecture-specific, the model is **exported once** (architecture-
agnostic) and then **AOTI-compiled per target** (sm_120 for the 5090, sm_89 for the L40S) — see
[`runtime/ARTIFACTS.md`](runtime/ARTIFACTS.md).

## Troubleshooting

**LLM crashes or stalls**:
- The buffered LLM service uses single-slot operation (`--parallel 1`)
- Ensure adequate VRAM for context size (default 16384 tokens)
- Check for httpx connection issues if generation hangs

**vLLM takes 10-15 minutes to start**:
- This is normal for first startup (model loading, kernel compilation)
- Set `SERVICE_TIMEOUT=900` if needed

**vLLM DNS resolution issues**:
- The container uses `--network=host` in vLLM mode to avoid DNS issues with HuggingFace

## License

The code in this repository is licensed under the Apache License 2.0 — see [LICENSE](LICENSE).

The NVIDIA models this sample uses (Nemotron Speech ASR, Nemotron 3 Nano LLM) are distributed under
their own NVIDIA model licenses on HuggingFace; review and accept those terms on each model's page
before downloading or deploying. Pocket TTS is a separate Kyutai package under its own license.

