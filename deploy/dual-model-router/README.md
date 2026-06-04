# Dual-model ASR serving (English + multilingual) — router + 2 backends

Serve the **English specialist** (`nvidia/nemotron-speech-streaming-en-0.6b`) and the
**multilingual** checkpoint (`nvidia/NVIDIA-Nemotron-3.5-ASR-Streaming-Multilingual-0.6b`) behind
**one WebSocket endpoint**, with each model at parity on the high-throughput batched path — without
putting English on an unvalidated runtime.

## Why a router instead of one process loading both

The two checkpoints need **different, incompatible NeMo runtimes**:

| | English backend | Multilingual backend |
|---|---|---|
| Model class | `EncDecHybridRNNTCTCBPEModel` | `EncDecRNNTBPEModelWithPrompt` (prompted, 128 langs) |
| NeMo / torch | stock NeMo 2.8.x / torch 2.x | **EA NeMo build** / torch 2.12 |
| Right context | rc1 (`[70,1]`) | rc3 (`[56,3]`) — rc1 is rejected by the model |

Hosting both in one process would force English onto the EA runtime (a different model class — its
streaming/numerics are no longer the validated English path) and require a route-aware rewrite of the
batched scheduler. Instead, run **two single-model servers** (each on its own venv, each running the
full batched scheduler) and a thin **router** that picks the backend per connection by `?language=`.
English byte-identity is preserved because its runtime is untouched.

> `server.py` still has an in-process `--multilingual-model` flag (loads both in one process, serial
> path only). It is **not recommended** for production for the reasons above; prefer this router.

## Topology

```
client ──▶ ws://host:8080  (router.py — routes by ?language=, pipes frames)
                 │  en / en-* / (none) ─▶ EN backend  :8081  (standard venv, rc1, batched)
                 │  any other language ─▶ ML backend  :8082  (EA venv,       rc3, batched)
```

The EN backend rejects a `?language=` query, so the router strips it for English routes; the ML
backend needs it for its per-language prompt, so the router forwards it. Language is a **connection
handshake contract** (it cannot be changed mid-session under `NEMOTRON_CONTINUOUS=1`).

## Run

```bash
EN_PY=/path/to/standard-nemo-venv/bin/python \
ML_PY=/path/to/ea-nemo-venv/bin/python \
ML_MODEL=/path/to/nemotron-asr-streaming-multilingual-0.6b.nemo \
  deploy/dual-model-router/run_dual_model.sh
```

- **EN_PY** — python in a venv with the stock NeMo runtime (English checkpoint).
- **ML_PY** — python in a venv with the **EA NeMo build** that provides `EncDecRNNTBPEModelWithPrompt`.
  The multilingual checkpoint will not load under stock NeMo. (The router itself needs only `websockets`.)
- **ML_MODEL** — local `.nemo` path is most reliable; the HF repo's `.nemo` filename differs from the
  repo id, so `from_pretrained(repo_id)` may not resolve it.

The Pipecat ASR service (`pipecat_bots/nvidia_stt.py`) already connects to a single URL and sends a
`language`, so point it at the router (`ws://host:8080`) and set the language per session.

## Performance (RTX 5090, real audio through the router)

| Metric | English (rc1) | Multilingual (real es, rc3) |
|---|--:|--:|
| Router tax | ~0 ms | — |
| N=1 finalize latency | 13 ms | 18 ms |
| p95 @ 16 concurrent | 367 ms (< 400) | 420 ms |

Router adds ~0 ms; per-utterance latency is within a few ms; the multilingual SLO knee is marginally
below English (its mandated rc3 = more compute per step). Near-parity at the serving path; for a
production capacity number, measure a sustained multiprocess EN+ES mixed-load knee on the target GPU.

## Robustness note

These servers run with `NEMOTRON_FINALIZE_SILENCE_MS=0`. The server is patched to always emit one
terminal `is_final && finalize` event per client-awaited finalize even when a turn produced no text
(silence / VAD false-trigger), so clients never hang waiting for a final that was suppressed.
