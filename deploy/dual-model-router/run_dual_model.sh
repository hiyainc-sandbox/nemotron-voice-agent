#!/usr/bin/env bash
# Dual-model ASR (English + multilingual) via a language-routing front-end + 2 single-model
# backends. One public endpoint; English stays on its validated runtime. See README.md.
#
# Configure via env vars (examples shown). The two backends need DIFFERENT NeMo runtimes:
#   EN  = stock NeMo / torch 2.x  (the English specialist checkpoint)
#   ML  = the EA NeMo build that provides EncDecRNNTBPEModelWithPrompt (the multilingual checkpoint)
# Run each backend from its own virtualenv.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"

EN_PY="${EN_PY:?set EN_PY to the python in the standard-NeMo venv (English backend)}"
ML_PY="${ML_PY:?set ML_PY to the python in the EA-NeMo venv (multilingual backend)}"
ROUTER_PY="${ROUTER_PY:-$EN_PY}"   # any python with the `websockets` package

EN_MODEL="${EN_MODEL:-nvidia/nemotron-speech-streaming-en-0.6b}"
ML_MODEL="${ML_MODEL:?set ML_MODEL to the multilingual .nemo path or HF id (e.g. nvidia/nemotron-3.5-asr-streaming-0.6b)}"

EN_PORT="${EN_PORT:-8081}"; ML_PORT="${ML_PORT:-8082}"; ROUTER_PORT="${ROUTER_PORT:-8080}"

# Batched scheduler flags (single-model mode on each backend — the full high-throughput path).
COMMON_ENV=(NEMOTRON_CONTINUOUS=1 NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1
            NEMOTRON_BATCH_MAX_SIZE=32 NEMOTRON_BATCH_MAX_WAIT_MS=8
            NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200)

cd "$REPO"
echo "[dual] EN backend :$EN_PORT (rc1, standard venv)"
env -u LD_LIBRARY_PATH "${COMMON_ENV[@]}" "$EN_PY" src/nemotron_speech/server.py \
    --model "$EN_MODEL" --host 127.0.0.1 --port "$EN_PORT" --right-context 1 &

echo "[dual] ML backend :$ML_PORT (rc3, EA venv, prompted)"
env -u LD_LIBRARY_PATH "${COMMON_ENV[@]}" NEMOTRON_MODEL_NAME=multilingual "$ML_PY" src/nemotron_speech/server.py \
    --model "$ML_MODEL" --host 127.0.0.1 --port "$ML_PORT" --right-context 3 &

# Wait for both backends to log "ASR server listening" before starting the router (omitted for brevity).
sleep 5
echo "[dual] router :$ROUTER_PORT  (en/en-* -> :$EN_PORT, other languages -> :$ML_PORT)"
exec env -u LD_LIBRARY_PATH "$ROUTER_PY" "$HERE/router.py" \
    --host 0.0.0.0 --port "$ROUTER_PORT" \
    --en "ws://127.0.0.1:$EN_PORT" --ml "ws://127.0.0.1:$ML_PORT" -v
