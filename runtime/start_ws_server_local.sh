#!/usr/bin/env bash
# Start the C++ ws_server locally on the RTX 5090 (sm_120), on the HOST (build is in-container).
# Exact env/flags mirror tests/server_compat/run_compat.py (the oracle-verified launch).
# See README.md. Overrides via env: PORT, CAP, LANES, FINALIZE_RUNNERS, BUILD_DIR, STEADY_BATCH_DIR, SHADOW.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
RT="$(pwd -P)"
REPO="$(cd ../.. && pwd -P)"

BUILD_DIR="${BUILD_DIR:-cpp/build_step10}"
BIN="$RT/$BUILD_DIR/ws_server"
[ -x "$BIN" ] || { echo "ws_server not built at $BIN — run: ./container/enter.sh cmake --build $BUILD_DIR --target ws_server -j8"; exit 1; }

# torch libs the host binary links against (must be the cu128 2.8.0 venv)
TORCH_LIB=$(ls -d "$RT"/.venv/lib/python*/site-packages/torch/lib 2>/dev/null | head -1)
[ -n "$TORCH_LIB" ] || { echo "venv torch/lib not found under $RT/.venv — create it with: bash setup-venv.sh"; exit 1; }

PORT="${PORT:-8081}"
CAP="${CAP:-64}"                       # admission active cap == lane count K
LANES="${LANES:-$CAP}"
FINALIZE_RUNNERS="${FINALIZE_RUNNERS:-2}"
STEADY_BATCH_DIR="${STEADY_BATCH_DIR:-$RT/steady_b_artifacts_b16}"

ENV=(
  HF_HUB_OFFLINE=1
  "PYTHONPATH=$REPO/src"
  "LD_LIBRARY_PATH=$TORCH_LIB:${LD_LIBRARY_PATH:-}"
  NEMOTRON_CONTINUOUS=1
  NEMOTRON_FINALIZE_SILENCE_MS=0
  "NEMOTRON_ARTIFACT_DIR=$RT/artifacts"
  NEMOTRON_WS_SCHEDULER=1
  NEMOTRON_DENSITY_BATCH_STEADY=1
  NEMOTRON_DENSITY_BATCH_MAX=16
  NEMOTRON_DENSITY_BATCH_WINDOW_MS=10
  NEMOTRON_DENSITY_BATCH_LONE_TIMEOUT_MS=0
  NEMOTRON_DENSITY_BATCH_MAX_QUEUE_DELAY_MS=2
  NEMOTRON_DENSITY_BATCH_MIN_FILL=1
  NEMOTRON_DENSITY_BATCH_QUEUE_CAPACITY=32
  "NEMOTRON_DENSITY_ADMISSION_ACTIVE_CAP=$CAP"
  "NEMOTRON_WS_LANES=$LANES"
  "NEMOTRON_WS_FINALIZE_RUNNERS=$FINALIZE_RUNNERS"
  "NEMOTRON_DENSITY_FINALIZE_RUNNERS=$FINALIZE_RUNNERS"
)
[ "${SHADOW:-0}" = 1 ] && ENV+=(NEMOTRON_WS_STEADY_SHADOW=1)   # diagnostic only (~2x work)

echo "[start] ws_server port=$PORT cap=$CAP lanes=$LANES finalize_runners=$FINALIZE_RUNNERS queue_capacity=32 bin=$BIN steady_batch_dir=$STEADY_BATCH_DIR"
echo "[start] cold AOTI load ~5-6min; ready when stdout prints 'ws_server listening on 127.0.0.1:$PORT'"
exec env "${ENV[@]}" "$BIN" \
  --port "$PORT" \
  --admission-active-cap "$CAP" \
  --steady-batch-dir "$STEADY_BATCH_DIR"
