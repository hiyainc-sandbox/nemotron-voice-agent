# Runbook — start the C++ `ws_server` locally on the RTX 5090 (sm_120)

How to build and run the from-scratch C++ streaming-ASR WebSocket server (`ws_server`) on this
dev box. This is the exact recipe the compat oracle (`tests/server_compat/run_compat.py`) uses to
launch the C++ server — extracted into a standalone procedure. **The server runs on the HOST**
(not inside the container); only the *build* uses the container.

The L40S/sm_89 equivalent is `run_l40s_density.README.md` (different arch → AOTI artifacts must be
recompiled there; see that doc).

---

## 0. Prerequisites (one-time)

- **GPU**: RTX 5090 (sm_120). `nvidia-smi` should show the 5090.
- **Build container**: `nemotron-aoti:cu128` (torch 2.8.0+cu128, nvcc). Entered via
  `runtime/container/enter.sh` (mounts the repo, GPU, HF cache).
- **Host venv with torch libs** at `runtime/.venv/lib/python3.12/site-packages/torch/lib`
  (torch 2.8.0+cu128 — same ABI the binary links against). Created by `bash setup-venv.sh`.
- **Artifacts (sm_120)** under `runtime/`. These are **not committed** (large + arch-specific) —
  regenerate them once from the public HF checkpoint per [`ARTIFACTS.md`](ARTIFACTS.md). You need:
  - `artifacts/enc_steady_aoti.pt2` (steady encoder AOTI), `artifacts/enc_first.ts`,
    `artifacts/joint_step.ts`, `artifacts/predict_step.ts`, `artifacts/preproc.ts`,
    `artifacts/stripped_finalize_buckets/` (finalize bucket loaders).
  - `steady_b_artifacts_b16/enc_steady_aoti_b{1,2,4,8,16}.pt2` (the production-default
    BatchedSteadyScheduler bucket loaders). A `{1,2,4}`-only dir is not sufficient
    for the default server because default `B_max=16` requires bucket 16.

> All `*.pt2` here are compiled for **sm_120**. They are NOT portable to L40S/sm_89 — that needs a
> fresh AOTI compile (see `run_l40s_density.README.md`). The `*.ts` modules are TorchScript and are
> arch-independent.

---

## 1. Build (in the container)

```bash
cd runtime    # from the repo root
./container/enter.sh cmake --build cpp/build_step10 --target ws_server -j8
# Step-10 gate build:
./container/enter.sh cmake --build cpp/build_step10 --target ws_server density_main ws_framing_selftest -j8
```

To serve the **multilingual** model (`nvidia/nemotron-3.5-asr-streaming-0.6b`) instead, configure a
separate build tree with `-DNEMOTRON_PROFILE=ml` (one model per binary; see
`cpp/lib/session/model_constants.h` and `ARTIFACTS.md` §"Model profiles") and point it at
`ml`-profile artifacts (e.g. `artifacts_ml/`). The ml server accepts `?language=<locale>` (any key
of the model's prompt dictionary) or `?language=auto` per connection, and transcript events carry a
`language` field.

Produces `cpp/build_step10/ws_server`. (The oracle's default path is `cpp/build/ws_server`; this
runbook uses `build_step10`, the dir built in the scheduler-integration work.)

---

## 2. Start the server (on the host)

Copy-paste block (or just run `bash start_ws_server_local.sh` — see §6):

```bash
cd runtime    # from the repo root
RT="$PWD"
REPO="$(cd .. && pwd)"
TORCH_LIB="$RT/.venv/lib/python3.12/site-packages/torch/lib"

PORT=8081
CAP=64          # NEMOTRON_DENSITY_ADMISSION_ACTIVE_CAP — concurrent-session cap (= lane count K)
LANES=$CAP      # NEMOTRON_WS_LANES (defaults to CAP)
FINALIZE_RUNNERS=2

env \
  HF_HUB_OFFLINE=1 \
  PYTHONPATH="$REPO/src" \
  LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}" \
  NEMOTRON_CONTINUOUS=1 \
  NEMOTRON_FINALIZE_SILENCE_MS=0 \
  NEMOTRON_ARTIFACT_DIR="$RT/artifacts" \
  NEMOTRON_WS_SCHEDULER=1 \
  NEMOTRON_DENSITY_BATCH_STEADY=1 \
  NEMOTRON_DENSITY_BATCH_MAX=16 \
  NEMOTRON_DENSITY_BATCH_WINDOW_MS=10 \
  NEMOTRON_DENSITY_BATCH_LONE_TIMEOUT_MS=0 \
  NEMOTRON_DENSITY_BATCH_MAX_QUEUE_DELAY_MS=2 \
  NEMOTRON_DENSITY_BATCH_MIN_FILL=1 \
  NEMOTRON_DENSITY_BATCH_QUEUE_CAPACITY=32 \
  NEMOTRON_DENSITY_ADMISSION_ACTIVE_CAP=$CAP \
  NEMOTRON_WS_LANES=$LANES \
  NEMOTRON_WS_FINALIZE_RUNNERS=$FINALIZE_RUNNERS \
  NEMOTRON_DENSITY_FINALIZE_RUNNERS=$FINALIZE_RUNNERS \
  "$RT/cpp/build_step10/ws_server" \
    --port $PORT \
    --admission-active-cap $CAP \
    --steady-batch-dir "$RT/steady_b_artifacts_b16"
```

**Cold start is slow (~5–6 min):** the server eagerly loads the SharedRuntime AOTI packages
(`enc_first`, `enc_steady`, the scheduler bucket loaders, and the finalize buckets) + warms the
lanes. It is ready when stdout prints:

```
ws_server listening on 127.0.0.1:8081
```

### Key env / flags
| knob | meaning | default here |
|---|---|---|
| `--port` | listen port | 8081 |
| `--admission-active-cap` / `NEMOTRON_DENSITY_ADMISSION_ACTIVE_CAP` | max concurrent admitted sessions = **lane count K** | 64 |
| `NEMOTRON_WS_LANES` | warm InferenceLane pool size (= K) | = cap |
| `NEMOTRON_WS_FINALIZE_RUNNERS` | concurrent finalize runners on the shared loader pool | 2 |
| `NEMOTRON_WS_SCHEDULER` | route steady through the BatchedSteadyScheduler (the concurrency win) | 1 (on) |
| `--steady-batch-dir` | dir of `enc_steady_aoti_b{1,2,4,8,16}.pt2` scheduler buckets | `steady_b_artifacts_b16` |
| `NEMOTRON_ARTIFACT_DIR` | dir of `enc_steady_aoti.pt2`, `enc_first.ts`, `*.ts`, `stripped_finalize_buckets/` | `artifacts` |
| `NEMOTRON_FINALIZE_SILENCE_MS` | finalize debounce (silence0_warm200 = 0) | 0 |
| `NEMOTRON_DENSITY_BATCH_MAX` | scheduler B_max | 16 |
| `NEMOTRON_DENSITY_BATCH_WINDOW_MS` | scheduler batch window (kept at W=10; W=0 regressed) | 10 |
| `NEMOTRON_DENSITY_BATCH_MAX_QUEUE_DELAY_MS` | adaptive max queue delay | 2 |
| `NEMOTRON_DENSITY_BATCH_MIN_FILL` | adaptive min-fill policy | 1 |
| `NEMOTRON_DENSITY_BATCH_QUEUE_CAPACITY` | scheduler queue cap | 32 |
| `NEMOTRON_WS_BACKGROUND_WARMUP` | serve after sync lane warmup while the rest warms in background | 1 (on) |
| `NEMOTRON_WS_PREWARM` | startup page-cache prewarm for cold artifacts | 1 (on) |
| `NEMOTRON_WS_STEADY_CUDAGRAPH` | steady CUDA graph replay | off |
| `NEMOTRON_WS_STEADY_SHADOW=1` | (diagnostic) run inline+scheduler in isolation & compare per chunk — ~2× work, do NOT use for perf | off |

The queue-capacity default is 32: cap 16->32 removes the Wall-1 admission
limiter and moved the L40S SLO-robust knee from 88 to roughly 112-120
streams/box (~+30%) with byte-identical outputs. It is only an admission/in-flight
buffer size; cap 32->64 gave no further gain because the next limiter is the
single dispatch stream.

---

## 3. Verify it's up

```bash
curl -s http://127.0.0.1:8081/health        # -> ok
curl -s http://127.0.0.1:8081/stats          # session/finalize telemetry JSON
curl -s http://127.0.0.1:8081/scheduler_telemetry   # batched-steady scheduler counters
```

WebSocket endpoint (16 kHz mono s16le PCM, 640-byte/20ms frames):

```
ws://127.0.0.1:8081/?model=en
```

---

## 4. Smoke / correctness (optional, recommended)

Run the compat oracle, which launches its OWN Python + C++ servers and checks byte-exact parity
8/8 + the WS-overhead perf gate (so stop any server you started on 8081/8080 first):

```bash
# from the repo root
runtime/.venv/bin/python tests/server_compat/run_compat.py \
  --cpp-server runtime/cpp/build_step10/ws_server \
  --server-start-timeout-s 600
# Expect: "COMPAT PASS" + "PASS perf-gate ... overhead_ms=<~0 or negative>"
```

Or the in-binary smokes (in-container):

```bash
./container/enter.sh cpp/build_step10/density_main artifacts --mode runtime-smoke
./container/enter.sh cpp/build_step10/density_main artifacts --mode b2-t1 --correctness-rows 4
```

---

## 5. Stop

The server runs in the foreground of the launching shell — `Ctrl-C` for a clean shutdown (it
drains in-flight sessions). If backgrounded, `kill <pid>` (SIGINT/SIGTERM → graceful drain). Avoid
`pkill -f ws_server` patterns that can match your own shell command.

---

## 6. Helper script

`start_ws_server_local.sh` (next to this file) wraps §2 with env overrides:

```bash
PORT=8081 CAP=64 FINALIZE_RUNNERS=2 bash start_ws_server_local.sh
```

---

## Troubleshooting

- **`libcudart.so.* not found` / torch symbol errors**: `LD_LIBRARY_PATH` is not pointing at the
  venv `torch/lib`. Confirm `runtime/.venv/lib/python3.12/site-packages/torch/lib` exists.
- **Hangs ~5 min then "listening"**: normal cold AOTI load. Watch stdout; don't kill before the
  "listening on" line. `/health` 404/refused until then.
- **`--admission-active-cap is required`**: pass `--admission-active-cap` (or the env). Required.
- **Port in use**: change `--port`, or free 8081.
- **Artifacts missing**: ensure `artifacts/enc_steady_aoti.pt2` + `stripped_finalize_buckets/` and
  `steady_b_artifacts_b16/enc_steady_aoti_b{1,2,4,8,16}.pt2` exist (sm_120 builds). A
  `{1,2,4}`-only dir now fails closed with a bucket-16 error under the default B16 policy.
- **Wrong GPU arch**: these `.pt2` are sm_120. On any other GPU the AOTI load fails → recompile.
