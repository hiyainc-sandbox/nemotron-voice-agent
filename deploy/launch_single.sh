#!/usr/bin/env bash
# Single-process Nemotron ASR launcher for Phase 1 L40S deployments.
#
# Purpose:
#   Run exactly one production Python Nemotron streaming-ASR server process on this
#   host, without CUDA MPS and without an in-script restart supervisor. This is
#   the Phase 1 primary topology: one full L40S per process, fronted by a
#   least-connections load balancer. The launcher keeps the production server
#   tuning stack from deploy/launch_multiproc.sh:49-67, expressed only with
#   canonical NEMOTRON_* names so future operators can diff this file against the
#   K=3+MPS fallback cleanly.
#
# Usage:
#   # Normal path: invoked by deploy/nemotron-asr.service after bootstrap.
#   deploy/launch_single.sh
#
#   # Manual smoke run from a bootstrapped checkout:
#   NEMOTRON_APP_DIR="$HOME/nemotron" \
#   NEMOTRON_VENV="$HOME/nemo-venv" \
#   NEMOTRON_PORT=8080 \
#   deploy/launch_single.sh
#
# CLI flags:
#   None. This launcher accepts no arguments; server configuration is via the
#   environment variables below so systemd can own the runtime contract.
#
# Environment consumed by the launcher:
#   HOME
#     Default: inherited from the service/user environment.
#     Meaning: base directory for path defaults when NEMOTRON_APP_DIR,
#       NEMOTRON_VENV, or HF_HOME are unset.
#     Why: matches the validated bootstrap layout under the ubuntu user's home.
#   NEMOTRON_APP_DIR
#     Default: $HOME/nemotron.
#     Meaning: application checkout; the launcher cd's here before module launch.
#     Why: matches the runbook rsync target and editable install source.
#   NEMOTRON_VENV
#     Default: $HOME/nemo-venv.
#     Meaning: Python virtualenv containing torch, NeMo, and the editable package.
#     Why: ec2-bench/bootstrap.sh creates this venv; the runbook runs
#       "uv pip install -e" there. This launcher assumes that install exists.
#   NEMOTRON_MODEL
#     Default: nvidia/nemotron-speech-streaming-en-0.6b.
#     Meaning: model id/path passed to nemotron_speech.server.
#     Why: production checkpoint validated for this deployment plan.
#   NEMOTRON_PORT
#     Default: 8080.
#     Meaning: TCP/WebSocket listen port for this single server process.
#     Why: Phase 1 exposes one backend per box on :8080.
#   NEMOTRON_RIGHT_CONTEXT
#     Default: 1.
#     Meaning: server right-context argument.
#     Why: keeps parity with the validated launch_multiproc.sh invocation.
#   HF_HOME
#     Default: $HOME/hf.
#     Meaning: Hugging Face cache root inherited by the Python process.
#     Why: keeps model artifacts outside the repo and matches bootstrap/runbook.
#   NEMOTRON_ADMISSION_MAX_BACKLOG
#     Default: 12.
#     Meaning: server-side admission cap; excess WebSocket attempts are closed
#       by the server with 1013.
#     Why: protects admitted latency under overload; HAProxy mode tcp cannot read
#       WebSocket close codes, so backpressure lives in server.py.
#
# Production NEMOTRON_* server env set by this launcher, with defaults:
#   Each entry can be overridden by setting the same canonical NEMOTRON_* name;
#   no unprefixed aliases are read.
#   NEMOTRON_CONTINUOUS=1
#     Streaming/continuous mode for the production WebSocket server.
#   NEMOTRON_FINALIZE_SILENCE_MS=0
#     Finalize immediately at endpoint; validated low-latency setting.
#   NEMOTRON_WARMUP_MS=200
#     Short warmup window before serving; preserves validated startup behavior.
#   NEMOTRON_SCHEDULER_B1=1
#     Scheduler B=1 mode used by the measured production path.
#   NEMOTRON_BATCH_SCHED=1
#     Enables batch scheduler path.
#   NEMOTRON_BATCH_MAX_SIZE=32
#     Batch-size ceiling from the production SRV_ENV stack.
#   NEMOTRON_BATCH_MAX_WAIT_MS=8
#     Batch wait budget from the production SRV_ENV stack.
#   NEMOTRON_MODEL_LANES=2
#     Per-process model lanes; >2 regressed in validation, 1 wastes overhead.
#   NEMOTRON_BATCH_BARRIER_DRAIN=1
#     Enables barrier drain behavior from the measured production stack.
#   NEMOTRON_BATCH_FINALIZE=1
#     Keeps finalize work on the batched path.
#   NEMOTRON_ENCODER_CUDAGRAPH=1
#     Enables steady encoder CUDA graphs.
#   NEMOTRON_ENCODER_CUDAGRAPH_MAX_B=8
#     Steady graph max batch from the production stack.
#   NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE=1
#     Enables finalize encoder CUDA graph buckets.
#   NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_PADDED=1
#     Uses the padded T_max finalize bucket to reduce graph-pool memory.
#   NEMOTRON_SYNC_COMPRESS=1
#     Drops measured redundant telemetry pre-syncs.
#   NEMOTRON_FINALIZE_PRIORITY=1
#     Prioritizes finalize lane work to bound cross-session tail latency.
#
# MPS and multi-process absence:
#   This launcher intentionally has no MPS daemon, CUDA_MPS_* exports, K
#   auto-selection, K loop, log redirect, SIGTERM trap, pkill, or shell
#   supervisor loop. The 2026-05-27 L40S lead showed one full-GPU process holding
#   roughly the same ~16-20 streams/box as K=3+MPS while avoiding the ~190ms
#   median-latency MPS tax. If sustained multi-turn load later proves that the
#   single asyncio intake path is the bottleneck, use deploy/launch_multiproc.sh
#   as the K=3+MPS fallback.
#
# Signals and drain:
#   The final command uses exec, so the launcher PID becomes python. A systemd
#   SIGTERM therefore kills/signals the server process directly. This is NOT a
#   graceful WebSocket drain contract: operators MUST run deploy/drain.sh first,
#   wait for active sessions to empty, and only then restart/stop the service.
#
# Exit codes / failure modes:
#   0      Python server exited normally; unusual for a long-running service.
#   64     Bad launcher invocation, or HOME is required for defaults but unset.
#   72     NEMOTRON_APP_DIR could not be entered.
#   126    "$NEMOTRON_VENV/bin/python" exists but cannot be executed.
#   127    "$NEMOTRON_VENV/bin/python" was not found.
#   other  Inherited from python/nemotron_speech.server after exec.
#
# Further documentation:
#   Runbook/how: deploy/RUNBOOK.md
#   Rationale/why: deploy/DEPLOYMENT.md

set -uo pipefail

if [[ "$#" -ne 0 ]]; then
  printf '[launch_single] no arguments are supported; configure via environment\n' >&2
  exit 64
fi

DEFAULT_HOME="${HOME:-}"
if { [[ -z "${NEMOTRON_APP_DIR:-}" ]] || [[ -z "${NEMOTRON_VENV:-}" ]] || [[ -z "${HF_HOME:-}" ]]; } &&
  [[ -z "$DEFAULT_HOME" ]]; then
  printf '[launch_single] HOME is unset; set NEMOTRON_APP_DIR, NEMOTRON_VENV, and HF_HOME explicitly\n' >&2
  exit 64
fi

APP_DIR="${NEMOTRON_APP_DIR:-$DEFAULT_HOME/nemotron}"
VENV="${NEMOTRON_VENV:-$DEFAULT_HOME/nemo-venv}"
MODEL="${NEMOTRON_MODEL:-nvidia/nemotron-speech-streaming-en-0.6b}"
export HF_HOME="${HF_HOME:-$DEFAULT_HOME/hf}"

SRV_ENV=(
  "NEMOTRON_CONTINUOUS=${NEMOTRON_CONTINUOUS:-1}"
  "NEMOTRON_FINALIZE_SILENCE_MS=${NEMOTRON_FINALIZE_SILENCE_MS:-0}"
  "NEMOTRON_WARMUP_MS=${NEMOTRON_WARMUP_MS:-200}"
  "NEMOTRON_SCHEDULER_B1=${NEMOTRON_SCHEDULER_B1:-1}"
  "NEMOTRON_BATCH_SCHED=${NEMOTRON_BATCH_SCHED:-1}"
  "NEMOTRON_BATCH_MAX_SIZE=${NEMOTRON_BATCH_MAX_SIZE:-32}"
  "NEMOTRON_BATCH_MAX_WAIT_MS=${NEMOTRON_BATCH_MAX_WAIT_MS:-8}"
  "NEMOTRON_MODEL_LANES=${NEMOTRON_MODEL_LANES:-2}"
  "NEMOTRON_BATCH_BARRIER_DRAIN=${NEMOTRON_BATCH_BARRIER_DRAIN:-1}"
  "NEMOTRON_BATCH_FINALIZE=${NEMOTRON_BATCH_FINALIZE:-1}"
  "NEMOTRON_ENCODER_CUDAGRAPH=${NEMOTRON_ENCODER_CUDAGRAPH:-1}"
  "NEMOTRON_ENCODER_CUDAGRAPH_MAX_B=${NEMOTRON_ENCODER_CUDAGRAPH_MAX_B:-8}"
  "NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE=${NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE:-1}"
  "NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_PADDED=${NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_PADDED:-1}"
  "NEMOTRON_SYNC_COMPRESS=${NEMOTRON_SYNC_COMPRESS:-1}"
  "NEMOTRON_FINALIZE_PRIORITY=${NEMOTRON_FINALIZE_PRIORITY:-1}"
  "NEMOTRON_ADMISSION_MAX_BACKLOG=${NEMOTRON_ADMISSION_MAX_BACKLOG:-12}"
)

if ! cd "$APP_DIR"; then
  printf '[launch_single] cannot cd to NEMOTRON_APP_DIR=%s\n' "$APP_DIR" >&2
  exit 72
fi

exec env -u LD_LIBRARY_PATH "${SRV_ENV[@]}" "$VENV/bin/python" -m nemotron_speech.server \
  --host 0.0.0.0 \
  --port "${NEMOTRON_PORT:-8080}" \
  --right-context "${NEMOTRON_RIGHT_CONTEXT:-1}" \
  --model "$MODEL"
