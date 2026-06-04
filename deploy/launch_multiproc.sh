#!/usr/bin/env bash
# Production multi-process launcher  —  DESIGN ARTIFACT for Step 2 of proj-2026-05-21-inference-opt/PLAN.md.
#
# Starts CUDA MPS + K Nemotron ASR server processes (each NEMOTRON_MODEL_LANES=2) on one box, with
# crash-restart supervision. K + config come from the per-GPU matrix (see deploy/DEPLOYMENT.md), which the
# benchmarks established: L4 -> K~2 (~32/box). L40S -> K=3 (~48/box) WITH the finalize encoder graph on (the
# default 246/279 latency win): each proc is ~11 GB (model + 2-lane STEADY + FINALIZE graph pools), so K=4 OOMs
# the 44 GB L40S (4x11~=44 GB; measured 2026-05-23). L40S is GPU-COMPUTE-capable of K~4/~64 but MEMORY-bound to
# K=3 with the finalize graph. lanes=2/process is the unit.
#
# This is the REFERENCE launcher to adapt to your substrate (systemd template / container entrypoint / ECS
# task). Substrate-dependent + production-hardening items are marked TODO and discussed in DEPLOYMENT.md.
set -uo pipefail

APP_DIR="${NEMOTRON_APP_DIR:-$HOME/nemotron}"          # holds server.py + batch_primitives.py
VENV="${NEMOTRON_VENV:-$HOME/nemo-venv}"
MODEL="${NEMOTRON_MODEL:-nvidia/nemotron-speech-streaming-en-0.6b}"
# Step 4: per-GPU config matrix + guarded auto-select (override with NEMOTRON_PROCS). Measured matrix:
#   L4 -> K=2 (~7/box, keep-up-bound — NOT GPU-compute-bound; encoder is mem-BW-bound, ~3x worse than L40S);
#   L40S -> K=3 (~48 GPU-fit but in-budget ~20/box, keep-up-bound) — MEMORY-bound to K=3: each proc ~11 GB with the
#   FINALIZE encoder graph on (default), so K=4 OOMs the 44 GB L40S (measured 2026-05-23). To run K=4, shrink the
#   per-proc graph pool with the padded-T_max bucket FINALIZE_PADDED=1 (preferred; ~19x less finalize pool) and
#   re-verify it fits — auto-select stays K=3 until that cloud no-OOM check lands (proj-2026-05-24-0859 Step 6).
auto_pick_K(){ local g; g=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  case "$g" in
    *L40S*) echo 3 ;;                                   # MUST precede *L4* (the glob *L4* also matches "L40S").
                                                        # memory-bound by the finalize graph (was 4 pre-finalize-graph)
    *L4*)   echo 2 ;;
    *)      echo 2 ;;                                   # unknown GPU: conservative — measure with ec2-bench first
  esac; }
K="${NEMOTRON_PROCS:-$(auto_pick_K)}"                   # processes/box (auto from GPU+vCPU; see DEPLOYMENT.md)
LANES="${NEMOTRON_MODEL_LANES:-2}"                      # within-process sweet spot (>2 regresses; GIL)
BASE_PORT="${NEMOTRON_BASE_PORT:-8080}"
export HF_HOME="${HF_HOME:-$HOME/hf}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}"

# silence0_warm200 + batching + barrier-drain + encoder CUDA graphs (STEADY + FINALIZE). The finalize encoder
# graph is the byte-exact 274/401 -> 246/279 TTFS win (frontier-competitive); it REQUIRES the steady graph (it
# warms its finalize buckets into the steady graph managers). The scheduler event-loop-starvation livelock fix is
# unconditional in server.py (ships automatically). MEMORY: the finalize graph adds ~2-3 GB graph-pool/proc
# (16 per-T buckets, B=1 x T=42..60). To shrink it (L4/24GB K=2, or L40S K=4) the PREFERRED lever is the
# padded-T_max bucket FINALIZE_PADDED=1 below (one B=1 x T_max bucket, ~19x less finalize pool, full T coverage,
# byte-exact); the per-T T_MAX/_MAX_B trim is the fallback. See DEPLOYMENT.md.
# proj-2026-05-24-0859 TAIL levers — SHIPPED ON by default (byte-exact, cloud-proven 2026-05-24; toggle off via env
# for A/B, e.g. SYNC_COMPRESS=0): PADDED = single padded-T_max finalize bucket (memory/L4-fit + K=3 headroom, NOT a
# density lever — L40S stays ~16-20/box regardless of K); SYNC_COMPRESS = drop 2 telemetry pre-syncs;
# FINALIZE_PRIORITY = cross-session priority finalize-lane (bounds the finalize lane-HOL tail).
SRV_ENV=(NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200
         NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1 NEMOTRON_BATCH_MAX_SIZE=32 NEMOTRON_BATCH_MAX_WAIT_MS=8
         "NEMOTRON_MODEL_LANES=$LANES" NEMOTRON_BATCH_BARRIER_DRAIN=1 NEMOTRON_BATCH_FINALIZE=1
         NEMOTRON_ENCODER_CUDAGRAPH=1 NEMOTRON_ENCODER_CUDAGRAPH_MAX_B=8 NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE=1
         "NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_PADDED=${FINALIZE_PADDED:-1}"
         "NEMOTRON_SYNC_COMPRESS=${SYNC_COMPRESS:-1}"
         "NEMOTRON_FINALIZE_PRIORITY=${FINALIZE_PRIORITY:-1}")
# Optional measurement passthroughs (no effect unless set).
[ "${FINALIZE_PROFILE:-0}" = 1 ] && SRV_ENV+=(NEMOTRON_FINALIZE_PROFILE=1)   # adds ~2x intake tax on L40S — measurement only
[ "${FAULTHANDLER:-0}" = 1 ] && SRV_ENV+=(NEMOTRON_FAULTHANDLER=1)
# Admission/backpressure — OPT-IN (client-facing: rejects past the cap with WS-close 1013, so the LB must DRAIN on 1013).
# RECOMMENDED prod cap NEMOTRON_ADMISSION_MAX_BACKLOG~8-12 (the backlog-count signal — protects admitted p50+p95 under
# overload; the *_READY_AGE_MS age signal does NOT track this intake-bound overload, confirmed 2026-05-24).
[ -n "${ADMISSION_MAX_BACKLOG:-}" ] && SRV_ENV+=("NEMOTRON_ADMISSION_MAX_BACKLOG=$ADMISSION_MAX_BACKLOG")
[ -n "${ADMISSION_MAX_READY_AGE_MS:-}" ] && SRV_ENV+=("NEMOTRON_ADMISSION_MAX_READY_AGE_MS=$ADMISSION_MAX_READY_AGE_MS")
# Finalize-graph T-range trim (the documented L4/24GB fit lever — shrinks the finalize graph pool; finalize calls
# with T outside [MIN,MAX] fail-closed to eager). Observed finalize T is 43-58 (flat); default is 42-60.
[ -n "${FINALIZE_T_MIN:-}" ] && SRV_ENV+=("NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_T_MIN=$FINALIZE_T_MIN")
[ -n "${FINALIZE_T_MAX:-}" ] && SRV_ENV+=("NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_T_MAX=$FINALIZE_T_MAX")

cd "$APP_DIR"

start_mps(){ mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"; nvidia-cuda-mps-control -d && echo "[mps] daemon up"; }
stop_mps(){ echo quit | nvidia-cuda-mps-control 2>/dev/null || true; }
trap 'echo "[launcher] SIGTERM — stopping"; pkill -f "server.py --model" 2>/dev/null; stop_mps; exit 0' INT TERM

declare -A PIDS
launch(){ local k=$1; local port=$((BASE_PORT+k))
  env -u LD_LIBRARY_PATH "${SRV_ENV[@]}" "$VENV/bin/python" server.py --model "$MODEL" \
      --host 0.0.0.0 --port "$port" --right-context 1 > "server_$k.log" 2>&1 &
  PIDS[$k]=$!; echo "[launch] proc $k -> pid ${PIDS[$k]} port $port"; }

start_mps                                              # NOTE: unset LD_LIBRARY_PATH for torch's bundled cuDNN
for k in $(seq 0 $((K-1))); do launch "$k"; done

# Supervisor: restart a crashed process. CRITICAL with MPS — a CUDA fault in one client can corrupt the shared
# context and take down the others (blast-radius; see DEPLOYMENT.md "MPS hardening"). For correctness, the LB
# should DRAIN the dead backend before kill and re-add it only after /health passes.
while true; do
  sleep 10
  for k in $(seq 0 $((K-1))); do
    if ! kill -0 "${PIDS[$k]}" 2>/dev/null; then
      echo "[supervisor] proc $k (pid ${PIDS[$k]}) DIED — restarting"   # TODO: drain LB backend + alert/metric
      # TODO: if MPS context is corrupt (multiple procs died together), restart MPS + all procs.
      launch "$k"
    fi
  done
done
