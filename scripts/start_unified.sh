#!/bin/bash
# Start services in the unified container with flexible service selection
#
# This script manages the ASR and LLM services in a single container.
# Any combination can be enabled/disabled. ASR starts first so it claims GPU memory
# before the LLM. (TTS is on-device Pocket TTS, run separately: `uvx pocket-tts serve`.)
#
# Service Selection:
#   ENABLE_LLM   - "true" (default) or "false" - Enable LLM server
#   ENABLE_ASR   - "true" (default) or "false" - Enable ASR server
#
# LLM Configuration:
#   LLM_MODE                      - "llamacpp-q8" (default), "llamacpp-q4", or "vllm"
#   LLAMA_MODEL                   - Path to GGUF model (required for llamacpp modes)
#   LLAMA_PARALLEL                - Number of parallel slots (default: 1 for buffered LLM mode)
#   LLAMA_CTX_SIZE                - Context size (default: 16384, enough for multi-turn voice)
#   LLAMA_REASONING_BUDGET        - Thinking mode for Q4: 0=disabled, -1=unlimited (default: 0)
#   VLLM_MODEL                    - Path to HF model dir (required for vllm mode)
#   VLLM_GPU_MEMORY_UTILIZATION   - GPU memory fraction (default: 0.60)
#
# General:
#   SERVICE_TIMEOUT               - Seconds to wait for each service to start (default: 60)
#   HUGGINGFACE_ACCESS_TOKEN      - HuggingFace token for gated models
#
# Logs are written to /var/log/nemotron/{asr,llm}.log for external access.
#
# Examples:
#   # Default: all services with llama.cpp Q8
#   LLAMA_MODEL=/path/to/Q8.gguf bash scripts/start_unified.sh
#
#   # llama.cpp Q4 mode
#   LLM_MODE=llamacpp-q4 LLAMA_MODEL=/path/to/Q4.gguf bash scripts/start_unified.sh
#
#   # vLLM mode
#   LLM_MODE=vllm VLLM_MODEL=/path/to/model bash scripts/start_unified.sh
#
#   # LLM only (no ASR)
#   ENABLE_ASR=false LLAMA_MODEL=/path/to/model.gguf bash scripts/start_unified.sh
#
#   # ASR only (no LLM)
#   ENABLE_LLM=false bash scripts/start_unified.sh

set -e

# =============================================================================
# Configuration with defaults
# =============================================================================
ENABLE_LLM="${ENABLE_LLM:-true}"
ENABLE_ASR="${ENABLE_ASR:-true}"
LLM_MODE="${LLM_MODE:-llamacpp-q8}"
# vLLM needs ~15 minutes to load the model, llama.cpp only needs ~60s
if [[ "$LLM_MODE" == "vllm" ]]; then
    SERVICE_TIMEOUT="${SERVICE_TIMEOUT:-900}"
else
    SERVICE_TIMEOUT="${SERVICE_TIMEOUT:-60}"
fi
LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"
LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE:-16384}"
LLAMA_REASONING_BUDGET="${LLAMA_REASONING_BUDGET:-0}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.60}"

# Log directory
LOG_DIR="/var/log/nemotron"
mkdir -p "$LOG_DIR"

# =============================================================================
# Display configuration
# =============================================================================
echo "============================================"
echo "Starting Nemotron Unified Container"
echo "============================================"
echo "  Date: $(date)"
echo ""
echo "  Services:"
echo "    ASR: $([ "$ENABLE_ASR" = "true" ] && echo "ENABLED (port 8080)" || echo "DISABLED")"
echo "    LLM: $([ "$ENABLE_LLM" = "true" ] && echo "ENABLED (port 8000, mode: $LLM_MODE)" || echo "DISABLED")"
echo ""
echo "  Logs: $LOG_DIR/{asr,llm}.log"
echo "  (TTS is on-device Pocket TTS, run separately: uvx pocket-tts serve --port 8001)"
echo "============================================"

# Check if at least one service is enabled
if [ "$ENABLE_LLM" != "true" ] && [ "$ENABLE_ASR" != "true" ]; then
    echo "ERROR: At least one service must be enabled"
    echo "  Set ENABLE_LLM=true or ENABLE_ASR=true"
    exit 1
fi

# =============================================================================
# Validate LLM configuration if enabled
# =============================================================================
if [ "$ENABLE_LLM" = "true" ]; then
    case "$LLM_MODE" in
        llamacpp-q8|llamacpp-q4)
            if [ -z "$LLAMA_MODEL" ]; then
                echo "ERROR: LLAMA_MODEL must be set for $LLM_MODE mode"
                echo "Example: LLAMA_MODEL=/root/.cache/huggingface/hub/models--unsloth--Nemotron-3-Nano-30B-A3B-GGUF/snapshots/.../Q8_0.gguf"
                exit 1
            fi
            if [ ! -f "$LLAMA_MODEL" ]; then
                echo "WARNING: LLAMA_MODEL file may not exist: $LLAMA_MODEL"
            fi
            echo "  llama.cpp Model: $LLAMA_MODEL"
            echo "  llama.cpp Parallel: $LLAMA_PARALLEL"
            echo "  llama.cpp Context: $LLAMA_CTX_SIZE"
            ;;
        vllm)
            if [ -z "$VLLM_MODEL" ]; then
                echo "ERROR: VLLM_MODEL must be set for vllm mode"
                echo "Example: VLLM_MODEL=nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
                exit 1
            fi
            echo "  vLLM Model: $VLLM_MODEL"
            echo "  vLLM GPU Utilization: $VLLM_GPU_MEMORY_UTILIZATION"
            ;;
        *)
            echo "ERROR: Unknown LLM_MODE: $LLM_MODE"
            echo "  Valid modes: llamacpp-q8, llamacpp-q4, vllm"
            exit 1
            ;;
    esac
fi
echo "============================================"

# =============================================================================
# Track PIDs for cleanup
# =============================================================================
ASR_PID=""
LLM_PID=""

# =============================================================================
# Health check functions
# =============================================================================

# HTTP health check - for services with a /health endpoint (LLM)
wait_for_http_health() {
    local name=$1
    local url=$2
    local timeout=$3
    local pid=$4

    echo -n "  Waiting for $name to be ready"
    for i in $(seq 1 $timeout); do
        # Check if process is still running
        if ! kill -0 $pid 2>/dev/null; then
            echo " FAILED"
            echo "ERROR: $name process exited unexpectedly"
            echo "Check logs: $LOG_DIR/$(echo $name | tr '[:upper:]' '[:lower:]').log"
            return 1
        fi

        # Try HTTP health check
        if curl -sf "$url" >/dev/null 2>&1; then
            echo " ready (${i}s)"
            return 0
        fi

        echo -n "."
        sleep 1
    done

    echo " TIMEOUT"
    echo "ERROR: $name failed to respond within ${timeout}s"
    echo "Check logs: $LOG_DIR/$(echo $name | tr '[:upper:]' '[:lower:]').log"
    return 1
}

# TCP port check - for WebSocket-only services (ASR)
# Waits for specific log message indicating warmup complete
wait_for_warmup_log() {
    local name=$1
    local log_file=$2
    local pattern=$3
    local timeout=$4
    local pid=$5

    echo -n "  Waiting for $name to be ready"
    for i in $(seq 1 $timeout); do
        # Check if process is still running
        if ! kill -0 $pid 2>/dev/null; then
            echo " FAILED"
            echo "ERROR: $name process exited unexpectedly"
            echo "Check logs: $log_file"
            return 1
        fi

        # Check for warmup complete message in logs
        if grep -q "$pattern" "$log_file" 2>/dev/null; then
            echo " ready (${i}s)"
            return 0
        fi

        echo -n "."
        sleep 1
    done

    echo " TIMEOUT"
    echo "ERROR: $name failed to start within ${timeout}s"
    echo "Check logs: $log_file"
    return 1
}

# =============================================================================
# Graceful shutdown handler
# =============================================================================
cleanup() {
    echo ""
    echo "============================================"
    echo "Shutting down services..."
    echo "============================================"

    # Send SIGTERM to all processes
    [ -n "$ASR_PID" ] && kill -TERM $ASR_PID 2>/dev/null && echo "  Stopping ASR (PID $ASR_PID)..."
    [ -n "$LLM_PID" ] && kill -TERM $LLM_PID 2>/dev/null && echo "  Stopping LLM (PID $LLM_PID)..."

    # Wait for processes to terminate (with timeout)
    for i in 1 2 3 4 5; do
        all_stopped=true
        [ -n "$ASR_PID" ] && kill -0 $ASR_PID 2>/dev/null && all_stopped=false
        [ -n "$LLM_PID" ] && kill -0 $LLM_PID 2>/dev/null && all_stopped=false
        if $all_stopped; then
            break
        fi
        sleep 1
    done

    # Force kill if still running
    [ -n "$ASR_PID" ] && kill -9 $ASR_PID 2>/dev/null || true
    [ -n "$LLM_PID" ] && kill -9 $LLM_PID 2>/dev/null || true

    echo "All services stopped."
    echo "============================================"
}

trap cleanup EXIT INT TERM

# =============================================================================
# Start services
# =============================================================================
STEP=0
TOTAL_STEPS=0
[ "$ENABLE_ASR" = "true" ] && TOTAL_STEPS=$((TOTAL_STEPS + 1))
[ "$ENABLE_LLM" = "true" ] && TOTAL_STEPS=$((TOTAL_STEPS + 1))

# Start ASR server (Nemotron streaming ASR via NeMo)
if [ "$ENABLE_ASR" = "true" ]; then
    STEP=$((STEP + 1))
    echo "[$STEP/$TOTAL_STEPS] Starting ASR server on port 8080..."
    python -m nemotron_speech.server --port 8080 > "$LOG_DIR/asr.log" 2>&1 &
    ASR_PID=$!
    echo "  ASR started (PID $ASR_PID, log: $LOG_DIR/asr.log)"
fi

# Wait for ASR to be ready (claims its GPU memory first) before starting LLM
if [ "$ENABLE_LLM" = "true" ]; then
    if [ -n "$ASR_PID" ]; then
        # ASR is WebSocket-only, so we wait for the warmup log message
        if ! wait_for_warmup_log "ASR" "$LOG_DIR/asr.log" "GPU memory claimed" "$SERVICE_TIMEOUT" "$ASR_PID"; then
            exit 1
        fi
        echo ""
    fi
fi

# Start LLM server based on mode
if [ "$ENABLE_LLM" = "true" ]; then
    STEP=$((STEP + 1))
    echo "[$STEP/$TOTAL_STEPS] Starting LLM server on port 8000..."

    case "$LLM_MODE" in
        llamacpp-q8)
            echo "  Mode: llama.cpp Q8 (best quality/VRAM balance)"
            llama-server \
                -m "${LLAMA_MODEL}" \
                --host 0.0.0.0 \
                --port 8000 \
                --n-gpu-layers 99 \
                --ctx-size "${LLAMA_CTX_SIZE}" \
                --flash-attn on \
                --parallel "${LLAMA_PARALLEL}" \
                > "$LOG_DIR/llm.log" 2>&1 &
            LLM_PID=$!
            ;;
        llamacpp-q4)
            echo "  Mode: llama.cpp Q4 (optimized for 32GB GPU)"
            # Q4 uses same context as Q8 (default 16384), all layers on GPU, quantized KV cache
            LLAMA_Q4_CTX_SIZE="${LLAMA_CTX_SIZE:-16384}"
            llama-server \
                -m "${LLAMA_MODEL}" \
                --host 0.0.0.0 \
                --port 8000 \
                --n-gpu-layers 99 \
                --ctx-size "${LLAMA_Q4_CTX_SIZE}" \
                --flash-attn on \
                --parallel "${LLAMA_PARALLEL}" \
                --cache-ram 0 \
                --reasoning-budget "${LLAMA_REASONING_BUDGET}" \
                -ctk q8_0 \
                -ctv q8_0 \
                > "$LOG_DIR/llm.log" 2>&1 &
            LLM_PID=$!
            ;;
        vllm)
            echo "  Mode: vLLM (BF16 full-precision inference)"
            python -m vllm.entrypoints.openai.api_server \
                --model "${VLLM_MODEL}" \
                --host 0.0.0.0 \
                --port 8000 \
                --dtype bfloat16 \
                --trust-remote-code \
                --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
                --max-num-seqs 1 \
                --max-model-len 100000 \
                --enforce-eager \
                --disable-log-requests \
                --enable-prefix-caching \
                > "$LOG_DIR/llm.log" 2>&1 &
            LLM_PID=$!
            ;;
    esac
    echo "  LLM started (PID $LLM_PID, log: $LOG_DIR/llm.log)"

    # Wait for LLM to be ready (both llama.cpp and vLLM have /health endpoint)
    echo ""
    if ! wait_for_http_health "LLM" "http://localhost:8000/health" "$SERVICE_TIMEOUT" "$LLM_PID"; then
        exit 1
    fi
fi

# =============================================================================
# Report startup status
# =============================================================================
echo ""
echo "============================================"
echo "All services started successfully!"
echo "============================================"
[ -n "$ASR_PID" ] && echo "  ASR: http://localhost:8080 (WebSocket) - PID $ASR_PID"
[ -n "$LLM_PID" ] && echo "  LLM: http://localhost:8000 (HTTP API) - PID $LLM_PID"
echo ""
echo "  Logs: $LOG_DIR/{asr,llm}.log"
echo "============================================"
echo ""
echo "Container is ready. Press Ctrl+C to stop all services."
echo ""

# =============================================================================
# Wait for any process to exit
# =============================================================================
# Build list of PIDs to wait for
PIDS=""
[ -n "$ASR_PID" ] && PIDS="$PIDS $ASR_PID"
[ -n "$LLM_PID" ] && PIDS="$PIDS $LLM_PID"

# Wait for the first process to exit
wait -n $PIDS
EXIT_CODE=$?

echo ""
echo "============================================"
echo "A service exited unexpectedly (code $EXIT_CODE)"
echo "============================================"

# Check which process died
[ -n "$ASR_PID" ] && ! kill -0 $ASR_PID 2>/dev/null && echo "  ASR server exited - check $LOG_DIR/asr.log"
[ -n "$LLM_PID" ] && ! kill -0 $LLM_PID 2>/dev/null && echo "  LLM server exited - check $LOG_DIR/llm.log"

exit $EXIT_CODE
