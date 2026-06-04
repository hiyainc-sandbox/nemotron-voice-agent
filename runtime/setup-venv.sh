#!/bin/bash
# Bootstrap script for the native runtime export + oracle venv (uv-native).
#
# Creates ./.venv at this directory and installs the deps declared in
# pyproject.toml + locked in uv.lock. Verifies torch+cu128, NeMo, CUDA.
#
# Prerequisites:
#   - uv (auto-installed if missing).
#   - Python 3.12.x — pinned in .python-version (uv can bootstrap it).
#   - CUDA 12.x on host (for torch+cu128 to load at runtime).
#   - NVIDIA driver compatible with CUDA 12.8.
#   - ~10 GiB disk for the venv (NeMo + torch + transformers + bitsandbytes +
#     pynini + janome + faiss + nvidia/* CUDA wheels).
#
# Usage (from this directory):
#   bash setup-venv.sh
#
# After setup, scripts run as (from this runtime/ directory):
#   HF_HUB_OFFLINE=1 ./.venv/bin/python export_steady_batched.py --out ./artifacts
#
# Or from the repo root as:
#   HF_HUB_OFFLINE=1 runtime/.venv/bin/python \
#       runtime/export_steady_batched.py --out ...
#
# Why this venv lives next to runtime/: the export scripts + finalize_ref.py
# (Python oracle) reference it; co-locating keeps the dependency boundary
# clear. The repo's top-level pyproject.toml + uv.lock are for the production
# WebSocket server (a much smaller dep set: pipecat-ai, nvidia-riva-client,
# websockets, aiohttp, numpy, loguru) — kept separate intentionally to avoid
# bloating that lockfile with NeMo+torch+CUDA wheels.
#
# Lockfile model: uv-native (pyproject.toml + uv.lock). To regenerate the
# lockfile after editing pyproject.toml: `uv lock`. To install exactly what's
# in the lockfile: `uv sync` (what this script does).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Nemotron native-runtime export venv setup (uv-native) ==="
echo "  project dir: $SCRIPT_DIR"
echo "  venv path:   .venv"
echo "  python:      $(cat .python-version 2>/dev/null || echo "(no .python-version)")"
echo "  uv source:   pyproject.toml + uv.lock"
echo

# Ensure uv is available.
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"
echo

# uv sync: creates .venv if missing, installs Python if needed, resolves +
# downloads + extracts wheels per uv.lock. Idempotent. ~10 min on a cold
# machine, ~10 sec on a warm uv cache.
echo "Running 'uv sync' (this can take 5-20 min on a cold machine; near-instant on a warm uv cache)..."
uv sync
echo

# Verify install.
echo "Verifying install..."
.venv/bin/python - <<'PY'
import sys
assert sys.version_info[:2] == (3, 12), f"Expected Python 3.12, got {sys.version_info}"
print(f"  ✓ Python {sys.version.split()[0]}")

import torch
assert torch.__version__.endswith("+cu128"), f"Expected torch+cu128, got {torch.__version__}"
print(f"  ✓ torch {torch.__version__}")

import nemo
print(f"  ✓ nemo {nemo.__version__}")

# Soft CUDA check (no GPU needed for export-only work on a CPU dev host, but warn).
if torch.cuda.is_available():
    print(f"  ✓ CUDA {torch.version.cuda} available, device: {torch.cuda.get_device_name(0)}")
else:
    print(f"  ⚠ CUDA not available on this host (export-only ok; oracle work needs a GPU)")

# Quick NeMo ASR import check (the most common failure mode if NeMo deps drift).
try:
    import nemo.collections.asr as nemo_asr  # noqa: F401
    print(f"  ✓ nemo.collections.asr imports cleanly")
except Exception as e:
    print(f"  ✗ nemo.collections.asr import failed: {e}")
    sys.exit(1)
PY

echo
echo "=== Setup Complete ==="
echo
echo "Run scripts from this directory (runtime/) as:"
echo "  HF_HUB_OFFLINE=1 ./.venv/bin/python <script>"
echo
echo "Or from the repo root as:"
echo "  HF_HUB_OFFLINE=1 runtime/.venv/bin/python <script>"
