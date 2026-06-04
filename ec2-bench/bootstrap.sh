#!/usr/bin/env bash
# Bootstrap the ASR runtime on a fresh DLAMI Base GPU box: standard torch + NeMo
# from the pinned git commit. L4 (Ada) -> standard CUDA torch (no Blackwell custom wheel).
set -euo pipefail
log(){ echo "[bootstrap $(date +%H:%M:%S)] $*"; }

NEMO_COMMIT=056d937544064df164b1751e9c8a1c3b597389fd
VENV=$HOME/nemo-venv
export HF_HOME=$HOME/hf
export HF_HUB_ENABLE_HF_TRANSFER=1

log "apt: git libsndfile1 ffmpeg"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git libsndfile1 ffmpeg >/dev/null

log "install uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

log "create venv (python ${PYVER:-3.11})"
uv venv --python "${PYVER:-3.11}" "$VENV"

log "pip: base runtime deps (incl. torch — unpinned, standard CUDA build)"
uv pip install --python "$VENV" \
  hf_transfer "huggingface_hub[hf-xet]==0.31.2" "numpy<2.0.0" torch \
  aiohttp loguru omegaconf Cython webdataset hydra-core websockets soundfile

log "pip: nemo_toolkit[asr] @ git ${NEMO_COMMIT:0:9} (the long build) ..."
uv pip install --python "$VENV" --no-cache \
  "nemo_toolkit[asr]@git+https://github.com/NVIDIA-NeMo/NeMo.git@${NEMO_COMMIT}"

log "pre-download checkpoint (public): nvidia/nemotron-speech-streaming-en-0.6b"
"$VENV/bin/huggingface-cli" download nvidia/nemotron-speech-streaming-en-0.6b >/dev/null 2>&1 \
  || log "WARN: hf pre-download failed (server.py will retry on load)"

log "smoke: import torch+nemo + GPU"
"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
import nemo, nemo.collections.asr  # noqa
print("nemo", getattr(nemo, "__version__", "?"), nemo.__file__)
PY
log "DONE"
