#!/usr/bin/env bash
# Drop into the CUDA 12.8 devel container (glibc 2.39 -> nvcc works) with the repo + HF model cache mounted, GPU visible.
# Usage: ./enter.sh [command...]   (no args = interactive bash)
set -euo pipefail
IMG="${NEMOTRON_CUDA_IMG:-nemotron-aoti:cu128}"   # built via Dockerfile (torch 2.8.0 baked); falls back below if absent
docker image inspect "$IMG" >/dev/null 2>&1 || IMG="nvidia/cuda:12.8.1-devel-ubuntu24.04"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"   # repo root (container/ -> runtime -> repo)
HF="${HF_HOME:-$HOME/.cache/huggingface}"
TI=""; [ -t 0 ] && [ -t 1 ] && TI="-it"   # interactive only when attached to a TTY; scriptable (no -it) otherwise
AWSV=""; [ -d "$HOME/.aws" ] && AWSV="-v $HOME/.aws:/root/.aws:ro"   # aws config/creds for export_model.sh S3 mode
exec docker run --rm $TI --gpus all \
  -v "$REPO":/work -w /work/runtime \
  -v "$HF":/root/.cache/huggingface \
  $AWSV \
  -e HF_HUB_OFFLINE=1 \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN \
  -e AWS_DEFAULT_REGION -e AWS_REGION -e AWS_PROFILE \
  "$IMG" "${@:-bash}"
