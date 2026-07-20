#!/usr/bin/env bash
# Generic on-box stage-2 recompile: produce the arch-specific serve artifacts for one
# GPU architecture and one model profile, from the arch-agnostic stage-1 exports.
# Generalizes run_l40s_density.sh (the sm_89/en recipe) over:
#
#   PROFILE=en|ml   model profile (see model_profile.py / runtime/ARTIFACTS.md):
#                     en  nvidia/nemotron-speech-streaming-en-0.6b   32 finalize buckets
#                     ml  nvidia/nemotron-3.5-asr-streaming-0.6b     64 finalize buckets
#   TARGET_CC       CUDA compute capability of THIS box (default: auto via nvidia-smi):
#                     8.6 -> A10G, 8.9 -> L40S/L4, 12.0 -> RTX 5090, ...
#                   Must run ON the target GPU — AOTI/Triton codegen targets the live
#                   device; you cannot cross-compile from a different arch.
#
# Produces into $ART_DIR (default artifacts_<profile>_sm<cc>/) the full serve set that
# scripts/backup_artifacts_s3.sh uploads as <arch>/:
#   enc_steady_aoti.pt2, enc_first_aoti.pt2, stripped_finalize_buckets/,
#   steady_b_artifacts_b16/enc_steady_aoti_b{1,2,4,8,16}.pt2 (+MANIFEST.json)
# plus the reused arch-agnostic *.ts modules downloaded alongside.
#
# Two ways to supply the stage-1 source EPs (see runtime/ARTIFACTS.md):
#   1. S3 (default): $S3_URI points at a base/ prefix laid out by
#      scripts/backup_artifacts_s3.sh; downloads are SHA-verified against
#      eps_manifest.json. Default derives from the profile's model id:
#      s3://audiointel-backups/STT/nvidia/<model>/base
#   2. EPS_LOCAL=1: export the EPs on any torch host, rsync them into $ART_DIR,
#      run with EPS_LOCAL=1. No bucket or credentials needed.
#
# Optional epilogue: RUN_DENSITY=1 builds density_main (-DNEMOTRON_PROFILE=$PROFILE)
# and runs the fresh-process-per-N density sweep, as run_l40s_density.sh did.
#
# Examples:
#   PROFILE=ml ./export_model.sh                     # ml serve set for this box's arch
#   PROFILE=en TARGET_CC=8.6 EPS_LOCAL=1 ./export_model.sh
#   PROFILE=ml RUN_DENSITY=1 ./export_model.sh       # also run the density sweep
set -euo pipefail
IFS=$'\n\t'

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT=$(pwd -P)

# --- model profile ------------------------------------------------------------
PROFILE=${PROFILE:-en}
case "$PROFILE" in
  en)
    MODEL_ID="nvidia/nemotron-speech-streaming-en-0.6b"
    EXPECTED_BUCKETS=32                # 2 * shift(16): drop0 + drop2 finalize buckets
    ;;
  ml)
    MODEL_ID="nvidia/nemotron-3.5-asr-streaming-0.6b"
    EXPECTED_BUCKETS=64                # 2 * shift(32)
    ;;
  *)
    echo "[export-model ERROR] PROFILE must be en or ml, got: $PROFILE" >&2
    exit 2
    ;;
esac
# model_profile.py-driven scripts (strip_bucket_weights.py, export_steady_batched.py,
# the contract epilogue below) all key off this.
export NEMOTRON_EXPORT_PROFILE="$PROFILE"

# --- target GPU architecture ----------------------------------------------------
TARGET_CC=${TARGET_CC:-auto}
if [[ "$TARGET_CC" == "auto" ]]; then
  TARGET_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  [[ -n "$TARGET_CC" ]] || { echo "[export-model ERROR] cannot auto-detect compute capability (nvidia-smi); set TARGET_CC=8.6|8.9|..." >&2; exit 2; }
fi
TARGET_SM="sm_${TARGET_CC/./}"                   # arch token: 8.6 -> sm_86 (matches torch get_arch_list)
TARGET_TAG="sm${TARGET_CC/./}"                   # dir/venv suffix (no underscore): 8.6 -> sm86

EPS_LOCAL=${EPS_LOCAL:-0}                        # 1 = use EPs already present in $ART_DIR (no S3)
S3_URI=${S3_URI:-"s3://audiointel-backups/STT/nvidia/${MODEL_ID#nvidia/}/base"}
ART_DIR=${ART_DIR:-"$ROOT/artifacts_${PROFILE}_${TARGET_TAG}"}
STEADY_B_DIR=${STEADY_B_DIR:-"$ART_DIR/steady_b_artifacts_b16"}
VENV=${VENV:-"$HOME/torch280-${TARGET_TAG}-venv"}
PYTHON_BIN=${PYTHON_BIN:-python3}
BUILD_DIR=${BUILD_DIR:-"$ROOT/cpp/build_${PROFILE}_${TARGET_TAG}_density"}
SELF_CHECK_ATOL=${SELF_CHECK_ATOL:-0.1}
STEADY_BATCHES=${STEADY_BATCHES:-"1,2,4,8,16"}
COMPILE_ENC_FIRST=${COMPILE_ENC_FIRST:-1}        # 1 = also AOTI-compile the first-chunk encoder
COMPILE_STEADY_BATCH=${COMPILE_STEADY_BATCH:-1}  # 1 = also AOTI-compile + strip the batched steady buckets
RUN_DENSITY=${RUN_DENSITY:-0}                    # 1 = build density_main and run the sweep after compiling
DENSITY_N_VALUES=${DENSITY_N_VALUES:-"1,8,16,24,32,40,48,64,80"}
DENSITY_SESSIONS_PER_WORKER=${DENSITY_SESSIONS_PER_WORKER:-0}
DENSITY_CHUNK_PERIOD_MS=${DENSITY_CHUNK_PERIOD_MS:-160}
DENSITY_TREAT_NO_PASS_AS_FAILURE=${DENSITY_TREAT_NO_PASS_AS_FAILURE:-0}
FORCE_S3_DOWNLOAD=${FORCE_S3_DOWNLOAD:-0}
KEEP_UNSTRIPPED_BUCKETS=${KEEP_UNSTRIPPED_BUCKETS:-0}
STRICT_EPS_MANIFEST=${STRICT_EPS_MANIFEST:-1}
ALLOW_UNMANIFESTED_AUX=${ALLOW_UNMANIFESTED_AUX:-1}
SKIP_EPS_VERIFY=${SKIP_EPS_VERIFY:-0}
# The AOTI .so inside each compiled .pt2 links the BUILD box's glibc, and glibc symbol
# versioning is backward- but not forward-compatible: the build glibc becomes the minimum
# glibc of every serve host (dlopen fails with "GLIBC_x.y not found" below it). The serve
# fleet floor is ubuntu 24.04 = glibc 2.39 (decision 2026-07-20); build at or below it —
# on a newer host, build inside the pinned container (runtime/container/enter.sh).
SERVE_GLIBC_MAX=${SERVE_GLIBC_MAX:-2.39}
ALLOW_INCOMPATIBLE_GLIBC=${ALLOW_INCOMPATIBLE_GLIBC:-0}  # 1 = skip both glibc gates

# Arch-agnostic auxiliary artifacts downloaded alongside the compile inputs. They are
# not compile inputs but complete the serve set backup_artifacts_s3.sh uploads.
SERVE_AUX_KEYS=(
  preproc.ts
  preproc.ts.manifest.json
  session_audio_bundle.ts
  session_audio_bundle.ts.audio_ci.json
)
if [[ "$PROFILE" == "ml" ]]; then
  SERVE_AUX_KEYS+=(prompt_apply.ts)              # post-encoder language-ID MLP (ml only)
fi

PY=""
CUDA_ROOT=${CUDA_ROOT:-}
SWEEP_STAMP=""
RUN_LOG_LIST=""

log() {
  printf '[export-model %(%H:%M:%S)T] %s\n' -1 "$*"
}

die() {
  printf '[export-model ERROR] %s\n' "$*" >&2
  exit 2
}

need_file() {
  [[ -f "$1" ]] || die "missing required file: $1"
}

need_dir() {
  [[ -d "$1" ]] || die "missing required directory: $1"
}

sudo_cmd() {
  if [[ ${EUID} -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    die "need root or sudo for: $*"
  fi
}

check_build_glibc() {
  local host_glibc
  host_glibc=$(ldd --version 2>/dev/null | awk 'NR==1{print $NF}')
  [[ "$host_glibc" =~ ^[0-9]+\.[0-9]+$ ]] || die "cannot parse host glibc version from ldd --version"
  if [[ "$(printf '%s\n' "$SERVE_GLIBC_MAX" "$host_glibc" | sort -V | tail -1)" != "$SERVE_GLIBC_MAX" ]]; then
    if [[ "$ALLOW_INCOMPATIBLE_GLIBC" == "1" ]]; then
      log "WARNING: host glibc $host_glibc > SERVE_GLIBC_MAX=$SERVE_GLIBC_MAX and ALLOW_INCOMPATIBLE_GLIBC=1 — artifacts built here will NOT load on serve hosts with glibc < $host_glibc"
      return 0
    fi
    die "host glibc $host_glibc > SERVE_GLIBC_MAX=$SERVE_GLIBC_MAX: AOTI .so files built here would fail dlopen on the serve fleet (GLIBC_x.y not found). Build inside the pinned container (runtime/container/enter.sh) or set ALLOW_INCOMPATIBLE_GLIBC=1 to override."
  fi
  log "host glibc $host_glibc <= serve floor $SERVE_GLIBC_MAX — OK"
}

# Fail-closed check of what actually matters at serve time: the max GLIBC_x.y version-need
# stamped into the .so members of each produced .pt2 package must not exceed SERVE_GLIBC_MAX.
verify_artifact_glibc() {
  if [[ "$ALLOW_INCOMPATIBLE_GLIBC" == "1" ]]; then
    log "ALLOW_INCOMPATIBLE_GLIBC=1; skipping stamped-GLIBC verification of $# package(s)"
    return 0
  fi
  log "verifying stamped GLIBC floor <= $SERVE_GLIBC_MAX in $# package(s)"
  "$PY" - "$SERVE_GLIBC_MAX" "$@" <<'PY'
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

limit = tuple(int(x) for x in sys.argv[1].split("."))
pkgs = [Path(p) for p in sys.argv[2:]]
ver_re = re.compile(r"\bGLIBC_(\d+)\.(\d+)\b")
failures = []
checked_sos = 0
worst = (0, 0)

for pkg in pkgs:
    if not pkg.is_file():
        failures.append(f"missing package: {pkg}")
        continue
    with zipfile.ZipFile(pkg) as zf:
        so_members = [m for m in zf.namelist() if m.endswith(".so") or ".so." in Path(m).name]
        if not so_members:
            failures.append(f"{pkg}: no .so members found (unexpected for an AOTI package)")
            continue
        with tempfile.TemporaryDirectory(prefix="glibc_scan_", dir=pkg.parent) as tmp:
            for member in so_members:
                so_path = Path(zf.extract(member, tmp))
                out = subprocess.run(
                    ["readelf", "-V", str(so_path)],
                    check=True, capture_output=True, text=True,
                ).stdout
                vers = {(int(a), int(b)) for a, b in ver_re.findall(out)}
                checked_sos += 1
                if not vers:
                    continue
                top = max(vers)
                worst = max(worst, top)
                if top > limit:
                    failures.append(
                        f"{pkg}:{member} requires GLIBC_{top[0]}.{top[1]} > serve floor "
                        f"GLIBC_{sys.argv[1]} — would fail dlopen on the serve fleet"
                    )

if failures:
    print("stamped-GLIBC verification FAILED:", file=sys.stderr)
    for failure in failures:
        print("  " + failure, file=sys.stderr)
    raise SystemExit(2)
print(
    f"stamped-GLIBC verification OK: packages={len(pkgs)} sos={checked_sos} "
    f"max_stamped=GLIBC_{worst[0]}.{worst[1]} floor=GLIBC_{sys.argv[1]}"
)
PY
}

install_os_deps() {
  log "installing/checking DL-AMI build deps, including python3-dev and awscli"
  sudo_cmd apt-get update -qq
  sudo_cmd env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ninja-build \
    python3-dev \
    python3-venv >/dev/null

  command -v cmake >/dev/null 2>&1 || die "cmake not found after apt install"
  command -v g++ >/dev/null 2>&1 || die "g++ not found after apt install"
  command -v aws >/dev/null 2>&1 || die "aws CLI not found after apt install"
}

setup_venv() {
  log "creating/updating torch 2.8.0 venv at $VENV"
  export PATH="$HOME/.local/bin:$PATH"
  # Idempotent across re-runs: reuse an existing venv that already has torch 2.8.0.
  if [[ -x "$VENV/bin/python" ]] && "$VENV/bin/python" -c "import torch,sys; sys.exit(0 if torch.__version__.split('+',1)[0]=='2.8.0' else 1)" >/dev/null 2>&1; then
    log "venv already has torch 2.8.0 — reusing $VENV"
    PY="$VENV/bin/python"
    export PY
    return
  fi
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi

  if command -v uv >/dev/null 2>&1; then
    uv venv --python "$PYTHON_BIN" "$VENV"
    uv pip install --python "$VENV/bin/python" --upgrade pip
    uv pip install --python "$VENV/bin/python" "torch==2.8.0"
  else
    "$PYTHON_BIN" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install "torch==2.8.0"
  fi
  PY="$VENV/bin/python"
  export PY
}

find_cuda_root() {
  if [[ -n "${CUDA_ROOT:-}" ]]; then
    [[ -x "$CUDA_ROOT/bin/nvcc" ]] || die "CUDA_ROOT is set but nvcc is missing: $CUDA_ROOT"
    printf '%s\n' "$CUDA_ROOT"
    return
  fi

  local -a candidates=()
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    candidates+=("/usr/local/cuda")
  fi
  while IFS= read -r nvcc_path; do
    candidates+=("$(dirname "$(dirname "$nvcc_path")")")
  done < <(find /usr/local -maxdepth 3 -path '/usr/local/cuda-*/bin/nvcc' 2>/dev/null | sort -V)

  if ((${#candidates[@]} == 0)); then
    die "could not find system CUDA nvcc under /usr/local/cuda or /usr/local/cuda-*"
  fi
  printf '%s\n' "${candidates[$((${#candidates[@]} - 1))]}"
}

find_driver_libcuda() {
  local path
  for path in \
    /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    /usr/lib/wsl/lib/libcuda.so.1; do
    if [[ -e "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  path=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF; exit}')
  [[ -n "$path" && -e "$path" ]] || return 1
  printf '%s\n' "$path"
}

prepare_cuda_link_env() {
  CUDA_ROOT=$(find_cuda_root)
  export CUDA_ROOT CUDA_HOME="$CUDA_ROOT"
  export PATH="$CUDA_ROOT/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
  command -v nvcc >/dev/null 2>&1 || die "nvcc not found under CUDA_ROOT=$CUDA_ROOT"
  log "cuda_root=$CUDA_ROOT"
  log "nvcc: $(nvcc --version | tail -1)"

  local driver_lib
  driver_lib=$(find_driver_libcuda) || die "libcuda.so.1 driver library not found; is the NVIDIA driver loaded?"
  local driver_dir
  driver_dir=$(dirname "$driver_lib")

  if [[ ! -e "$driver_dir/libcuda.so" ]]; then
    log "creating unversioned libcuda.so symlink for Triton link probes: $driver_dir/libcuda.so -> $driver_lib"
    sudo_cmd ln -s "$driver_lib" "$driver_dir/libcuda.so"
  fi

  local stub_dir="$CUDA_ROOT/lib64/stubs"
  if [[ ! -d "$stub_dir" ]]; then
    sudo_cmd mkdir -p "$stub_dir"
  fi
  if [[ ! -e "$stub_dir/libcuda.so" ]]; then
    log "creating CUDA stub libcuda.so symlink: $stub_dir/libcuda.so -> $driver_dir/libcuda.so"
    sudo_cmd ln -s "$driver_dir/libcuda.so" "$stub_dir/libcuda.so"
  fi

  export LIBRARY_PATH="$driver_dir:$stub_dir:$CUDA_ROOT/lib64:${LIBRARY_PATH:-}"
  log "libcuda link inputs: driver=$driver_lib stub=$stub_dir/libcuda.so"
}

configure_aoti_env() {
  export TORCHINDUCTOR_CACHE_DIR="$ART_DIR/torchinductor_cache"
  export TORCHINDUCTOR_MAX_AUTOTUNE=0
  export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=0
  export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=0
  export TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0
  export TORCH_CUDA_ARCH_LIST="$TARGET_CC"
  unset TORCHINDUCTOR_FREEZING
  log "AOTI compile env: autotune OFF, TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
}

check_torch_cuda() {
  log "checking torch 2.8.0 CUDA device and $TARGET_SM capability"
  "$PY" - "$TARGET_CC" "$TARGET_SM" <<'PY'
import os
import platform
import sys
import torch
import torch._inductor.config as config

cc_want = tuple(int(x) for x in sys.argv[1].split("."))
sm_want = sys.argv[2]

print("python_machine", platform.machine())
print("torch", torch.__version__, "torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if platform.machine() != "x86_64":
    raise SystemExit(f"expected x86_64 DL AMI, got {platform.machine()}")
if torch.__version__.split("+", 1)[0] != "2.8.0":
    raise SystemExit(f"expected torch==2.8.0, got {torch.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda is not available")
cc = tuple(torch.cuda.get_device_capability())
arch = torch.cuda.get_arch_list()
print("device", torch.cuda.get_device_name(0), "cc", cc)
print("arch_list", arch)
if cc != cc_want:
    raise SystemExit(f"expected {sm_want} device capability {cc_want}, got {cc}")
# The wheel arch_list is about PREBUILT kernels and does NOT gate AOTI codegen:
# Inductor/Triton compile hot kernels for the LIVE device, and the cpp/nvcc parts honor
# TORCH_CUDA_ARCH_LIST. Device cc==cc_want is the real hardware gate (asserted above);
# the meaningful AOTI-target gate is that TORCH_CUDA_ARCH_LIST requests the target cc.
# (e.g. the stock 2.8.0+cu128 wheel ships SASS for sm_86/sm_90/sm_120 but NOT sm_89.)
cc_str = sys.argv[1]
arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
print("TORCH_CUDA_ARCH_LIST", repr(arch_list_env))
if sm_want not in arch:
    print(f"WARNING: wheel arch_list lacks {sm_want} ({arch}); AOTI targets {sm_want} via TORCH_CUDA_ARCH_LIST")
    if cc_str not in arch_list_env:
        raise SystemExit(
            f"{sm_want} absent from wheel arch_list AND TORCH_CUDA_ARCH_LIST does not request {cc_str}: {arch_list_env!r}"
        )
if not hasattr(torch._inductor, "aoti_compile_and_package"):
    raise SystemExit("torch._inductor.aoti_compile_and_package is missing")
bad = []
for name in ("max_autotune", "coordinate_descent_tuning"):
    if bool(getattr(config, name, False)):
        bad.append(name)
if bad:
    raise SystemExit(f"autotune config unexpectedly enabled: {bad}")
print("aoti_compile_and_package OK")
print("_GLIBCXX_USE_CXX11_ABI", getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", "unknown"))
PY
}

download_s3_object() {
  local key=$1
  local dst=$2
  mkdir -p "$(dirname "$dst")"
  if [[ -f "$dst" && "$FORCE_S3_DOWNLOAD" != "1" ]]; then
    log "S3 download exists, keeping: $key -> $dst"
    return
  fi
  log "S3 download: $S3_URI/$key -> $dst"
  aws s3 cp "$S3_URI/$key" "$dst" --only-show-errors
}

parse_bucket_manifest() {
  local manifest=$1
  "$PY" - "$manifest" "$EXPECTED_BUCKETS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
data = json.loads(path.read_text())
if isinstance(data, dict):
    entries = data.get("buckets") or data.get("contract") or data.get("files")
else:
    entries = data
if not isinstance(entries, list):
    raise SystemExit(f"{path} does not contain a bucket list")
names = []
for item in entries:
    if not isinstance(item, dict):
        continue
    ep = item.get("ep")
    if ep is None and "drop" in item and "T" in item:
        ep = f"enc_finalize_d{int(item['drop'])}_T{int(item['T'])}_ep.pt2"
    if isinstance(ep, str) and ep.endswith("_ep.pt2"):
        names.append(ep)
names = sorted(set(names))
if len(names) != expected:
    raise SystemExit(f"expected {expected} finalize bucket EPs, got {len(names)} from {path}")
for name in names:
    print(name)
PY
}

steady_batch_values() {
  local -a batches=()
  IFS=',' read -r -a batches <<< "$STEADY_BATCHES"
  local b
  for b in "${batches[@]}"; do
    [[ "$b" =~ ^[0-9]+$ ]] || die "invalid STEADY_BATCHES entry: $b"
    printf '%s\n' "$b"
  done
}

download_artifacts_from_s3() {
  mkdir -p "$ART_DIR/finalize_buckets" "$ART_DIR/logs"

  local -a root_keys=(
    "enc_steady_t2a.pt2"
    "session_bundle.ts"
    "finalize_shared_weights.pt"
    "finalize_shared_weights.ts"
    "enc_first.ts"
    "t2a_io.pt"
    "joint_step.ts"
    "predict_step.ts"
  )
  root_keys+=("${SERVE_AUX_KEYS[@]}")
  if [[ "$COMPILE_ENC_FIRST" == "1" ]]; then
    root_keys+=("enc_first_t2a.pt2" "enc_first_t2a_io.pt")
  fi

  if [[ "$EPS_LOCAL" == "1" ]]; then
    # No S3: the EPs were exported on a torch host and rsynced into $ART_DIR.
    log "EPS_LOCAL=1; using pre-placed source EPs in $ART_DIR (no S3 download)"
    local req missing=0
    for req in "${root_keys[@]}" finalize_buckets/buckets_manifest.json; do
      [[ -f "$ART_DIR/$req" ]] || { log "MISSING required EP: $ART_DIR/$req"; missing=1; }
    done
    if [[ "$COMPILE_STEADY_BATCH" == "1" ]]; then
      local b
      while IFS= read -r b; do
        [[ -f "$STEADY_B_DIR/enc_steady_t2a_b${b}.pt2" ]] || { log "MISSING required EP: $STEADY_B_DIR/enc_steady_t2a_b${b}.pt2"; missing=1; }
      done < <(steady_batch_values)
    fi
    local n_bucket_eps
    n_bucket_eps=$(find "$ART_DIR/finalize_buckets" -maxdepth 1 -type f -name 'enc_finalize_d*_T*_ep.pt2' 2>/dev/null | wc -l)
    [[ "$n_bucket_eps" == "$EXPECTED_BUCKETS" ]] || { log "expected $EXPECTED_BUCKETS finalize bucket EPs in finalize_buckets/, found $n_bucket_eps"; missing=1; }
    [[ "$missing" == "0" ]] || die "EPS_LOCAL=1 but the local export is incomplete in $ART_DIR (see runtime/ARTIFACTS.md for the export + rsync manifest)"
    return 0
  fi

  log "downloading source artifacts from $S3_URI"

  download_s3_object "eps_manifest.json" "$ART_DIR/eps_manifest.json"
  download_s3_object "buckets_manifest.json" "$ART_DIR/buckets_manifest.json"
  cp -f "$ART_DIR/buckets_manifest.json" "$ART_DIR/finalize_buckets/buckets_manifest.json"

  local key
  for key in "${root_keys[@]}"; do
    download_s3_object "$key" "$ART_DIR/$key"
  done

  if [[ "$COMPILE_STEADY_BATCH" == "1" ]]; then
    local b
    while IFS= read -r b; do
      download_s3_object "steady_b_artifacts_b16/enc_steady_t2a_b${b}.pt2" "$STEADY_B_DIR/enc_steady_t2a_b${b}.pt2"
    done < <(steady_batch_values)
  fi

  local bucket_list="$ART_DIR/finalize_buckets/.bucket_eps"
  parse_bucket_manifest "$ART_DIR/buckets_manifest.json" > "$bucket_list"
  while IFS= read -r ep_name; do
    [[ -n "$ep_name" ]] || continue
    download_s3_object "finalize_buckets/$ep_name" "$ART_DIR/finalize_buckets/$ep_name"
  done < "$bucket_list"

  verify_s3_artifacts
}

verify_s3_artifacts() {
  if [[ "$SKIP_EPS_VERIFY" == "1" ]]; then
    log "SKIP_EPS_VERIFY=1; skipping eps_manifest SHA256 verification"
    return 0
  fi
  # Auxiliary keys mirror exactly what download_artifacts_from_s3 fetched beyond the
  # core compile inputs, so the SHA sweep covers the whole download set.
  local aux_csv
  aux_csv=$(IFS=','; printf '%s' "${SERVE_AUX_KEYS[*]}")
  if [[ "$COMPILE_ENC_FIRST" == "1" ]]; then
    aux_csv+=",enc_first_t2a.pt2,enc_first_t2a_io.pt"
  fi
  if [[ "$COMPILE_STEADY_BATCH" == "1" ]]; then
    local b
    while IFS= read -r b; do
      aux_csv+=",steady_b_artifacts_b16/enc_steady_t2a_b${b}.pt2"
    done < <(steady_batch_values)
  fi
  log "verifying downloaded artifacts against eps_manifest.json"
  "$PY" - "$ART_DIR" "$STRICT_EPS_MANIFEST" "$ALLOW_UNMANIFESTED_AUX" "$EXPECTED_BUCKETS" "$aux_csv" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

art = Path(sys.argv[1])
strict = sys.argv[2] == "1"
allow_unmanifested_aux = sys.argv[3] == "1"
expected_buckets = int(sys.argv[4])
aux_keys = [k for k in sys.argv[5].split(",") if k]
manifest_path = art / "eps_manifest.json"
manifest = json.loads(manifest_path.read_text())

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def norm_key(value: str) -> str:
    value = value.strip()
    if value.startswith("s3://"):
        marker = "/base/"
        if marker in value:
            value = value.split(marker, 1)[1]
        else:
            value = value.rsplit("/", 1)[-1]
    while value.startswith("./"):
        value = value[2:]
    # backup_artifacts_s3.sh manifests key by S3 layout ("base/..."); older/local
    # manifests may key by repo path.
    for prefix in ("base/", "runtime/artifacts/", "artifacts/"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    return value

def looks_sha(text: str) -> bool:
    return len(text) == 64 and all(c in "0123456789abcdefABCDEF" for c in text)

entries = []
if isinstance(manifest, dict):
    raw_files = manifest.get("files") or manifest.get("artifacts") or manifest.get("objects")
    if isinstance(raw_files, list):
        entries.extend(raw_files)
    for k, v in manifest.items():
        if isinstance(v, str) and looks_sha(v):
            entries.append({"path": k, "sha256": v})
        elif isinstance(v, dict) and any(x in v for x in ("sha256", "sha", "digest")):
            item = dict(v)
            item.setdefault("path", k)
            entries.append(item)
elif isinstance(manifest, list):
    entries.extend(manifest)

by_key = {}
by_base = {}
for entry in entries:
    if not isinstance(entry, dict):
        continue
    path = entry.get("path") or entry.get("key") or entry.get("name") or entry.get("s3_key")
    sha = entry.get("sha256") or entry.get("sha") or entry.get("digest")
    if not path or not isinstance(sha, str) or not looks_sha(sha):
        continue
    key = norm_key(str(path))
    by_key[key] = entry
    by_base.setdefault(Path(key).name, entry)

bucket_manifest = json.loads((art / "buckets_manifest.json").read_text())
if isinstance(bucket_manifest, dict):
    bucket_entries = bucket_manifest.get("buckets") or bucket_manifest.get("contract") or []
else:
    bucket_entries = bucket_manifest
bucket_eps = []
for item in bucket_entries:
    if not isinstance(item, dict):
        continue
    ep = item.get("ep")
    if ep is None and "drop" in item and "T" in item:
        ep = f"enc_finalize_d{int(item['drop'])}_T{int(item['T'])}_ep.pt2"
    if isinstance(ep, str) and ep.endswith("_ep.pt2"):
        bucket_eps.append(ep)
bucket_eps = sorted(set(bucket_eps))
if len(bucket_eps) != expected_buckets:
    raise SystemExit(f"expected {expected_buckets} bucket EPs in buckets_manifest.json, got {len(bucket_eps)}")

required = [
    ("enc_steady_t2a.pt2", art / "enc_steady_t2a.pt2", True),
    ("session_bundle.ts", art / "session_bundle.ts", True),
    ("finalize_shared_weights.pt", art / "finalize_shared_weights.pt", True),
    ("finalize_shared_weights.ts", art / "finalize_shared_weights.ts", True),
    ("enc_first.ts", art / "enc_first.ts", True),
]
required.extend((f"finalize_buckets/{name}", art / "finalize_buckets" / name, True) for name in bucket_eps)

auxiliary = [
    ("t2a_io.pt", art / "t2a_io.pt", False),
    ("joint_step.ts", art / "joint_step.ts", False),
    ("predict_step.ts", art / "predict_step.ts", False),
]
auxiliary.extend((key, art / key, False) for key in aux_keys)

errors = []
verified = 0
unmanifested_aux = []
for key, path, must_manifest in required + auxiliary:
    if not path.is_file():
        errors.append(f"missing downloaded artifact: {path}")
        continue
    entry = by_key.get(key) or by_key.get(f"artifacts/{key}") or by_base.get(Path(key).name)
    if entry is None:
        if must_manifest or (strict and not allow_unmanifested_aux):
            errors.append(f"eps_manifest.json has no SHA entry for required key: {key}")
        else:
            unmanifested_aux.append(key)
        continue
    expected = (entry.get("sha256") or entry.get("sha") or entry.get("digest")).lower()
    actual = sha256_file(path)
    if actual.lower() != expected:
        errors.append(f"sha256 mismatch for {key}: manifest={expected} actual={actual}")
    else:
        verified += 1

bucket_entry = by_key.get("buckets_manifest.json") or by_key.get("finalize_buckets/buckets_manifest.json")
if bucket_entry is not None:
    expected = (bucket_entry.get("sha256") or bucket_entry.get("sha") or bucket_entry.get("digest")).lower()
    actual = sha256_file(art / "buckets_manifest.json")
    if actual.lower() != expected:
        errors.append(f"sha256 mismatch for buckets_manifest.json: manifest={expected} actual={actual}")
    else:
        verified += 1
elif isinstance(manifest, dict) and "contract" in manifest:
    if manifest["contract"] != bucket_manifest:
        errors.append("eps_manifest.json contract does not match buckets_manifest.json")
else:
    errors.append("eps_manifest.json has no SHA or contract for buckets_manifest.json")

if errors:
    print("eps_manifest verification failed:", file=sys.stderr)
    for error in errors:
        print("  " + error, file=sys.stderr)
    raise SystemExit(2)
if unmanifested_aux:
    print("AUX_UNMANIFESTED " + ",".join(unmanifested_aux))
print(f"eps_manifest verification OK: verified_sha256={verified} buckets={len(bucket_eps)}")
PY
}

prepare_artifact_workdir() {
  log "preparing native $TARGET_SM artifact workdir: $ART_DIR"
  need_file "$ART_DIR/enc_steady_t2a.pt2"
  need_file "$ART_DIR/t2a_io.pt"
  need_file "$ART_DIR/session_bundle.ts"
  need_file "$ART_DIR/finalize_shared_weights.pt"
  need_file "$ART_DIR/finalize_shared_weights.ts"
  need_file "$ART_DIR/enc_first.ts"
  need_file "$ART_DIR/joint_step.ts"
  need_file "$ART_DIR/predict_step.ts"
  need_dir "$ART_DIR/finalize_buckets"
  need_file "$ART_DIR/finalize_buckets/buckets_manifest.json"
  if [[ "$PROFILE" == "ml" ]]; then
    need_file "$ART_DIR/prompt_apply.ts"
  fi

  rm -f "$ART_DIR/enc_steady_aoti.pt2"
  # Keep stripped_finalize_buckets across re-runs (compile_strip_bucket skips ones already present);
  # only the per-bucket scratch is wiped. The manifest epilogue revalidates all-present + SHAs.
  rm -rf "$ART_DIR/finalize_compile_work"
  mkdir -p "$ART_DIR/stripped_finalize_buckets" "$ART_DIR/finalize_compile_work/manifests"
  find "$ART_DIR/finalize_buckets" -maxdepth 1 -type f \
    -name 'enc_finalize_d*_T*.pt2' ! -name '*_ep.pt2' -delete
}

compile_steady() {
  log "native $TARGET_SM AOTI compile, autotune OFF: enc_steady_t2a.pt2 -> enc_steady_aoti.pt2"
  NEMOTRON_ART_DIR="$ART_DIR" "$PY" "$ROOT/aot_compile.py"
  need_file "$ART_DIR/enc_steady_aoti.pt2"
  verify_artifact_glibc "$ART_DIR/enc_steady_aoti.pt2"
}

compile_enc_first() {
  if [[ "$COMPILE_ENC_FIRST" != "1" ]]; then
    log "COMPILE_ENC_FIRST=0; skipping first-chunk encoder AOTI compile"
    return 0
  fi
  # Idempotent: without --force an existing enc_first_aoti.pt2 is self-checked, not recompiled.
  log "native $TARGET_SM AOTI compile, autotune OFF: enc_first_t2a.pt2 -> enc_first_aoti.pt2"
  "$PY" "$ROOT/aot_compile_enc_first.py" \
    --artifacts "$ART_DIR" \
    --self-check-atol "$SELF_CHECK_ATOL"
  need_file "$ART_DIR/enc_first_aoti.pt2"
  verify_artifact_glibc "$ART_DIR/enc_first_aoti.pt2"
}

compile_steady_batched() {
  if [[ "$COMPILE_STEADY_BATCH" != "1" ]]; then
    log "COMPILE_STEADY_BATCH=0; skipping batched steady-encoder AOTI compile"
    return 0
  fi
  log "native $TARGET_SM AOTI compile, autotune OFF: batched steady buckets B={$STEADY_BATCHES}"
  "$PY" "$ROOT/export_steady_batched.py" \
    --out "$STEADY_B_DIR" \
    --compile-only \
    --batches "$STEADY_BATCHES" \
    --shared-weights "$ART_DIR/finalize_shared_weights.ts" \
    --self-check-weights "$ART_DIR/finalize_shared_weights.pt" \
    --production-b1 "$ART_DIR/enc_steady_aoti.pt2"

  log "strip batched steady buckets to the shared weight set"
  "$PY" "$ROOT/strip_steady_buckets.py" \
    --dir "$STEADY_B_DIR" \
    --shared-weights "$ART_DIR/finalize_shared_weights.ts" \
    --buckets "$STEADY_BATCHES"

  # MANIFEST.json embeds package SHA256s; re-emit after stripping so the manifest
  # describes the stripped packages the server actually loads.
  "$PY" "$ROOT/export_steady_batched.py" \
    --out "$STEADY_B_DIR" \
    --manifest-only \
    --batches "$STEADY_BATCHES" \
    --shared-weights "$ART_DIR/finalize_shared_weights.ts" \
    --production-b1 "$ART_DIR/enc_steady_aoti.pt2"

  local b
  local -a batch_pkgs=()
  while IFS= read -r b; do
    need_file "$STEADY_B_DIR/enc_steady_aoti_b${b}.pt2"
    batch_pkgs+=("$STEADY_B_DIR/enc_steady_aoti_b${b}.pt2")
  done < <(steady_batch_values)
  need_file "$STEADY_B_DIR/MANIFEST.json"
  verify_artifact_glibc "${batch_pkgs[@]}"
}

compile_strip_bucket() {
  local ep=$1
  local base key pkg one_dir
  base=$(basename "$ep")
  key=${base%_ep.pt2}
  pkg=${base/_ep.pt2/.pt2}
  one_dir="$ART_DIR/finalize_compile_work/$key"

  # Idempotent across re-runs: a stripped bucket that already compiled+self-checked is
  # deterministic, so reuse it instead of burning compile minutes (e.g. an atol re-run that
  # only needs to recompile one outlier bucket). A clean rebuild = rm -rf stripped_finalize_buckets.
  if [[ -f "$ART_DIR/stripped_finalize_buckets/$pkg" ]]; then
    log "stripped bucket exists, skipping recompile: $pkg"
    return
  fi

  rm -rf "$one_dir"
  mkdir -p "$one_dir"
  ln -sfn "$(realpath "$ep")" "$one_dir/$base"

  log "native $TARGET_SM AOTI compile bucket, autotune OFF: $base"
  "$PY" "$ROOT/aot_compile_buckets.py" \
    --dir "$one_dir" \
    --shared-weights "$ART_DIR/finalize_shared_weights.pt" \
    --force \
    --self-check-atol "$SELF_CHECK_ATOL"

  need_file "$one_dir/$pkg"
  cp -f "$one_dir/buckets_manifest.json" "$ART_DIR/finalize_compile_work/manifests/$key.json"

  log "strip bucket weights: $pkg"
  "$PY" "$ROOT/strip_bucket_weights.py" \
    --bucket "$one_dir/$pkg" \
    --out-dir "$ART_DIR/stripped_finalize_buckets" \
    --shared-weights "$ART_DIR/finalize_shared_weights.pt" \
    --bundle "$ART_DIR/session_bundle.ts" \
    --joint "$ART_DIR/joint_step.ts" \
    --predict "$ART_DIR/predict_step.ts" \
    --strip-only \
    --force

  if [[ "$KEEP_UNSTRIPPED_BUCKETS" == "1" ]]; then
    mv -f "$one_dir/$pkg" "$ART_DIR/finalize_buckets/$pkg"
  fi
  rm -rf "$one_dir"
}

compile_finalize_buckets() {
  log "native $TARGET_SM AOTI compile + strip for $EXPECTED_BUCKETS finalize buckets, self-check-atol=$SELF_CHECK_ATOL"
  local -a eps=()
  while IFS= read -r path; do
    eps+=("$path")
  done < <(find "$ART_DIR/finalize_buckets" -maxdepth 1 -type f -name 'enc_finalize_d*_T*_ep.pt2' | sort)
  ((${#eps[@]} == EXPECTED_BUCKETS)) || die "expected $EXPECTED_BUCKETS finalize bucket EPs, found ${#eps[@]} in $ART_DIR/finalize_buckets"

  local ep
  for ep in "${eps[@]}"; do
    compile_strip_bucket "$ep"
  done

  local -a stripped_pkgs=()
  while IFS= read -r path; do
    stripped_pkgs+=("$path")
  done < <(find "$ART_DIR/stripped_finalize_buckets" -maxdepth 1 -type f -name 'enc_finalize_d*_T*.pt2' | sort)
  verify_artifact_glibc "${stripped_pkgs[@]}"

  "$PY" - "$ART_DIR" "$ROOT" "$EXPECTED_BUCKETS" "$TARGET_TAG" <<'PY'
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

art = Path(sys.argv[1])
sys.path.insert(0, sys.argv[2])
expected_buckets = int(sys.argv[3])
target_tag = sys.argv[4]
bucket_re = re.compile(r"^enc_finalize_d(?P<drop>\d+)_T(?P<T>\d+)\.pt2$")

# CONTRACT constants come from the profile (NEMOTRON_EXPORT_PROFILE), the same source
# of truth the export scripts and cpp/lib/session/model_constants.h mirror.
from model_profile import get_profile
profile = get_profile()
if 2 * profile.shift != expected_buckets:
    raise SystemExit(
        f"profile {profile.name!r} implies {2 * profile.shift} finalize buckets, script expects {expected_buckets}"
    )

source_manifest = json.loads((art / "buckets_manifest.json").read_text())
if isinstance(source_manifest, dict):
    source_entries = source_manifest.get("buckets") or source_manifest.get("contract") or []
else:
    source_entries = source_manifest
expected = {(int(b["drop"]), int(b["T"])) for b in source_entries if "drop" in b and "T" in b}
if len(expected) != expected_buckets:
    raise SystemExit(f"expected {expected_buckets} source bucket keys, got {len(expected)}")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

stripped_dir = art / "stripped_finalize_buckets"
buckets = []
seen = set()
for path in sorted(stripped_dir.glob("enc_finalize_d*_T*.pt2")):
    m = bucket_re.match(path.name)
    if not m:
        continue
    key = (int(m.group("drop")), int(m.group("T")))
    seen.add(key)
    buckets.append({
        "drop": key[0],
        "T": key[1],
        "pkg": path.name,
        "pkg_sha256": sha256_file(path),
    })
if seen != expected:
    raise SystemExit(f"stripped bucket key mismatch: missing={sorted(expected-seen)} extra={sorted(seen-expected)}")

contract = {
    "att_context": list(profile.att_context),
    "blank": profile.blank,
    "drop_extra": profile.drop,
    "final_padding_frames": profile.final_padding_frames,
    "max_symbols": profile.max_symbols,
    "model_id": profile.model_id,
    "pre_encode_cache": profile.pre,
    "right_context": profile.right_context,
    "shift": profile.shift,
    "weights_sha256": sha256_file(art / "finalize_shared_weights.pt"),
}
manifest = {
    "schema_version": 1,
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    "CONTRACT": contract,
    "buckets": buckets,
}
(stripped_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

compile_records = []
for manifest_path in sorted((art / "finalize_compile_work" / "manifests").glob("*.json")):
    compile_records.extend(json.loads(manifest_path.read_text()))
(art / "finalize_buckets" / f"buckets_manifest_{target_tag}.json").write_text(
    json.dumps(compile_records, indent=2, sort_keys=True) + "\n"
)
print(f"wrote stripped manifest: buckets={len(buckets)} path={stripped_dir / 'manifest.json'}")
print(f"wrote compile self-check manifest: buckets={len(compile_records)}")
PY
}

build_density_main() {
  log "building density_main (NEMOTRON_PROFILE=$PROFILE) with pip torch libtorch and system CUDA"
  need_file "$ROOT/cpp/density_main.cpp"
  need_file "$ROOT/cpp/session_main.cpp"
  need_file "$ROOT/cpp/CMakeLists.txt"

  local torch_root
  torch_root=$("$PY" - <<'PY'
import os
import torch
print(os.path.dirname(torch.__file__))
PY
)
  export TORCH_ROOT="$torch_root"
  export LD_LIBRARY_PATH="$TORCH_ROOT/lib:$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
  log "torch_root=$TORCH_ROOT"
  # CUDA-runtime unification (DL AMI ships system CUDA 13 = cudart-13; pip torch bundles cudart-12).
  # Linking density_main's explicit CUDA calls (cudaStreamCreate*, sync) against cudart-13 while torch's
  # AOTI run() uses cudart-12 loads BOTH runtimes and DEADLOCKS the multi-thread explicit-stream path
  # (GPU idle, main thread futex-waiting). Expose an unversioned libcudart.so pointing at torch's
  # cudart-12 in TORCH_ROOT/lib (which is FIRST in the cmake link dirs), so -lcudart resolves there and
  # the whole process uses ONE CUDA runtime.
  # pip torch ships cudart-12 under site-packages/nvidia/cuda_runtime/lib (NOT torch/lib). Symlink it into
  # torch/lib as the unversioned libcudart.so so -lcudart (torch/lib is FIRST link dir) resolves to
  # cudart-12, and add its dir to LD_LIBRARY_PATH so the SONAME libcudart.so.12 is found at runtime.
  local torch_cudart torch_cudart_dir
  torch_cudart=$(find "$TORCH_ROOT/lib" "$(dirname "$TORCH_ROOT")/nvidia/cuda_runtime/lib" -maxdepth 1 -name 'libcudart.so.*' 2>/dev/null | head -1 || true)
  if [[ -n "$torch_cudart" ]]; then
    torch_cudart_dir=$(dirname "$torch_cudart")
    ln -sf "$torch_cudart" "$TORCH_ROOT/lib/libcudart.so"
    export LD_LIBRARY_PATH="$TORCH_ROOT/lib:$torch_cudart_dir:$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
    log "cudart unify: link+runtime -> $torch_cudart"
  else
    log "WARNING: bundled libcudart.so.* not found; cudart unify skipped (deadlock risk)"
  fi
  cmake -S "$ROOT/cpp" -B "$BUILD_DIR" -G Ninja \
    -DTORCH_ROOT="$TORCH_ROOT" \
    -DCUDA_ROOT="$CUDA_ROOT" \
    -DNEMOTRON_PROFILE="$PROFILE" \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build "$BUILD_DIR" --target density_main -j"$(nproc)"
  # fail-closed: the binary must link ONLY torch's cudart-12, never the system cudart-13 (the deadlock).
  local linked_cudart
  linked_cudart=$(ldd "$BUILD_DIR/density_main" 2>/dev/null | grep -oE 'libcudart\.so\.[0-9]+' | sort -u | tr '\n' ' ')
  log "cudart check: density_main links [$linked_cudart]"
  if printf '%s' "$linked_cudart" | grep -q 'libcudart.so.13'; then
    die "density_main links libcudart.so.13 (cudart-mix) — would deadlock the multi-thread path; cudart unify failed"
  fi
}

run_density_one_n() {
  local n=$1
  local run_log="$ART_DIR/logs/density_N${n}_${SWEEP_STAMP}.stdout.log"
  local -a cmd=(
    "$BUILD_DIR/density_main"
    --mode density-sweep
    --n-values "$n"
    --density-chunk-period-ms "$DENSITY_CHUNK_PERIOD_MS"
  )
  if [[ "$DENSITY_SESSIONS_PER_WORKER" != "0" ]]; then
    cmd+=(--density-sessions-per-worker "$DENSITY_SESSIONS_PER_WORKER")
  fi
  cmd+=("$ART_DIR")
  if (($# > 1)); then
    cmd+=("${@:2}")
  fi

  log "fresh-process density run N=$n command: ${cmd[*]}"
  set +e
  "${cmd[@]}" 2>&1 | tee "$run_log"
  local rc=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$run_log" >> "$RUN_LOG_LIST"
  log "N=$n density_main exit code: $rc (0=multi-N PASS, 1=completed no-pass/row-bound, 2=setup/runtime failure)"
  if [[ $rc -eq 2 ]]; then
    die "density_main setup/runtime failure at N=$n; see $run_log"
  fi
  if [[ $rc -gt 2 ]]; then
    die "density_main unexpected exit code $rc at N=$n; see $run_log"
  fi
}

aggregate_density_results() {
  log "aggregating fresh-process sweep rows"
  "$PY" - "$RUN_LOG_LIST" "$ART_DIR/logs/density_summary_${SWEEP_STAMP}.json" "$DENSITY_TREAT_NO_PASS_AS_FAILURE" <<'PY'
import json
import re
import sys
from pathlib import Path

log_list = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
treat_no_pass_as_failure = sys.argv[3] == "1"

rows = {}
for line in log_list.read_text().splitlines():
    log_path = Path(line.strip())
    if not log_path.is_file():
        continue
    for raw in log_path.read_text(errors="replace").splitlines():
        marker = "DENSITY_TELEMETRY path="
        if marker not in raw or " json=" not in raw:
            continue
        payload = raw.split(" json=", 1)[1]
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if obj.get("check") == "1a_density_sweep_full_session":
            n = int(obj.get("workers") or obj.get("num_runners") or 0)
            if n > 0:
                rows[n] = obj

if not rows:
    print("DENSITY_RESULT status=NO_ROWS")
    raise SystemExit(2)

ordered = [rows[n] for n in sorted(rows)]

def stat(obj, key, pct, default=0.0):
    value = obj.get(key) or {}
    if not isinstance(value, dict):
        return default
    return float(value.get(pct, default) or default)

def resource(obj, key, default=0.0):
    value = obj.get("resource") or {}
    if not isinstance(value, dict):
        return default
    return value.get(key, default)

def gib(bytes_value):
    return float(bytes_value or 0) / (1024.0 ** 3)

def row_n(obj):
    return int(obj.get("workers") or obj.get("num_runners") or 0)

def row_slo(obj):
    return bool(obj.get("slo_robust"))

knee = 0
for row in ordered:
    n = row_n(row)
    if row_slo(row):
        knee = max(knee, n)

base = rows.get(1)
base_steady_p50 = stat(base, "steady_gpu", "p50") if base else 0.0
first_bound = None
for row in ordered:
    if row_n(row) > knee:
        first_bound = row
        break
observed_bound = first_bound is not None

def infer_binding_slo(row):
    if row is None:
        return "not_observed"
    if row.get("oom"):
        return "memory_oom"
    if int(row.get("mismatches", 0) or 0) or int(row.get("errors", 0) or 0):
        return "correctness_or_runtime"
    ttfs_p95 = stat(row, "ttfs", "p95")
    ttfs_p99 = stat(row, "ttfs", "p99")
    lag_p95 = stat(row, "lag", "p95")
    if ttfs_p95 > 175.0 or ttfs_p99 > 250.0:
        return "ttfs_p95_or_p99"
    if lag_p95 >= 500.0:
        return "keepup_lag_p95"
    return "none_observed_in_sweep"

def infer_binding_resource(row):
    if row is None:
        return "not_observed"
    peak = float(row.get("peak_gpu_mem_bytes", 0) or 0)
    total = float(row.get("total_gpu_mem_bytes", 0) or 0)
    if row.get("oom") or (total > 0 and peak / total >= 0.92):
        return "memory"
    cpu_cores = float(resource(row, "cpu_cores_used", 0.0) or 0.0)
    cpu_threads = float(resource(row, "cpu_threads", 0.0) or 0.0)
    if cpu_threads > 0 and cpu_cores >= 0.85 * cpu_threads:
        return "CPU cores"
    steady_p50 = stat(row, "steady_gpu", "p50")
    gpu_util = float(resource(row, "gpu_util_mean_pct", 0.0) or 0.0)
    if (base_steady_p50 > 0 and steady_p50 / base_steady_p50 >= 1.5) or gpu_util >= 80.0:
        return "GPU encoder contention"
    slo = infer_binding_slo(row)
    if slo.startswith("ttfs"):
        return "finalize/TTFS tail"
    if slo.startswith("keepup"):
        return "mixed_or_unknown_keepup"
    return "not_observed"

if observed_bound:
    binding_slo = infer_binding_slo(first_bound)
    binding_resource = infer_binding_resource(first_bound)
else:
    binding_slo = "none_observed_in_sweep"
    binding_resource = "not_observed_in_sweep"
knee_row = rows.get(knee)
per_stream_mem = gib(knee_row.get("worker_context_delta_per_worker_bytes", 0) if knee_row else 0)
if per_stream_mem == 0.0 and knee_row and row_n(knee_row) > 0:
    per_stream_mem = gib(knee_row.get("peak_gpu_mem_bytes", 0)) / float(row_n(knee_row))

summary_rows = []
for row in ordered:
    n = row_n(row)
    ttfs_p50 = stat(row, "ttfs", "p50")
    ttfs_p95 = stat(row, "ttfs", "p95")
    ttfs_p99 = stat(row, "ttfs", "p99")
    lag_p50 = stat(row, "lag", "p50")
    lag_p95 = stat(row, "lag", "p95")
    enc_first_lock_p95 = stat(row, "enc_first_lock_wait", "p95")
    peak_gib = gib(row.get("peak_gpu_mem_bytes", 0))
    per_stream_gib = gib(row.get("worker_context_delta_per_worker_bytes", 0))
    gpu_util = float(resource(row, "gpu_util_mean_pct", 0.0) or 0.0)
    cpu_cores = float(resource(row, "cpu_cores_used", 0.0) or 0.0)
    cpu_threads = int(resource(row, "cpu_threads", 0) or 0)
    mismatches = int(row.get("mismatches", 0) or 0)
    errors = int(row.get("errors", 0) or 0)
    correctness = mismatches == 0 and errors == 0
    print(
        "DENSITY_ROW "
        f"N={n} "
        f"slo_robust={str(row_slo(row)).lower()} "
        f"ttfs_p50_ms={ttfs_p50:.3f} ttfs_p95_ms={ttfs_p95:.3f} ttfs_p99_ms={ttfs_p99:.3f} "
        f"lag_p50_ms={lag_p50:.3f} lag_p95_ms={lag_p95:.3f} "
        f"peak_mem_GiB={peak_gib:.3f} per_stream_mem_GiB={per_stream_gib:.3f} "
        f"gpu_util_mean_pct={gpu_util:.1f} cpu_cores={cpu_cores:.2f}/{cpu_threads} "
        f"enc_first_lock_wait_p95_ms={enc_first_lock_p95:.3f} "
        f"correctness_0_mismatch={str(correctness).lower()} mismatches={mismatches} errors={errors} "
        f"oom={str(bool(row.get('oom'))).lower()}"
    )
    summary_rows.append({
        "N": n,
        "slo_robust": row_slo(row),
        "ttfs_p50_ms": ttfs_p50,
        "ttfs_p95_ms": ttfs_p95,
        "ttfs_p99_ms": ttfs_p99,
        "lag_p50_ms": lag_p50,
        "lag_p95_ms": lag_p95,
        "peak_mem_GiB": peak_gib,
        "per_stream_mem_GiB": per_stream_gib,
        "gpu_util_mean_pct": gpu_util,
        "cpu_cores_used": cpu_cores,
        "cpu_threads": cpu_threads,
        "enc_first_lock_wait_p95_ms": enc_first_lock_p95,
        "mismatches": mismatches,
        "errors": errors,
        "oom": bool(row.get("oom")),
    })

result = {
    "status": "PASS" if knee > 0 else "NO_SLO_ROBUST_N",
    "knee_N": knee,
    "binding_slo": binding_slo,
    "binding_resource": binding_resource,
    "per_stream_mem_GiB_at_knee": per_stream_mem,
    "rows": summary_rows,
}
summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(
    "DENSITY_RESULT "
    f"status={result['status']} knee_N={knee} "
    f"binding_resource={binding_resource} binding_slo={binding_slo} "
    f"per_stream_mem_GiB_at_knee={per_stream_mem:.3f} "
    f"summary={summary_path}"
)
if treat_no_pass_as_failure and knee <= 0:
    raise SystemExit(1)
PY
}

run_density_sweep_fresh_process_per_n() {
  log "running density sweep fresh-process-per-N: N=$DENSITY_N_VALUES"
  mkdir -p "$ART_DIR/logs"
  SWEEP_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  RUN_LOG_LIST="$ART_DIR/logs/density_run_${SWEEP_STAMP}.logs"
  : > "$RUN_LOG_LIST"

  local -a n_values=()
  IFS=',' read -r -a n_values <<< "$DENSITY_N_VALUES"
  local n
  for n in "${n_values[@]}"; do
    [[ "$n" =~ ^[0-9]+$ ]] || die "invalid DENSITY_N_VALUES entry: $n"
    run_density_one_n "$n" "$@"
  done
  aggregate_density_results
}

main() {
  log "starting on-box $TARGET_SM (cc=$TARGET_CC) $PROFILE-profile export, root=$ROOT"
  log "model=$MODEL_ID buckets=$EXPECTED_BUCKETS art_dir=$ART_DIR"
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader || true

  check_build_glibc
  install_os_deps
  prepare_cuda_link_env
  setup_venv
  configure_aoti_env
  check_torch_cuda
  download_artifacts_from_s3
  prepare_artifact_workdir
  compile_steady
  compile_finalize_buckets
  compile_enc_first
  compile_steady_batched
  if [[ "$RUN_DENSITY" == "1" ]]; then
    build_density_main
    run_density_sweep_fresh_process_per_n "$@"
  else
    log "RUN_DENSITY=0; skipping density_main build + sweep"
  fi
  log "DONE: serve set in $ART_DIR (+ $STEADY_B_DIR); back up with scripts/backup_artifacts_s3.sh ARCH=$TARGET_TAG ART_DIR=$ART_DIR"
}

if [[ "${EXPORT_MODEL_SOURCE_ONLY:-0}" != "1" ]]; then
  main "$@"
fi
