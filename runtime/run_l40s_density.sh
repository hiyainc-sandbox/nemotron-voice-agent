#!/usr/bin/env bash
# On-box density sweep for a g6e/L40S (sm_89) DL Base AMI.
#
# Run after this runtime subset is rsynced to the g6e. The script obtains the
# arch-agnostic source ExportedPrograms (EPs), compiles native sm_89 autotune-off
# AOTI packages into artifacts_sm89/, builds density_main, and runs each N in a
# fresh process.
#
# Two ways to supply the source EPs (see runtime/ARTIFACTS.md):
#   1. EPS_LOCAL=1 (recommended, no S3): export the EPs from the HF checkpoint on
#      any torch host (the 5090 dev box works), rsync them into artifacts_sm89/,
#      then run with EPS_LOCAL=1. No bucket or credentials needed.
#   2. S3: set S3_URI to your own bucket prefix; the script downloads + SHA-verifies
#      eps_manifest.json. (You produce the bucket yourself; none is shipped.)
set -euo pipefail
IFS=$'\n\t'

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT=$(pwd -P)

EPS_LOCAL=${EPS_LOCAL:-0}                       # 1 = use EPs already present in artifacts_sm89/ (no S3)
S3_URI=${S3_URI:-"s3://YOUR-BUCKET/density"}     # only used when EPS_LOCAL != 1; point at your own bucket
ART_SM89=${ART_SM89:-"$ROOT/artifacts_sm89"}
VENV=${VENV:-"$HOME/torch280-sm89-venv"}
PYTHON_BIN=${PYTHON_BIN:-python3}
BUILD_DIR=${BUILD_DIR:-"$ROOT/cpp/build_l40s_density"}
SELF_CHECK_ATOL=${SELF_CHECK_ATOL:-0.1}
DENSITY_N_VALUES=${DENSITY_N_VALUES:-"1,8,16,24,32,40,48,64,80"}
DENSITY_SESSIONS_PER_WORKER=${DENSITY_SESSIONS_PER_WORKER:-0}
DENSITY_CHUNK_PERIOD_MS=${DENSITY_CHUNK_PERIOD_MS:-160}
DENSITY_TREAT_NO_PASS_AS_FAILURE=${DENSITY_TREAT_NO_PASS_AS_FAILURE:-0}
FORCE_S3_DOWNLOAD=${FORCE_S3_DOWNLOAD:-0}
KEEP_UNSTRIPPED_BUCKETS=${KEEP_UNSTRIPPED_BUCKETS:-0}
STRICT_EPS_MANIFEST=${STRICT_EPS_MANIFEST:-1}
ALLOW_UNMANIFESTED_AUX=${ALLOW_UNMANIFESTED_AUX:-1}
SKIP_EPS_VERIFY=${SKIP_EPS_VERIFY:-0}

PY=""
CUDA_ROOT=${CUDA_ROOT:-}
SWEEP_STAMP=""
RUN_LOG_LIST=""

log() {
  printf '[l40s-density %(%H:%M:%S)T] %s\n' -1 "$*"
}

die() {
  printf '[l40s-density ERROR] %s\n' "$*" >&2
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
    python3-venv \
    awscli >/dev/null

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
  export TORCHINDUCTOR_CACHE_DIR="$ART_SM89/torchinductor_cache"
  export TORCHINDUCTOR_MAX_AUTOTUNE=0
  export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=0
  export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=0
  export TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0
  export TORCH_CUDA_ARCH_LIST="8.9"
  unset TORCHINDUCTOR_FREEZING
  log "AOTI compile env: autotune OFF, TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
}

check_torch_cuda() {
  log "checking torch 2.8.0 CUDA device and sm_89 capability"
  "$PY" - <<'PY'
import os
import platform
import torch
import torch._inductor.config as config

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
if cc != (8, 9):
    raise SystemExit(f"expected L40S sm_89 device capability (8, 9), got {cc}")
# The stock torch 2.8.0+cu128 wheel ships SASS for sm_86/sm_90/sm_120 but NOT sm_89.
# That arch_list is about PREBUILT kernels and does NOT gate AOTI codegen: Inductor/Triton
# compile hot kernels for the LIVE device (sm_89), and the cpp/nvcc parts honor
# TORCH_CUDA_ARCH_LIST. Device cc==(8,9) is the real hardware gate (asserted above); the
# meaningful AOTI-target gate is that TORCH_CUDA_ARCH_LIST requests 8.9.
arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
print("TORCH_CUDA_ARCH_LIST", repr(arch_list_env))
if "sm_89" not in arch:
    print(f"WARNING: wheel arch_list lacks sm_89 ({arch}); AOTI targets sm_89 via TORCH_CUDA_ARCH_LIST")
    if "8.9" not in arch_list_env:
        raise SystemExit(
            f"sm_89 absent from wheel arch_list AND TORCH_CUDA_ARCH_LIST does not request 8.9: {arch_list_env!r}"
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
  "$PY" - "$manifest" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
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
if len(names) != 32:
    raise SystemExit(f"expected 32 finalize bucket EPs, got {len(names)} from {path}")
for name in names:
    print(name)
PY
}

download_artifacts_from_s3() {
  mkdir -p "$ART_SM89/finalize_buckets" "$ART_SM89/logs"

  if [[ "$EPS_LOCAL" == "1" ]]; then
    # No S3: the EPs were exported on a torch host and rsynced into artifacts_sm89/.
    log "EPS_LOCAL=1; using pre-placed source EPs in $ART_SM89 (no S3 download)"
    local req missing=0
    for req in enc_steady_t2a.pt2 session_bundle.ts finalize_shared_weights.pt \
               finalize_shared_weights.ts enc_first.ts t2a_io.pt joint_step.ts predict_step.ts \
               finalize_buckets/buckets_manifest.json; do
      [[ -f "$ART_SM89/$req" ]] || { log "MISSING required EP: $ART_SM89/$req"; missing=1; }
    done
    local n_bucket_eps
    n_bucket_eps=$(find "$ART_SM89/finalize_buckets" -maxdepth 1 -type f -name 'enc_finalize_d*_T*_ep.pt2' 2>/dev/null | wc -l)
    [[ "$n_bucket_eps" == "32" ]] || { log "expected 32 finalize bucket EPs in finalize_buckets/, found $n_bucket_eps"; missing=1; }
    [[ "$missing" == "0" ]] || die "EPS_LOCAL=1 but the local export is incomplete in $ART_SM89 (see runtime/ARTIFACTS.md for the export + rsync manifest)"
    return 0
  fi

  log "downloading source artifacts from $S3_URI"

  download_s3_object "eps_manifest.json" "$ART_SM89/eps_manifest.json"
  download_s3_object "buckets_manifest.json" "$ART_SM89/buckets_manifest.json"
  cp -f "$ART_SM89/buckets_manifest.json" "$ART_SM89/finalize_buckets/buckets_manifest.json"

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
  local key
  for key in "${root_keys[@]}"; do
    download_s3_object "$key" "$ART_SM89/$key"
  done

  local bucket_list="$ART_SM89/finalize_buckets/.bucket_eps"
  parse_bucket_manifest "$ART_SM89/buckets_manifest.json" > "$bucket_list"
  while IFS= read -r ep_name; do
    [[ -n "$ep_name" ]] || continue
    download_s3_object "finalize_buckets/$ep_name" "$ART_SM89/finalize_buckets/$ep_name"
  done < "$bucket_list"

  verify_s3_artifacts
}

verify_s3_artifacts() {
  if [[ "$SKIP_EPS_VERIFY" == "1" ]]; then
    log "SKIP_EPS_VERIFY=1; skipping eps_manifest SHA256 verification"
    return 0
  fi
  log "verifying downloaded artifacts against eps_manifest.json"
  "$PY" - "$ART_SM89" "$STRICT_EPS_MANIFEST" "$ALLOW_UNMANIFESTED_AUX" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

art = Path(sys.argv[1])
strict = sys.argv[2] == "1"
allow_unmanifested_aux = sys.argv[3] == "1"
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
        marker = "/density/"
        if marker in value:
            value = value.split(marker, 1)[1]
        else:
            value = value.rsplit("/", 1)[-1]
    while value.startswith("./"):
        value = value[2:]
    for prefix in ("runtime/artifacts/", "artifacts/"):
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
if len(bucket_eps) != 32:
    raise SystemExit(f"expected 32 bucket EPs in buckets_manifest.json, got {len(bucket_eps)}")

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
  log "preparing native sm_89 artifact workdir: $ART_SM89"
  need_file "$ART_SM89/enc_steady_t2a.pt2"
  need_file "$ART_SM89/t2a_io.pt"
  need_file "$ART_SM89/session_bundle.ts"
  need_file "$ART_SM89/finalize_shared_weights.pt"
  need_file "$ART_SM89/finalize_shared_weights.ts"
  need_file "$ART_SM89/enc_first.ts"
  need_file "$ART_SM89/joint_step.ts"
  need_file "$ART_SM89/predict_step.ts"
  need_dir "$ART_SM89/finalize_buckets"
  need_file "$ART_SM89/finalize_buckets/buckets_manifest.json"

  rm -f "$ART_SM89/enc_steady_aoti.pt2"
  # Keep stripped_finalize_buckets across re-runs (compile_strip_bucket skips ones already present);
  # only the per-bucket scratch is wiped. The manifest epilogue revalidates all-32-present + SHAs.
  rm -rf "$ART_SM89/finalize_compile_work"
  mkdir -p "$ART_SM89/stripped_finalize_buckets" "$ART_SM89/finalize_compile_work/manifests"
  find "$ART_SM89/finalize_buckets" -maxdepth 1 -type f \
    -name 'enc_finalize_d*_T*.pt2' ! -name '*_ep.pt2' -delete
}

compile_steady_sm89() {
  log "native sm_89 AOTI compile, autotune OFF: enc_steady_t2a.pt2 -> enc_steady_aoti.pt2"
  local script_art="$ROOT/artifacts"
  local art_real script_real
  art_real=$(realpath "$ART_SM89")
  script_real=""
  if [[ ! -e "$script_art" && ! -L "$script_art" ]]; then
    ln -s "$art_real" "$script_art"
  fi
  if [[ -e "$script_art" || -L "$script_art" ]]; then
    script_real=$(realpath "$script_art" 2>/dev/null || true)
  fi
  if [[ "$script_real" == "$art_real" ]]; then
    log "using aot_compile.py against $script_art -> $ART_SM89"
    "$PY" "$ROOT/aot_compile.py"
    need_file "$ART_SM89/enc_steady_aoti.pt2"
    return
  fi

  log "existing $script_art is not $ART_SM89; using embedded aot_compile.py-equivalent steady compiler"
  "$PY" - "$ART_SM89" <<'PY'
import os
import sys
import torch

art = sys.argv[1]

def force_noexecstack_on_link():
    import torch._inductor.cpp_builder as cb
    orig = cb.CppBuilder.get_command_line
    seen = {"flagged": False}
    def patched(self):
        cmd = orig(self)
        if getattr(self, "_do_link", False) and "-shared" in cmd:
            if "-Wl,-z,noexecstack" not in cmd:
                cmd += " -Wl,-z,noexecstack"
            seen["flagged"] = True
            print("[noexecstack] injected into shared-lib link:", cmd[-160:])
        return cmd
    cb.CppBuilder.get_command_line = patched
    return seen

print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "cc", torch.cuda.get_device_capability())
seen = force_noexecstack_on_link()
ep = torch.export.load(os.path.join(art, "enc_steady_t2a.pt2"))
pkg = os.path.join(art, "enc_steady_aoti.pt2")
out_path = torch._inductor.aoti_compile_and_package(ep, package_path=pkg)
if not seen["flagged"]:
    raise SystemExit("noexecstack shim never fired on a shared-lib link")
print("AOTI package:", out_path)

runner = torch._inductor.aoti_load_package(out_path)
io = torch.load(os.path.join(art, "t2a_io.pt"), weights_only=False)
inputs = [io["chunk"].cuda(), io["L"].cuda(), io["clc"].cuda(), io["clt"].cuda(), io["clcl"].cuda()]
with torch.inference_mode():
    out = runner(*inputs)
outs = list(out) if isinstance(out, (list, tuple)) else [out]
ref = [t.cuda() for t in io["out"]]
names = ["enc_out", "enc_len", "cache_ch", "cache_t", "cache_ch_len"]
maxd = 0.0
for name, expected, actual in zip(names, ref, outs):
    if not torch.is_tensor(expected) or not torch.is_tensor(actual):
        continue
    if expected.shape != actual.shape:
        raise SystemExit(f"{name} shape mismatch: {tuple(expected.shape)} vs {tuple(actual.shape)}")
    diff = (expected.float() - actual.float()).abs().max().item() if expected.numel() else 0.0
    maxd = max(maxd, diff)
    print(f"  {name}: byte_equal={torch.equal(expected, actual)} max_abs_diff={diff:.3e} shape={tuple(actual.shape)}")
print(f"=== steady AOTI load+run OK: max_abs_diff={maxd:.3e} ===")
PY
}

compile_strip_bucket() {
  local ep=$1
  local base key pkg one_dir
  base=$(basename "$ep")
  key=${base%_ep.pt2}
  pkg=${base/_ep.pt2/.pt2}
  one_dir="$ART_SM89/finalize_compile_work/$key"

  # Idempotent across re-runs: a stripped bucket that already compiled+self-checked is
  # deterministic, so reuse it instead of burning compile minutes (e.g. an atol re-run that
  # only needs to recompile one outlier bucket). A clean rebuild = rm -rf stripped_finalize_buckets.
  if [[ -f "$ART_SM89/stripped_finalize_buckets/$pkg" ]]; then
    log "stripped bucket exists, skipping recompile: $pkg"
    return
  fi

  rm -rf "$one_dir"
  mkdir -p "$one_dir"
  ln -sfn "$(realpath "$ep")" "$one_dir/$base"

  log "native sm_89 AOTI compile bucket, autotune OFF: $base"
  "$PY" "$ROOT/aot_compile_buckets.py" \
    --dir "$one_dir" \
    --shared-weights "$ART_SM89/finalize_shared_weights.pt" \
    --force \
    --self-check-atol "$SELF_CHECK_ATOL"

  need_file "$one_dir/$pkg"
  cp -f "$one_dir/buckets_manifest.json" "$ART_SM89/finalize_compile_work/manifests/$key.json"

  log "strip bucket weights: $pkg"
  "$PY" "$ROOT/strip_bucket_weights.py" \
    --bucket "$one_dir/$pkg" \
    --out-dir "$ART_SM89/stripped_finalize_buckets" \
    --shared-weights "$ART_SM89/finalize_shared_weights.pt" \
    --bundle "$ART_SM89/session_bundle.ts" \
    --joint "$ART_SM89/joint_step.ts" \
    --predict "$ART_SM89/predict_step.ts" \
    --strip-only \
    --force

  if [[ "$KEEP_UNSTRIPPED_BUCKETS" == "1" ]]; then
    mv -f "$one_dir/$pkg" "$ART_SM89/finalize_buckets/$pkg"
  fi
  rm -rf "$one_dir"
}

compile_finalize_buckets_sm89() {
  log "native sm_89 AOTI compile + strip for 32 finalize buckets, self-check-atol=$SELF_CHECK_ATOL"
  local -a eps=()
  while IFS= read -r path; do
    eps+=("$path")
  done < <(find "$ART_SM89/finalize_buckets" -maxdepth 1 -type f -name 'enc_finalize_d*_T*_ep.pt2' | sort)
  ((${#eps[@]} == 32)) || die "expected 32 finalize bucket EPs, found ${#eps[@]} in $ART_SM89/finalize_buckets"

  local ep
  for ep in "${eps[@]}"; do
    compile_strip_bucket "$ep"
  done

  "$PY" - "$ART_SM89" <<'PY'
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

art = Path(sys.argv[1])
bucket_re = re.compile(r"^enc_finalize_d(?P<drop>\d+)_T(?P<T>\d+)\.pt2$")

source_manifest = json.loads((art / "buckets_manifest.json").read_text())
if isinstance(source_manifest, dict):
    source_entries = source_manifest.get("buckets") or source_manifest.get("contract") or []
else:
    source_entries = source_manifest
expected = {(int(b["drop"]), int(b["T"])) for b in source_entries if "drop" in b and "T" in b}
if len(expected) != 32:
    raise SystemExit(f"expected 32 source bucket keys, got {len(expected)}")

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
    "att_context": [70, 1],
    "blank": 1024,
    "drop_extra": 2,
    "final_padding_frames": 32,
    "max_symbols": 10,
    "model_id": "nvidia/nemotron-speech-streaming-en-0.6b",
    "pre_encode_cache": 9,
    "right_context": 1,
    "shift": 16,
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
(art / "finalize_buckets" / "buckets_manifest_sm89.json").write_text(
    json.dumps(compile_records, indent=2, sort_keys=True) + "\n"
)
print(f"wrote stripped manifest: buckets={len(buckets)} path={stripped_dir / 'manifest.json'}")
print(f"wrote compile self-check manifest: buckets={len(compile_records)}")
PY
}

build_density_main() {
  log "building current Fix-2 + unique-streams density_main with pip torch libtorch and system CUDA"
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
  local run_log="$ART_SM89/logs/l40s_density_N${n}_${SWEEP_STAMP}.stdout.log"
  local -a cmd=(
    "$BUILD_DIR/density_main"
    --mode density-sweep
    --n-values "$n"
    --density-chunk-period-ms "$DENSITY_CHUNK_PERIOD_MS"
  )
  if [[ "$DENSITY_SESSIONS_PER_WORKER" != "0" ]]; then
    cmd+=(--density-sessions-per-worker "$DENSITY_SESSIONS_PER_WORKER")
  fi
  cmd+=("$ART_SM89")
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
  "$PY" - "$RUN_LOG_LIST" "$ART_SM89/logs/l40s_density_summary_${SWEEP_STAMP}.json" "$DENSITY_TREAT_NO_PASS_AS_FAILURE" <<'PY'
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
        elif obj.get("check") == "1a_density_sweep_full_session" and "N" in obj:
            rows[int(obj["N"])] = obj

if not rows:
    print("L40S_DENSITY_RESULT status=NO_ROWS")
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
        "L40S_DENSITY_ROW "
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
    "L40S_DENSITY_RESULT "
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
  log "running W3 density sweep fresh-process-per-N: N=$DENSITY_N_VALUES"
  mkdir -p "$ART_SM89/logs"
  SWEEP_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  RUN_LOG_LIST="$ART_SM89/logs/l40s_density_run_${SWEEP_STAMP}.logs"
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
  log "starting on-box L40S W3 run, root=$ROOT"
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader || true

  install_os_deps
  prepare_cuda_link_env
  setup_venv
  configure_aoti_env
  check_torch_cuda
  download_artifacts_from_s3
  prepare_artifact_workdir
  compile_steady_sm89
  compile_finalize_buckets_sm89
  build_density_main
  run_density_sweep_fresh_process_per_n "$@"
  log "DONE"
}

if [[ "${RUN_L40S_DENSITY_SOURCE_ONLY:-0}" != "1" ]]; then
  main "$@"
fi
