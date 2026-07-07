# L40S W3 Density Sweep Prep

Run `run_l40s_density.sh` on the g6e/L40S box after rsyncing the code subset below to `~/density/`.
The script is an on-box workflow: it installs/checks the DL-AMI build environment, obtains the arch-agnostic
source ExportedPrograms (EPs), compiles native AOTI packages with autotune off (sm_89 by default), builds
`cpp/density_main`, and runs the density sweep fresh-process-per-N.

> **Multi-GPU:** the script is arch-parameterized via `TARGET_CC` (the CUDA compute capability of the box it
> runs on). It defaults to `8.9` (L40S / L4). Set `TARGET_CC=8.6` to target an **A10G** (Ampere), `8.0` for an
> **A100**, etc. See [Retargeting to another GPU architecture](#retargeting-to-another-gpu-architecture) below.

**Source EPs — two ways (see [`ARTIFACTS.md`](ARTIFACTS.md)):**

1. **`EPS_LOCAL=1` (recommended, no S3):** export the EPs once from the HF checkpoint on any torch host (the 5090
   dev box works), rsync them into `artifacts_sm89/` on the g6e, then run `EPS_LOCAL=1 ./run_l40s_density.sh`. No
   bucket or credentials required. The EPs are arch-agnostic; only the AOTI compile below is sm_89-specific.
2. **S3 (optional):** stage the EPs in your own bucket and set `S3_URI=s3://YOUR-BUCKET/density`; the script then
   `aws s3 cp`s them and SHA-verifies `eps_manifest.json`. No bucket is shipped with this repo.

It does not launch an instance.

## Retargeting to another GPU architecture

The model is compiled into **AOTInductor `.pt2` packages that are GPU-architecture-specific** — an sm_89
package will not load on an sm_86 device and vice versa. Retargeting to a new GPU is driven by a single
variable, `TARGET_CC`, the CUDA compute capability of the box the script runs on.

```bash
# A10G (Ampere, sm_86):
cd ~/density && TARGET_CC=8.6 EPS_LOCAL=1 ./run_l40s_density.sh
# L40S / L4 (Ada, sm_89) — the default, unchanged:
cd ~/density && EPS_LOCAL=1 ./run_l40s_density.sh
```

### What `TARGET_CC` drives

Setting `TARGET_CC=X.Y` derives two tokens and threads them through the whole compile:

| derived | value for `8.6` | value for `8.9` (default) | used for |
|---|---|---|---|
| `TARGET_SM` (arch token, `sm_${cc}`) | `sm_86` | `sm_89` | `TORCH_CUDA_ARCH_LIST`, device-capability assertion, wheel `get_arch_list()` membership check, log lines |
| `TARGET_TAG` (dir suffix, no underscore) | `sm86` | `sm89` | `artifacts_sm86/`, `torch280-sm86-venv`, `cpp/build_sm86_density/` |

Concretely, `TARGET_CC=8.6`:
- exports `TORCH_CUDA_ARCH_LIST=8.6` for the AOTI compile,
- asserts the live device is compute capability `(8, 6)` (fail-closed — a mismatched box is rejected before
  compiling, e.g. `expected sm_86 device capability (8, 6), got (8, 9)`),
- writes all artifacts to `artifacts_sm86/`, uses venv `~/torch280-sm86-venv`, builds in `cpp/build_sm86_density/`.

Override any of these individually with the usual env vars (`ART_SM89=…`, `VENV=…`, `BUILD_DIR=…`) if you need
non-default paths; `TARGET_CC` only sets their defaults.

### Why this is the *only* change needed

The two-stage artifact pipeline (see [`ARTIFACTS.md`](ARTIFACTS.md)) is built for exactly this. Stage-1 **export**
(`export_*.py`) produces **architecture-agnostic** intermediates — all `*.ts` TorchScript modules and all `*.pt2`
**ExportedPrograms** (the steady encoder EP, the 32 finalize-bucket EPs, shared weights, session bundle). These are
portable across GPUs: export once on any torch+NeMo host (the 5090 dev box is fine), rsync into the target's
`artifacts_<tag>/`, done. Only stage-2 (AOTI compile + strip) is arch-specific, and it is entirely driven by the
live device + `TORCH_CUDA_ARCH_LIST` — the compile/strip Python (`aot_compile*.py`, `aot_compile_buckets.py`,
`strip_*.py`) contains **no** arch tokens, and the **C++/CMake build is arch-neutral** (`cpp/CMakeLists.txt` is
`LANGUAGES C CXX` — no nvcc, no `-arch`, no `CMAKE_CUDA_ARCHITECTURES`; the binaries just `dlopen` the `.pt2` at
runtime). So `run_l40s_density.sh` is the single place the arch is gated.

### Per-GPU notes

- **A10G (sm_86):** the stock `torch 2.8.0+cu128` wheel already ships prebuilt SASS for sm_86 (unlike sm_89), so the
  wheel-`arch_list` sanity check is a clean no-op — sm_86 is actually better-supported than sm_89 was. **24 GB** of
  VRAM (vs the L40S's 48 GB) means the density knee-N is lower and you will hit the memory-bound regime sooner; that
  is a capacity difference, not a compile problem. AWS: `g5` instances.
- **L4 (sm_89):** L4 is the *same* architecture as the L40S — it needs **no** new compile at all. The existing sm_89
  artifacts run on an L4 as-is; only capacity/density-N differs (L4 is also 24 GB). Just run the default.
- **Must compile on the target GPU.** AOTI/Triton codegen targets the *live* device — you cannot cross-compile sm_86
  from a 5090/L40S box. Run the script on the actual A10G.

### Optional — full `ws_server` serving on the new arch

This script covers the density sweep, which needs only the steady encoder + the 32 finalize buckets. Serving via
`ws_server` additionally needs the **batched-steady buckets** (`steady_b_artifacts_b16/enc_steady_aoti_b{1,2,4,8,16}.pt2`),
which are also arch-specific and are **not** produced here. Compile them on the target GPU:

```bash
TORCH_CUDA_ARCH_LIST=8.6 python export_steady_batched.py --out ./artifacts_sm86 --compile-only
python strip_steady_buckets.py    # rebinds the shared weight set
```

(`enc_first` is consumed by the server as arch-agnostic `enc_first.ts`, so it needs no AOTI recompile.)

## Run Command

```bash
cd ~/density
./run_l40s_density.sh
```

Default N sweep:

```text
1,8,16,24,32,40,48,64,80
```

Useful overrides:

```bash
EPS_LOCAL=1 ./run_l40s_density.sh                              # no-S3 path: use pre-placed EPs in artifacts_sm89/
TARGET_CC=8.6 EPS_LOCAL=1 ./run_l40s_density.sh                # retarget A10G/sm_86 (EPs pre-placed in artifacts_sm86/)
DENSITY_N_VALUES=1,8,16,24,32,40,48,64,80 ./run_l40s_density.sh
FORCE_S3_DOWNLOAD=1 ./run_l40s_density.sh                      # S3 mode only: re-download even if files exist
DENSITY_TREAT_NO_PASS_AS_FAILURE=1 ./run_l40s_density.sh
```

## Profiling on the L40S (ncu + nsys) — the SM-work attribution

Profile **on the L40S target** (arch-specific; the 5090 wouldn't transfer). nsys gives the contention *timeline*
(launch gaps / GPU-idle% / stream overlap under load → launch-bound vs compute-bound); ncu gives the per-kernel
*roofline* (occupancy / DRAM throughput → is the steady encoder mem-BW-bound?).

### One-time tooling setup (verified 2026-05-27 on the g6e DL AMI)
```bash
# ncu: already present at /usr/local/cuda/bin/ncu, BUT the driver restricts perf counters to root
#   (cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly  -> 1)  => run ncu via sudo.
# nsys: install from the configured CUDA apt repo (pick a version matching the runtime's CUDA 12.8):
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nsight-systems-2024.6.2
sudo chmod o+rx /opt/nvidia /opt/nvidia/nsight-systems          # the install leaves /opt/nvidia/nsight-systems as drwx------; this lets non-root run nsys
sudo ln -sf /opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys /usr/local/bin/nsys
nsys --version    # NVIDIA Nsight Systems version 2024.6.2...
```

### The env (REQUIRED — the cudart-12-vs-13 fix)
`density_main`'s RUNPATH points at cuda-13 (cudart-13), but torch 2.8.0+cu128 needs **cudart-12** (bundled in the
venv). A bare profiler invocation hits `libcudart.so.12 => not found`. Set `LD_LIBRARY_PATH` to the venv torch/lib +
the venv `nvidia/*/lib` (where cudart-12 lives) + cuda-13/lib64:
```bash
cd ~/density
LD=/home/ubuntu/torch280-sm89-venv/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$LD/torch/lib:$(ls -d $LD/nvidia/*/lib | tr '\n' ':')/usr/local/cuda-13.0/lib64"
BIN=./cpp/build_sm89_density/density_main    # build dir is arch-tagged: build_sm86_density on an A10G, etc.
# The default density policy counts cross-arch interim event-timing drifts but does not gate on them.
# Set DENSITY_GOLD_EVENTS_TOLERANT=1 only for opt-in strict byte-exact event debug runs.
```

### nsys — multi-stream timeline (launch-bound vs compute-bound at the knee)
Low-overhead (CUPTI); safe to run on a real multi-stream sweep. Use a SHORT config (`--density-sessions-per-worker 2`)
and skip the serial-oracle/warmup setup with `--delay` so the trace captures the **measured gate** (the setup is
~250s = oracle ~176s + warmup ~66s; tune `--delay` from the run's `DENSITY_PHASE_TIMING` lines):
```bash
LD_LIBRARY_PATH="$LD_LIBRARY_PATH" SELF_CHECK_ATOL=0.2 \
  nsys profile --trace=cuda --sample=none --force-overwrite=true --delay=255 --duration=45 \
    -o ~/density/nsys_n38 \
    "$BIN" --n-values 38 --density-sessions-per-worker 2 artifacts_sm89
# summarize without the GUI:
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum ~/density/nsys_n38.nsys-rep
# GPU-idle% / launch gaps: inspect the kernel timeline gaps in the report (or `nsys stats --report cuda_gpu_trace`).
```

### ncu — single-stream roofline (is the encoder mem-BW-bound?)
ncu **serializes + replays** each profiled kernel → run **single-stream (N=1)** and cap with `--launch-count`; never
run ncu on a multi-stream sweep. Needs **root** (perf counters), and `sudo` drops env → pass it with `sudo env`:
```bash
sudo env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" SELF_CHECK_ATOL=0.2 \
  /usr/local/cuda/bin/ncu --set roofline --launch-count 40 --target-processes all --force-overwrite \
    -o ~/density/ncu_enc \
    "$BIN" --n-values 1 --density-sessions-per-worker 1 artifacts_sm89
ncu -i ~/density/ncu_enc.ncu-rep --page details --csv | head   # occupancy, DRAM throughput, SOL%, roofline
```

**Notes:** if `RmProfilingAdminOnly` is `0` (driver flag `NVreg_RestrictProfilingToAdminUsers=0` + reboot), ncu runs
without sudo. The `--delay/--duration` window must land in the measured gate — confirm against `DENSITY_PHASE_TIMING
phase=measured-gate`. Keep autotune OFF (same as the sweep). Both tools profile the *same* binary the gate uses.

## Script Flow

1. Installs/checks DL-AMI dependencies: `build-essential`, `cmake`, `ninja-build`, `python3-dev`, `python3-venv`,
   `awscli`, `curl`, and certs.
2. Locates system CUDA and `nvcc` under `/usr/local/cuda*`.
3. Creates a `torch==2.8.0` venv with `uv` when available, otherwise `venv` plus `pip`.
4. Confirms the box is x86_64 and `torch.cuda.get_device_capability()` matches `TARGET_CC` (default `(8, 9)`;
   `(8, 6)` when `TARGET_CC=8.6`). Fail-closed: a device whose capability differs from `TARGET_CC` is rejected
   before any compile.
5. Ports the bare-AMI AOTI fixes: `python3-dev`, unversioned `libcuda.so` plus CUDA stub symlink for Triton link
   probes, and fail-closed `-Wl,-z,noexecstack` injection during AOTI shared-library links.
6. Obtains source EPs into the arch-tagged artifacts dir (`artifacts_sm89/` by default, `artifacts_sm86/` when
   `TARGET_CC=8.6`): with `EPS_LOCAL=1` it uses the pre-placed EPs (no S3); otherwise it downloads from `$S3_URI`
   and verifies SHA256s against `eps_manifest.json`.
7. Compiles `enc_steady_t2a.pt2` to `artifacts_sm89/enc_steady_aoti.pt2`, autotune off.
8. Compiles the 32 finalize bucket EPs with `aot_compile_buckets.py`, runs per-bucket self-checks at
   `SELF_CHECK_ATOL=0.1`, strips weights with `strip_bucket_weights.py`, and writes
   `artifacts_sm89/stripped_finalize_buckets/manifest.json`.
9. Builds the current shared+locked `enc_first.ts`, explicit-stream, `num_runners=N`, capped-finalize-pool
   `density_main`.
10. Runs each N in a fresh process and prints `L40S_DENSITY_ROW` plus `L40S_DENSITY_RESULT`.

## Rsync From This Box

Rsync these paths under `runtime/` to `~/density/` on the g6e:

```bash
rsync -av \
  runtime/run_l40s_density.sh \
  runtime/aot_compile.py \
  runtime/aot_compile_buckets.py \
  runtime/strip_bucket_weights.py \
  g6e:~/density/

rsync -av \
  runtime/cpp/CMakeLists.txt \
  runtime/cpp/density_main.cpp \
  runtime/cpp/session_main.cpp \
  g6e:~/density/cpp/
```

| Path under `runtime/` | Bytes | Size |
|---|---:|---:|
| `run_l40s_density.sh` | 32,468 | 0.032 MB |
| `aot_compile.py` | 3,960 | 0.004 MB |
| `aot_compile_buckets.py` | 14,907 | 0.015 MB |
| `strip_bucket_weights.py` | 24,930 | 0.025 MB |
| `cpp/density_main.cpp` | 178,074 | 0.178 MB |
| `cpp/session_main.cpp` | 209,192 | 0.209 MB |
| `cpp/CMakeLists.txt` | 4,831 | 0.005 MB |

Rsync subtotal: **468,362 bytes = 0.468 MB**.

## Source artifact set (the EPs the sweep needs)

These are the keys placed under the arch-tagged artifacts dir (`artifacts_sm89/` shown; use `artifacts_sm86/` for
an A10G target) — by `EPS_LOCAL=1` rsync (recommended) or by S3 download into your own bucket. The EPs are
arch-agnostic, so the *same* files are staged regardless of `TARGET_CC`; only the AOTI compile output differs. Sizes below are from a validated artifact set. In S3 mode the script fail-closes on SHA mismatches
for the required artifacts in `eps_manifest.json`; the small helper artifacts are SHA-verified when the manifest
lists them. Note the EPs are large (~84 GiB total) — this is why they are regenerated/staged rather than committed.

| key under `artifacts_sm89/` | Bytes | Size |
|---|---:|---:|
| `eps_manifest.json` | manifest | 0.009 MB expected |
| `buckets_manifest.json` | 2,435 | 0.002 MB |
| `enc_steady_t2a.pt2` | 2,490,209,376 | 2,490.209 MB |
| `session_bundle.ts` | 667,603,966 | 667.604 MB |
| `finalize_shared_weights.pt` | 2,477,736,629 | 2,477.737 MB |
| `finalize_shared_weights.ts` | 2,477,725,779 | 2,477.726 MB |
| `enc_first.ts` | 2,478,955,502 | 2,478.956 MB |
| `t2a_io.pt` | 15,359,925 | 15.360 MB |
| `joint_step.ts` | 6,909,312 | 6.909 MB |
| `predict_step.ts` | 28,890,948 | 28.891 MB |
| `finalize_buckets/enc_finalize_d0_T34_ep.pt2` | 2,490,205,928 | 2,490.206 MB |
| `finalize_buckets/enc_finalize_d0_T35_ep.pt2` | 2,490,206,440 | 2,490.206 MB |
| `finalize_buckets/enc_finalize_d0_T36_ep.pt2` | 2,490,206,952 | 2,490.207 MB |
| `finalize_buckets/enc_finalize_d0_T37_ep.pt2` | 2,490,207,464 | 2,490.207 MB |
| `finalize_buckets/enc_finalize_d0_T38_ep.pt2` | 2,490,207,976 | 2,490.208 MB |
| `finalize_buckets/enc_finalize_d0_T39_ep.pt2` | 2,490,208,488 | 2,490.208 MB |
| `finalize_buckets/enc_finalize_d0_T40_ep.pt2` | 2,490,209,000 | 2,490.209 MB |
| `finalize_buckets/enc_finalize_d0_T41_ep.pt2` | 2,490,209,512 | 2,490.210 MB |
| `finalize_buckets/enc_finalize_d0_T42_ep.pt2` | 2,490,210,280 | 2,490.210 MB |
| `finalize_buckets/enc_finalize_d0_T43_ep.pt2` | 2,490,210,792 | 2,490.211 MB |
| `finalize_buckets/enc_finalize_d0_T44_ep.pt2` | 2,490,211,304 | 2,490.211 MB |
| `finalize_buckets/enc_finalize_d0_T45_ep.pt2` | 2,490,211,816 | 2,490.212 MB |
| `finalize_buckets/enc_finalize_d0_T46_ep.pt2` | 2,490,212,328 | 2,490.212 MB |
| `finalize_buckets/enc_finalize_d0_T47_ep.pt2` | 2,490,212,840 | 2,490.213 MB |
| `finalize_buckets/enc_finalize_d0_T48_ep.pt2` | 2,490,213,352 | 2,490.213 MB |
| `finalize_buckets/enc_finalize_d0_T49_ep.pt2` | 2,490,213,864 | 2,490.214 MB |
| `finalize_buckets/enc_finalize_d2_T43_ep.pt2` | 2,490,216,104 | 2,490.216 MB |
| `finalize_buckets/enc_finalize_d2_T44_ep.pt2` | 2,490,216,616 | 2,490.217 MB |
| `finalize_buckets/enc_finalize_d2_T45_ep.pt2` | 2,490,217,128 | 2,490.217 MB |
| `finalize_buckets/enc_finalize_d2_T46_ep.pt2` | 2,490,217,640 | 2,490.218 MB |
| `finalize_buckets/enc_finalize_d2_T47_ep.pt2` | 2,490,218,152 | 2,490.218 MB |
| `finalize_buckets/enc_finalize_d2_T48_ep.pt2` | 2,490,218,664 | 2,490.219 MB |
| `finalize_buckets/enc_finalize_d2_T49_ep.pt2` | 2,490,219,176 | 2,490.219 MB |
| `finalize_buckets/enc_finalize_d2_T50_ep.pt2` | 2,490,219,688 | 2,490.220 MB |
| `finalize_buckets/enc_finalize_d2_T51_ep.pt2` | 2,490,220,200 | 2,490.220 MB |
| `finalize_buckets/enc_finalize_d2_T52_ep.pt2` | 2,490,220,712 | 2,490.221 MB |
| `finalize_buckets/enc_finalize_d2_T53_ep.pt2` | 2,490,221,224 | 2,490.221 MB |
| `finalize_buckets/enc_finalize_d2_T54_ep.pt2` | 2,490,221,736 | 2,490.222 MB |
| `finalize_buckets/enc_finalize_d2_T55_ep.pt2` | 2,490,222,248 | 2,490.222 MB |
| `finalize_buckets/enc_finalize_d2_T56_ep.pt2` | 2,490,222,760 | 2,490.223 MB |
| `finalize_buckets/enc_finalize_d2_T57_ep.pt2` | 2,490,223,272 | 2,490.223 MB |
| `finalize_buckets/enc_finalize_d2_T58_ep.pt2` | 2,490,224,040 | 2,490.224 MB |

S3 subtotal excluding `eps_manifest.json`: **90,330,271,568 bytes = 90.330 GB = 84.127 GiB**.

`t2a_io.pt`, `joint_step.ts`, and `predict_step.ts` are small runtime/helper artifacts required by the current
compile self-check, strip path, and `density_main`, even though the W3 gate artifacts are the EPs, bundle, shared
weights, bucket manifest, and `enc_first.ts`.

## Bare-AMI Risks

- `eps_manifest.json` must include SHA entries for the W3 required artifacts, including `enc_first.ts`; otherwise
  the script fail-closes before compile.
- The current `density_main` requires `joint_step.ts` and `predict_step.ts` at runtime. If those were not uploaded
  to S3 with the current artifact set, upload them or place them under `artifacts_sm89/` before running.
- The AOTI link path needs a visible unversioned `libcuda.so`. The script creates both the driver-lib symlink and
  CUDA stub symlink, then adds them to `LIBRARY_PATH`.
- Autotune is intentionally off. Do not set max-autotune env vars for this W3 run; the script asserts Inductor
  `max_autotune` and `coordinate_descent_tuning` are disabled.
