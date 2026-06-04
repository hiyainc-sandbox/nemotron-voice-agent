# CUDA devel container — the glibc-compatible build env (unblocks AOTInductor + Wave-2 kernels)

## Why
The local host is Ubuntu 25.04 (**glibc 2.41**), which breaks `nvcc` for every installed CUDA (12.8 + 13.x) via the
`math_functions.h` `noexcept` conflict — so we can't compile CUDA *kernels* on the host (manual-link of prebuilt
libtorch works for graph-replay/decode, but NOT AOTInductor or custom kernels). The fix: build in an **Ubuntu-24.04
CUDA devel container (glibc 2.39)**, where `nvcc` works.

## Image — VERIFIED WORKING
**`nvidia/cuda:12.8.1-devel-ubuntu24.04`** (pulled + verified) — matches torch's cu128 and sm_120 (5090).

## Smoke test — PASSED (2026-05-25), both images
Verified command + result on the pinned **12.8** image:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-devel-ubuntu24.04 bash -c \
  'ldd --version | head -1; nvcc --version | grep release;
   echo "int main(){return 0;}" > /tmp/t.cu;
   nvcc -arch=sm_120 /tmp/t.cu -o /tmp/t && /tmp/t && echo NVCC_sm120_COMPILE_OK;
   nvidia-smi --query-gpu=name,compute_cap --format=csv | sed -n 2p'
# ->  glibc 2.39  | nvcc release 12.8, V12.8.93  | NVCC_sm120_COMPILE_OK  | NVIDIA GeForce RTX 5090, 12.0
```
- glibc **2.39** (vs host 2.41) → no `math_functions.h` `noexcept` conflict.
- **nvcc compiles for sm_120 (Blackwell)** — which **fails on the host at all CUDA versions**.
- GPU visible: **RTX 5090, sm_120**. (The 13.0/13.1-devel-ubuntu24.04 images also pass the nvcc smoke test; 12.8
  matches the torch wheel and is the one to use for AOTInductor.)

Host prerequisites (present): docker 29, nvidia-container-toolkit 1.19.1, nvidia runtime + CDI (`nvidia.com/gpu=all`),
1.5 TB free.

## Built dev image `nemotron-aoti:cu128` — VERIFIED WORKING (2026-05-25)
`docker build -t nemotron-aoti:cu128 runtime/container/` (CUDA 12.8 devel + torch 2.8.0 + cmake/g++ + soundfile).
Smoke (`docker run --rm --gpus all nemotron-aoti:cu128 ...`):
- **torch 2.8.0+cu128, cuda 12.8, cuda.is_available()=True, device cap (12,0)=sm_120** (5090 visible)
- **nvcc 12.8.93 compiles `-arch=sm_120`** → `NVCC_sm120_OK` (fails on the host)
- **`torch._inductor.aot_compile` present** → AOTInductor available

## Enter the container (project + HF model cache mounted, GPU visible)
```bash
./enter.sh                 # uses nemotron-aoti:cu128 if built, else the base CUDA image; --gpus all; cwd=runtime/
./enter.sh python3 ref_decode.py     # run a script directly (note: nemo is NOT in the image — host-export the .pt2/.ts first)
```
The image has **torch + nvcc + AOTInductor**, NOT nemo (model export/fixtures are produced on the host with nemo;
the container consumes the exported `.pt2`/`.ts` for AOTI compilation + kernel builds).

## CANONICAL build + run of the streaming density runtime (`density_main`)
**Always build AND run in this container** — the host (Ubuntu 25.04 / glibc 2.41) can't build, and its torch
(pyenv 2.10.x) is ABI-incompatible with the **2.8.0** the EPs + binaries use (every `cpp/build_*/density_main`
has RUNPATH = the container's `/usr/local/.../torch`, so it only runs in here). `enter.sh` auto-drops `-it` when
not on a TTY, so it is scriptable. Canonical settings: `TORCH_ROOT=/usr/local/lib/python3.12/dist-packages/torch`
(torch 2.8.0+cu128, sm_120), `CUDA_ROOT` defaults to `/usr/local/cuda-12.8`; `density_main` is **pure C++** (links
prebuilt libtorch, no nvcc).
```bash
# build (from runtime/)
./container/enter.sh bash -lc 'cmake -S cpp -B cpp/build_density \
  -DTORCH_ROOT=/usr/local/lib/python3.12/dist-packages/torch -DCMAKE_BUILD_TYPE=Release \
  && cmake --build cpp/build_density -j --target density_main'
# run a 5090 density sweep (sm_120 EPs in artifacts_at_sm120/; 8 sessions/worker matches the L40S gate)
./container/enter.sh bash -lc './cpp/build_density/density_main --mode density-sweep \
  --n-values 37 --density-chunk-period-ms 160 --density-sessions-per-worker 8 artifacts_at_sm120'
```
The L40S/g6e analog is the sm_89 build in `run_l40s_density.sh` (a torch-2.8.0 sm_89 venv) — same torch, different arch.

## What it's for (next steps)
1. **1.2b-wire — byte-exact C++ encoder via AOTInductor**: AOTI-compile the T2a `torch.export` steady encoder
   (`runtime/export_t2a.py` → `enc_steady_t2a.pt2`) to a `.so` (`torch._inductor.aot_compile`), load via the AOTI C++
   runtime in the streaming runtime. (`.pt2` can't be `torch::jit::load`ed; AOTI is the C++ path.)
2. **Wave-2 custom CUDA kernels** (fused encoder/decode, the 6–10 ms fusion path) — any hand-written kernel build.

## Note
GPU-byte-exact validation still happens here on the 5090 (the container sees the same GPU). The container only changes
the *build* toolchain (nvcc/glibc), not the GPU or the model.
