# Reproducible glibc-compatible build env for the from-scratch runtime (AOTInductor + Wave-2 kernels).
# Ubuntu 24.04 (glibc 2.39) -> nvcc works (the host's glibc 2.41 breaks it). CUDA 12.8 matches torch's cu128 + sm_120.
#   build: docker build -t nemotron-aoti:cu128 runtime/container/
#   run:   ./enter.sh   (uses this image if built; falls back to the base CUDA image)
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv python3-dev cmake g++ git libsndfile1 ffmpeg \
      libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# AOTInductor/Triton compile its cuda_utils.c with `gcc -lcuda`; the devel image ships only the stub libcuda.so.
# Make it linkable at compile time (the REAL driver libcuda.so.1 is dlopen'd at run via --gpus all).
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so

# Pinned torch (cu128) — matches the export producer (T2a byte-exact rule). soundfile for audio fixtures.
RUN pip install --break-system-packages --no-cache-dir torch==2.8.0 numpy soundfile

WORKDIR /work/runtime
