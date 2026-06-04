# Regenerating the native-runtime artifacts (step 0)

The native C++ `ws_server` loads compiled model artifacts that are **not committed** to this repo —
they are large (tens of GB) and GPU-architecture-specific. This document is how you regenerate them
from the **public** HuggingFace checkpoint. No private buckets or credentials are involved.

> Input checkpoint (public, no token): **`nvidia/nemotron-speech-streaming-en-0.6b`**

## The two stages

Artifact generation is two stages, and the split is what makes cross-GPU deployment work:

1. **Export** (on any torch + NeMo host — the RTX 5090 dev box is fine). Traces / `torch.export`s the
   model into **architecture-agnostic** intermediates:
   - TorchScript modules (`*.ts`) — preprocessor, decode steps, shared weights, the session bundle.
   - ExportedPrograms (`*.pt2` EPs) — the steady encoder and the 32 finalize buckets.

   These are portable across GPUs.

2. **AOTI compile + strip** (in the build container, **on the target GPU's architecture**).
   AOTInductor-compiles each ExportedProgram into a GPU-arch-specific package, validates it byte-exact
   against an eager reference, then strips the duplicated weights (the runtime binds one shared weight
   set). The resulting `*.pt2` packages are **sm-specific** — sm_120 for the RTX 5090, sm_89 for the
   L40S — and are not portable. The `*.ts` modules from stage 1 are reused as-is.

So: **export once, then AOTI-compile per target architecture.** To bring up an L40S cluster from an
RTX 5090 dev box, you export on the 5090, copy the EPs + `*.ts` to the L40S, and AOTI-compile there.

## Prerequisites

```bash
# from the repo root
cd runtime
bash setup-venv.sh                                   # host venv: torch 2.8.0+cu128 + NeMo + AOTI tools
docker build -t nemotron-aoti:cu128 container/        # the CUDA 12.8 build container (torch + nvcc)
```

The export scripts run in the **host venv** (they need NeMo to load the checkpoint). The AOTI compile
runs **in the container** (it needs nvcc + the target GPU), entered via `container/enter.sh`.

## Required artifacts

The server (see [`README.md`](README.md)) loads, under `runtime/`:

| artifact | kind | arch-specific? |
|---|---|---|
| `artifacts/preproc.ts` | TorchScript preprocessor | no |
| `artifacts/enc_first.ts` | TorchScript first-chunk encoder | no |
| `artifacts/joint_step.ts`, `artifacts/predict_step.ts` | TorchScript decode steps | no |
| `artifacts/enc_steady_aoti.pt2` | AOTI steady encoder | **yes** |
| `artifacts/stripped_finalize_buckets/` | 32 stripped AOTI finalize buckets | **yes** |
| `steady_b_artifacts_b16/enc_steady_aoti_b{1,2,4,8,16}.pt2` | AOTI batched-steady scheduler buckets | **yes** |

## Pipeline scripts

**Stage 1 — export (host venv, `HF_HUB_OFFLINE=1 ./.venv/bin/python <script>`):**

| script | produces |
|---|---|
| `export_pipeline.py` | `preproc.ts` (+ full-encoder self-check bundle) |
| `export_decode.py` | `joint_step.ts`, `predict_step.ts` (validates greedy decode byte-exact vs `fixtures/decode_*.pt`) |
| `export_t2a.py` | `enc_steady_t2a.pt2` — the streaming steady-encoder ExportedProgram |
| `export_enc_first_t2a.py` | `enc_first.ts` + the first-chunk encoder EP |
| `export_finalize_buckets.py` | the 32 finalize-bucket ExportedPrograms + `buckets_manifest.json` |
| `export_shared_weights.py` | `finalize_shared_weights.{pt,ts}` (one weight set shared by all buckets) |
| `export_session_bundle.py` | `session_bundle.ts` (the deterministic replay bundle the compat oracle uses) |

**Stage 2 — AOTI compile + strip (in the container, `./container/enter.sh python3 <script>`):**

| script | produces |
|---|---|
| `aot_compile.py` | `artifacts/enc_steady_aoti.pt2` from `enc_steady_t2a.pt2`, validated byte-exact |
| `aot_compile_enc_first.py` | the first-chunk encoder AOTI package (constants-on-disk, stripped) |
| `aot_compile_buckets.py` | AOTI-compiles each finalize bucket EP with fail-closed self-checks |
| `strip_bucket_weights.py` | `artifacts/stripped_finalize_buckets/` (weights zeroed; runtime binds the shared set) |
| `export_steady_batched.py` | `steady_b_artifacts_b16/enc_steady_aoti_b{1,2,4,8,16}.pt2` (the scheduler's batched buckets) |
| `strip_steady_buckets.py` | strips the batched steady buckets to share the same weight set |

## The runnable recipes

This document is the map; two scripts are the **tested, executable** recipes:

- **RTX 5090 (sm_120):** build + run is [`README.md`](README.md). It assumes the artifacts above are
  present under `runtime/`; produce them with the two stages on the 5090 itself.
- **L40S (sm_89):** [`run_l40s_density.sh`](run_l40s_density.sh) encodes the full stage-2 recompile on
  the box. Export the stage-1 EPs on a torch host, rsync them into `artifacts_sm89/`, and run with
  `EPS_LOCAL=1` (no S3). See [`run_l40s_density.README.md`](run_l40s_density.README.md).

## Notes

- Determinism: AOTI autotune is kept **off** so compiles are reproducible; the compile self-checks
  fail closed on any byte-level drift versus the eager reference.
- The `*.ts` TorchScript modules are architecture-agnostic — export them once and reuse across GPUs.
- Only the AOTI `*.pt2` packages must be recompiled per GPU architecture (sm_120 ≠ sm_89).
