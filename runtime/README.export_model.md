# `export_model.sh` — generic stage-2 AOTI recompile (any GPU arch × en/ml)

`export_model.sh` produces the **complete, arch-specific serve artifact set** for one GPU
architecture and one model profile, starting from the arch-agnostic stage-1 exports (see
[`ARTIFACTS.md`](ARTIFACTS.md) for the two-stage pipeline). It generalizes
[`run_l40s_density.sh`](run_l40s_density.sh) (the original sm_89 / English-only recipe) over two axes:

- **`PROFILE=en|ml`** — the model (see [`model_profile.py`](model_profile.py)):

  | profile | model | finalize buckets | extra artifacts |
  |---|---|---:|---|
  | `en` | `nvidia/nemotron-speech-streaming-en-0.6b` | 32 | — |
  | `ml` | `nvidia/nemotron-3.5-asr-streaming-0.6b` (multilingual, 40 locales) | 64 | `prompt_apply.ts` (language-ID MLP) |

- **`TARGET_CC`** — the CUDA compute capability of the box the script runs on
  (`8.6` = A10G, `8.9` = L40S/L4, `12.0` = RTX 5090, …). Defaults to **auto-detect** via
  `nvidia-smi`. AOTI/Triton codegen targets the *live* device — you **must run on the target
  GPU**; cross-compiling from a different arch is not possible.

Unlike `run_l40s_density.sh` (steady encoder + finalize buckets only), this script also compiles
the **first-chunk encoder** (`enc_first_aoti.pt2`) and the **batched-steady scheduler buckets**
(`enc_steady_aoti_b{1,2,4,8,16}.pt2`), so the output directory is the full serve set that
[`scripts/backup_artifacts_s3.sh`](scripts/backup_artifacts_s3.sh) uploads as `<arch>/` and that
`ws_server` mounts as `NEMOTRON_ARTIFACT_DIR` + `--steady-batch-dir`. The density sweep is still
available, but opt-in (`RUN_DENSITY=1`).

## Quick start

Bootstrap the multilingual model on a new architecture (e.g. an A10G, sm_86), pulling the stage-1
exports from the S3 backup:

```bash
cd runtime
PROFILE=ml ./export_model.sh
```

That is all: `TARGET_CC` auto-detects from the GPU, and for `PROFILE=ml` the source URI defaults to

```text
s3://audiointel-backups/STT/nvidia/nemotron-3.5-asr-streaming-0.6b/base/
```

which is the `base/` prefix produced by `scripts/backup_artifacts_s3.sh`. Equivalent explicit form:

```bash
PROFILE=ml TARGET_CC=8.6 \
  S3_URI=s3://audiointel-backups/STT/nvidia/nemotron-3.5-asr-streaming-0.6b/base \
  ./export_model.sh
```

Other common invocations:

```bash
# English model, EPs already rsynced into artifacts_en_sm86/ (no S3, no credentials):
PROFILE=en TARGET_CC=8.6 EPS_LOCAL=1 ./export_model.sh

# ml serve set + the on-box density sweep afterwards:
PROFILE=ml RUN_DENSITY=1 ./export_model.sh

# compile only the density-sweep subset (steady encoder + finalize buckets), like the old script:
PROFILE=ml COMPILE_ENC_FIRST=0 COMPILE_STEADY_BATCH=0 RUN_DENSITY=1 ./export_model.sh

# re-download sources even if present, and re-verify:
PROFILE=ml FORCE_S3_DOWNLOAD=1 ./export_model.sh
```

## Prerequisites

- An x86_64 DL-AMI-style box **with the target GPU installed** (the script asserts
  `torch.cuda.get_device_capability() == TARGET_CC`, fail-closed).
- The `runtime/` subset of this repo on the box (the script, `aot_compile*.py`,
  `strip_*.py`, `export_steady_batched.py`, `model_profile.py`; plus `cpp/` if `RUN_DENSITY=1`).
- AWS credentials able to read the bucket (S3 mode only; `EPS_LOCAL=1` needs none).
- **Disk:** the ml source download is ~172 GiB (en is roughly half — 32 buckets instead of 64);
  compile outputs and the Inductor cache add ~30 GiB. Budget **≥ 250 GiB free** for ml.
- Root/sudo for the one-time OS deps and `libcuda.so` symlinks (see Script flow).
- **glibc ≤ 2.39 (Ubuntu ≤ 24.04) on the build box** — see
  [glibc / serve compatibility](#glibc--serve-compatibility). On a newer host, run the whole
  export inside the pinned container: `./container/enter.sh bash -lc 'PROFILE=ml ./export_model.sh'`.

The script itself installs the rest: build tools via apt, a `torch==2.8.0` venv (reused across
runs), and the CUDA link fixes.

## Parameters

Primary:

| variable | default | meaning |
|---|---|---|
| `PROFILE` | `en` | model profile: `en` or `ml`; also exported as `NEMOTRON_EXPORT_PROFILE` to the compile/strip scripts |
| `TARGET_CC` | `auto` | compute capability of this box (`8.6`, `8.9`, …); `auto` = read from `nvidia-smi` |
| `EPS_LOCAL` | `0` | `1` = use pre-placed source EPs in `$ART_DIR`, skip S3 entirely |
| `S3_URI` | derived | stage-1 source prefix; for `ml` it is `s3://audiointel-backups/STT/nvidia/nemotron-3.5-asr-streaming-0.6b/base` |
| `ART_DIR` | `artifacts_<profile>_<tag>/` | working + output dir, e.g. `artifacts_ml_sm86/` |
| `COMPILE_ENC_FIRST` | `1` | also AOTI-compile the first-chunk encoder (`enc_first_aoti.pt2`) |
| `COMPILE_STEADY_BATCH` | `1` | also AOTI-compile + strip the batched steady buckets (`steady_b_artifacts_b16/`) |
| `RUN_DENSITY` | `0` | `1` = build `density_main` (`-DNEMOTRON_PROFILE=$PROFILE`) and run the sweep after compiling |

Advanced (defaults preserve `run_l40s_density.sh` behavior):

| variable | default | meaning |
|---|---|---|
| `STEADY_BATCHES` | `1,2,4,8,16` | batched-steady bucket set |
| `SELF_CHECK_ATOL` | `0.1` | per-package compile self-check tolerance (sanity gate, not byte-exactness) |
| `VENV` | `~/torch280-<tag>-venv` | torch 2.8.0 venv (arch-tagged, profile-independent) |
| `BUILD_DIR` | `cpp/build_<profile>_<tag>_density` | cmake build dir for `density_main` |
| `FORCE_S3_DOWNLOAD` | `0` | re-download objects that already exist locally |
| `SKIP_EPS_VERIFY` | `0` | skip the SHA256 sweep against `eps_manifest.json` |
| `STRICT_EPS_MANIFEST` / `ALLOW_UNMANIFESTED_AUX` | `1` / `1` | manifest strictness for auxiliary (non-compile-input) files |
| `KEEP_UNSTRIPPED_BUCKETS` | `0` | keep the pre-strip finalize bucket packages in `finalize_buckets/` |
| `SERVE_GLIBC_MAX` | `2.39` | serve-fleet glibc floor (Ubuntu 24.04); both glibc gates check against it |
| `ALLOW_INCOMPATIBLE_GLIBC` | `0` | `1` = bypass the glibc gates (artifacts may not load on the serve fleet) |
| `DENSITY_N_VALUES` etc. | see script | density-sweep knobs, unchanged from `run_l40s_density.sh` |

## Source artifacts (what gets downloaded)

All keys are relative to `$S3_URI` — for ml,
`s3://audiointel-backups/STT/nvidia/nemotron-3.5-asr-streaming-0.6b/base/`. Every download is
SHA256-verified against `eps_manifest.json` (fail-closed; `SKIP_EPS_VERIFY=1` to bypass). Sizes
below are the ml set; en has 32 finalize buckets instead of 64 and no `prompt_apply.ts`.

| group | keys | ml size |
|---|---|---:|
| compile inputs (core) | `enc_steady_t2a.pt2`, `enc_first.ts`, `session_bundle.ts`, `finalize_shared_weights.{pt,ts}`, `t2a_io.pt`, `joint_step.ts`, `predict_step.ts` | 9.4 GiB |
| finalize bucket EPs | `finalize_buckets/enc_finalize_d{0,2}_T*_ep.pt2` (64 for ml: d0 T130–161, d2 T139–170) + `buckets_manifest.json` | 148.4 GiB |
| batched steady EPs (`COMPILE_STEADY_BATCH=1`) | `steady_b_artifacts_b16/enc_steady_t2a_b{1,2,4,8,16}.pt2` | 11.7 GiB |
| first-chunk EP (`COMPILE_ENC_FIRST=1`) | `enc_first_t2a.pt2`, `enc_first_t2a_io.pt` | 2.3 GiB |
| serve-set pass-through (arch-agnostic, reused as-is) | `preproc.ts` (+`.manifest.json`), `session_audio_bundle.ts` (+`.audio_ci.json`); ml only: `prompt_apply.ts` | 0.1 GiB |
| manifests | `eps_manifest.json`, `buckets_manifest.json` | <1 MiB |
| **total (ml, all stages on)** | **86 objects** | **≈ 172 GiB** |

With `EPS_LOCAL=1` the same file set must be pre-placed under `$ART_DIR` (bucket EPs in
`finalize_buckets/`, batched EPs in `steady_b_artifacts_b16/`); the script verifies presence and
the per-profile bucket count, then proceeds without S3.

## Script flow

1. **OS deps** — apt-installs `build-essential`, `cmake`, `ninja-build`, `python3-dev`,
   `python3-venv`, `curl` (idempotent).
2. **CUDA link env** — locates system CUDA/`nvcc`, creates the unversioned `libcuda.so`
   driver + stub symlinks Triton's link probes need.
3. **Venv** — creates (or reuses) the `torch==2.8.0` venv.
4. **Fail-closed device check** — x86_64, torch 2.8.0, live device capability == `TARGET_CC`,
   `TORCH_CUDA_ARCH_LIST` requests the target when the wheel lacks its SASS, Inductor autotune
   confirmed **off** (compiles must stay reproducible).
5. **Sources** — downloads (or, with `EPS_LOCAL=1`, checks) the stage-1 set into `$ART_DIR` and
   SHA-verifies it against `eps_manifest.json`, including the per-profile finalize-bucket count
   (32 en / 64 ml).
6. **Steady encoder** — `aot_compile.py` (via `NEMOTRON_ART_DIR=$ART_DIR`) →
   `enc_steady_aoti.pt2`, self-checked against `t2a_io.pt`.
7. **Finalize buckets** — per bucket: `aot_compile_buckets.py` (self-check at
   `SELF_CHECK_ATOL`) then `strip_bucket_weights.py` →
   `stripped_finalize_buckets/enc_finalize_d*_T*.pt2` (weights zeroed; the runtime binds
   `finalize_shared_weights.ts`). Writes `stripped_finalize_buckets/manifest.json` whose
   `CONTRACT` block (att context, shift, blank, model id, …) is **derived from
   `model_profile.py`** — the same source of truth mirrored by
   `cpp/lib/session/model_constants.h`, so profile/binary mismatches fail closed at load. Also
   writes the compile self-check record `finalize_buckets/buckets_manifest_<tag>.json`.
8. **First-chunk encoder** (`COMPILE_ENC_FIRST=1`) — `aot_compile_enc_first.py` →
   `enc_first_aoti.pt2` (constants-on-disk, stripped, self-checked).
9. **Batched steady buckets** (`COMPILE_STEADY_BATCH=1`) — `export_steady_batched.py
   --compile-only` → `enc_steady_aoti_b{B}.pt2`, then `strip_steady_buckets.py` rebinds the shared
   weight set, then a `--manifest-only` pass re-emits `MANIFEST.json` (it embeds package SHA256s,
   which stripping changes).
10. **Optional density sweep** (`RUN_DENSITY=1`) — builds `density_main` with
    `-DNEMOTRON_PROFILE=$PROFILE` (plus the cudart-12/13 unification fix) and runs
    fresh-process-per-N, printing `DENSITY_ROW` / `DENSITY_RESULT` lines and writing
    `logs/density_summary_*.json`.

## Outputs

Everything lands under `$ART_DIR` (default `artifacts_<profile>_<tag>/`, e.g. `artifacts_ml_sm86/`):

| output | arch-specific? | serve role |
|---|---|---|
| `enc_steady_aoti.pt2` | **yes** | streaming steady encoder |
| `enc_first_aoti.pt2` | **yes** | first-chunk encoder |
| `stripped_finalize_buckets/` (+`manifest.json`) | **yes** | finalize buckets, weights stripped |
| `steady_b_artifacts_b16/enc_steady_aoti_b{1,2,4,8,16}.pt2` (+`MANIFEST.json`) | **yes** | batched-steady scheduler buckets |
| `preproc.ts`, `enc_first.ts`, `joint_step.ts`, `predict_step.ts`, `prompt_apply.ts` (ml), `finalize_shared_weights.{pt,ts}`, `session_audio_bundle.ts*` | no | downloaded alongside — reused as-is on every arch |

This is exactly the `<arch>/` serve contract of `scripts/backup_artifacts_s3.sh`: mount `$ART_DIR`
as `NEMOTRON_ARTIFACT_DIR` and `$ART_DIR/steady_b_artifacts_b16` as `--steady-batch-dir`
(pair with a `ws_server` binary built with the matching `-DNEMOTRON_PROFILE`). To publish the new
arch back to the bucket:

```bash
ARCH=sm86 ART_DIR=$PWD/artifacts_ml_sm86 \
  S3_BASE=s3://audiointel-backups/STT/nvidia/nemotron-3.5-asr-streaming-0.6b \
  EXPECTED_BUCKETS=64 \
  ./scripts/backup_artifacts_s3.sh
```

after which bringing up another box of the same arch is just two `aws s3 sync` calls (see the
backup script header).

## Idempotency & re-runs

The script is safe to re-run after an interruption:

- downloads are skipped when the local file exists (`FORCE_S3_DOWNLOAD=1` to override);
- the venv is reused if it already has torch 2.8.0;
- already-stripped finalize buckets are **not** recompiled (compiles are deterministic,
  autotune off) — a clean rebuild is `rm -rf $ART_DIR/stripped_finalize_buckets`;
- `enc_first_aoti.pt2` is self-checked and kept if present;
- `enc_steady_aoti.pt2` is always recompiled (single package, cheap relative to the buckets).

## glibc / serve compatibility

The AOTI compile embeds a natively linked `.so` inside every produced `.pt2`. The linker stamps
the versioned glibc symbols it references (e.g. `__isoc23_strtol@GLIBC_2.38` when built on
glibc ≥ 2.38), and glibc is **backward- but not forward-compatible** — so the build box's glibc
silently becomes the *minimum* glibc of every serve host. Below it, `ws_server` fails at load
with `dlopen: ... version 'GLIBC_2.38' not found`.

**The serve fleet floor is glibc 2.39 = Ubuntu 24.04** (decision 2026-07-20: serving uses the
Ubuntu-24.04-based `nemotron-ws-server` image; the older 22.04-based `pytorch/pytorch` runtime
image is retired). Newer serve OSes (26.04, …) are always fine; older ones are not.

The script enforces this fail-closed, at two points:

1. **Preflight** — if the build host's glibc (from `ldd --version`) exceeds `SERVE_GLIBC_MAX`
   (default `2.39`), it dies before compiling anything and points at the pinned container:
   `./container/enter.sh bash -lc 'PROFILE=ml ./export_model.sh'`
   (Ubuntu 24.04 / glibc 2.39 / CUDA 12.8 / torch 2.8.0 / awscli; `enter.sh` passes AWS
   credentials and `~/.aws` through for the S3 source mode).
2. **Post-compile** — after each compile stage it unzips the `.so` members of the produced
   `.pt2` packages and verifies the max stamped `GLIBC_x.y` version-need is ≤ `SERVE_GLIBC_MAX`
   (`readelf -V` over the extracted members). This certifies the property that actually matters
   at `dlopen` time, independent of what host the compile ran on.

`ALLOW_INCOMPATIBLE_GLIBC=1` bypasses both gates (e.g. artifacts intentionally targeting a
newer private fleet). Lower `SERVE_GLIBC_MAX` if an older serve OS ever rejoins the fleet —
e.g. `SERVE_GLIBC_MAX=2.35` for Ubuntu 22.04-based serve images.

## Notes & troubleshooting

- **Run on the target GPU.** The capability assertion rejects a mismatched box before any
  compile: `expected sm_86 device capability (8, 6), got (8, 9)`.
- **The en `base/` prefix is not populated yet.** Only the ml model is currently backed up under
  `s3://audiointel-backups/STT/nvidia/`. For `PROFILE=en`, use `EPS_LOCAL=1` (export on a torch
  host per [`ARTIFACTS.md`](ARTIFACTS.md), rsync into `$ART_DIR`) or point `S3_URI` at a prefix
  you populated with `scripts/backup_artifacts_s3.sh`.
- **Autotune stays off** (`TORCHINDUCTOR_MAX_AUTOTUNE=0` etc., asserted at startup) so compiles
  are reproducible and the self-checks are meaningful. Do not enable max-autotune env vars.
- **Self-check tolerance:** cross-arch fp drift versus the export-host references is expected;
  `SELF_CHECK_ATOL=0.1` is a sanity gate (loads, runs, not garbage), not a byte-exactness claim.
  If a single bucket trips the gate, bump `SELF_CHECK_ATOL` for a re-run — only the failed bucket
  recompiles.
- **`libcuda.so` link probes:** on bare AMIs the AOTI link path needs an unversioned
  `libcuda.so`; the script creates the driver-lib and CUDA-stub symlinks itself (needs sudo once).
- **cudart 12 vs 13 (density sweep only):** DL AMIs ship system CUDA 13 while pip torch bundles
  cudart-12; mixing both in `density_main` deadlocks the multi-stream path. The build step unifies
  on torch's cudart-12 and fail-closes if `libcudart.so.13` is still linked.
- **Wheel SASS coverage:** the stock torch 2.8.0+cu128 wheel ships prebuilt SASS for sm_86/sm_90/
  sm_120 but not sm_89 — either way AOTI targets the live device via `TORCH_CUDA_ARCH_LIST`; the
  script only warns when the wheel lacks the arch.
