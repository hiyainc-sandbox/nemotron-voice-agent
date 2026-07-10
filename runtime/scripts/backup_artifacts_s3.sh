#!/usr/bin/env bash
# Back up the ml-profile model artifacts to S3, split into:
#
#   $S3_BASE/base/         arch-agnostic stage-1 exports (TorchScript + ExportedPrograms
#                          + fixtures/manifests) — the starting point to AOTI-compile
#                          for ANY new GPU architecture (see runtime/ARTIFACTS.md).
#   $S3_BASE/<arch>/       self-contained serve set for one GPU arch (sm89, sm120, ...):
#     artifacts/           -> mount as NEMOTRON_ARTIFACT_DIR
#     steady-batch/        -> mount as --steady-batch-dir
#
# Bringing up a new ws_server instance is then just:
#   aws s3 sync $S3_BASE/<arch>/artifacts/    <dir>/artifacts
#   aws s3 sync $S3_BASE/<arch>/steady-batch/ <dir>/steady-batch
# and a NEW architecture is bootstrapped by syncing base/ and running the
# stage-2 AOTI compile there (run_l40s_density.sh encodes the sm89 recipe).
#
# Every uploaded file is SHA256-hashed into a manifest (base/eps_manifest.json,
# <arch>/serve_manifest.json), flat {key: sha256} — the format
# run_l40s_density.sh verifies fail-closed.
#
# Usage:
#   DRY_RUN=1 ./backup_artifacts_s3.sh     # list what would upload, verify nothing is missing
#   ./backup_artifacts_s3.sh               # upload base/ + auto-detected arch dir
#   UPLOAD_BASE=0 ./backup_artifacts_s3.sh # arch serve set only
#   INCLUDE_HEAVY=1 ...                    # also back up the big validation-only intermediates

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNTIME_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

S3_BASE=${S3_BASE:-"s3://audiointel-backups/STT/nvidia/nemotron-3.5-asr-streaming-0.6b"}
ART_DIR=${ART_DIR:-"$RUNTIME_DIR/artifacts_ml"}
STEADY_B_DIR=${STEADY_B_DIR:-"$ART_DIR/steady_b_artifacts_b16"}
EXPECTED_BUCKETS=${EXPECTED_BUCKETS:-64}   # ml: 64 finalize buckets (en would be 32)

UPLOAD_BASE=${UPLOAD_BASE:-1}
UPLOAD_SERVE=${UPLOAD_SERVE:-1}
INCLUDE_HEAVY=${INCLUDE_HEAVY:-0}          # encoder_full.ts, enc_steady.ts, enc_finalize_drop{0,2}.ts, finalize_fixture.pt
DRY_RUN=${DRY_RUN:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}          # skip upload when S3 object exists with the same size

# --- arch detection -----------------------------------------------------------
ARCH=${ARCH:-auto}
if [[ "$ARCH" == "auto" ]]; then
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
  [[ -n "$cap" ]] || { echo "FATAL: cannot auto-detect GPU arch (nvidia-smi); set ARCH=sm89|sm120|..." >&2; exit 1; }
  ARCH="sm${cap}"
fi

log() { echo "[backup] $*" >&2; }
die() { echo "[backup] FATAL: $*" >&2; exit 1; }

command -v aws >/dev/null || die "awscli not found"
[[ -d "$ART_DIR" ]] || die "artifact dir not found: $ART_DIR"

BUCKET=${S3_BASE#s3://}; BUCKET=${BUCKET%%/*}
KEY_PREFIX=${S3_BASE#s3://$BUCKET/}

MANIFEST_TMP=$(mktemp)
trap 'rm -f "$MANIFEST_TMP"' EXIT
UPLOADED=0 SKIPPED=0 MISSING=0

s3_size() {  # object size or empty if absent
  aws s3api head-object --bucket "$BUCKET" --key "$KEY_PREFIX/$1" \
    --query ContentLength --output text 2>/dev/null || true
}

# put <local-file> <s3-key-relative-to-S3_BASE> [manifest-key]
# Uploads, and records "manifest-key sha256" for the manifest of the current section.
put() {
  local src=$1 key=$2 mkey=${3:-$2}
  if [[ ! -f "$src" ]]; then
    log "MISSING: $src  (-> $key)"; MISSING=$((MISSING + 1)); return 0
  fi
  local size; size=$(stat -c %s "$src")
  if [[ "$DRY_RUN" == "1" ]]; then
    log "would upload: $key ($(numfmt --to=iec "$size"))"
  elif [[ "$SKIP_EXISTING" == "1" && "$(s3_size "$key")" == "$size" ]]; then
    log "exists, same size, skipping: $key"
    SKIPPED=$((SKIPPED + 1))
  else
    log "uploading: $key ($(numfmt --to=iec "$size"))"
    aws s3 cp "$src" "$S3_BASE/$key" --only-show-errors
    UPLOADED=$((UPLOADED + 1))
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    sha256sum "$src" | awk -v k="$mkey" '{print k, $1}' >> "$MANIFEST_TMP"
  fi
}

# flush_manifest <s3-key> — write accumulated hashes as flat {key: sha256} JSON
flush_manifest() {
  local key=$1
  [[ "$DRY_RUN" == "1" ]] && { : > "$MANIFEST_TMP"; return 0; }
  local json="$MANIFEST_TMP.json"
  awk 'BEGIN{print "{"} {printf "%s  \"%s\": \"%s\"", (NR>1?",\n":""), $1, $2} END{print "\n}"}' \
    "$MANIFEST_TMP" > "$json"
  log "uploading manifest: $key ($(wc -l < "$MANIFEST_TMP") entries)"
  aws s3 cp "$json" "$S3_BASE/$key" --only-show-errors
  rm -f "$json"; : > "$MANIFEST_TMP"
}

# --- base: arch-agnostic exports (compile inputs for any new GPU arch) --------
upload_base() {
  log "=== base/ (arch-agnostic exports) from $ART_DIR ==="

  # stage-2 compile inputs (ExportedPrograms)
  put "$ART_DIR/enc_steady_t2a.pt2"  "base/enc_steady_t2a.pt2"
  put "$ART_DIR/enc_first_t2a.pt2"   "base/enc_first_t2a.pt2"
  local b
  for b in 1 2 4 8 16; do
    put "$STEADY_B_DIR/enc_steady_t2a_b${b}.pt2" "base/steady_b_artifacts_b16/enc_steady_t2a_b${b}.pt2"
  done

  # finalize bucket EPs + manifest (root copy too, matching run_l40s_density.sh)
  local n_eps=0 ep
  while IFS= read -r ep; do
    put "$ep" "base/finalize_buckets/$(basename "$ep")"
    n_eps=$((n_eps + 1))
  done < <(find "$ART_DIR/finalize_buckets" -maxdepth 1 -name '*_ep.pt2' | sort)
  [[ "$n_eps" == "$EXPECTED_BUCKETS" ]] || die "expected $EXPECTED_BUCKETS finalize bucket EPs, found $n_eps"
  put "$ART_DIR/finalize_buckets/buckets_manifest.json" "base/finalize_buckets/buckets_manifest.json"
  put "$ART_DIR/finalize_buckets/buckets_manifest.json" "base/buckets_manifest.json" "buckets_manifest.json"

  # server-loaded TorchScript, reused as-is on every arch
  put "$ART_DIR/preproc.ts"                   "base/preproc.ts"
  put "$ART_DIR/preproc.ts.manifest.json"     "base/preproc.ts.manifest.json"
  put "$ART_DIR/enc_first.ts"                 "base/enc_first.ts"
  put "$ART_DIR/joint_step.ts"                "base/joint_step.ts"
  put "$ART_DIR/predict_step.ts"              "base/predict_step.ts"
  put "$ART_DIR/prompt_apply.ts"              "base/prompt_apply.ts"
  put "$ART_DIR/finalize_shared_weights.pt"   "base/finalize_shared_weights.pt"
  put "$ART_DIR/finalize_shared_weights.ts"   "base/finalize_shared_weights.ts"

  # fixtures / validation IO for the compile self-checks and gates
  put "$ART_DIR/t2a_io.pt"                    "base/t2a_io.pt"
  put "$ART_DIR/enc_first_t2a_io.pt"          "base/enc_first_t2a_io.pt"
  put "$ART_DIR/decode_init.pt"               "base/decode_init.pt"
  put "$ART_DIR/session_bundle.ts"            "base/session_bundle.ts"
  put "$ART_DIR/session_audio_bundle.ts"      "base/session_audio_bundle.ts"
  put "$ART_DIR/session_audio_bundle.ts.audio_ci.json" "base/session_audio_bundle.ts.audio_ci.json"
  put "$ART_DIR/stream_bundle.ts"             "base/stream_bundle.ts"
  put "$ART_DIR/pipeline_bundle.ts"           "base/pipeline_bundle.ts"
  put "$ART_DIR/cpp_bundle.ts"                "base/cpp_bundle.ts"
  put "$ART_DIR/finalize_bundle.ts"           "base/finalize_bundle.ts"
  put "$ART_DIR/finalize_export_report.txt"   "base/finalize_export_report.txt"

  if [[ "$INCLUDE_HEAVY" == "1" ]]; then
    put "$ART_DIR/encoder_full.ts"        "base/encoder_full.ts"
    put "$ART_DIR/enc_steady.ts"          "base/enc_steady.ts"
    put "$ART_DIR/enc_finalize_drop0.ts"  "base/enc_finalize_drop0.ts"
    put "$ART_DIR/enc_finalize_drop2.ts"  "base/enc_finalize_drop2.ts"
    put "$ART_DIR/finalize_fixture.pt"    "base/finalize_fixture.pt"
  fi

  flush_manifest "base/eps_manifest.json"
}

# --- <arch>: self-contained serve set (nemotron-ws-server artifacts contract) --
upload_serve() {
  log "=== $ARCH/ (serve set for NEMOTRON_ARTIFACT_DIR + --steady-batch-dir) ==="

  local f
  for f in preproc.ts preproc.ts.manifest.json enc_first.ts joint_step.ts predict_step.ts \
           prompt_apply.ts enc_first_aoti.pt2 enc_steady_aoti.pt2 \
           finalize_shared_weights.pt finalize_shared_weights.ts \
           session_audio_bundle.ts session_audio_bundle.ts.audio_ci.json; do
    put "$ART_DIR/$f" "$ARCH/artifacts/$f"
  done

  local n=0 p
  while IFS= read -r p; do
    put "$p" "$ARCH/artifacts/stripped_finalize_buckets/$(basename "$p")"
    n=$((n + 1))
  done < <(find "$ART_DIR/stripped_finalize_buckets" -maxdepth 1 -name '*.pt2' | sort)
  [[ "$n" == "$EXPECTED_BUCKETS" ]] || die "expected $EXPECTED_BUCKETS stripped buckets, found $n"
  put "$ART_DIR/stripped_finalize_buckets/manifest.json" "$ARCH/artifacts/stripped_finalize_buckets/manifest.json"

  local b
  for b in 1 2 4 8 16; do
    put "$STEADY_B_DIR/enc_steady_aoti_b${b}.pt2" "$ARCH/steady-batch/enc_steady_aoti_b${b}.pt2"
  done
  put "$STEADY_B_DIR/MANIFEST.json" "$ARCH/steady-batch/MANIFEST.json"

  flush_manifest "$ARCH/serve_manifest.json"
}

log "S3 base:   $S3_BASE"
log "artifacts: $ART_DIR"
log "arch:      $ARCH   (dry-run=$DRY_RUN, skip-existing=$SKIP_EXISTING)"

[[ "$UPLOAD_BASE"  == "1" ]] && upload_base
[[ "$UPLOAD_SERVE" == "1" ]] && upload_serve

if [[ "$MISSING" -gt 0 ]]; then
  die "$MISSING required file(s) missing locally — nothing partial was manifested; see log above"
fi
log "done: uploaded=$UPLOADED skipped=$SKIPPED"
log "new-instance sync:"
log "  aws s3 sync $S3_BASE/$ARCH/artifacts/    <dir>/artifacts"
log "  aws s3 sync $S3_BASE/$ARCH/steady-batch/ <dir>/steady-batch"
