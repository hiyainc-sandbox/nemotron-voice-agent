"""Continuous-batching primitives for cache-aware streaming ASR (Step 5 of proj-2026-05-21-0410).

These pure functions encode the strict recipe for stacking N independent per-stream cache-aware states
into one `conformer_stream_step(B)` call and scattering the results back. Probe B validated text-only
B=2 behavior; `proj-2026-05-21-0410/test_batch_state.py` is the blocking full-state Probe B2 gate.

CRITICAL axes (NeMo conformer_encoder.py:1087-1125):
  cache_last_channel : [layers, B, cache_T, d_model]   -> stack/scatter on dim 1
  cache_last_time    : [layers, B, d_model, time_T]    -> stack/scatter on dim 1
  cache_last_channel_len : [B]                          -> stack/scatter on dim 0

Hypotheses: a FLAT per-row list (one Hypothesis object per row, each UNIQUE — the batched label-looping
decoder mutates them in place, rnnt_greedy_decoding.py:825-831). Scatter back as a one-element list per
session so the existing B=1 finalize/fork code sees the same shape.

These are deliberately dependency-light (only torch) so they unit-test without loading NeMo. The
scheduler (Steps 6-7) wires them into the live path under `NEMOTRON_BATCH_SCHED`.
"""
from typing import Any, List, Optional, Sequence, Tuple

import torch


def batch_group_key(target_lang: Optional[str], keep_all_outputs: bool, drop_extra: int,
                    chunk_T: int, decoder_mode: str) -> tuple:
    """Sessions sharing this key may be batched in one conformer_stream_step call.

    Different target_lang (model-global prompt), keep_all_outputs, drop_extra (scalar/global per call),
    chunk_T (input width), or decoder_mode must NOT share a batched call.
    """
    return (target_lang, bool(keep_all_outputs), int(drop_extra), int(chunk_T), str(decoder_mode))


def ready_predicate(
    *,
    synthetic_prefix_samples: int,
    total_audio_samples: int,
    emitted_frames: int,
    shift_frames: int,
    hop_samples: int,
    pending_audio_len: int,
    preprocess_new_audio_samples: int,
) -> bool:
    """Exact normal-tick readiness guard from server.py.

    A stream is ready only when the timeline has enough audio to expose the next
    shift of mel frames and the pending audio buffer has one fixed preprocessor
    window of new audio. The scheduler must not pad or coerce rows that fail
    either guard.
    """
    timeline_samples = int(synthetic_prefix_samples) + int(total_audio_samples)
    needed_samples = (int(emitted_frames) + int(shift_frames) + 1) * int(hop_samples)
    return (
        timeline_samples >= needed_samples
        and int(pending_audio_len) >= int(preprocess_new_audio_samples)
    )


def stack_processed(chunk_mels: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack per-session chunk mels [1,F,T] -> processed_signal [B,F,T] + lengths [B] (batch on dim 0)."""
    assert chunk_mels, "empty batch"
    T = chunk_mels[0].shape[-1]
    F = chunk_mels[0].shape[1]
    dtype = chunk_mels[0].dtype
    device = chunk_mels[0].device
    for cm in chunk_mels:
        assert cm.dim() == 3 and cm.shape[0] == 1, (
            f"each row must be [1,F,T], got {tuple(cm.shape)}")
        assert cm.dtype == dtype, f"dtype mismatch: expected {dtype}, got {cm.dtype}"
        assert cm.device == device, f"device mismatch: expected {device}, got {cm.device}"
        assert cm.shape[1] == F and cm.shape[-1] == T, (
            f"ragged batch: expected [*,{F},{T}], got {tuple(cm.shape)} — group by chunk_T first")
    processed = torch.cat(list(chunk_mels), dim=0)
    lengths = torch.full((processed.shape[0],), T, dtype=torch.long, device=processed.device)
    return processed, lengths


def stack_caches(caches: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack N per-session caches (each [layers,1,...]/[layers,1,...]/[1]) into a B=N batch.

    channel & time caches concat on dim 1 (batch INSIDE layers); len concat on dim 0.
    """
    clc = torch.cat([c[0] for c in caches], dim=1)
    clt = torch.cat([c[1] for c in caches], dim=1)
    clcl = torch.cat([c[2] for c in caches], dim=0)
    return clc, clt, clcl


def scatter_cache_row(clc: torch.Tensor, clt: torch.Tensor, clcl: torch.Tensor, i: int
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract row i's cache back to a B=1 shape ([layers,1,...]/[layers,1,...]/[1])."""
    return (
        clc[:, i:i + 1, ...].detach().clone(),
        clt[:, i:i + 1, ...].detach().clone(),
        clcl[i:i + 1].detach().clone(),
    )


def _assert_uniform_none(values: Sequence[Any], label: str) -> None:
    assert values, f"empty {label} batch"
    none_count = sum(1 for value in values if value is None)
    assert none_count in (0, len(values)), (
        f"{label} must be uniformly None-or-not within a batch; "
        f"got {none_count} None rows out of {len(values)}"
    )


def stack_hypotheses(per_session_hyps: Sequence[Optional[list]]) -> List[Any]:
    """Build the flat per-row hypothesis list for conformer_stream_step.

    Each session stores `previous_hypotheses` as a NeMo list of length 1 (or None for a fresh stream).
    The batched call wants a flat list [hyp_or_None, ...]; objects must be unique per row (caller's
    responsibility — they already are, being distinct sessions' state).
    """
    _assert_uniform_none(per_session_hyps, "previous_hypotheses")
    out = []
    for h in per_session_hyps:
        out.append(h[0] if h else None)
    # alias guard: no two non-None rows may be the same object (would corrupt via in-place merge_)
    seen = set()
    for x in out:
        if x is not None:
            assert id(x) not in seen, "hypothesis aliasing: two rows share one Hypothesis object"
            seen.add(id(x))
    return out


def stack_pred_out(per_session_pred: Sequence[Optional[list]], *, rnnt: bool = True) -> Optional[List[Any]]:
    """Flat per-row previous_pred_out list, or None when every row is fresh.

    This path is only validated for RNNT. CTC previous_pred_out is a cumulative
    prediction history and needs its own ragged-state proof before batching.
    """
    assert rnnt, "stack_pred_out is only validated for RNNT batches"
    _assert_uniform_none(per_session_pred, "previous_pred_out")
    if per_session_pred[0] is None:
        return None
    return [p[0] for p in per_session_pred]


def conformer_stream_step_restoring_drop_extra(model: Any, **kwargs: Any) -> tuple:
    """Call NeMo conformer_stream_step and restore encoder drop_extra on all exits."""
    streaming_cfg = model.encoder.streaming_cfg
    original_drop_extra = streaming_cfg.drop_extra_pre_encoded
    try:
        return model.conformer_stream_step(**kwargs)
    finally:
        streaming_cfg.drop_extra_pre_encoded = original_drop_extra
