#!/usr/bin/env python3
"""Export finalize-geometry encoder attempts + eager fixtures.

The C++ finalize port needs the keep_all_outputs=True encoder geometry, whose T
is dynamic at debounce time:

* first finalize:      chunk_mel T = remaining, drop_extra=0
* continuation final: chunk_mel T = pre_encode_cache_size + remaining, drop_extra=2

This script traces both scalar drop geometries to TorchScript when possible and
always writes an eager fixture bundle containing inputs, eager reference outputs,
carried decoder state, and gold final token sequences.

Run:
  HF_HUB_OFFLINE=1 ./.venv/bin/python export_finalize_encoder.py --out ./artifacts
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import torch

from finalize_ref import (
    ContinuousFinalizeRef,
    FinalizeInputs,
    load_benchmark_dataset,
    load_model,
    load_wav,
)


class FinalizeStep(torch.nn.Module):
    def __init__(self, encoder, drop_extra: int):
        super().__init__()
        self.encoder = encoder
        self.drop_extra = int(drop_extra)

    def forward(self, chunk, length, cache_last_channel, cache_last_time, cache_last_channel_len):
        return self.encoder.cache_aware_stream_step(
            processed_signal=chunk,
            processed_signal_length=length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=True,
            drop_extra_pre_encoded=self.drop_extra,
        )


def _cpu_tree(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, list):
        return [_cpu_tree(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_cpu_tree(item) for item in obj)
    if isinstance(obj, dict):
        return {key: _cpu_tree(value) for key, value in obj.items()}
    return obj


def _input_tuple(inputs: FinalizeInputs):
    return (
        inputs.chunk_mel,
        inputs.chunk_len,
        inputs.cache_last_channel,
        inputs.cache_last_time,
        inputs.cache_last_channel_len,
    )


def _tensor_outputs_equal(lhs, rhs) -> tuple[bool, float]:
    ok = True
    max_diff = 0.0
    for a, b in zip(lhs, rhs):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            ok = ok and (a == b)
            continue
        if not torch.equal(a, b):
            ok = False
            max_diff = max(max_diff, float((a.float() - b.float()).abs().max().item()))
    return ok, max_diff


def _first_token_diff(lhs: list[int], rhs: list[int]) -> str:
    for index, (a, b) in enumerate(zip(lhs, rhs)):
        if a != b:
            return f"first_diff={index} finalize_ref={a} nemo_oracle={b}"
    if len(lhs) != len(rhs):
        return f"prefix_equal len_finalize_ref={len(lhs)} len_nemo_oracle={len(rhs)}"
    return "identical"


def _fit_length(wav: np.ndarray, n: int) -> np.ndarray:
    """Tile-or-clip a real utterance to exactly n samples. Audio content is
    irrelevant to the finalize geometry (T depends only on length), and the
    oracle gate is self-consistent on whatever audio we feed."""
    if len(wav) == 0:
        raise ValueError("empty source wav")
    if len(wav) >= n:
        return wav[:n].copy()
    reps = -(-n // len(wav))  # ceil
    return np.tile(wav, reps)[:n].copy()


def _prepare_session(
    rt: ContinuousFinalizeRef,
    ds,
    sample_index: int,
    kind: str,
    target_samples: Optional[int] = None,
):
    """Build a session fed to the target audio length. Returns (ex, wav, session)
    or None if the resulting steady-emission regime doesn't match the requested
    kind (first => emitted_frames==0/drop=0; continuation => emitted_frames>0/drop=2)."""
    ex = ds[sample_index]
    wav = load_wav(ex)

    if kind == "first":
        # Production can only finalize with emitted_frames==0 before the first
        # steady chunk is ready, so clip below preprocess_new_audio_samples.
        limit = rt.geometry.preprocess_new_audio_samples - 1
        n = limit if target_samples is None else min(int(target_samples), limit)
        wav = wav[:n].copy()
    elif kind == "continuation":
        if target_samples is not None:
            wav = _fit_length(wav, int(target_samples))
    else:
        raise ValueError(f"unknown fixture kind {kind!r}")

    session = rt.new_session(f"fixture-{kind}-{sample_index}")
    rt.append_audio(session, wav)

    if kind == "first" and session.emitted_frames != 0:
        return None
    if kind == "continuation" and session.emitted_frames == 0:
        return None
    return ex, wav, session


def _make_row(rt: ContinuousFinalizeRef, ds, sample_index: int, kind: str) -> dict[str, Any]:
    prepared = _prepare_session(rt, ds, sample_index, kind)
    if prepared is None:
        raise RuntimeError(
            f"regime mismatch building fixture: sample_index={sample_index} kind={kind}"
        )
    ex, wav, session = prepared
    return _finish_row(rt, ex, wav, session, kind, sample_index)


def _finish_row(rt, ex, wav, session, kind: str, sample_index: int) -> dict[str, Any]:
    fork = rt.build_continuous_finalize_fork(session)
    pre_final_tokens = list(fork.hyp_tokens)
    pre_final_decoder_state = _cpu_tree(fork.decoder_state)
    pre_final_pred_out = _cpu_tree(fork.pred_out_stream)
    flush = rt.flush_finalize_fork(fork)
    inputs = flush["inputs"]
    eager_outputs = flush["encoder_outputs"]
    if inputs is None or eager_outputs is None:
        raise RuntimeError(f"empty finalize fixture for sample_index={sample_index}")

    finalize_ref_tokens = list(flush["final_tokens"])
    oracle_tokens = rt.nemo_stream_finalize_tokens(wav)
    # Diagnostic-only reference (not used by the bucket export or the oracle gate).
    # Sweep rows can be as short as ~160 samples, which underflows the offline
    # preprocessor's STFT (n_fft padding > input); zero-pad to the runtime's
    # constant preprocess floor for the offline call only. The streaming session
    # above already produced the actual fixture geometry from the un-padded clip.
    offline_wav = wav
    floor = rt.geometry.constant_preprocess_samples
    if len(offline_wav) < floor:
        offline_wav = np.concatenate(
            [offline_wav, np.zeros(floor - len(offline_wav), dtype=np.float32)]
        )
    offline_tokens = rt.offline_full_greedy_tokens(offline_wav)
    oracle_match = finalize_ref_tokens == oracle_tokens
    new_tokens = oracle_tokens[len(pre_final_tokens) :]
    return {
        "kind": kind,
        "sample_index": sample_index,
        "sample_id": ex.get("sample_id"),
        "transcription": ex.get("transcription"),
        "drop_extra": int(inputs.drop_extra),
        "chunk_T": int(inputs.chunk_mel.shape[-1]),
        "remaining_frames": int(inputs.remaining_frames),
        "padded_total_samples": int(inputs.padded_total_samples),
        "chunk_mel": inputs.chunk_mel.detach().cpu().clone(),
        "chunk_len": inputs.chunk_len.detach().cpu().clone(),
        "cache_last_channel": inputs.cache_last_channel.detach().cpu().clone(),
        "cache_last_time": inputs.cache_last_time.detach().cpu().clone(),
        "cache_last_channel_len": inputs.cache_last_channel_len.detach().cpu().clone(),
        "eager_outputs": [item.detach().cpu().clone() for item in eager_outputs],
        "pre_final_tokens": torch.tensor(pre_final_tokens, dtype=torch.int64),
        "gold_new_tokens": torch.tensor(new_tokens, dtype=torch.int64),
        "gold_final_tokens": torch.tensor(oracle_tokens, dtype=torch.int64),
        "nemo_stream_finalize_tokens": torch.tensor(oracle_tokens, dtype=torch.int64),
        "finalize_ref_final_tokens": torch.tensor(finalize_ref_tokens, dtype=torch.int64),
        "offline_full_greedy_tokens": torch.tensor(offline_tokens, dtype=torch.int64),
        "nemo_oracle_match": oracle_match,
        "nemo_oracle_mismatch": (
            "" if oracle_match else _first_token_diff(finalize_ref_tokens, oracle_tokens)
        ),
        "pre_final_decoder_state": pre_final_decoder_state,
        "pre_final_pred_out": pre_final_pred_out,
        "steady_text": rt.text(pre_final_tokens),
        "final_text": rt.text(oracle_tokens),
        "finalize_ref_text": rt.text(finalize_ref_tokens),
        "offline_full_greedy_text": rt.text(offline_tokens),
    }


def _row_inputs_cuda(row: dict[str, Any], device):
    return (
        row["chunk_mel"].to(device),
        row["chunk_len"].to(device),
        row["cache_last_channel"].to(device),
        row["cache_last_time"].to(device),
        row["cache_last_channel_len"].to(device),
    )


# The finalize AOTI bucket grid: both drop regimes reduce to the same window of
# remaining_frames (34..49). drop=0 => T == remaining_frames; drop=2 =>
# T == pre_encode_cache_size + remaining_frames. See ARTIFACTS.md ("32 finalize buckets").
DROP0_T = range(34, 50)  # 16 buckets, T == remaining_frames
DROP2_T = range(43, 59)  # 16 buckets, T == pre_encode_cache_size + remaining_frames
TARGETS: set[tuple[int, int]] = {(0, t) for t in DROP0_T} | {(2, t) for t in DROP2_T}


def _sweep_rows(rt: ContinuousFinalizeRef, ds) -> list[dict[str, Any]]:
    """Sweep audio lengths to materialize one fixture row per (drop, T) bucket in
    the full 32-bucket grid. remaining_frames = len(pending_audio)//hop + 1, so
    stepping the fed length by hop_samples walks T by 1; each row is binned by its
    observed (drop, T) and kept only if it fills a needed, still-empty bucket."""
    g = rt.geometry
    hop = g.hop_samples
    pad = g.final_padding_frames * hop  # fixed final-pad samples in every fork
    rows: dict[tuple[int, int], dict[str, Any]] = {}

    # drop=0: emitted_frames stays 0, so clip length sets T directly and exactly.
    for T in DROP0_T:
        clip = (T - 1) * hop - pad  # invert remaining = (clip+pad)//hop + 1
        if not 0 < clip < g.preprocess_new_audio_samples:
            raise RuntimeError(f"drop=0 T={T} clip={clip} outside first-chunk regime")
        prepared = _prepare_session(rt, ds, 5, "first", target_samples=clip)
        if prepared is None:
            continue
        row = _finish_row(rt, *prepared, "first", 5)
        key = (row["drop_extra"], row["chunk_T"])
        if key == (0, T):
            rows[key] = row

    # drop=2: past the first steady chunk, remaining cycles through 16 consecutive
    # values per shift period; sweep 2 periods (safety margin) and bin what lands.
    base = 6 * g.shift_frames * hop
    for k in range(2 * g.shift_frames):
        prepared = _prepare_session(rt, ds, 4, "continuation", target_samples=base + k * hop)
        if prepared is None:
            continue
        row = _finish_row(rt, *prepared, "continuation", 4)
        key = (row["drop_extra"], row["chunk_T"])
        if key in TARGETS and key not in rows:
            rows[key] = row
        if TARGETS <= rows.keys():
            break

    missing = sorted(TARGETS - rows.keys())
    if missing:
        raise RuntimeError(f"length sweep failed to cover {len(missing)} buckets: {missing}")
    return [rows[key] for key in sorted(rows)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./artifacts")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("loading model + building finalize fixtures...")
    model = load_model()
    rt = ContinuousFinalizeRef(model)
    ds = load_benchmark_dataset()
    device = rt.device

    rows = _sweep_rows(rt, ds)

    fixture_path = os.path.join(args.out, "finalize_fixture.pt")
    meta = {
        "model": "nvidia/nemotron-speech-streaming-en-0.6b",
        "att_context_size": [70, 1],
        "keep_all_outputs": True,
        "geometry": asdict(rt.geometry),
        "notes": (
            "Rows contain finalize encoder inputs, eager encoder outputs, "
            "carried decoder state, and NeMo chunked stream+finalize oracle "
            "gold final/new token sequences."
        ),
    }
    torch.save({"meta": meta, "rows": rows}, fixture_path)
    print(f"saved fixture: {fixture_path}")
    for row in rows:
        print(
            f"  row kind={row['kind']} idx={row['sample_index']} "
            f"drop={row['drop_extra']} T={row['chunk_T']} "
            f"pre/final/offline tok={len(row['pre_final_tokens'])}/"
            f"{len(row['gold_final_tokens'])}/{len(row['offline_full_greedy_tokens'])} "
            f"nemo_oracle_match={row['nemo_oracle_match']}"
        )
        if not row["nemo_oracle_match"]:
            print(f"    mismatch: {row['nemo_oracle_mismatch']}")

    gold_mismatches = [row for row in rows if not row["nemo_oracle_match"]]
    if gold_mismatches:
        print("=== FINALIZE_ENCODER_EXPORT FAIL nemo oracle gold mismatch ===")
        for row in gold_mismatches:
            print(
                f"  row kind={row['kind']} idx={row['sample_index']}: "
                f"{row['nemo_oracle_mismatch']}"
            )
        return 1

    report_lines: list[str] = []
    all_trace_byte_exact = True
    for drop_extra in (0, 2):
        drop_rows = [row for row in rows if row["drop_extra"] == drop_extra]
        if not drop_rows:
            continue
        example = drop_rows[0]
        module = FinalizeStep(rt.encoder, drop_extra).cuda().eval()
        ts_name = f"enc_finalize_drop{drop_extra}.ts"
        ts_path = os.path.join(args.out, ts_name)
        try:
            with torch.inference_mode():
                traced = torch.jit.trace(
                    module,
                    _row_inputs_cuda(example, device),
                    check_trace=False,
                )
            traced.save(ts_path)
            print(f"saved trace: {ts_path}")
        except Exception as exc:
            all_trace_byte_exact = False
            line = (
                f"drop={drop_extra}: torch.jit.trace FAILED: "
                f"{type(exc).__name__}: {str(exc)[:500]}"
            )
            print(line)
            report_lines.append(line)
            continue

        drop_ok = True
        drop_max_diff = 0.0
        for row in drop_rows:
            cuda_inputs = _row_inputs_cuda(row, device)
            eager = [item.to(device) for item in row["eager_outputs"]]
            try:
                with torch.inference_mode():
                    traced_out = traced(*cuda_inputs)
                ok, max_diff = _tensor_outputs_equal(eager, traced_out)
            except Exception as exc:
                ok = False
                max_diff = float("inf")
                report_lines.append(
                    f"drop={drop_extra} T={row['chunk_T']}: traced call FAILED: "
                    f"{type(exc).__name__}: {str(exc)[:500]}"
                )
            drop_ok = drop_ok and ok
            drop_max_diff = max(drop_max_diff, max_diff)
            print(
                f"  trace check drop={drop_extra} T={row['chunk_T']}: "
                f"byte_exact={ok} max_abs_diff={max_diff:.3e}"
            )

        all_trace_byte_exact = all_trace_byte_exact and drop_ok
        if drop_ok:
            report_lines.append(
                f"drop={drop_extra}: TorchScript trace byte-exact for "
                f"T={[row['chunk_T'] for row in drop_rows]}"
            )
        else:
            report_lines.append(
                f"drop={drop_extra}: TorchScript trace NOT byte-exact/dynamic-faithful "
                f"for T={[row['chunk_T'] for row in drop_rows]} "
                f"(max_abs_diff={drop_max_diff:.3e}); C++ port should use "
                "torch.export/AOTI or native encoder step for byte-exact finalize."
            )

    report_path = os.path.join(args.out, "finalize_export_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"saved report: {report_path}")
    print(
        "=== FINALIZE_ENCODER_EXPORT "
        f"{'PASS byte-exact trace' if all_trace_byte_exact else 'PASS fixture; trace needs torch.export/AOTI'} "
        "==="
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
