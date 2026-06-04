#!/usr/bin/env python3
"""Export a deterministic continuous-session replay bundle.

The C++ session harness consumes this bundle to replay the verified
``finalize_ref.ContinuousFinalizeRef`` state machine.  In the default MEL-fed
mode, C++ consumes Python-produced mels directly.  With ``--audio``, C++ consumes
the same PCM appended to the Python session and treats Python-produced mels and
finalize geometry as gold checks only.  Each utterance contains:

* ordered steady ``new_mel`` chunks plus geometry flags;
* the single finalize remainder ``chunk_mel`` plus ``drop_extra`` and ``T``;
* gold cumulative token ids from ``ContinuousFinalizeRef.debounce_expire``.
* gold emitted-event stream from the finalize_ref/server WORD/TEXT semantics.
  Text is packed as UTF-8 bytes and C++ compares it directly.  Interim events
  are generated from the same AOTI steady encoder used by the C++ session, so
  this gate checks event logic rather than eager-vs-AOTI timing drift.
* the tokenizer id->piece table plus Python ``ids_to_text`` self-test sequences
  so C++ can prove its detokenizer matches the reference at load.
* in ``--audio`` mode: raw PCM per appended turn, fixed-preprocessor geometry,
  per-chunk gold mel tensors, and final remainder geometry/tensors.

Run from runtime/:
  HF_HUB_OFFLINE=1 ./.venv/bin/python export_session_bundle.py --n 20
  HF_HUB_OFFLINE=1 ./.venv/bin/python export_session_bundle.py --multiturn --n 8
  HF_HUB_OFFLINE=1 ./.venv/bin/python export_session_bundle.py --audio --n 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from finalize_ref import (
    BLANK,
    MAX_SYMBOLS,
    RIGHT_CONTEXT,
    ContinuousFinalizeRef,
    ContinuousSession,
    _continuous_append_only_delta,
    load_benchmark_dataset,
    load_model,
    load_wav,
)
from ref_decode import ref_greedy_range


ART = os.path.join(os.path.dirname(__file__), "artifacts")
EVENT_INTERIM = 0
EVENT_FINAL = 1
EVENT_SUPPRESSED = 2
PREPROC_CI_REL_EPS = 1.0e-6
PREPROC_CI_MEL_MAX_HEADROOM = 2.0
PREPROC_CI_MEL_P99_HEADROOM = 16.0
PREPROC_CI_CACHE_MAX_HEADROOM = 2.0
MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
PREPROC_MANIFEST_SCHEMA = 1


def _as_cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"expected tensor, got {type(value).__name__}")
    return value.detach().cpu().clone()


def _audio_tensor(wav: Any) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).clone()


def _scalar(value: int | bool) -> torch.Tensor:
    return torch.tensor([int(value)], dtype=torch.int64)


def _pack_i64_lists(values: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    flat: list[int] = []
    for item in values:
        flat.extend(int(v) for v in item)
        offsets.append(len(flat))
    return (
        torch.tensor(flat, dtype=torch.int64),
        torch.tensor(offsets, dtype=torch.int64),
    )


def _pack_utf8(strings: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    flat = bytearray()
    for text in strings:
        encoded = text.encode("utf-8")
        flat.extend(encoded)
        offsets.append(len(flat))
    return (
        torch.tensor(list(flat), dtype=torch.uint8),
        torch.tensor(offsets, dtype=torch.int64),
    )


def _sample_id_for(example: dict[str, Any], sample_index: int) -> str:
    return str(example.get("sample_id", sample_index))


def _tokenizer_vocab_size(tokenizer: Any) -> int:
    for attr in ("original_vocab_size", "vocab_size"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    sp = getattr(tokenizer, "tokenizer", None)
    if sp is not None and hasattr(sp, "get_piece_size"):
        return int(sp.get_piece_size())
    raise TypeError("cannot determine tokenizer vocabulary size")


def _tokenizer_pieces(tokenizer: Any) -> list[str]:
    vocab = _tokenizer_vocab_size(tokenizer)
    return [str(tokenizer.ids_to_tokens([idx])[0]) for idx in range(vocab)]


def _build_detok_selftest(rows: list[dict[str, Any]], tokenizer: Any) -> tuple[list[list[int]], list[str]]:
    vocab = _tokenizer_vocab_size(tokenizer)
    sequences: list[list[int]] = [[]]
    sequences.extend([[idx] for idx in range(vocab)])
    for row in rows:
        sequences.append(row["steady_tokens"].cpu().to(torch.int64).tolist())
        sequences.append(row["gold_tokens"].cpu().to(torch.int64).tolist())
        for event in row["events"]:
            sequences.append(list(event["tokens"]))
            sequences.append(list(event["collector_tokens"]))

    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for seq in sequences:
        key = tuple(int(v) for v in seq)
        if key in seen:
            continue
        seen.add(key)
        unique.append(list(key))

    texts = [tokenizer.ids_to_text(seq) if seq else "" for seq in unique]
    return unique, texts


def _append_only_delta_tokens(final_tokens: list[int], emitted_tokens: list[int]) -> list[int]:
    """Legacy token payload helper; text fields are the authoritative oracle."""
    common = 0
    for emitted_token, final_token in zip(emitted_tokens, final_tokens):
        if emitted_token != final_token:
            break
        common += 1

    if common == len(emitted_tokens):
        delta_tokens = final_tokens[common:]
    elif len(final_tokens) <= len(emitted_tokens):
        delta_tokens = []
    else:
        delta_tokens = final_tokens[len(emitted_tokens) :]
        max_overlap = min(len(emitted_tokens), len(delta_tokens))
        for overlap in range(max_overlap, 0, -1):
            if emitted_tokens[-overlap:] == delta_tokens[:overlap]:
                delta_tokens = delta_tokens[overlap:]
                break

    return list(delta_tokens)


def _decoder_state_hc(state: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(state, (tuple, list)) and len(state) == 2:
        h, c = state
        if torch.is_tensor(h) and torch.is_tensor(c):
            return h, c
    raise TypeError(f"unsupported decoder_state shape for export: {type(state).__name__}")


def _register_event_buffers(module: torch.nn.Module, prefix: str, events: list[dict[str, Any]]) -> None:
    module.register_buffer(
        f"{prefix}_event_kinds",
        torch.tensor([event["kind"] for event in events], dtype=torch.int64),
    )
    event_tokens, event_token_offsets = _pack_i64_lists(
        [event["tokens"] for event in events]
    )
    collector_tokens, collector_token_offsets = _pack_i64_lists(
        [event["collector_tokens"] for event in events]
    )
    event_text_bytes, event_text_offsets = _pack_utf8(
        [event["text"] for event in events]
    )
    collector_text_bytes, collector_text_offsets = _pack_utf8(
        [event["collector_text"] for event in events]
    )
    module.register_buffer(f"{prefix}_event_tokens", event_tokens)
    module.register_buffer(f"{prefix}_event_token_offsets", event_token_offsets)
    module.register_buffer(f"{prefix}_event_collector_tokens", collector_tokens)
    module.register_buffer(
        f"{prefix}_event_collector_token_offsets",
        collector_token_offsets,
    )
    module.register_buffer(f"{prefix}_event_text_bytes", event_text_bytes)
    module.register_buffer(f"{prefix}_event_text_offsets", event_text_offsets)
    module.register_buffer(
        f"{prefix}_event_collector_text_bytes",
        collector_text_bytes,
    )
    module.register_buffer(
        f"{prefix}_event_collector_text_offsets",
        collector_text_offsets,
    )


def _register_steady_chunks(
    module: torch.nn.Module,
    prefix: str,
    steady_chunks: list[dict[str, Any]],
) -> None:
    for j, chunk in enumerate(steady_chunks):
        cprefix = f"{prefix}_chunk{j}"
        module.register_buffer(f"{cprefix}_new_mel", chunk["new_mel"].cpu())
        module.register_buffer(f"{cprefix}_is_first", _scalar(chunk["is_first"]))
        module.register_buffer(f"{cprefix}_drop_extra", _scalar(chunk["drop_extra"]))
        module.register_buffer(f"{cprefix}_chunk_T", _scalar(chunk["chunk_T"]))
        module.register_buffer(
            f"{cprefix}_emitted_before",
            _scalar(chunk["emitted_before"]),
        )
        if "first_eager_enc_out" in chunk:
            module.register_buffer(
                f"{cprefix}_first_eager_enc_out",
                chunk["first_eager_enc_out"].cpu(),
            )
            module.register_buffer(
                f"{cprefix}_first_eager_enc_len",
                chunk["first_eager_enc_len"].cpu().to(torch.int64),
            )


def _register_audio_geometry(
    module: torch.nn.Module,
    geometry,
    *,
    audio: bool,
) -> None:
    module.register_buffer("audio_bundle_mode", _scalar(audio))
    module.register_buffer(
        "audio_geometry",
        torch.tensor(
            [
                int(geometry.shift_frames),
                int(geometry.pre_encode_cache_size),
                int(geometry.drop_extra),
                int(geometry.final_padding_frames),
                RIGHT_CONTEXT,
                int(geometry.first_preprocess_mel_frame),
                int(geometry.hop_samples),
                int(geometry.raw_audio_ring_samples),
                int(geometry.preprocess_align_pad_samples),
                int(geometry.preprocess_new_audio_samples),
                int(geometry.stream_preprocess_valid_samples),
                int(geometry.constant_preprocess_frames),
                int(geometry.constant_preprocess_samples),
            ],
            dtype=torch.int64,
        ),
    )


def _register_audio_ci(
    module: torch.nn.Module,
    audio_ci: dict[str, Any] | None,
    *,
    audio: bool,
) -> None:
    if not audio:
        return
    if audio_ci is None:
        raise ValueError("audio bundle requires measured audio CI thresholds")
    module.register_buffer(
        "mel_ci_atol",
        torch.tensor([float(audio_ci["mel_ci_atol"])], dtype=torch.float64),
    )
    module.register_buffer(
        "cache_ci_atol",
        torch.tensor([float(audio_ci["cache_ci_atol"])], dtype=torch.float64),
    )
    module.register_buffer(
        "audio_ci_stats",
        torch.tensor(
            [
                float(audio_ci["mel"]["abs"]["max"]),
                float(audio_ci["mel"]["abs"]["mean"]),
                float(audio_ci["mel"]["abs"]["p99"]),
                float(audio_ci["mel"]["rel"]["max"]),
                float(audio_ci["mel"]["rel"]["mean"]),
                float(audio_ci["mel"]["rel"]["p99"]),
                float(audio_ci["cache"]["cache_last_channel"]["max_abs"]),
                float(audio_ci["cache"]["cache_last_time"]["max_abs"]),
                float(audio_ci["cache"]["cache_last_channel_len"]["max_abs"]),
                float(audio_ci["mel"]["checks"]),
                float(audio_ci["cache"]["checks"]),
                float(audio_ci["headroom"]["mel_max"]),
                float(audio_ci["headroom"]["mel_p99"]),
                float(audio_ci["headroom"]["cache_max"]),
            ],
            dtype=torch.float64,
        ),
    )
    rationale_bytes, rationale_offsets = _pack_utf8([str(audio_ci["rationale"])])
    module.register_buffer("audio_ci_rationale_bytes", rationale_bytes)
    module.register_buffer("audio_ci_rationale_offsets", rationale_offsets)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _audio_geometry_manifest_values(geometry) -> list[int]:
    return [
        int(geometry.shift_frames),
        int(geometry.pre_encode_cache_size),
        int(geometry.drop_extra),
        int(geometry.final_padding_frames),
        RIGHT_CONTEXT,
        int(geometry.first_preprocess_mel_frame),
        int(geometry.hop_samples),
        int(geometry.raw_audio_ring_samples),
        int(geometry.preprocess_align_pad_samples),
        int(geometry.preprocess_new_audio_samples),
        int(geometry.stream_preprocess_valid_samples),
        int(geometry.constant_preprocess_frames),
        int(geometry.constant_preprocess_samples),
    ]


def _preprocessor_config(model) -> Any:
    pp = model.preprocessor
    candidates = [
        getattr(pp, "cfg", None),
        getattr(pp, "_cfg", None),
        getattr(getattr(pp, "featurizer", None), "cfg", None),
        getattr(getattr(pp, "featurizer", None), "_cfg", None),
    ]
    for cfg in candidates:
        if cfg is None:
            continue
        try:
            return _jsonable(OmegaConf.to_container(cfg, resolve=True))
        except Exception:
            return _jsonable(cfg)

    featurizer = getattr(pp, "featurizer", None)
    if featurizer is None:
        return {}
    fallback: dict[str, Any] = {}
    for name in (
        "sample_rate",
        "n_window_size",
        "n_window_stride",
        "window",
        "normalize",
        "n_fft",
        "nfilt",
        "features",
        "dither",
        "pad_to",
        "frame_splicing",
        "stft_exact_pad",
        "stft_conv",
        "mag_power",
        "log_zero_guard_type",
        "log_zero_guard_value",
    ):
        if hasattr(featurizer, name):
            fallback[name] = _jsonable(getattr(featurizer, name))
    return fallback


def _preproc_manifest_payload(model, geometry, preproc_path: str) -> dict[str, Any]:
    dither = getattr(getattr(model.preprocessor, "featurizer", None), "dither", None)
    return {
        "CONTRACT": {
            "schema": PREPROC_MANIFEST_SCHEMA,
            "model_id": MODEL_ID,
            "dither": None if dither is None else float(dither),
            "geometry": _audio_geometry_manifest_values(geometry),
            "trace_k": int(geometry.constant_preprocess_samples),
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "cudnn_version": None
            if torch.backends.cudnn.version() is None
            else int(torch.backends.cudnn.version()),
            "preproc_ts_sha256": _sha256_file(preproc_path),
        },
        "preprocessor_config": _preprocessor_config(model),
    }


def _preproc_manifest_matches(manifest: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, str]:
    for key in ("CONTRACT", "preprocessor_config"):
        if key not in manifest or key not in expected:
            return False, key
    for key in (
        "schema",
        "model_id",
        "dither",
        "geometry",
        "trace_k",
        "torch_version",
        "cuda_version",
        "cudnn_version",
        "preproc_ts_sha256",
    ):
        if manifest["CONTRACT"].get(key) != expected["CONTRACT"].get(key):
            return False, f"CONTRACT.{key}"
    if manifest.get("preprocessor_config") != expected.get("preprocessor_config"):
        return False, "preprocessor_config"
    return True, ""


def _trace_preproc_ts(model, artifacts_dir: str, path: str) -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    pp = model.preprocessor

    class PP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pp = pp

        def forward(self, sig, length):
            return self.pp(input_signal=sig, length=length)

    # The traced graph is shape-flexible for the NeMo preprocessor, but tracing
    # at the fixed runtime K makes the artifact provenance match the audio gate.
    dev = next(model.parameters()).device
    g = ContinuousFinalizeRef(model).geometry
    audio = torch.zeros((1, g.constant_preprocess_samples), device=dev)
    length = torch.full((1,), g.constant_preprocess_samples, device=dev, dtype=torch.long)
    with torch.inference_mode():
        ts = torch.jit.trace(PP().to(dev).eval(), (audio, length), check_trace=False)
    ts.save(path)
    print(f"exported preprocessor TorchScript: {path}")


def _preproc_ts_vs_eager_self_check(model, preproc_path: str, geometry) -> dict[str, Any]:
    dev = next(model.parameters()).device
    preproc_ts = torch.jit.load(preproc_path).to(dev).eval()
    audio = torch.linspace(
        -0.25,
        0.25,
        int(geometry.constant_preprocess_samples),
        device=dev,
        dtype=torch.float32,
    ).unsqueeze(0)
    length = torch.full(
        (1,),
        int(geometry.stream_preprocess_valid_samples),
        device=dev,
        dtype=torch.long,
    )
    with torch.inference_mode():
        eager_mel, eager_len = model.preprocessor(input_signal=audio, length=length)
        ts_mel, ts_len = preproc_ts(audio, length)
    byte_equal = bool(torch.equal(eager_mel, ts_mel) and torch.equal(eager_len, ts_len))
    max_abs = float((eager_mel.float() - ts_mel.float()).abs().max().item()) if eager_mel.numel() else 0.0
    if not byte_equal:
        raise AssertionError(
            "preproc.ts export-time TS-vs-eager self-check failed: "
            f"byte_equal={byte_equal} max_abs={max_abs:.6e}"
        )
    print(
        "preproc.ts TS-vs-eager self-check PASS: "
        f"K={geometry.constant_preprocess_samples} valid={geometry.stream_preprocess_valid_samples} "
        f"mel_shape={tuple(ts_mel.shape)} max_abs={max_abs:.6e}"
    )
    return {
        "byte_equal": byte_equal,
        "max_abs": max_abs,
        "K": int(geometry.constant_preprocess_samples),
        "valid_samples": int(geometry.stream_preprocess_valid_samples),
    }


def _ensure_preproc_ts(model, artifacts_dir: str) -> None:
    path = os.path.join(artifacts_dir, "preproc.ts")
    manifest_path = path + ".manifest.json"
    geometry = ContinuousFinalizeRef(model).geometry

    if os.path.exists(path) or os.path.exists(manifest_path):
        print(f"regenerating preproc.ts and manifest unconditionally for audio export: {path}")

    _trace_preproc_ts(model, artifacts_dir, path)
    self_check = _preproc_ts_vs_eager_self_check(model, path, geometry)
    manifest = _preproc_manifest_payload(model, geometry, path)
    manifest["export_time_self_check"] = self_check
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    expected = _preproc_manifest_payload(model, geometry, path)
    ok, mismatch_key = _preproc_manifest_matches(manifest, {**expected, "export_time_self_check": self_check})
    if not ok:
        raise AssertionError(f"internal preproc manifest verification failed at {mismatch_key}")
    print(
        "wrote preproc.ts manifest: "
        f"{manifest_path} sha256={manifest['CONTRACT']['preproc_ts_sha256']} "
        f"trace_k={manifest['CONTRACT']['trace_k']}"
    )


def _append_diff(
    abs_parts: list[torch.Tensor],
    rel_parts: list[torch.Tensor],
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    if actual.shape != expected.shape:
        raise ValueError(f"CI tensor shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    diff = (actual.float() - expected.float()).abs().detach().cpu().reshape(-1)
    denom = expected.detach().cpu().float().abs().reshape(-1).clamp_min(PREPROC_CI_REL_EPS)
    abs_parts.append(diff)
    rel_parts.append(diff / denom)


def _summary(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {"max": 0.0, "mean": 0.0, "p99": 0.0}
    values = values.reshape(-1).to(torch.float64)
    kth = max(1, min(values.numel(), int(np.ceil(0.99 * values.numel()))))
    return {
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "p99": float(values.kthvalue(kth).values.item()),
    }


def _ci_preproc_new_mel(
    rt: "RecordingContinuousFinalizeRef",
    preproc_ts: torch.jit.ScriptModule,
    raw_audio_ring: np.ndarray,
    new_audio: np.ndarray,
) -> torch.Tensor:
    g = rt.geometry
    fixed_audio, valid_samples = rt._build_fixed_preprocess_audio(raw_audio_ring, new_audio)
    audio = torch.from_numpy(np.ascontiguousarray(fixed_audio)).unsqueeze(0).to(rt.device)
    audio_len = torch.tensor([valid_samples], device=rt.device, dtype=torch.long)
    ts_mel, _ts_len = preproc_ts(audio, audio_len)
    return ts_mel[
        :,
        :,
        g.first_preprocess_mel_frame : g.first_preprocess_mel_frame + g.shift_frames,
    ].contiguous()


def _ci_process_steady_chunk(
    rt: "RecordingContinuousFinalizeRef",
    session: ContinuousSession,
    valid_new_mel: torch.Tensor,
) -> None:
    g = rt.geometry
    is_first = session.emitted_frames == 0
    if is_first:
        chunk_mel = valid_new_mel
        drop_extra = 0
    else:
        if session.mel_frame_ring is None:
            raise RuntimeError("CI steady continuation missing mel ring")
        chunk_mel = torch.cat((session.mel_frame_ring, valid_new_mel), dim=-1)
        drop_extra = int(g.drop_extra)

    _enc_out, _enc_len, clc, clt, clcl = rt._run_aoti_consistent_steady_encoder(
        session,
        chunk_mel.contiguous(),
        drop_extra,
    )
    session.cache_last_channel = clc
    session.cache_last_time = clt
    session.cache_last_channel_len = clcl

    consumed_audio = session.pending_audio[: g.shift_frames * g.hop_samples]
    session.raw_audio_ring = rt._advance_raw_ring(session.raw_audio_ring, consumed_audio)
    session.pending_audio = session.pending_audio[g.shift_frames * g.hop_samples :]
    rt._update_mel_frame_ring(session, valid_new_mel)
    session.emitted_frames += g.shift_frames


def _ci_prepare_finalize_inputs(
    rt: "RecordingContinuousFinalizeRef",
    preproc_ts: torch.jit.ScriptModule,
    parent: ContinuousSession,
) -> dict[str, Any]:
    g = rt.geometry
    pending = np.asarray(parent.pending_audio, dtype=np.float32).copy()
    if parent.total_audio_samples > 0:
        pending = np.concatenate(
            [pending, np.zeros(g.final_padding_frames * g.hop_samples, dtype=np.float32)]
        ).astype(np.float32, copy=False)

    padded_total_samples = int(parent.emitted_frames * g.hop_samples + len(pending))
    if len(pending) == 0:
        padded_total_samples = int(parent.emitted_frames * g.hop_samples)
        return {
            "has_inputs": False,
            "padded_total_samples": padded_total_samples,
            "total_mel_frames": 0,
            "remaining_frames": 0,
            "drop_extra": -1,
            "final_T": 0,
            "new_mel": torch.empty((1, 128, 0), device=rt.device),
            "chunk_mel": torch.empty((1, 128, 0), device=rt.device),
        }

    total_mel_frames = int(padded_total_samples // g.hop_samples + 1)
    remaining_frames = int(total_mel_frames - parent.emitted_frames)
    if remaining_frames <= 0:
        return {
            "has_inputs": False,
            "padded_total_samples": padded_total_samples,
            "total_mel_frames": total_mel_frames,
            "remaining_frames": remaining_frames,
            "drop_extra": -1,
            "final_T": 0,
            "new_mel": torch.empty((1, 128, 0), device=rt.device),
            "chunk_mel": torch.empty((1, 128, 0), device=rt.device),
        }

    raw_ring = np.asarray(parent.raw_audio_ring, dtype=np.float32).copy()
    new_mels: list[torch.Tensor] = []
    frames_collected = 0
    while frames_collected < remaining_frames:
        frames_this_call = min(int(g.shift_frames), remaining_frames - frames_collected)
        needed_new_samples = min(len(pending), int(g.preprocess_new_audio_samples))
        new_audio = pending[:needed_new_samples]
        block_mel = _ci_preproc_new_mel(rt, preproc_ts, raw_ring, new_audio)
        new_mels.append(block_mel[:, :, :frames_this_call].contiguous())

        if frames_this_call == g.shift_frames:
            consumed_samples = min(int(g.shift_frames * g.hop_samples), len(pending))
            consumed = pending[:consumed_samples]
            raw_ring = rt._advance_raw_ring(raw_ring, consumed)
            pending = pending[consumed_samples:]
        frames_collected += frames_this_call

    new_mel = torch.cat(new_mels, dim=2).contiguous()
    if parent.emitted_frames == 0:
        chunk_mel = new_mel
        drop_extra = 0
    else:
        if parent.mel_frame_ring is None:
            raise RuntimeError("CI finalize continuation missing mel ring")
        chunk_mel = torch.cat((parent.mel_frame_ring.to(rt.device), new_mel), dim=2).contiguous()
        drop_extra = int(g.drop_extra)
    return {
        "has_inputs": True,
        "padded_total_samples": padded_total_samples,
        "total_mel_frames": total_mel_frames,
        "remaining_frames": remaining_frames,
        "drop_extra": drop_extra,
        "final_T": int(chunk_mel.shape[-1]),
        "new_mel": new_mel,
        "chunk_mel": chunk_mel,
    }


def _ci_cache_max(cache_max: dict[str, float], actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    if actual.shape != expected.shape:
        raise ValueError(f"CI cache shape mismatch for {name}: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    max_abs = float((actual.float() - expected.float()).abs().max().item()) if actual.numel() else 0.0
    cache_max[name] = max(cache_max[name], max_abs)


def _measure_audio_ci(
    rt: "RecordingContinuousFinalizeRef",
    *,
    artifacts_dir: str,
    rows: list[dict[str, Any]] | None = None,
    streams: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    preproc_path = os.path.join(artifacts_dir, "preproc.ts")
    preproc_ts = torch.jit.load(preproc_path).to(rt.device).eval()
    abs_parts: list[torch.Tensor] = []
    rel_parts: list[torch.Tensor] = []
    cache_max = {
        "cache_last_channel": 0.0,
        "cache_last_time": 0.0,
        "cache_last_channel_len": 0.0,
    }
    mel_checks = 0
    cache_checks = 0

    def process_segment(
        ts_state: ContinuousSession,
        gold_state: ContinuousSession,
        segment: dict[str, Any],
        label: str,
    ) -> None:
        nonlocal mel_checks, cache_checks
        wav = segment["audio"].detach().cpu().numpy().astype(np.float32, copy=True)
        for state in (ts_state, gold_state):
            state.pending_audio = np.concatenate([state.pending_audio, wav]).astype(np.float32, copy=False)
            state.total_audio_samples += int(wav.shape[0])

        for chunk_index, chunk in enumerate(segment["steady_chunks"]):
            if not rt._session_ready(ts_state) or not rt._session_ready(gold_state):
                raise RuntimeError(f"CI state not ready for {label}.chunk{chunk_index}")
            new_audio = ts_state.pending_audio[: rt.geometry.preprocess_new_audio_samples]
            ts_new_mel = _ci_preproc_new_mel(rt, preproc_ts, ts_state.raw_audio_ring, new_audio)
            gold_new_mel = chunk.get("eager_new_mel", chunk["new_mel"]).to(rt.device).contiguous()
            _append_diff(abs_parts, rel_parts, ts_new_mel, gold_new_mel)
            mel_checks += 1
            _ci_process_steady_chunk(rt, ts_state, ts_new_mel)
            _ci_process_steady_chunk(rt, gold_state, gold_new_mel)

        fin = _ci_prepare_finalize_inputs(rt, preproc_ts, ts_state)
        expected_keys = {
            "padded_total_samples": "final_padded_total_samples",
            "total_mel_frames": "final_total_mel_frames",
            "remaining_frames": "final_remaining_frames",
            "drop_extra": "final_drop_extra",
            "final_T": "final_T",
        }
        for name, expected_key in expected_keys.items():
            if int(fin[name]) != int(segment[expected_key]):
                raise RuntimeError(f"CI finalize geometry mismatch for {label}.{name}: {fin[name]} vs {segment[expected_key]}")
        if fin["has_inputs"]:
            _append_diff(
                abs_parts,
                rel_parts,
                fin["new_mel"],
                segment.get("final_eager_new_mel", segment["final_new_mel"]).to(rt.device).contiguous(),
            )
            _append_diff(
                abs_parts,
                rel_parts,
                fin["chunk_mel"],
                segment.get("final_eager_chunk_mel", segment["final_chunk_mel"]).to(rt.device).contiguous(),
            )
            mel_checks += 2

        _ci_cache_max(cache_max, ts_state.cache_last_channel, gold_state.cache_last_channel, "cache_last_channel")
        _ci_cache_max(cache_max, ts_state.cache_last_time, gold_state.cache_last_time, "cache_last_time")
        _ci_cache_max(cache_max, ts_state.cache_last_channel_len, gold_state.cache_last_channel_len, "cache_last_channel_len")
        cache_checks += 3

    with torch.inference_mode():
        if rows is not None:
            for row_index, row in enumerate(rows):
                ts_state = rt.new_session(f"audio-ci-ts-row-{row_index}")
                gold_state = rt.new_session(f"audio-ci-gold-row-{row_index}")
                process_segment(ts_state, gold_state, row, f"row{row_index}")
        if streams is not None:
            for stream_index, stream in enumerate(streams):
                ts_state = rt.new_session(f"audio-ci-ts-stream-{stream_index}")
                gold_state = rt.new_session(f"audio-ci-gold-stream-{stream_index}")
                for turn_index, turn in enumerate(stream["turns"]):
                    process_segment(ts_state, gold_state, turn, f"stream{stream_index}.turn{turn_index}")
                end = stream["end"]
                fin = _ci_prepare_finalize_inputs(rt, preproc_ts, ts_state)
                expected_keys = {
                    "padded_total_samples": "final_padded_total_samples",
                    "total_mel_frames": "final_total_mel_frames",
                    "remaining_frames": "final_remaining_frames",
                    "drop_extra": "final_drop_extra",
                    "final_T": "final_T",
                }
                for name, expected_key in expected_keys.items():
                    if int(fin[name]) != int(end[expected_key]):
                        raise RuntimeError(
                            f"CI true-boundary geometry mismatch for stream{stream_index}.{name}: "
                            f"{fin[name]} vs {end[expected_key]}"
                        )
                if fin["has_inputs"]:
                    _append_diff(
                        abs_parts,
                        rel_parts,
                        fin["new_mel"],
                        end.get("final_eager_new_mel", end["final_new_mel"]).to(rt.device).contiguous(),
                    )
                    _append_diff(
                        abs_parts,
                        rel_parts,
                        fin["chunk_mel"],
                        end.get("final_eager_chunk_mel", end["final_chunk_mel"]).to(rt.device).contiguous(),
                    )
                    mel_checks += 2

    abs_values = torch.cat(abs_parts) if abs_parts else torch.empty((0,), dtype=torch.float32)
    rel_values = torch.cat(rel_parts) if rel_parts else torch.empty((0,), dtype=torch.float32)
    abs_stats = _summary(abs_values)
    rel_stats = _summary(rel_values)
    mel_ci_atol = max(
        abs_stats["max"] * PREPROC_CI_MEL_MAX_HEADROOM,
        abs_stats["p99"] * PREPROC_CI_MEL_P99_HEADROOM,
    )
    cache_ci_atol = max(cache_max["cache_last_channel"], cache_max["cache_last_time"]) * PREPROC_CI_CACHE_MAX_HEADROOM
    rationale = (
        "Fixed-K preproc.ts is measured against eager preprocessing on the bundle PCM. "
        "mel_ci_atol=max(abs_max*{mel_max:g}, abs_p99*{mel_p99:g}); "
        "cache_ci_atol=max(retained cache_last_channel/time max_abs)*{cache_max:g}. "
        "This documents the cuFFT fixed-plan cross-process CI envelope; token/event exactness remains the semantic gate."
    ).format(
        mel_max=PREPROC_CI_MEL_MAX_HEADROOM,
        mel_p99=PREPROC_CI_MEL_P99_HEADROOM,
        cache_max=PREPROC_CI_CACHE_MAX_HEADROOM,
    )
    report = {
        "mel_ci_atol": float(mel_ci_atol),
        "cache_ci_atol": float(cache_ci_atol),
        "mel": {
            "checks": int(mel_checks),
            "rel_eps": PREPROC_CI_REL_EPS,
            "abs": abs_stats,
            "rel": rel_stats,
        },
        "cache": {
            "checks": int(cache_checks),
            "cache_last_channel": {"max_abs": float(cache_max["cache_last_channel"])},
            "cache_last_time": {"max_abs": float(cache_max["cache_last_time"])},
            "cache_last_channel_len": {"max_abs": float(cache_max["cache_last_channel_len"])},
        },
        "headroom": {
            "mel_max": PREPROC_CI_MEL_MAX_HEADROOM,
            "mel_p99": PREPROC_CI_MEL_P99_HEADROOM,
            "cache_max": PREPROC_CI_CACHE_MAX_HEADROOM,
        },
        "rationale": rationale,
    }
    print(
        "audio CI: mel_abs max={max:.6e} mean={mean:.6e} p99={p99:.6e}; "
        "mel_rel max={rmax:.6e} mean={rmean:.6e} p99={rp99:.6e}; "
        "cache_last_channel={clc:.6e} cache_last_time={clt:.6e}; "
        "thresholds mel_ci_atol={matol:.6e} cache_ci_atol={catol:.6e}".format(
            max=abs_stats["max"],
            mean=abs_stats["mean"],
            p99=abs_stats["p99"],
            rmax=rel_stats["max"],
            rmean=rel_stats["mean"],
            rp99=rel_stats["p99"],
            clc=cache_max["cache_last_channel"],
            clt=cache_max["cache_last_time"],
            matol=mel_ci_atol,
            catol=cache_ci_atol,
        )
    )
    return report


def _write_audio_ci_sidecar(bundle_path: str, audio_ci: dict[str, Any]) -> None:
    sidecar = bundle_path + ".audio_ci.json"
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(audio_ci, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote audio CI sidecar: {sidecar}")


def _events_semantically_equal(lhs: list[dict[str, Any]], rhs: list[dict[str, Any]]) -> bool:
    if len(lhs) != len(rhs):
        return False
    for a, b in zip(lhs, rhs):
        for key in ("kind", "text", "tokens", "collector_text", "collector_tokens"):
            if a[key] != b[key]:
                return False
    return True


def _assert_segment_semantics_match(label: str, shipped: dict[str, Any], eager: dict[str, Any]) -> None:
    checks = (
        ("steady_tokens", shipped["steady_tokens"], eager["steady_tokens"]),
        ("gold_tokens", shipped["gold_tokens"], eager["gold_tokens"]),
        ("finalize_new_tokens", shipped["finalize_new_tokens"], eager["finalize_new_tokens"]),
    )
    for name, expected, actual in checks:
        if not torch.equal(expected.cpu().to(torch.int64), actual.cpu().to(torch.int64)):
            raise AssertionError(f"audio eager semantic subset mismatch for {label}.{name}")
    if not _events_semantically_equal(shipped["events"], eager["events"]):
        raise AssertionError(f"audio eager semantic subset mismatch for {label}.events")


def _run_eager_audio_subset_gate(
    model,
    *,
    artifacts_dir: str,
    dataset: Any,
    start: int,
    n: int,
    multiturn: bool,
    shipped_rows: list[dict[str, Any]] | None = None,
    shipped_streams: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    subset = min(n, 8)
    eager_rt = RecordingContinuousFinalizeRef(
        model,
        artifacts_dir=artifacts_dir,
        use_preproc_ts=False,
        warm_encoder=multiturn,
    )
    segments = 0
    if multiturn:
        if shipped_streams is None:
            raise ValueError("shipped_streams is required for multiturn eager subset gate")
        for stream_index in range(subset):
            sample_a_index = start + 2 * stream_index
            sample_b_index = sample_a_index + 1
            eager_stream = _build_multiturn_stream(
                eager_rt,
                load_wav(dataset[sample_a_index]),
                load_wav(dataset[sample_b_index]),
                stream_index=stream_index,
                sample_a_index=sample_a_index,
                sample_b_index=sample_b_index,
            )
            shipped_stream = shipped_streams[stream_index]
            for turn_index, (shipped_turn, eager_turn) in enumerate(
                zip(shipped_stream["turns"], eager_stream["turns"])
            ):
                _assert_segment_semantics_match(
                    f"stream{stream_index}.turn{turn_index}",
                    shipped_turn,
                    eager_turn,
                )
                segments += 1
            _assert_segment_semantics_match(
                f"stream{stream_index}.true_boundary",
                shipped_stream["end"],
                eager_stream["end"],
            )
            segments += 1
    else:
        if shipped_rows is None:
            raise ValueError("shipped_rows is required for single eager subset gate")
        for offset in range(subset):
            sample_index = start + offset
            example = dataset[sample_index]
            eager_row = _build_row(
                eager_rt,
                load_wav(example),
                sample_index=sample_index,
                sample_id=_sample_id_for(example, sample_index),
                reference_text=str(example.get("transcription", "")),
            )
            _assert_segment_semantics_match(f"row{offset}", shipped_rows[offset], eager_row)
            segments += 1

    report = {
        "status": "PASS",
        "mode": "multiturn" if multiturn else "single",
        "subset": int(subset),
        "segments": int(segments),
        "claim": "eager-preproc tokens/events match the shipped preproc.ts runtime bundle on this subset",
    }
    print(
        "audio eager token/event subset PASS: "
        f"mode={report['mode']} subset={subset} segments={segments} "
        "eager_preproc_vs_shipped_preproc_ts_runtime=token_event_exact"
    )
    return report


def _capture_finalize_payload(
    rt: RecordingContinuousFinalizeRef,
    session: ContinuousSession,
    steady_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    fork = rt.build_continuous_finalize_fork(session)
    inputs = rt.prepare_finalize_inputs(fork)
    eager_inputs = None
    if getattr(rt, "use_preproc_ts", False) and inputs is not None:
        eager_fork = rt.build_continuous_finalize_fork(session)
        eager_inputs = rt.prepare_finalize_inputs_eager(eager_fork)

    if inputs is None:
        mel_dim = (
            int(steady_chunks[0]["new_mel"].shape[1])
            if steady_chunks
            else 128
        )
        final_chunk_mel = torch.empty((1, mel_dim, 0), dtype=torch.float32)
        final_new_mel = torch.empty((1, mel_dim, 0), dtype=torch.float32)
        final_drop_extra = -1
        final_T = 0
        remaining_frames = 0
        padded_total_samples = int(session.emitted_frames * rt.geometry.hop_samples)
        total_mel_frames = 0
        final_eager_chunk_mel = final_chunk_mel
        final_eager_new_mel = final_new_mel
    else:
        final_chunk_mel = _as_cpu_tensor(inputs.chunk_mel)
        final_new_mel = _as_cpu_tensor(inputs.new_mel)
        final_drop_extra = int(inputs.drop_extra)
        final_T = int(inputs.chunk_mel.shape[-1])
        remaining_frames = int(inputs.remaining_frames)
        padded_total_samples = int(inputs.padded_total_samples)
        total_mel_frames = int(padded_total_samples // rt.geometry.hop_samples + 1)
        if eager_inputs is not None:
            final_eager_chunk_mel = _as_cpu_tensor(eager_inputs.chunk_mel)
            final_eager_new_mel = _as_cpu_tensor(eager_inputs.new_mel)
        else:
            final_eager_chunk_mel = final_chunk_mel
            final_eager_new_mel = final_new_mel

    return {
        "final_chunk_mel": final_chunk_mel,
        "final_new_mel": final_new_mel,
        "final_eager_chunk_mel": final_eager_chunk_mel,
        "final_eager_new_mel": final_eager_new_mel,
        "final_drop_extra": final_drop_extra,
        "final_T": final_T,
        "final_remaining_frames": remaining_frames,
        "final_total_mel_frames": total_mel_frames,
        "final_padded_total_samples": padded_total_samples,
    }


def _capture_retained_state(
    session: ContinuousSession,
    recording_continuous_emitted_tokens: list[int],
) -> dict[str, Any]:
    h, c = _decoder_state_hc(session.decoder_state)
    ring = (
        torch.empty((0,), dtype=torch.float32)
        if session.mel_frame_ring is None
        else _as_cpu_tensor(session.mel_frame_ring)
    )
    return {
        "retained_clc": _as_cpu_tensor(session.cache_last_channel),
        "retained_clt": _as_cpu_tensor(session.cache_last_time),
        "retained_clcl": _as_cpu_tensor(session.cache_last_channel_len),
        "retained_g": _as_cpu_tensor(session.pred_out_stream),
        "retained_h": _as_cpu_tensor(h),
        "retained_c": _as_cpu_tensor(c),
        "retained_ring": ring,
        "retained_ring_defined": session.mel_frame_ring is not None,
        "retained_emitted": int(session.emitted_frames),
        "retained_hyp_tokens": torch.tensor(session.hyp_tokens, dtype=torch.int64),
        "retained_collector_tokens": torch.tensor(
            recording_continuous_emitted_tokens,
            dtype=torch.int64,
        ),
        "retained_collector_text": session.continuous_emitted_text,
        "retained_pending_audio": _audio_tensor(session.pending_audio),
        "retained_raw_audio_ring": _audio_tensor(session.raw_audio_ring),
        "retained_total_audio_samples": int(session.total_audio_samples),
        "retained_synthetic_prefix_samples": int(session.synthetic_prefix_samples),
    }


class RecordingContinuousFinalizeRef(ContinuousFinalizeRef):
    """Reference runtime with non-invasive steady chunk capture."""

    def __init__(
        self,
        model,
        *,
        artifacts_dir: str = ART,
        use_preproc_ts: bool = False,
        warm_encoder: bool = False,
    ):
        super().__init__(model)
        self.use_preproc_ts = bool(use_preproc_ts)
        self._force_eager_preproc = False
        self.enc_first = torch.jit.load(os.path.join(artifacts_dir, "enc_first.ts")).to(self.device)
        self.enc_first.eval()
        self.enc_steady_aoti = torch._inductor.aoti_load_package(
            os.path.join(artifacts_dir, "enc_steady_aoti.pt2")
        )
        self.preproc_ts = None
        if self.use_preproc_ts:
            self.preproc_ts = torch.jit.load(os.path.join(artifacts_dir, "preproc.ts")).to(self.device).eval()
        if warm_encoder:
            self._warm_stream_encoder_artifacts()

    @torch.inference_mode()
    def _warm_stream_encoder_artifacts(self) -> None:
        warm = self.new_session("export-encoder-warmup")
        g = self.geometry
        first_mel = torch.zeros((1, 128, int(g.shift_frames)), device=self.device)
        first_len = torch.tensor([int(g.shift_frames)], device=self.device)
        first_out = self.enc_first(
            first_mel.contiguous(),
            first_len.contiguous(),
            warm.cache_last_channel.contiguous(),
            warm.cache_last_time.contiguous(),
            warm.cache_last_channel_len.contiguous(),
        )
        steady_mel = torch.zeros(
            (1, 128, int(g.pre_encode_cache_size + g.shift_frames)),
            device=self.device,
        )
        steady_len = torch.tensor([int(g.pre_encode_cache_size + g.shift_frames)], device=self.device)
        _ = self.enc_steady_aoti(
            steady_mel.contiguous(),
            steady_len.contiguous(),
            first_out[2].contiguous(),
            first_out[3].contiguous(),
            first_out[4].contiguous(),
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _eager_fixed_mel_from_new_audio(
        self,
        raw_audio_ring: np.ndarray,
        new_audio: np.ndarray,
    ) -> torch.Tensor:
        return super()._fixed_mel_from_new_audio(raw_audio_ring, new_audio)

    def _ts_fixed_mel_from_new_audio(
        self,
        raw_audio_ring: np.ndarray,
        new_audio: np.ndarray,
    ) -> torch.Tensor:
        if self.preproc_ts is None:
            raise RuntimeError("preproc.ts requested but not loaded")
        fixed_audio, valid_samples = self._build_fixed_preprocess_audio(
            raw_audio_ring,
            new_audio,
        )
        audio = torch.from_numpy(np.ascontiguousarray(fixed_audio)).unsqueeze(0).to(self.device)
        audio_len = torch.tensor([valid_samples], device=self.device, dtype=torch.long)
        mel, _mel_len = self.preproc_ts(audio, audio_len)
        return mel

    def _fixed_mel_from_new_audio(
        self,
        raw_audio_ring: np.ndarray,
        new_audio: np.ndarray,
    ) -> torch.Tensor:
        if self.use_preproc_ts and not self._force_eager_preproc:
            return self._ts_fixed_mel_from_new_audio(raw_audio_ring, new_audio)
        return self._eager_fixed_mel_from_new_audio(raw_audio_ring, new_audio)

    def prepare_finalize_inputs_eager(self, fork: ContinuousSession):
        old = self._force_eager_preproc
        self._force_eager_preproc = True
        try:
            return super().prepare_finalize_inputs(fork)
        finally:
            self._force_eager_preproc = old

    def begin_recording(self) -> None:
        self.recorded_steady_chunks: list[dict[str, Any]] = []
        self.recorded_events: list[dict[str, Any]] = []
        self.recording_continuous_emitted_tokens: list[int] = []

    def _record_event(
        self,
        *,
        kind: int,
        text: str,
        tokens: list[int],
        collector_text: str,
        collector_tokens: list[int],
    ) -> None:
        self.recorded_events.append(
            {
                "kind": int(kind),
                "text": text,
                "tokens": list(tokens),
                "collector_text": collector_text,
                "collector_tokens": list(collector_tokens),
            }
        )

    @torch.inference_mode()
    def _run_aoti_consistent_steady_encoder(
        self,
        session: ContinuousSession,
        chunk_mel: torch.Tensor,
        drop_extra: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk_len = torch.tensor([chunk_mel.shape[-1]], device=self.device)
        if drop_extra == 0:
            out = self.enc_first(
                chunk_mel.contiguous(),
                chunk_len.contiguous(),
                session.cache_last_channel.contiguous(),
                session.cache_last_time.contiguous(),
                session.cache_last_channel_len.contiguous(),
            )
            return tuple(out)  # type: ignore[return-value]
        out = self.enc_steady_aoti(
            chunk_mel.contiguous(),
            chunk_len.contiguous(),
            session.cache_last_channel.contiguous(),
            session.cache_last_time.contiguous(),
            session.cache_last_channel_len.contiguous(),
        )
        return tuple(out)  # type: ignore[return-value]

    @torch.inference_mode()
    def _run_eager_steady_encoder(
        self,
        session: ContinuousSession,
        chunk_mel: torch.Tensor,
        drop_extra: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk_len = torch.tensor([chunk_mel.shape[-1]], device=self.device)
        return self.encoder.cache_aware_stream_step(
            processed_signal=chunk_mel.contiguous(),
            processed_signal_length=chunk_len.contiguous(),
            cache_last_channel=session.cache_last_channel.contiguous(),
            cache_last_time=session.cache_last_time.contiguous(),
            cache_last_channel_len=session.cache_last_channel_len.contiguous(),
            keep_all_outputs=False,
            drop_extra_pre_encoded=drop_extra,
        )

    @torch.inference_mode()
    def _process_one_steady_chunk(self, session: ContinuousSession) -> None:
        g = self.geometry
        new_audio = session.pending_audio[: g.preprocess_new_audio_samples]
        mel = self._fixed_mel_from_new_audio(session.raw_audio_ring, new_audio)
        valid_new_mel = mel[
            :,
            :,
            g.first_preprocess_mel_frame : g.first_preprocess_mel_frame
            + g.shift_frames,
        ]
        valid_eager_new_mel = None
        if self.use_preproc_ts:
            eager_mel = self._eager_fixed_mel_from_new_audio(session.raw_audio_ring, new_audio)
            valid_eager_new_mel = eager_mel[
                :,
                :,
                g.first_preprocess_mel_frame : g.first_preprocess_mel_frame
                + g.shift_frames,
            ]

        is_first = session.emitted_frames == 0
        chunk_T = (
            int(valid_new_mel.shape[-1])
            if is_first
            else int(session.mel_frame_ring.shape[-1] + valid_new_mel.shape[-1])
        )
        self.recorded_steady_chunks.append(
            {
                "new_mel": _as_cpu_tensor(valid_new_mel),
                "is_first": is_first,
                "drop_extra": 0 if is_first else int(g.drop_extra),
                "chunk_T": chunk_T,
                "emitted_before": int(session.emitted_frames),
            }
        )
        if valid_eager_new_mel is not None:
            self.recorded_steady_chunks[-1]["eager_new_mel"] = _as_cpu_tensor(valid_eager_new_mel)

        old_text = session.current_text
        if is_first:
            chunk_mel = valid_new_mel
            drop_extra = 0
        else:
            chunk_mel = torch.cat((session.mel_frame_ring, valid_new_mel), dim=-1)
            drop_extra = g.drop_extra

        if is_first:
            eager_out = self._run_eager_steady_encoder(session, chunk_mel, int(drop_extra))
            self.recorded_steady_chunks[-1]["first_eager_enc_out"] = _as_cpu_tensor(eager_out[0])
            self.recorded_steady_chunks[-1]["first_eager_enc_len"] = _as_cpu_tensor(eager_out[1])

        enc_out, enc_len, clc, clt, clcl = self._run_aoti_consistent_steady_encoder(
            session,
            chunk_mel,
            drop_extra,
        )
        tokens, decoder_state, pred_out = ref_greedy_range(
            self.decoder,
            self.joint,
            enc_out.transpose(1, 2).contiguous(),
            0,
            int(enc_len[0]),
            session.decoder_state,
            session.pred_out_stream,
        )
        session.hyp_tokens.extend(tokens)
        session.decoder_state = decoder_state
        session.pred_out_stream = pred_out
        session.cache_last_channel = clc
        session.cache_last_time = clt
        session.cache_last_channel_len = clcl

        consumed_audio = session.pending_audio[: g.shift_frames * g.hop_samples]
        session.raw_audio_ring = self._advance_raw_ring(
            session.raw_audio_ring,
            consumed_audio,
        )
        session.pending_audio = session.pending_audio[g.shift_frames * g.hop_samples :]
        self._update_mel_frame_ring(session, valid_new_mel)
        session.emitted_frames += g.shift_frames
        session.current_text = self.tokenizer.ids_to_text(session.hyp_tokens)

        text_changed = session.current_text != old_text
        if text_changed:
            self._record_event(
                kind=EVENT_INTERIM,
                text=session.current_text,
                tokens=list(session.hyp_tokens),
                collector_text=session.continuous_emitted_text,
                collector_tokens=list(self.recording_continuous_emitted_tokens),
            )

    def _finalize_and_emit(
        self,
        session: ContinuousSession,
        *,
        reason: str,
    ):
        emitted_before = list(self.recording_continuous_emitted_tokens)
        emitted_text_before = session.continuous_emitted_text
        result = super()._finalize_and_emit(session, reason=reason)
        token_delta = _append_only_delta_tokens(result.final_tokens, emitted_before)
        text_delta = _continuous_append_only_delta(result.final_text, emitted_text_before)
        if text_delta != result.delta_text:
            raise AssertionError(
                "finalize_ref delta oracle mismatch during export: "
                f"computed={text_delta!r} result={result.delta_text!r}"
            )
        if result.delta_text:
            collector_tokens = emitted_before + token_delta
            kind = EVENT_FINAL
            text = result.delta_text
        else:
            collector_tokens = emitted_before
            kind = EVENT_SUPPRESSED
            text = ""
        self._record_event(
            kind=kind,
            text=text,
            tokens=token_delta,
            collector_text=session.continuous_emitted_text,
            collector_tokens=collector_tokens,
        )
        self.recording_continuous_emitted_tokens = list(collector_tokens)
        return result


def _build_row(
    rt: RecordingContinuousFinalizeRef,
    wav,
    *,
    sample_index: int,
    sample_id: str,
    reference_text: str,
) -> dict[str, Any]:
    session = rt.new_session(f"session-bundle-{sample_index}")
    rt.begin_recording()
    rt.append_audio(session, wav)
    steady_chunks = list(rt.recorded_steady_chunks)

    rt.vad_stop(session)
    finalize_payload = _capture_finalize_payload(rt, session, steady_chunks)

    result = rt.debounce_expire(session)
    if not result.fork_assert_passed:
        raise AssertionError(f"finalize_ref FORK_ASSERT failed for sample {sample_index}")
    events = list(rt.recorded_events)
    if not events:
        raise AssertionError(f"no emitted events recorded for sample {sample_index}")
    if events[-1]["kind"] not in (EVENT_FINAL, EVENT_SUPPRESSED):
        raise AssertionError(
            f"sample {sample_index} event stream did not end in final/suppressed"
        )

    steady_tokens = torch.tensor(result.steady_tokens, dtype=torch.int64)
    gold_tokens = torch.tensor(result.final_tokens, dtype=torch.int64)
    finalize_new_tokens = torch.tensor(
        result.final_tokens[len(result.steady_tokens) :],
        dtype=torch.int64,
    )

    return {
        "sample_index": int(sample_index),
        "sample_id": str(sample_id),
        "reference_text": str(reference_text),
        "finalize_ref_text": result.final_text,
        "audio_samples": int(len(wav)),
        "audio": _audio_tensor(wav),
        "steady_chunks": steady_chunks,
        "steady_tokens": steady_tokens,
        "gold_tokens": gold_tokens,
        "finalize_new_tokens": finalize_new_tokens,
        "events": events,
        **finalize_payload,
        "finalize_ref_meta": dict(result.meta),
    }


def _build_multiturn_finalize_record(
    rt: RecordingContinuousFinalizeRef,
    session: ContinuousSession,
    *,
    stream_index: int,
    turn_index: int,
    sample_index: int,
    audio: Any,
    steady_chunks: list[dict[str, Any]],
    event_start: int,
) -> dict[str, Any]:
    collector_before = session.continuous_emitted_text
    rt.vad_stop(session)
    finalize_payload = _capture_finalize_payload(rt, session, steady_chunks)
    result = rt.debounce_expire(session)
    if not result.fork_assert_passed:
        raise AssertionError(
            f"finalize_ref FORK_ASSERT failed for stream {stream_index} turn {turn_index}"
        )
    events = list(rt.recorded_events[event_start:])
    if not events or events[-1]["kind"] not in (EVENT_FINAL, EVENT_SUPPRESSED):
        raise AssertionError(
            f"stream {stream_index} turn {turn_index} event stream did not end in final/suppressed"
        )

    return {
        "stream_index": int(stream_index),
        "turn_index": int(turn_index),
        "sample_index": int(sample_index),
        "audio_samples": int(len(audio)),
        "audio": _audio_tensor(audio),
        "collector_text_before_finalize": collector_before,
        "steady_chunks": steady_chunks,
        "steady_tokens": torch.tensor(result.steady_tokens, dtype=torch.int64),
        "gold_tokens": torch.tensor(result.final_tokens, dtype=torch.int64),
        "finalize_new_tokens": torch.tensor(
            result.final_tokens[len(result.steady_tokens) :],
            dtype=torch.int64,
        ),
        "events": events,
        **finalize_payload,
        **_capture_retained_state(session, rt.recording_continuous_emitted_tokens),
        "finalize_ref_meta": dict(result.meta),
    }


def _build_multiturn_stream(
    rt: RecordingContinuousFinalizeRef,
    turn_a_wav,
    turn_b_wav,
    *,
    stream_index: int,
    sample_a_index: int,
    sample_b_index: int,
) -> dict[str, Any]:
    session = rt.new_session(f"session-multiturn-{stream_index}")
    rt.begin_recording()

    turns: list[dict[str, Any]] = []
    for turn_index, (sample_index, wav) in enumerate(
        ((sample_a_index, turn_a_wav), (sample_b_index, turn_b_wav))
    ):
        chunk_start = len(rt.recorded_steady_chunks)
        event_start = len(rt.recorded_events)
        rt.append_audio(session, wav)
        steady_chunks = list(rt.recorded_steady_chunks[chunk_start:])
        if not steady_chunks:
            raise AssertionError(
                f"stream {stream_index} turn {turn_index} produced no steady chunks"
            )
        turns.append(
            _build_multiturn_finalize_record(
                rt,
                session,
                stream_index=stream_index,
                turn_index=turn_index,
                sample_index=sample_index,
                audio=wav,
                steady_chunks=steady_chunks,
                event_start=event_start,
            )
        )

    end_event_start = len(rt.recorded_events)
    end_collector_before = session.continuous_emitted_text
    end_payload = _capture_finalize_payload(
        rt,
        session,
        list(rt.recorded_steady_chunks),
    )
    end_result = rt.force_finalize_end(session)
    if not end_result.fork_assert_passed:
        raise AssertionError(f"finalize_ref FORK_ASSERT failed for stream {stream_index} true boundary")
    end_events = list(rt.recorded_events[end_event_start:])
    if not end_events or end_events[-1]["kind"] not in (EVENT_FINAL, EVENT_SUPPRESSED):
        raise AssertionError(
            f"stream {stream_index} true-boundary event stream did not end in final/suppressed"
        )

    end = {
        "collector_text_before_finalize": end_collector_before,
        "steady_tokens": torch.tensor(end_result.steady_tokens, dtype=torch.int64),
        "gold_tokens": torch.tensor(end_result.final_tokens, dtype=torch.int64),
        "finalize_new_tokens": torch.tensor(
            end_result.final_tokens[len(end_result.steady_tokens) :],
            dtype=torch.int64,
        ),
        "events": end_events,
        **end_payload,
        "post_reset_emitted": int(session.emitted_frames),
        "post_reset_hyp_tokens": torch.tensor(session.hyp_tokens, dtype=torch.int64),
        "post_reset_collector_text": session.continuous_emitted_text,
        "post_reset_collector_tokens": torch.tensor([], dtype=torch.int64),
        "post_reset_pending_audio": _audio_tensor(session.pending_audio),
        "post_reset_raw_audio_ring": _audio_tensor(session.raw_audio_ring),
        "post_reset_total_audio_samples": int(session.total_audio_samples),
        "post_reset_synthetic_prefix_samples": int(session.synthetic_prefix_samples),
        "finalize_ref_meta": dict(end_result.meta),
    }

    return {
        "stream_index": int(stream_index),
        "sample_indices": [int(sample_a_index), int(sample_b_index)],
        "turns": turns,
        "end": end,
    }


class SessionBundle(torch.nn.Module):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        init_session: ContinuousSession,
        geometry,
        tokenizer_pieces: list[str],
        detok_sequences: list[list[int]],
        detok_texts: list[str],
        *,
        audio: bool = False,
        audio_ci: dict[str, Any] | None = None,
    ):
        super().__init__()
        init_h, init_c = _decoder_state_hc(init_session.decoder_state)
        self.register_buffer("num_utts", torch.tensor([len(rows)], dtype=torch.int64))
        self.register_buffer(
            "meta",
            torch.tensor(
                [
                    len(rows),
                    BLANK,
                    MAX_SYMBOLS,
                    int(geometry.shift_frames),
                    int(geometry.pre_encode_cache_size),
                    int(geometry.drop_extra),
                    int(geometry.final_padding_frames),
                    RIGHT_CONTEXT,
                    int(geometry.first_preprocess_mel_frame),
                    int(geometry.hop_samples),
                ],
                dtype=torch.int64,
            ),
        )
        _register_audio_geometry(self, geometry, audio=audio)
        _register_audio_ci(self, audio_ci, audio=audio)
        self.register_buffer("init_clc", _as_cpu_tensor(init_session.cache_last_channel))
        self.register_buffer("init_clt", _as_cpu_tensor(init_session.cache_last_time))
        self.register_buffer("init_clcl", _as_cpu_tensor(init_session.cache_last_channel_len))
        self.register_buffer("init_g", _as_cpu_tensor(init_session.pred_out_stream))
        self.register_buffer("init_h", _as_cpu_tensor(init_h))
        self.register_buffer("init_c", _as_cpu_tensor(init_c))
        piece_bytes, piece_offsets = _pack_utf8(tokenizer_pieces)
        self.register_buffer("token_piece_bytes", piece_bytes)
        self.register_buffer("token_piece_offsets", piece_offsets)
        detok_tokens, detok_token_offsets = _pack_i64_lists(detok_sequences)
        detok_text_bytes, detok_text_offsets = _pack_utf8(detok_texts)
        self.register_buffer("detok_selftest_tokens", detok_tokens)
        self.register_buffer("detok_selftest_token_offsets", detok_token_offsets)
        self.register_buffer("detok_selftest_text_bytes", detok_text_bytes)
        self.register_buffer("detok_selftest_text_offsets", detok_text_offsets)

        for i, row in enumerate(rows):
            prefix = f"utt{i}"
            steady_chunks = row["steady_chunks"]
            self.register_buffer(f"{prefix}_sample_index", _scalar(row["sample_index"]))
            sample_id_bytes, sample_id_offsets = _pack_utf8([row["sample_id"]])
            reference_text_bytes, reference_text_offsets = _pack_utf8(
                [row["reference_text"]]
            )
            finalize_ref_text_bytes, finalize_ref_text_offsets = _pack_utf8(
                [row["finalize_ref_text"]]
            )
            self.register_buffer(f"{prefix}_sample_id_bytes", sample_id_bytes)
            self.register_buffer(f"{prefix}_sample_id_offsets", sample_id_offsets)
            self.register_buffer(
                f"{prefix}_reference_text_bytes",
                reference_text_bytes,
            )
            self.register_buffer(
                f"{prefix}_reference_text_offsets",
                reference_text_offsets,
            )
            self.register_buffer(
                f"{prefix}_finalize_ref_text_bytes",
                finalize_ref_text_bytes,
            )
            self.register_buffer(
                f"{prefix}_finalize_ref_text_offsets",
                finalize_ref_text_offsets,
            )
            self.register_buffer(f"{prefix}_audio_samples", _scalar(row["audio_samples"]))
            if audio:
                self.register_buffer(f"{prefix}_audio", row["audio"].cpu())
            self.register_buffer(f"{prefix}_num_steady", _scalar(len(steady_chunks)))
            self.register_buffer(f"{prefix}_steady_tokens", row["steady_tokens"].cpu().to(torch.int64))
            self.register_buffer(f"{prefix}_gold_tokens", row["gold_tokens"].cpu().to(torch.int64))
            self.register_buffer(
                f"{prefix}_finalize_new_tokens",
                row["finalize_new_tokens"].cpu().to(torch.int64),
            )
            _register_event_buffers(self, prefix, row["events"])
            self.register_buffer(f"{prefix}_final_chunk_mel", row["final_chunk_mel"].cpu())
            self.register_buffer(f"{prefix}_final_new_mel", row["final_new_mel"].cpu())
            self.register_buffer(f"{prefix}_final_drop_extra", _scalar(row["final_drop_extra"]))
            self.register_buffer(f"{prefix}_final_T", _scalar(row["final_T"]))
            self.register_buffer(
                f"{prefix}_final_remaining_frames",
                _scalar(row["final_remaining_frames"]),
            )
            self.register_buffer(
                f"{prefix}_final_total_mel_frames",
                _scalar(row["final_total_mel_frames"]),
            )
            self.register_buffer(
                f"{prefix}_final_padded_total_samples",
                _scalar(row["final_padded_total_samples"]),
            )

            _register_steady_chunks(self, prefix, steady_chunks)

    def forward(self):
        return self.num_utts


class MultiTurnSessionBundle(torch.nn.Module):
    def __init__(
        self,
        streams: list[dict[str, Any]],
        init_session: ContinuousSession,
        geometry,
        tokenizer_pieces: list[str],
        detok_sequences: list[list[int]],
        detok_texts: list[str],
        *,
        audio: bool = False,
        audio_ci: dict[str, Any] | None = None,
    ):
        super().__init__()
        init_h, init_c = _decoder_state_hc(init_session.decoder_state)
        self.register_buffer("num_streams", torch.tensor([len(streams)], dtype=torch.int64))
        self.register_buffer(
            "meta",
            torch.tensor(
                [
                    len(streams),
                    BLANK,
                    MAX_SYMBOLS,
                    int(geometry.shift_frames),
                    int(geometry.pre_encode_cache_size),
                    int(geometry.drop_extra),
                    int(geometry.final_padding_frames),
                    RIGHT_CONTEXT,
                    int(geometry.first_preprocess_mel_frame),
                    int(geometry.hop_samples),
                    2,
                ],
                dtype=torch.int64,
            ),
        )
        _register_audio_geometry(self, geometry, audio=audio)
        _register_audio_ci(self, audio_ci, audio=audio)
        self.register_buffer("init_clc", _as_cpu_tensor(init_session.cache_last_channel))
        self.register_buffer("init_clt", _as_cpu_tensor(init_session.cache_last_time))
        self.register_buffer("init_clcl", _as_cpu_tensor(init_session.cache_last_channel_len))
        self.register_buffer("init_g", _as_cpu_tensor(init_session.pred_out_stream))
        self.register_buffer("init_h", _as_cpu_tensor(init_h))
        self.register_buffer("init_c", _as_cpu_tensor(init_c))
        piece_bytes, piece_offsets = _pack_utf8(tokenizer_pieces)
        self.register_buffer("token_piece_bytes", piece_bytes)
        self.register_buffer("token_piece_offsets", piece_offsets)
        detok_tokens, detok_token_offsets = _pack_i64_lists(detok_sequences)
        detok_text_bytes, detok_text_offsets = _pack_utf8(detok_texts)
        self.register_buffer("detok_selftest_tokens", detok_tokens)
        self.register_buffer("detok_selftest_token_offsets", detok_token_offsets)
        self.register_buffer("detok_selftest_text_bytes", detok_text_bytes)
        self.register_buffer("detok_selftest_text_offsets", detok_text_offsets)

        for stream_i, stream in enumerate(streams):
            sprefix = f"stream{stream_i}"
            self.register_buffer(
                f"{sprefix}_sample_indices",
                torch.tensor(stream["sample_indices"], dtype=torch.int64),
            )
            self.register_buffer(f"{sprefix}_num_turns", torch.tensor([2], dtype=torch.int64))
            for turn_i, turn in enumerate(stream["turns"]):
                prefix = f"{sprefix}_turn{turn_i}"
                steady_chunks = turn["steady_chunks"]
                self.register_buffer(f"{prefix}_sample_index", _scalar(turn["sample_index"]))
                self.register_buffer(f"{prefix}_audio_samples", _scalar(turn["audio_samples"]))
                if audio:
                    self.register_buffer(f"{prefix}_audio", turn["audio"].cpu())
                self.register_buffer(f"{prefix}_num_steady", _scalar(len(steady_chunks)))
                self.register_buffer(f"{prefix}_steady_tokens", turn["steady_tokens"].cpu().to(torch.int64))
                self.register_buffer(f"{prefix}_gold_tokens", turn["gold_tokens"].cpu().to(torch.int64))
                self.register_buffer(
                    f"{prefix}_finalize_new_tokens",
                    turn["finalize_new_tokens"].cpu().to(torch.int64),
                )
                collector_before_bytes, collector_before_offsets = _pack_utf8(
                    [turn["collector_text_before_finalize"]]
                )
                self.register_buffer(
                    f"{prefix}_collector_before_text_bytes",
                    collector_before_bytes,
                )
                self.register_buffer(
                    f"{prefix}_collector_before_text_offsets",
                    collector_before_offsets,
                )
                _register_event_buffers(self, prefix, turn["events"])
                self.register_buffer(f"{prefix}_final_chunk_mel", turn["final_chunk_mel"].cpu())
                self.register_buffer(f"{prefix}_final_new_mel", turn["final_new_mel"].cpu())
                self.register_buffer(f"{prefix}_final_drop_extra", _scalar(turn["final_drop_extra"]))
                self.register_buffer(f"{prefix}_final_T", _scalar(turn["final_T"]))
                self.register_buffer(
                    f"{prefix}_final_remaining_frames",
                    _scalar(turn["final_remaining_frames"]),
                )
                self.register_buffer(
                    f"{prefix}_final_total_mel_frames",
                    _scalar(turn["final_total_mel_frames"]),
                )
                self.register_buffer(
                    f"{prefix}_final_padded_total_samples",
                    _scalar(turn["final_padded_total_samples"]),
                )
                for name in (
                    "retained_clc",
                    "retained_clt",
                    "retained_clcl",
                    "retained_g",
                    "retained_h",
                    "retained_c",
                    "retained_ring",
                ):
                    self.register_buffer(f"{prefix}_{name}", turn[name].cpu())
                if audio:
                    self.register_buffer(
                        f"{prefix}_retained_pending_audio",
                        turn["retained_pending_audio"].cpu(),
                    )
                    self.register_buffer(
                        f"{prefix}_retained_raw_audio_ring",
                        turn["retained_raw_audio_ring"].cpu(),
                    )
                    self.register_buffer(
                        f"{prefix}_retained_total_audio_samples",
                        _scalar(turn["retained_total_audio_samples"]),
                    )
                    self.register_buffer(
                        f"{prefix}_retained_synthetic_prefix_samples",
                        _scalar(turn["retained_synthetic_prefix_samples"]),
                    )
                self.register_buffer(
                    f"{prefix}_retained_ring_defined",
                    _scalar(turn["retained_ring_defined"]),
                )
                self.register_buffer(
                    f"{prefix}_retained_emitted",
                    _scalar(turn["retained_emitted"]),
                )
                self.register_buffer(
                    f"{prefix}_retained_hyp_tokens",
                    turn["retained_hyp_tokens"].cpu().to(torch.int64),
                )
                self.register_buffer(
                    f"{prefix}_retained_collector_tokens",
                    turn["retained_collector_tokens"].cpu().to(torch.int64),
                )
                retained_text_bytes, retained_text_offsets = _pack_utf8(
                    [turn["retained_collector_text"]]
                )
                self.register_buffer(
                    f"{prefix}_retained_collector_text_bytes",
                    retained_text_bytes,
                )
                self.register_buffer(
                    f"{prefix}_retained_collector_text_offsets",
                    retained_text_offsets,
                )
                _register_steady_chunks(self, prefix, steady_chunks)

            end = stream["end"]
            eprefix = f"{sprefix}_end"
            self.register_buffer(f"{eprefix}_steady_tokens", end["steady_tokens"].cpu().to(torch.int64))
            self.register_buffer(f"{eprefix}_gold_tokens", end["gold_tokens"].cpu().to(torch.int64))
            self.register_buffer(
                f"{eprefix}_finalize_new_tokens",
                end["finalize_new_tokens"].cpu().to(torch.int64),
            )
            end_collector_before_bytes, end_collector_before_offsets = _pack_utf8(
                [end["collector_text_before_finalize"]]
            )
            self.register_buffer(
                f"{eprefix}_collector_before_text_bytes",
                end_collector_before_bytes,
            )
            self.register_buffer(
                f"{eprefix}_collector_before_text_offsets",
                end_collector_before_offsets,
            )
            _register_event_buffers(self, eprefix, end["events"])
            self.register_buffer(f"{eprefix}_final_chunk_mel", end["final_chunk_mel"].cpu())
            self.register_buffer(f"{eprefix}_final_new_mel", end["final_new_mel"].cpu())
            self.register_buffer(f"{eprefix}_final_drop_extra", _scalar(end["final_drop_extra"]))
            self.register_buffer(f"{eprefix}_final_T", _scalar(end["final_T"]))
            self.register_buffer(
                f"{eprefix}_final_remaining_frames",
                _scalar(end["final_remaining_frames"]),
            )
            self.register_buffer(
                f"{eprefix}_final_total_mel_frames",
                _scalar(end["final_total_mel_frames"]),
            )
            self.register_buffer(
                f"{eprefix}_final_padded_total_samples",
                _scalar(end["final_padded_total_samples"]),
            )
            self.register_buffer(f"{eprefix}_post_reset_emitted", _scalar(end["post_reset_emitted"]))
            self.register_buffer(
                f"{eprefix}_post_reset_hyp_tokens",
                end["post_reset_hyp_tokens"].cpu().to(torch.int64),
            )
            post_reset_text_bytes, post_reset_text_offsets = _pack_utf8(
                [end["post_reset_collector_text"]]
            )
            self.register_buffer(
                f"{eprefix}_post_reset_collector_text_bytes",
                post_reset_text_bytes,
            )
            self.register_buffer(
                f"{eprefix}_post_reset_collector_text_offsets",
                post_reset_text_offsets,
            )
            self.register_buffer(
                f"{eprefix}_post_reset_collector_tokens",
                end["post_reset_collector_tokens"].cpu().to(torch.int64),
            )
            if audio:
                self.register_buffer(
                    f"{eprefix}_post_reset_pending_audio",
                    end["post_reset_pending_audio"].cpu(),
                )
                self.register_buffer(
                    f"{eprefix}_post_reset_raw_audio_ring",
                    end["post_reset_raw_audio_ring"].cpu(),
                )
                self.register_buffer(
                    f"{eprefix}_post_reset_total_audio_samples",
                    _scalar(end["post_reset_total_audio_samples"]),
                )
                self.register_buffer(
                    f"{eprefix}_post_reset_synthetic_prefix_samples",
                    _scalar(end["post_reset_synthetic_prefix_samples"]),
                )

    def forward(self):
        return self.num_streams


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="number of stt-benchmark utterances")
    parser.add_argument("--start", type=int, default=0, help="dataset start index")
    parser.add_argument("--out", default=os.path.join(ART, "session_bundle.ts"))
    parser.add_argument(
        "--multiturn",
        action="store_true",
        help="export retained-context two-turn streams plus a true-boundary reset",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="export PCM-fed bundle; C++ computes mels and uses bundle mels/geometry as gold",
    )
    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError("--n must be positive")
    if args.out == os.path.join(ART, "session_bundle.ts"):
        if args.multiturn and args.audio:
            args.out = os.path.join(ART, "session_multiturn_audio_bundle.ts")
        elif args.multiturn:
            args.out = os.path.join(ART, "session_multiturn_bundle.ts")
        elif args.audio:
            args.out = os.path.join(ART, "session_audio_bundle.ts")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model = load_model()
    artifacts_dir = os.path.dirname(args.out) or ART
    if args.audio:
        _ensure_preproc_ts(model, artifacts_dir)
        print(
            "audio bundle oracle: C++ runtime vs the shipped preproc.ts artifact; "
            "a separate eager-preproc token/event subset gate runs during export"
        )
    rt = RecordingContinuousFinalizeRef(
        model,
        artifacts_dir=artifacts_dir,
        use_preproc_ts=args.audio,
        warm_encoder=args.multiturn,
    )
    dataset = load_benchmark_dataset()
    end_index = args.start + (args.n * 2 if args.multiturn else args.n)
    if args.start < 0 or args.start >= len(dataset) or end_index > len(dataset):
        raise ValueError(
            f"requested range start={args.start} n={args.n} "
            f"multiturn={args.multiturn} exceeds dataset length {len(dataset)}"
        )

    if args.multiturn:
        streams: list[dict[str, Any]] = []
        for stream_index in range(args.n):
            sample_a_index = args.start + 2 * stream_index
            sample_b_index = sample_a_index + 1
            turn_a = load_wav(dataset[sample_a_index])
            turn_b = load_wav(dataset[sample_b_index])
            stream = _build_multiturn_stream(
                rt,
                turn_a,
                turn_b,
                stream_index=stream_index,
                sample_a_index=sample_a_index,
                sample_b_index=sample_b_index,
            )
            streams.append(stream)
            turn_summaries = []
            for turn in stream["turns"]:
                nonempty_before = bool(turn["collector_text_before_finalize"])
                final_event = turn["events"][-1]
                turn_summaries.append(
                    "turn{turn} sample={sample} steady_chunks={chunks} "
                    "steady_tok={steady_tok} gold_tok={gold_tok} events={events} "
                    "collector_before={collector_before} final_kind={kind} "
                    "retained_emitted={emitted} final_drop={drop} final_T={T}".format(
                        turn=turn["turn_index"],
                        sample=turn["sample_index"],
                        chunks=len(turn["steady_chunks"]),
                        steady_tok=turn["steady_tokens"].numel(),
                        gold_tok=turn["gold_tokens"].numel(),
                        events=len(turn["events"]),
                        collector_before="nonempty" if nonempty_before else "empty",
                        kind=(
                            "final"
                            if final_event["kind"] == EVENT_FINAL
                            else "suppressed"
                        ),
                        emitted=turn["retained_emitted"],
                        drop=turn["final_drop_extra"],
                        T=turn["final_T"],
                    )
                )
            end = stream["end"]
            print(
                f"stream{stream_index} samples={stream['sample_indices']} "
                + " | ".join(turn_summaries)
                + f" | true_boundary events={len(end['events'])} "
                f"kind={'final' if end['events'][-1]['kind'] == EVENT_FINAL else 'suppressed'} "
                f"post_reset_emitted={end['post_reset_emitted']} "
                f"final_drop={end['final_drop_extra']} final_T={end['final_T']}"
            )

        init_session = rt.new_session("session-multiturn-init")
        detok_rows: list[dict[str, Any]] = []
        for stream in streams:
            detok_rows.extend(stream["turns"])
            detok_rows.append(stream["end"])
        detok_sequences, detok_texts = _build_detok_selftest(detok_rows, rt.tokenizer)
        audio_ci = None
        if args.audio:
            audio_ci = _measure_audio_ci(
                rt,
                artifacts_dir=artifacts_dir,
                streams=streams,
            )
            audio_ci["eager_token_event_subset"] = _run_eager_audio_subset_gate(
                model,
                artifacts_dir=artifacts_dir,
                dataset=dataset,
                start=args.start,
                n=args.n,
                multiturn=True,
                shipped_streams=streams,
            )
            _write_audio_ci_sidecar(args.out, audio_ci)
        bundle = torch.jit.script(
            MultiTurnSessionBundle(
                streams,
                init_session,
                rt.geometry,
                _tokenizer_pieces(rt.tokenizer),
                detok_sequences,
                detok_texts,
                audio=args.audio,
                audio_ci=audio_ci,
            )
        )
        bundle.save(args.out)
        print(
            f"wrote {args.out} ({len(streams)} multi-turn streams, "
            f"detok_selftests={len(detok_sequences)}, audio={args.audio})"
        )
        print(
            "schema: meta, init_*; stream{i}_{sample_indices,num_turns}; "
            "stream{i}_turn{j}_{num_steady,steady_tokens,gold_tokens,event_*,"
            "final_chunk_mel,final_drop_extra,final_T,retained_{clc,clt,clcl,g,h,c,"
            "ring,emitted,hyp_tokens,collector_tokens,collector_text}}; "
            "stream{i}_end_{gold_tokens,event_*,final_chunk_mel,final_drop_extra,"
            "final_T,post_reset_*}; tokenizer token_piece_* and detok_selftest_*"
            + ("; AUDIO audio_geometry + audio CI thresholds/stats + per-turn *_audio + retained/post-reset raw audio state" if args.audio else "")
        )
        return 0

    rows: list[dict[str, Any]] = []
    for offset in range(args.n):
        sample_index = args.start + offset
        example = dataset[sample_index]
        wav = load_wav(example)
        row = _build_row(
            rt,
            wav,
            sample_index=sample_index,
            sample_id=_sample_id_for(example, sample_index),
            reference_text=str(example.get("transcription", "")),
        )
        rows.append(row)
        print(
            f"row{offset} sample={sample_index} id={row['sample_id']} audio={row['audio_samples']} "
            f"steady_chunks={len(row['steady_chunks'])} "
            f"steady_tok={row['steady_tokens'].numel()} gold_tok={row['gold_tokens'].numel()} "
            f"events={len(row['events'])} "
            f"final_drop={row['final_drop_extra']} final_T={row['final_T']}"
        )

    init_session = rt.new_session("session-bundle-init")
    detok_sequences, detok_texts = _build_detok_selftest(rows, rt.tokenizer)
    audio_ci = None
    if args.audio:
        audio_ci = _measure_audio_ci(
            rt,
            artifacts_dir=artifacts_dir,
            rows=rows,
        )
        audio_ci["eager_token_event_subset"] = _run_eager_audio_subset_gate(
            model,
            artifacts_dir=artifacts_dir,
            dataset=dataset,
            start=args.start,
            n=args.n,
            multiturn=False,
            shipped_rows=rows,
        )
        _write_audio_ci_sidecar(args.out, audio_ci)
    bundle = torch.jit.script(
        SessionBundle(
            rows,
            init_session,
            rt.geometry,
            _tokenizer_pieces(rt.tokenizer),
            detok_sequences,
            detok_texts,
            audio=args.audio,
            audio_ci=audio_ci,
        )
    )
    bundle.save(args.out)
    print(
        f"wrote {args.out} ({len(rows)} utterances, "
        f"detok_selftests={len(detok_sequences)}, audio={args.audio})"
    )
    print(
        "schema: meta, init_*; utt{i}_{sample_id,reference_text,finalize_ref_text,"
        "num_steady,steady_tokens,gold_tokens,"
        "event_kinds,event_tokens,event_token_offsets,event_collector_tokens,"
        "event_collector_token_offsets,event_text_bytes,event_text_offsets,"
        "event_collector_text_bytes,event_collector_text_offsets,"
        "final_chunk_mel,final_drop_extra,final_T}; tokenizer token_piece_* "
        "and detok_selftest_*; utt{i}_chunk{j}_*"
        + ("; AUDIO audio_geometry + audio CI thresholds/stats + utt{i}_audio + final geometry gold" if args.audio else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
