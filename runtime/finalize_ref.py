#!/usr/bin/env python3
"""1.3 finalize reference: single-stream continuous FINALIZE executable spec.

This intentionally mirrors the production continuous-finalize path in
src/nemotron_speech/server.py, but keeps it synchronous and single-stream:

* STREAMING / PENDING_FINALIZE / FINALIZED state transitions.
* normal streaming uses keep_all_outputs=False.
* finalize forks the live state, appends right-context silence, runs one
  keep_all_outputs=True remainder encoder pass, and continues greedy RNNT decode
  with the carried decoder state.
* FORK_ASSERT proves the parent cache/decoder/hyp state is byte-identical after
  flushing the fork.

Run:
  HF_HUB_OFFLINE=1 ./.venv/bin/python finalize_ref.py
"""
from __future__ import annotations

import copy
import dataclasses
import io
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

import nemo.collections.asr as nemo_asr
from model_profile import apply_prompt, get_profile, load_profile_model
from ref_decode import ref_greedy, ref_greedy_range


STREAMING = "STREAMING"
PENDING_FINALIZE = "PENDING_FINALIZE"
FINALIZED = "FINALIZED"

PROFILE = get_profile()
BLANK = PROFILE.blank
MAX_SYMBOLS = PROFILE.max_symbols
FINALIZE_SILENCE_MS = 150
RIGHT_CONTEXT = PROFILE.right_context
CANARY_INDICES = (4, 9, 2, 3)


def tensor_clone(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if tensor is None else tensor.detach().clone()


def clone_tree(obj: Any, memo: Optional[dict[int, Any]] = None) -> Any:
    """Tensor-aware deepcopy for disposable ASR state."""
    if memo is None:
        memo = {}
    oid = id(obj)
    if oid in memo:
        return memo[oid]

    if torch.is_tensor(obj):
        return tensor_clone(obj)
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return obj
    if isinstance(obj, list):
        out: list[Any] = []
        memo[oid] = out
        out.extend(clone_tree(item, memo) for item in obj)
        return out
    if isinstance(obj, tuple):
        placeholder: list[Any] = []
        memo[oid] = placeholder
        out = tuple(clone_tree(item, memo) for item in obj)
        memo[oid] = out
        return out
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        memo[oid] = out
        for key, value in obj.items():
            out[clone_tree(key, memo)] = clone_tree(value, memo)
        return out
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = copy.copy(obj)
        memo[oid] = out
        for fld in dataclasses.fields(obj):
            setattr(out, fld.name, clone_tree(getattr(obj, fld.name), memo))
        return out
    try:
        return copy.deepcopy(obj, memo)
    except Exception:
        return obj


def assert_tree_equal(name: str, expected: Any, actual: Any) -> None:
    """Byte-exact recursive equality check for FORK_ASSERT state."""
    if torch.is_tensor(expected) or torch.is_tensor(actual):
        if not (torch.is_tensor(expected) and torch.is_tensor(actual)):
            raise AssertionError(f"{name}: tensor/non-tensor mismatch")
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise AssertionError(
                f"{name}: tensor metadata mismatch {expected.shape}/{expected.dtype} "
                f"vs {actual.shape}/{actual.dtype}"
            )
        if not torch.equal(expected, actual):
            raise AssertionError(f"{name}: tensor values differ")
        return
    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        if not (isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray)):
            raise AssertionError(f"{name}: ndarray/non-ndarray mismatch")
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise AssertionError(f"{name}: ndarray metadata mismatch")
        if not np.array_equal(expected, actual):
            raise AssertionError(f"{name}: ndarray values differ")
        return
    if isinstance(expected, (list, tuple)) or isinstance(actual, (list, tuple)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise AssertionError(f"{name}: sequence mismatch")
        for index, (lhs, rhs) in enumerate(zip(expected, actual)):
            assert_tree_equal(f"{name}[{index}]", lhs, rhs)
        return
    if isinstance(expected, dict) or isinstance(actual, dict):
        if not (isinstance(expected, dict) and isinstance(actual, dict)):
            raise AssertionError(f"{name}: dict/non-dict mismatch")
        if expected.keys() != actual.keys():
            raise AssertionError(f"{name}: dict keys differ")
        for key in expected:
            assert_tree_equal(f"{name}.{key}", expected[key], actual[key])
        return
    if dataclasses.is_dataclass(expected) or dataclasses.is_dataclass(actual):
        if type(expected) is not type(actual):
            raise AssertionError(f"{name}: dataclass type mismatch")
        for fld in dataclasses.fields(expected):
            assert_tree_equal(
                f"{name}.{fld.name}",
                getattr(expected, fld.name),
                getattr(actual, fld.name),
            )
        return
    if expected != actual:
        raise AssertionError(f"{name}: {expected!r} != {actual!r}")


def _continuous_append_only_delta(final_text: str, emitted_text: str) -> str:
    """Return the collector-safe word suffix to append for a cumulative final."""
    final_tokens = final_text.split()
    emitted_tokens = emitted_text.split()

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

    return " ".join(delta_tokens)


def _load_normalizer():
    try:
        from whisper_normalizer.english import EnglishTextNormalizer

        normalizer = EnglishTextNormalizer()
        return lambda text: normalizer(text).strip()
    except Exception:
        return lambda text: re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def load_model():
    model = load_profile_model(PROFILE)
    model.change_decoding_strategy(
        decoding_cfg=OmegaConf.create(
            {
                "strategy": "greedy_batch",
                "greedy": {
                    "max_symbols": MAX_SYMBOLS,
                    "loop_labels": True,
                    "use_cuda_graph_decoder": False,
                },
            }
        )
    )
    return model


@dataclass(frozen=True)
class RuntimeGeometry:
    shift_frames: int
    pre_encode_cache_size: int
    drop_extra: int
    final_padding_frames: int
    hop_samples: int
    window_size_samples: int
    raw_audio_ring_samples: int
    preprocess_align_pad_samples: int
    preprocess_new_audio_samples: int
    stream_preprocess_valid_samples: int
    first_preprocess_mel_frame: int
    constant_preprocess_frames: int
    constant_preprocess_samples: int


@dataclass
class FinalizeInputs:
    chunk_mel: torch.Tensor
    chunk_len: torch.Tensor
    cache_last_channel: torch.Tensor
    cache_last_time: torch.Tensor
    cache_last_channel_len: torch.Tensor
    drop_extra: int
    new_mel: torch.Tensor
    remaining_frames: int
    padded_total_samples: int


@dataclass
class FinalizeResult:
    final_text: str
    delta_text: str
    steady_text: str
    final_tokens: list[int]
    steady_tokens: list[int]
    fork_assert_passed: bool
    state_after: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class NemoStreamFinalizeResult:
    final_text: str
    steady_text: str
    final_tokens: list[int]
    steady_tokens: list[int]
    steady_chunks: int
    emitted_frames: int
    steady_tokens_by_emitted_frames: dict[int, list[int]]
    remaining_frames: Optional[int]
    final_chunk_T: Optional[int]
    drop_extra: Optional[int]


@dataclass
class ContinuousSession:
    id: str
    cache_last_channel: torch.Tensor
    cache_last_time: torch.Tensor
    cache_last_channel_len: torch.Tensor
    decoder_state: Any
    pred_out_stream: torch.Tensor
    pending_audio: np.ndarray
    raw_audio_ring: np.ndarray
    state: str = STREAMING
    debounce_armed: bool = False
    continuous_stop_seq: int = 0
    continuous_reset_seen: bool = False
    continuous_post_stop_audio: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    total_audio_samples: int = 0
    synthetic_prefix_samples: int = 0
    mel_frame_ring: Optional[torch.Tensor] = None
    emitted_frames: int = 0
    hyp_tokens: list[int] = field(default_factory=list)
    current_text: str = ""
    last_emitted_text: str = ""
    committed_text: str = ""
    continuous_emitted_text: str = ""


class ContinuousFinalizeRef:
    def __init__(self, model):
        self.model = model
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.joint = model.joint
        self.tokenizer = model.tokenizer
        self.device = next(model.parameters()).device
        self.geometry = self._derive_geometry()

    def _derive_geometry(self) -> RuntimeGeometry:
        scfg = self.encoder.streaming_cfg
        to_int = lambda value: int(value[1]) if isinstance(value, (list, tuple)) else int(value)
        shift_frames = to_int(scfg.shift_size)
        pre_encode_cache_size = to_int(scfg.pre_encode_cache_size)
        drop_extra = int(scfg.drop_extra_pre_encoded)
        featurizer = self.model.preprocessor.featurizer
        hop_samples = int(getattr(featurizer, "hop_length", 160))
        window_size_samples = int(getattr(featurizer, "win_length", 400))
        raw_audio_ring_samples = window_size_samples - hop_samples
        preprocess_align_pad_samples = (
            hop_samples - (raw_audio_ring_samples % hop_samples)
        ) % hop_samples
        preprocess_new_audio_samples = (shift_frames + 1) * hop_samples
        stream_preprocess_valid_samples = (
            preprocess_align_pad_samples
            + raw_audio_ring_samples
            + preprocess_new_audio_samples
        )
        prefix_samples = preprocess_align_pad_samples + raw_audio_ring_samples
        if prefix_samples % hop_samples != 0:
            raise RuntimeError("fixed preprocessor prefix is not hop-aligned")
        first_preprocess_mel_frame = prefix_samples // hop_samples
        final_padding_frames = (RIGHT_CONTEXT + 1) * shift_frames
        min_plan_frames = (
            first_preprocess_mel_frame
            + pre_encode_cache_size
            + shift_frames
            + final_padding_frames
            + 1
        )
        constant_preprocess_frames = 1 << (min_plan_frames - 1).bit_length()
        constant_preprocess_samples = (constant_preprocess_frames - 1) * hop_samples
        return RuntimeGeometry(
            shift_frames=shift_frames,
            pre_encode_cache_size=pre_encode_cache_size,
            drop_extra=drop_extra,
            final_padding_frames=final_padding_frames,
            hop_samples=hop_samples,
            window_size_samples=window_size_samples,
            raw_audio_ring_samples=raw_audio_ring_samples,
            preprocess_align_pad_samples=preprocess_align_pad_samples,
            preprocess_new_audio_samples=preprocess_new_audio_samples,
            stream_preprocess_valid_samples=stream_preprocess_valid_samples,
            first_preprocess_mel_frame=first_preprocess_mel_frame,
            constant_preprocess_frames=constant_preprocess_frames,
            constant_preprocess_samples=constant_preprocess_samples,
        )

    def new_session(self, session_id: str = "s0") -> ContinuousSession:
        cache = self.encoder.get_initial_cache_state(batch_size=1)
        state = self.decoder.initialize_state(
            torch.zeros(1, 1, dtype=torch.float32, device=self.device)
        )
        pred_out, state = self.decoder.predict(
            None,
            state,
            add_sos=False,
            batch_size=1,
        )
        return ContinuousSession(
            id=session_id,
            cache_last_channel=cache[0].detach().clone(),
            cache_last_time=cache[1].detach().clone(),
            cache_last_channel_len=cache[2].detach().clone(),
            decoder_state=clone_tree(state),
            pred_out_stream=pred_out.detach().clone(),
            pending_audio=np.array([], dtype=np.float32),
            raw_audio_ring=np.zeros(
                self.geometry.raw_audio_ring_samples,
                dtype=np.float32,
            ),
        )

    def _session_ready(self, session: ContinuousSession) -> bool:
        g = self.geometry
        timeline_samples = session.synthetic_prefix_samples + session.total_audio_samples
        needed_samples = (session.emitted_frames + g.shift_frames + 1) * g.hop_samples
        return (
            timeline_samples >= needed_samples
            and len(session.pending_audio) >= g.preprocess_new_audio_samples
        )

    def _flush_post_stop_audio(self, session: ContinuousSession) -> None:
        if len(session.continuous_post_stop_audio) == 0:
            return
        held = session.continuous_post_stop_audio
        session.continuous_post_stop_audio = np.array([], dtype=np.float32)
        session.pending_audio = np.concatenate([session.pending_audio, held]).astype(
            np.float32,
            copy=False,
        )
        session.total_audio_samples += int(held.shape[0])
        self.drain_steady(session)

    def append_audio(self, session: ContinuousSession, audio: np.ndarray) -> None:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError(f"expected mono audio, got shape={audio.shape}")
        if session.state == PENDING_FINALIZE:
            session.continuous_post_stop_audio = np.concatenate(
                [session.continuous_post_stop_audio, audio]
            ).astype(np.float32, copy=False)
            return
        if session.state != STREAMING:
            raise RuntimeError(f"append_audio in state={session.state}")
        self._flush_post_stop_audio(session)
        session.pending_audio = np.concatenate([session.pending_audio, audio]).astype(
            np.float32,
            copy=False,
        )
        session.total_audio_samples += int(audio.shape[0])
        self.drain_steady(session)

    def drain_steady(self, session: ContinuousSession) -> None:
        while self._session_ready(session):
            self._process_one_steady_chunk(session)

    def vad_stop(self, session: ContinuousSession) -> None:
        if session.state == FINALIZED:
            raise RuntimeError("vad_stop called before finalize finish")
        session.continuous_stop_seq += 1
        session.state = PENDING_FINALIZE
        session.debounce_armed = True
        session.continuous_reset_seen = False

    def vad_start(self, session: ContinuousSession) -> None:
        if session.state == PENDING_FINALIZE:
            session.state = STREAMING
            session.debounce_armed = False
            session.continuous_reset_seen = False
            self._flush_post_stop_audio(session)
            return
        if session.state == STREAMING and len(session.continuous_post_stop_audio) > 0:
            self._flush_post_stop_audio(session)

    def debounce_expire(self, session: ContinuousSession) -> FinalizeResult:
        if session.state != PENDING_FINALIZE:
            raise RuntimeError(f"debounce_expire in state={session.state}")
        parent_snapshot = self._snapshot_parent_asr_state(session)
        session.debounce_armed = False
        session.state = FINALIZED
        result = self._finalize_and_emit(session, reason="debounce_expired")
        self._finish_speculative(session, result)
        self._assert_parent_asr_state_unchanged(
            session,
            parent_snapshot,
            "parent_after_speculative_finish",
        )
        return result

    def force_finalize_end(self, session: ContinuousSession) -> FinalizeResult:
        if len(session.continuous_post_stop_audio) > 0:
            held = session.continuous_post_stop_audio
            session.continuous_post_stop_audio = np.array([], dtype=np.float32)
            session.pending_audio = np.concatenate([session.pending_audio, held]).astype(
                np.float32,
                copy=False,
            )
            session.total_audio_samples += int(held.shape[0])
            self.drain_steady(session)
        session.debounce_armed = False
        session.state = FINALIZED
        result = self._finalize_and_emit(session, reason="end")
        self._cold_reset_after_finalize(session)
        return result

    def _build_fixed_preprocess_audio(
        self,
        raw_audio_ring: np.ndarray,
        new_audio: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        g = self.geometry
        if len(raw_audio_ring) != g.raw_audio_ring_samples:
            raise ValueError(
                f"expected raw ring {g.raw_audio_ring_samples}, got {len(raw_audio_ring)}"
            )
        prefix_len = g.preprocess_align_pad_samples + g.raw_audio_ring_samples
        valid_samples = prefix_len + len(new_audio)
        if valid_samples > g.constant_preprocess_samples:
            raise ValueError(
                f"fixed preprocessor valid span {valid_samples} exceeds "
                f"K={g.constant_preprocess_samples}"
            )
        audio = np.zeros(g.constant_preprocess_samples, dtype=np.float32)
        cursor = g.preprocess_align_pad_samples
        audio[cursor : cursor + g.raw_audio_ring_samples] = raw_audio_ring
        cursor += g.raw_audio_ring_samples
        audio[cursor : cursor + len(new_audio)] = new_audio
        return audio, valid_samples

    @torch.inference_mode()
    def _preprocess_fixed_audio(
        self,
        audio: np.ndarray,
        valid_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.geometry
        if len(audio) != g.constant_preprocess_samples:
            raise ValueError(
                f"expected fixed preprocessor input {g.constant_preprocess_samples}, "
                f"got {len(audio)}"
            )
        audio_tensor = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0).to(
            self.device
        )
        audio_len = torch.tensor([valid_samples], device=self.device, dtype=torch.long)
        return self.model.preprocessor(input_signal=audio_tensor, length=audio_len)

    def _fixed_mel_from_new_audio(
        self,
        raw_audio_ring: np.ndarray,
        new_audio: np.ndarray,
    ) -> torch.Tensor:
        fixed_audio, valid_samples = self._build_fixed_preprocess_audio(
            raw_audio_ring,
            new_audio,
        )
        mel, _mel_len = self._preprocess_fixed_audio(fixed_audio, valid_samples)
        return mel

    def _update_mel_frame_ring(
        self,
        session: ContinuousSession,
        new_mel: torch.Tensor,
    ) -> None:
        g = self.geometry
        if session.mel_frame_ring is None:
            combined = new_mel.detach()
        else:
            combined = torch.cat((session.mel_frame_ring, new_mel.detach()), dim=-1)
        session.mel_frame_ring = combined[:, :, -g.pre_encode_cache_size :].detach()

    def _advance_raw_ring(self, raw_ring: np.ndarray, consumed_audio: np.ndarray) -> np.ndarray:
        g = self.geometry
        if len(consumed_audio) >= g.raw_audio_ring_samples:
            return consumed_audio[-g.raw_audio_ring_samples :].copy()
        if len(consumed_audio) == 0:
            return raw_ring
        keep = g.raw_audio_ring_samples - len(consumed_audio)
        return np.concatenate([raw_ring[-keep:], consumed_audio]).astype(
            np.float32,
            copy=False,
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

        if session.emitted_frames == 0:
            chunk_mel = valid_new_mel
            drop_extra = 0
        else:
            chunk_mel = torch.cat((session.mel_frame_ring, valid_new_mel), dim=-1)
            drop_extra = g.drop_extra

        chunk_len = torch.tensor([chunk_mel.shape[-1]], device=self.device)
        enc_out, enc_len, clc, clt, clcl = self.encoder.cache_aware_stream_step(
            processed_signal=chunk_mel,
            processed_signal_length=chunk_len,
            cache_last_channel=session.cache_last_channel,
            cache_last_time=session.cache_last_time,
            cache_last_channel_len=session.cache_last_channel_len,
            keep_all_outputs=False,
            drop_extra_pre_encoded=drop_extra,
        )
        enc_out = apply_prompt(self.model, enc_out)
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

    def _snapshot_parent_asr_state(self, session: ContinuousSession) -> dict[str, Any]:
        return {
            "cache_last_channel": tensor_clone(session.cache_last_channel),
            "cache_last_time": tensor_clone(session.cache_last_time),
            "cache_last_channel_len": tensor_clone(session.cache_last_channel_len),
            "decoder_state": clone_tree(session.decoder_state),
            "pred_out_stream": tensor_clone(session.pred_out_stream),
            "hyp_tokens": list(session.hyp_tokens),
            "pending_audio": session.pending_audio.copy(),
            "raw_audio_ring": session.raw_audio_ring.copy(),
            "mel_frame_ring": clone_tree(session.mel_frame_ring),
            "emitted_frames": int(session.emitted_frames),
            "total_audio_samples": int(session.total_audio_samples),
        }

    def _assert_parent_asr_state_unchanged(
        self,
        session: ContinuousSession,
        snapshot: dict[str, Any],
        name: str,
    ) -> None:
        assert_tree_equal(
            f"{name}.cache_last_channel",
            snapshot["cache_last_channel"],
            session.cache_last_channel,
        )
        assert_tree_equal(
            f"{name}.cache_last_time",
            snapshot["cache_last_time"],
            session.cache_last_time,
        )
        assert_tree_equal(
            f"{name}.cache_last_channel_len",
            snapshot["cache_last_channel_len"],
            session.cache_last_channel_len,
        )
        assert_tree_equal(
            f"{name}.decoder_state",
            snapshot["decoder_state"],
            session.decoder_state,
        )
        assert_tree_equal(
            f"{name}.pred_out_stream",
            snapshot["pred_out_stream"],
            session.pred_out_stream,
        )
        assert_tree_equal(f"{name}.hyp_tokens", snapshot["hyp_tokens"], session.hyp_tokens)
        assert_tree_equal(
            f"{name}.pending_audio",
            snapshot["pending_audio"],
            session.pending_audio,
        )
        assert_tree_equal(
            f"{name}.raw_audio_ring",
            snapshot["raw_audio_ring"],
            session.raw_audio_ring,
        )
        assert_tree_equal(
            f"{name}.mel_frame_ring",
            snapshot["mel_frame_ring"],
            session.mel_frame_ring,
        )
        assert_tree_equal(
            f"{name}.emitted_frames",
            snapshot["emitted_frames"],
            session.emitted_frames,
        )
        assert_tree_equal(
            f"{name}.total_audio_samples",
            snapshot["total_audio_samples"],
            session.total_audio_samples,
        )

    def build_continuous_finalize_fork(
        self,
        session: ContinuousSession,
        append_padding: bool = True,
    ) -> ContinuousSession:
        pending_audio = session.pending_audio.copy()
        padding_samples = 0
        if append_padding and session.total_audio_samples > 0:
            padding_samples = (
                self.geometry.final_padding_frames * self.geometry.hop_samples
            )
            pending_audio = np.concatenate(
                [pending_audio, np.zeros(padding_samples, dtype=np.float32)]
            ).astype(np.float32, copy=False)

        fork = ContinuousSession(
            id=f"{session.id}:fork",
            cache_last_channel=tensor_clone(session.cache_last_channel),
            cache_last_time=tensor_clone(session.cache_last_time),
            cache_last_channel_len=tensor_clone(session.cache_last_channel_len),
            decoder_state=clone_tree(session.decoder_state),
            pred_out_stream=tensor_clone(session.pred_out_stream),
            pending_audio=pending_audio,
            raw_audio_ring=session.raw_audio_ring.copy(),
            state=session.state,
            debounce_armed=session.debounce_armed,
            continuous_stop_seq=session.continuous_stop_seq,
            continuous_reset_seen=session.continuous_reset_seen,
            total_audio_samples=session.total_audio_samples + padding_samples,
            synthetic_prefix_samples=session.synthetic_prefix_samples,
            mel_frame_ring=clone_tree(session.mel_frame_ring),
            emitted_frames=session.emitted_frames,
            hyp_tokens=list(session.hyp_tokens),
            current_text=session.current_text,
            last_emitted_text=session.last_emitted_text,
            committed_text=session.committed_text,
            continuous_emitted_text=session.continuous_emitted_text,
        )
        return fork

    def prepare_finalize_inputs(self, fork: ContinuousSession) -> Optional[FinalizeInputs]:
        g = self.geometry
        if len(fork.pending_audio) == 0:
            return None
        padded_total_samples = fork.emitted_frames * g.hop_samples + len(
            fork.pending_audio
        )
        total_mel_frames = padded_total_samples // g.hop_samples + 1
        remaining_frames = total_mel_frames - fork.emitted_frames
        if remaining_frames <= 0:
            return None

        pending = fork.pending_audio
        raw_ring = fork.raw_audio_ring.copy()
        new_mels: list[torch.Tensor] = []
        frames_collected = 0
        while frames_collected < remaining_frames:
            frames_this_call = min(
                g.shift_frames,
                remaining_frames - frames_collected,
            )
            needed_new_samples = min(len(pending), g.preprocess_new_audio_samples)
            new_audio = pending[:needed_new_samples]
            mel = self._fixed_mel_from_new_audio(raw_ring, new_audio)
            start = g.first_preprocess_mel_frame
            new_mels.append(mel[:, :, start : start + frames_this_call])

            if frames_this_call == g.shift_frames:
                consumed_samples = min(g.shift_frames * g.hop_samples, len(pending))
                consumed_audio = pending[:consumed_samples]
                raw_ring = self._advance_raw_ring(raw_ring, consumed_audio)
                pending = pending[consumed_samples:]
            frames_collected += frames_this_call

        new_mel = torch.cat(new_mels, dim=-1)
        if fork.emitted_frames == 0:
            chunk_mel = new_mel
            drop_extra = 0
        else:
            chunk_mel = torch.cat((fork.mel_frame_ring, new_mel), dim=-1)
            drop_extra = g.drop_extra

        chunk_len = torch.tensor([chunk_mel.shape[-1]], device=self.device)
        return FinalizeInputs(
            chunk_mel=chunk_mel,
            chunk_len=chunk_len,
            cache_last_channel=tensor_clone(fork.cache_last_channel),
            cache_last_time=tensor_clone(fork.cache_last_time),
            cache_last_channel_len=tensor_clone(fork.cache_last_channel_len),
            drop_extra=int(drop_extra),
            new_mel=new_mel,
            remaining_frames=int(remaining_frames),
            padded_total_samples=int(padded_total_samples),
        )

    @torch.inference_mode()
    def flush_finalize_fork(self, fork: ContinuousSession) -> dict[str, Any]:
        g = self.geometry
        inputs = self.prepare_finalize_inputs(fork)
        if inputs is None:
            return {
                "final_text": fork.current_text,
                "final_tokens": list(fork.hyp_tokens),
                "inputs": None,
                "encoder_outputs": None,
            }

        enc_out, enc_len, clc, clt, clcl = self.encoder.cache_aware_stream_step(
            processed_signal=inputs.chunk_mel,
            processed_signal_length=inputs.chunk_len,
            cache_last_channel=inputs.cache_last_channel,
            cache_last_time=inputs.cache_last_time,
            cache_last_channel_len=inputs.cache_last_channel_len,
            keep_all_outputs=True,
            drop_extra_pre_encoded=inputs.drop_extra,
        )
        enc_out = apply_prompt(self.model, enc_out)
        tokens, decoder_state, pred_out = ref_greedy_range(
            self.decoder,
            self.joint,
            enc_out.transpose(1, 2).contiguous(),
            0,
            int(enc_len[0]),
            fork.decoder_state,
            fork.pred_out_stream,
        )
        fork.hyp_tokens.extend(tokens)
        fork.decoder_state = decoder_state
        fork.pred_out_stream = pred_out
        fork.cache_last_channel = clc
        fork.cache_last_time = clt
        fork.cache_last_channel_len = clcl
        fork.current_text = self.tokenizer.ids_to_text(fork.hyp_tokens)

        finalized_audio = fork.pending_audio
        fork.emitted_frames += inputs.remaining_frames
        fork.pending_audio = np.array([], dtype=np.float32)
        self._update_mel_frame_ring(fork, inputs.new_mel)
        if len(finalized_audio) >= g.raw_audio_ring_samples:
            fork.raw_audio_ring = finalized_audio[-g.raw_audio_ring_samples :].copy()
        elif len(finalized_audio) > 0:
            fork.raw_audio_ring = self._advance_raw_ring(
                fork.raw_audio_ring,
                finalized_audio,
            )

        return {
            "final_text": fork.current_text,
            "final_tokens": list(fork.hyp_tokens),
            "new_tokens": tokens,
            "inputs": inputs,
            "encoder_outputs": (enc_out, enc_len, clc, clt, clcl),
        }

    @staticmethod
    def _tokens_from_nemo_hypotheses(hypotheses: Any) -> list[int]:
        if hypotheses is None:
            return []
        y_sequence = hypotheses[0].y_sequence
        if torch.is_tensor(y_sequence):
            return y_sequence.detach().cpu().tolist()
        return list(y_sequence)

    @torch.inference_mode()
    def _process_one_steady_chunk_nemo(
        self,
        session: ContinuousSession,
        partial_hypotheses: Any,
    ) -> tuple[Any, list[int]]:
        g = self.geometry
        new_audio = session.pending_audio[: g.preprocess_new_audio_samples]
        mel = self._fixed_mel_from_new_audio(session.raw_audio_ring, new_audio)
        valid_new_mel = mel[
            :,
            :,
            g.first_preprocess_mel_frame : g.first_preprocess_mel_frame
            + g.shift_frames,
        ]

        if session.emitted_frames == 0:
            chunk_mel = valid_new_mel
            drop_extra = 0
        else:
            chunk_mel = torch.cat((session.mel_frame_ring, valid_new_mel), dim=-1)
            drop_extra = g.drop_extra

        chunk_len = torch.tensor([chunk_mel.shape[-1]], device=self.device)
        enc_out, enc_len, clc, clt, clcl = self.encoder.cache_aware_stream_step(
            processed_signal=chunk_mel,
            processed_signal_length=chunk_len,
            cache_last_channel=session.cache_last_channel,
            cache_last_time=session.cache_last_time,
            cache_last_channel_len=session.cache_last_channel_len,
            keep_all_outputs=False,
            drop_extra_pre_encoded=drop_extra,
        )
        enc_out = apply_prompt(self.model, enc_out)
        hypotheses = self.model.decoding.rnnt_decoder_predictions_tensor(
            enc_out,
            enc_len,
            return_hypotheses=True,
            partial_hypotheses=partial_hypotheses,
        )
        tokens = self._tokens_from_nemo_hypotheses(hypotheses)

        session.hyp_tokens = list(tokens)
        session.current_text = self.text(tokens)
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
        return hypotheses, tokens

    @torch.inference_mode()
    def nemo_stream_finalize(self, wav: np.ndarray) -> NemoStreamFinalizeResult:
        """NeMo chunked streaming + keep_all_outputs finalize oracle."""
        audio = np.asarray(wav, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError(f"expected mono audio, got shape={audio.shape}")

        session = self.new_session("nemo-oracle")
        session.pending_audio = audio.copy()
        session.total_audio_samples = int(audio.shape[0])

        partial_hypotheses = None
        steady_tokens: list[int] = []
        steady_tokens_by_emitted_frames: dict[int, list[int]] = {0: []}
        steady_chunks = 0
        while self._session_ready(session):
            partial_hypotheses, steady_tokens = self._process_one_steady_chunk_nemo(
                session,
                partial_hypotheses,
            )
            steady_chunks += 1
            steady_tokens_by_emitted_frames[session.emitted_frames] = list(steady_tokens)

        fork = self.build_continuous_finalize_fork(session)
        inputs = self.prepare_finalize_inputs(fork)
        remaining_frames = None
        final_chunk_T = None
        drop_extra = None
        if inputs is not None:
            enc_out, enc_len, _clc, _clt, _clcl = self.encoder.cache_aware_stream_step(
                processed_signal=inputs.chunk_mel,
                processed_signal_length=inputs.chunk_len,
                cache_last_channel=inputs.cache_last_channel,
                cache_last_time=inputs.cache_last_time,
                cache_last_channel_len=inputs.cache_last_channel_len,
                keep_all_outputs=True,
                drop_extra_pre_encoded=inputs.drop_extra,
            )
            enc_out = apply_prompt(self.model, enc_out)
            partial_hypotheses = self.model.decoding.rnnt_decoder_predictions_tensor(
                enc_out,
                enc_len,
                return_hypotheses=True,
                partial_hypotheses=partial_hypotheses,
            )
            final_tokens = self._tokens_from_nemo_hypotheses(partial_hypotheses)
            remaining_frames = int(inputs.remaining_frames)
            final_chunk_T = int(inputs.chunk_mel.shape[-1])
            drop_extra = int(inputs.drop_extra)
        else:
            final_tokens = list(steady_tokens)

        return NemoStreamFinalizeResult(
            final_text=self.text(final_tokens),
            steady_text=self.text(steady_tokens),
            final_tokens=list(final_tokens),
            steady_tokens=list(steady_tokens),
            steady_chunks=steady_chunks,
            emitted_frames=session.emitted_frames,
            steady_tokens_by_emitted_frames=steady_tokens_by_emitted_frames,
            remaining_frames=remaining_frames,
            final_chunk_T=final_chunk_T,
            drop_extra=drop_extra,
        )

    def nemo_stream_finalize_tokens(self, wav: np.ndarray) -> list[int]:
        return self.nemo_stream_finalize(wav).final_tokens

    def _finalize_and_emit(
        self,
        session: ContinuousSession,
        *,
        reason: str,
    ) -> FinalizeResult:
        parent_snapshot = self._snapshot_parent_asr_state(session)
        fork = self.build_continuous_finalize_fork(session)
        steady_tokens = list(session.hyp_tokens)
        steady_text = session.current_text
        flush = self.flush_finalize_fork(fork)
        self._assert_parent_asr_state_unchanged(
            session,
            parent_snapshot,
            "parent_after_fork_flush",
        )

        final_text = flush["final_text"]
        delta_text = _continuous_append_only_delta(
            final_text,
            session.continuous_emitted_text,
        )
        session.committed_text = final_text
        session.last_emitted_text = final_text
        if delta_text:
            session.continuous_emitted_text = (
                session.continuous_emitted_text + " " + delta_text
            ).strip()

        return FinalizeResult(
            final_text=final_text,
            delta_text=delta_text,
            steady_text=steady_text,
            final_tokens=list(fork.hyp_tokens),
            steady_tokens=steady_tokens,
            fork_assert_passed=True,
            state_after=session.state,
            meta={
                "reason": reason,
                "remaining_frames": (
                    None if flush["inputs"] is None else flush["inputs"].remaining_frames
                ),
                "final_chunk_T": (
                    None if flush["inputs"] is None else int(flush["inputs"].chunk_mel.shape[-1])
                ),
                "drop_extra": (
                    None if flush["inputs"] is None else int(flush["inputs"].drop_extra)
                ),
            },
        )

    def _finish_speculative(
        self,
        session: ContinuousSession,
        result: FinalizeResult,
    ) -> None:
        session.state = STREAMING
        session.debounce_armed = False
        session.continuous_reset_seen = False
        result.state_after = session.state

    def _cold_reset_after_finalize(self, session: ContinuousSession) -> None:
        committed = session.committed_text
        emitted = session.continuous_emitted_text
        fresh = self.new_session(session.id)
        session.cache_last_channel = fresh.cache_last_channel
        session.cache_last_time = fresh.cache_last_time
        session.cache_last_channel_len = fresh.cache_last_channel_len
        session.decoder_state = fresh.decoder_state
        session.pred_out_stream = fresh.pred_out_stream
        session.pending_audio = fresh.pending_audio
        session.raw_audio_ring = fresh.raw_audio_ring
        session.mel_frame_ring = None
        session.emitted_frames = 0
        session.hyp_tokens = []
        session.current_text = ""
        session.total_audio_samples = 0
        session.synthetic_prefix_samples = 0
        session.continuous_post_stop_audio = np.array([], dtype=np.float32)
        session.committed_text = ""
        session.last_emitted_text = ""
        session.continuous_emitted_text = ""
        session.state = STREAMING
        session.debounce_armed = False
        session.continuous_reset_seen = False
        _ = committed, emitted

    @torch.inference_mode()
    def offline_full_greedy_tokens(self, wav: np.ndarray) -> list[int]:
        audio = torch.tensor(wav, dtype=torch.float32, device=self.device).unsqueeze(0)
        audio_len = torch.tensor([wav.shape[0]], dtype=torch.long, device=self.device)
        # Run preprocessor+encoder directly (prompted models require
        # prompt_indices through model.forward; the inference-prompt path is
        # the post-encoder apply_prompt hook instead).
        proc, proc_len = self.model.preprocessor(input_signal=audio, length=audio_len)
        enc, enc_len = self.encoder(audio_signal=proc, length=proc_len)
        enc = apply_prompt(self.model, enc)
        return ref_greedy(self.decoder, self.joint, enc, enc_len)

    def text(self, tokens: list[int]) -> str:
        return self.tokenizer.ids_to_text(tokens)


def load_benchmark_dataset():
    import datasets

    return datasets.load_dataset(
        "pipecat-ai/stt-benchmark-data",
        split="train",
    ).cast_column("audio", datasets.Audio(decode=False))


def load_wav(example: dict[str, Any]) -> np.ndarray:
    wav, sr = sf.read(io.BytesIO(example["audio"]["bytes"]), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        n = int(len(wav) * 16000 / sr)
        wav = np.interp(
            np.linspace(0, len(wav), n, endpoint=False),
            np.arange(len(wav)),
            wav,
        ).astype(np.float32)
    return np.asarray(wav, dtype=np.float32)


def run_single_finalize(rt: ContinuousFinalizeRef, wav: np.ndarray, session_id: str) -> FinalizeResult:
    session = rt.new_session(session_id)
    rt.append_audio(session, wav)
    rt.vad_stop(session)
    return rt.debounce_expire(session)


def _first_token_diff(lhs: list[int], rhs: list[int]) -> str:
    for index, (a, b) in enumerate(zip(lhs, rhs)):
        if a != b:
            return f"first_diff={index} finalize={a} oracle={b}"
    if len(lhs) != len(rhs):
        return f"prefix_equal len_finalize={len(lhs)} len_oracle={len(rhs)}"
    return "identical"


def _concat_audio(*parts: np.ndarray) -> np.ndarray:
    non_empty = [np.asarray(part, dtype=np.float32) for part in parts if len(part) > 0]
    if not non_empty:
        return np.array([], dtype=np.float32)
    return np.concatenate(non_empty).astype(np.float32, copy=False)


def _clip_or_repeat(wav: np.ndarray, length: int) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=np.float32)
    if len(wav) >= length:
        return wav[:length].astype(np.float32, copy=True)
    repeats = (length + len(wav) - 1) // len(wav)
    return np.tile(wav, repeats)[:length].astype(np.float32, copy=True)


def _audio_needed_for_one_steady_chunk(
    rt: ContinuousFinalizeRef,
    session: ContinuousSession,
) -> int:
    g = rt.geometry
    needed_pending = g.preprocess_new_audio_samples - len(session.pending_audio)
    needed_timeline = (
        (session.emitted_frames + g.shift_frames + 1) * g.hop_samples
        - (session.synthetic_prefix_samples + session.total_audio_samples)
    )
    return max(1, int(needed_pending), int(needed_timeline))


def _assert_steady_matches_oracle(
    session: ContinuousSession,
    oracle: NemoStreamFinalizeResult,
    name: str,
) -> None:
    expected = oracle.steady_tokens_by_emitted_frames.get(session.emitted_frames)
    if expected is None:
        raise AssertionError(
            f"{name}: oracle has no steady snapshot for emitted_frames="
            f"{session.emitted_frames}; available={sorted(oracle.steady_tokens_by_emitted_frames)}"
        )
    assert_tree_equal(f"{name}.steady_tokens", expected, session.hyp_tokens)


def _run_parent_after_speculative_assert(
    rt: ContinuousFinalizeRef,
    wav: np.ndarray,
) -> bool:
    session = rt.new_session("parent-after-speculative")
    rt.append_audio(session, wav)
    rt.vad_stop(session)
    parent_snapshot = rt._snapshot_parent_asr_state(session)
    result = rt.debounce_expire(session)
    rt._assert_parent_asr_state_unchanged(
        session,
        parent_snapshot,
        "external_parent_after_speculative_finish",
    )
    return result.fork_assert_passed and result.state_after == STREAMING


def _run_reset_resume_test(
    rt: ContinuousFinalizeRef,
    turn_a: np.ndarray,
    turn_b: np.ndarray,
) -> tuple[bool, str]:
    session = rt.new_session("reset-resume")
    rt.append_audio(session, turn_a)
    rt.vad_stop(session)
    pre_finish_snapshot = rt._snapshot_parent_asr_state(session)
    first = rt.debounce_expire(session)
    rt._assert_parent_asr_state_unchanged(
        session,
        pre_finish_snapshot,
        "reset_resume_parent_after_first_finish",
    )

    need = _audio_needed_for_one_steady_chunk(rt, session)
    if len(turn_b) < need:
        raise AssertionError(f"reset_resume: continuation too short need={need} got={len(turn_b)}")
    continuation = turn_b[:need]
    before_frames = session.emitted_frames
    rt.append_audio(session, continuation)
    if session.emitted_frames <= before_frames:
        raise AssertionError("reset_resume: continuation did not produce the next steady chunk")

    oracle_audio = _concat_audio(turn_a, continuation)
    oracle = rt.nemo_stream_finalize(oracle_audio)
    _assert_steady_matches_oracle(session, oracle, "reset_resume_next_steady")

    rt.vad_stop(session)
    pre_second_finish_snapshot = rt._snapshot_parent_asr_state(session)
    second = rt.debounce_expire(session)
    rt._assert_parent_asr_state_unchanged(
        session,
        pre_second_finish_snapshot,
        "reset_resume_parent_after_second_finish",
    )
    final_ok = second.final_tokens == oracle.final_tokens
    detail = (
        f"first_state={first.state_after} next_steady_frames={session.emitted_frames} "
        f"final_tok={len(second.final_tokens)} oracle_tok={len(oracle.final_tokens)}"
    )
    if not final_ok:
        detail += f" {_first_token_diff(second.final_tokens, oracle.final_tokens)}"
    return final_ok, detail


def _run_vad_start_cancel_test(
    rt: ContinuousFinalizeRef,
    wav_a: np.ndarray,
    wav_b: np.ndarray,
) -> tuple[bool, str]:
    g = rt.geometry
    source = _concat_audio(wav_a, wav_b)
    pre_len = (
        g.preprocess_new_audio_samples
        + 8 * g.shift_frames * g.hop_samples
        + 3 * g.hop_samples
    )
    post_len = 2 * g.hop_samples
    min_len = pre_len + post_len + g.preprocess_new_audio_samples
    source = _clip_or_repeat(source, min_len)
    pre = source[:pre_len]
    post = source[pre_len : pre_len + post_len]

    session = rt.new_session("vad-start-cancel")
    rt.append_audio(session, pre)
    pre_stop_snapshot = rt._snapshot_parent_asr_state(session)
    rt.vad_stop(session)
    rt.append_audio(session, post)
    if session.committed_text or session.continuous_emitted_text:
        raise AssertionError("vad_start_cancel: emitted a final before debounce")
    if len(session.continuous_post_stop_audio) != len(post):
        raise AssertionError("vad_start_cancel: post-stop audio was not held while pending")
    rt._assert_parent_asr_state_unchanged(
        session,
        pre_stop_snapshot,
        "vad_start_cancel_parent_while_pending",
    )

    rt.vad_start(session)
    if session.committed_text or session.continuous_emitted_text:
        raise AssertionError("vad_start_cancel: emitted a final during vad_start cancellation")
    if session.state != STREAMING or session.debounce_armed:
        raise AssertionError(f"vad_start_cancel: bad state={session.state} debounce={session.debounce_armed}")
    if len(session.continuous_post_stop_audio) != 0:
        raise AssertionError("vad_start_cancel: held audio was not flushed")

    no_stop = rt.new_session("vad-start-cancel-no-stop")
    rt.append_audio(no_stop, pre)
    rt.append_audio(no_stop, post)
    rt._assert_parent_asr_state_unchanged(
        session,
        rt._snapshot_parent_asr_state(no_stop),
        "vad_start_cancel_matches_no_stop_after_flush",
    )

    cursor = pre_len + post_len
    need = _audio_needed_for_one_steady_chunk(rt, session)
    continuation = source[cursor : cursor + need]
    if len(continuation) < need:
        raise AssertionError("vad_start_cancel: continuation source too short")
    rt.append_audio(session, continuation)
    oracle_audio = _concat_audio(pre, post, continuation)
    oracle = rt.nemo_stream_finalize(oracle_audio)
    _assert_steady_matches_oracle(session, oracle, "vad_start_cancel_next_steady")

    rt.vad_stop(session)
    final = rt.debounce_expire(session)
    final_ok = final.final_tokens == oracle.final_tokens
    detail = (
        f"held={len(post)} flushed=True steady_frames={session.emitted_frames} "
        f"final_tok={len(final.final_tokens)} oracle_tok={len(oracle.final_tokens)}"
    )
    if not final_ok:
        detail += f" {_first_token_diff(final.final_tokens, oracle.final_tokens)}"
    return final_ok, detail


def _run_finalize_geometry_case(
    rt: ContinuousFinalizeRef,
    wav: np.ndarray,
    name: str,
    expect_emitted_zero: Optional[bool],
) -> dict[str, int]:
    g = rt.geometry
    session = rt.new_session(name)
    if len(wav) > 0:
        rt.append_audio(session, wav)
    if expect_emitted_zero is not None and (session.emitted_frames == 0) != expect_emitted_zero:
        raise AssertionError(
            f"{name}: emitted_frames={session.emitted_frames}, "
            f"expect_emitted_zero={expect_emitted_zero}"
        )

    fork = rt.build_continuous_finalize_fork(session)
    inputs = rt.prepare_finalize_inputs(fork)
    if inputs is None:
        if len(fork.pending_audio) != 0:
            raise AssertionError(f"{name}: missing inputs with pending={len(fork.pending_audio)}")
        final_tokens = list(fork.hyp_tokens)
        remaining_frames = 0
        chunk_T = 0
        drop_extra = 0
    else:
        expected_remaining = (
            fork.emitted_frames * g.hop_samples + len(fork.pending_audio)
        ) // g.hop_samples + 1 - fork.emitted_frames
        expected_chunk_T = (
            expected_remaining
            if fork.emitted_frames == 0
            else g.pre_encode_cache_size + expected_remaining
        )
        expected_drop = 0 if fork.emitted_frames == 0 else g.drop_extra
        if inputs.remaining_frames != expected_remaining:
            raise AssertionError(
                f"{name}: remaining_frames={inputs.remaining_frames} "
                f"expected={expected_remaining}"
            )
        if int(inputs.chunk_mel.shape[-1]) != expected_chunk_T:
            raise AssertionError(
                f"{name}: chunk_T={int(inputs.chunk_mel.shape[-1])} "
                f"expected={expected_chunk_T}"
            )
        if int(inputs.drop_extra) != expected_drop:
            raise AssertionError(
                f"{name}: drop_extra={inputs.drop_extra} expected={expected_drop}"
            )
        flush = rt.flush_finalize_fork(fork)
        final_tokens = flush["final_tokens"]
        remaining_frames = int(inputs.remaining_frames)
        chunk_T = int(inputs.chunk_mel.shape[-1])
        drop_extra = int(inputs.drop_extra)

    oracle = rt.nemo_stream_finalize(wav)
    if final_tokens != oracle.final_tokens:
        raise AssertionError(
            f"{name}: finalize tokens differ from oracle "
            f"{_first_token_diff(final_tokens, oracle.final_tokens)}"
        )
    return {
        "remaining_frames": remaining_frames,
        "chunk_T": chunk_T,
        "drop_extra": drop_extra,
        "emitted_frames": int(session.emitted_frames),
    }


def _run_boundary_sweep(
    rt: ContinuousFinalizeRef,
    wav: np.ndarray,
) -> tuple[bool, str]:
    g = rt.geometry
    base = wav if len(wav) > 0 else np.zeros(g.preprocess_new_audio_samples, dtype=np.float32)
    cases: list[tuple[str, int, Optional[bool]]] = []
    for residual in range(0, g.shift_frames + 1):
        cases.append((f"first_residual_{residual}", residual * g.hop_samples, True))
    for residual in range(1, g.shift_frames + 1):
        cases.append(
            (
                f"continuation_residual_{residual}",
                (g.shift_frames + residual) * g.hop_samples,
                False,
            )
        )

    boundary_lengths = {
        g.preprocess_new_audio_samples - 1,
        g.preprocess_new_audio_samples,
        g.preprocess_new_audio_samples + 1,
        g.shift_frames * g.hop_samples - 1,
        g.shift_frames * g.hop_samples,
        g.shift_frames * g.hop_samples + 1,
        2 * g.shift_frames * g.hop_samples - 1,
        2 * g.shift_frames * g.hop_samples,
        2 * g.shift_frames * g.hop_samples + 1,
    }
    for length in sorted(length for length in boundary_lengths if length >= 0):
        cases.append((f"boundary_len_{length}", length, None))

    seen: set[tuple[str, int]] = set()
    stats = {"first": 0, "continuation": 0, "empty": 0}
    for name, length, expect_zero in cases:
        key = (name, length)
        if key in seen:
            continue
        seen.add(key)
        wav_case = _clip_or_repeat(base, length)
        meta = _run_finalize_geometry_case(rt, wav_case, name, expect_zero)
        if length == 0:
            stats["empty"] += 1
        elif meta["emitted_frames"] == 0:
            stats["first"] += 1
        else:
            stats["continuation"] += 1

    return True, (
        f"cases={len(seen)} first={stats['first']} "
        f"continuation={stats['continuation']} empty={stats['empty']}"
    )


def validation_gate() -> bool:
    print("loading model + stt-benchmark canaries...")
    model = load_model()
    rt = ContinuousFinalizeRef(model)
    ds = load_benchmark_dataset()
    g = rt.geometry
    print(
        "geometry "
        f"shift={g.shift_frames} pre={g.pre_encode_cache_size} drop={g.drop_extra} "
        f"final_padding_frames={g.final_padding_frames} "
        f"fixed_preproc_samples={g.constant_preprocess_samples}"
    )

    canary_ok = True
    exact_count = 0
    recovered_count = 0
    print("\nT1 finalize canary (gate: token-exact vs NeMo chunked stream+finalize oracle):")
    for sample_index in CANARY_INDICES:
        ex = ds[sample_index]
        wav = load_wav(ex)
        result = run_single_finalize(rt, wav, f"canary-{sample_index}")
        oracle = rt.nemo_stream_finalize(wav)
        offline_tokens = rt.offline_full_greedy_tokens(wav)
        offline_text = rt.text(offline_tokens)
        exact = result.final_tokens == oracle.final_tokens
        offline_exact = result.final_tokens == offline_tokens
        recovered = result.final_tokens != result.steady_tokens
        exact_count += int(exact)
        recovered_count += int(recovered)
        canary_ok = canary_ok and exact
        status = "PASS" if exact else "FAIL"
        print(
            f"  [{status}] idx={sample_index} id={ex.get('sample_id')} "
            f"steady/final/oracle/offline tok={len(result.steady_tokens)}/"
            f"{len(result.final_tokens)}/{len(oracle.final_tokens)}/{len(offline_tokens)} "
            f"token_exact={exact} offline_full_greedy_exact={offline_exact} "
            f"recovered={recovered} "
            f"T={result.meta['final_chunk_T']} drop={result.meta['drop_extra']}"
        )
        if not exact:
            print(f"    residual: {_first_token_diff(result.final_tokens, oracle.final_tokens)}")
        print(f"    steady  : {result.steady_text!r}")
        print(f"    finalize: {result.final_text!r}")
        print(f"    oracle  : {oracle.final_text!r}")
        print(f"    offline_full_greedy diagnostic: {offline_text!r}")

    print(
        f"  summary: canary={'PASS' if canary_ok else 'FAIL'} "
        f"token_exact={exact_count}/{len(CANARY_INDICES)} "
        f"visible_recovery={recovered_count}/{len(CANARY_INDICES)}"
    )

    print("\nparent-byte-identical after speculative finish:")
    try:
        parent_ok = _run_parent_after_speculative_assert(rt, load_wav(ds[CANARY_INDICES[0]]))
        print(f"  [{'PASS' if parent_ok else 'FAIL'}] parent ASR state unchanged after _finish_speculative")
    except Exception as exc:
        parent_ok = False
        print(f"  [FAIL] {type(exc).__name__}: {exc}")

    turn_a = load_wav(ds[CANARY_INDICES[0]])
    turn_b = load_wav(ds[CANARY_INDICES[1]])

    print("\nreset/resume:")
    try:
        reset_ok, reset_detail = _run_reset_resume_test(rt, turn_a, turn_b)
        print(f"  [{'PASS' if reset_ok else 'FAIL'}] {reset_detail}")
    except Exception as exc:
        reset_ok = False
        print(f"  [FAIL] {type(exc).__name__}: {exc}")

    print("\nvad_start cancellation:")
    try:
        vad_ok, vad_detail = _run_vad_start_cancel_test(rt, turn_a, turn_b)
        print(f"  [{'PASS' if vad_ok else 'FAIL'}] {vad_detail}")
    except Exception as exc:
        vad_ok = False
        print(f"  [FAIL] {type(exc).__name__}: {exc}")

    print("\nresidual-frame boundary sweep:")
    try:
        boundary_ok, boundary_detail = _run_boundary_sweep(rt, turn_a)
        print(f"  [{'PASS' if boundary_ok else 'FAIL'}] {boundary_detail}")
    except Exception as exc:
        boundary_ok = False
        print(f"  [FAIL] {type(exc).__name__}: {exc}")

    ok = canary_ok and parent_ok and reset_ok and vad_ok and boundary_ok
    print(f"\n=== FINALIZE_REF {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if validation_gate() else 1)
