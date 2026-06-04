from __future__ import annotations

import dataclasses
import importlib.util
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
TBS_PATH = REPO / "proj-2026-05-21-0410" / "test_batch_state.py"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_speech.cudagraph_encoder import (  # noqa: E402
    BucketedCudaGraphEncoder,
    EncoderGraphInputs,
    FinalizeEncoderGraphKey,
    encoder_stream_step_restoring_drop_extra,
)


DEFAULT_K = int(os.environ.get("TEST_CUDAGRAPH_FINALIZE_MAX_B", "4"))
CAPTURE_WARMUP_ITERS = int(os.environ.get("TEST_CUDAGRAPH_FINALIZE_WARMUPS", "5"))
FINALIZE_T_VALUES = tuple(
    int(value)
    for value in os.environ.get("TEST_CUDAGRAPH_FINALIZE_T", "44,51,58").split(",")
    if value.strip()
)


@dataclasses.dataclass
class CompareResult:
    encoded_bit_equal: bool = True
    encoded_len_bit_equal: bool = True
    state_bit_equal: bool = True
    tensor_count: int = 0
    encoded_max_abs: float = 0.0
    state_max_abs: float = 0.0
    max_path: str = ""
    diffs: list[str] = dataclasses.field(default_factory=list)

    @property
    def hard_pass(self) -> bool:
        return (
            self.encoded_bit_equal
            and self.encoded_len_bit_equal
            and self.state_bit_equal
            and self.encoded_max_abs == 0.0
            and self.state_max_abs == 0.0
        )


def _import_harness() -> Any:
    spec = importlib.util.spec_from_file_location("cudagraph_finalize_tbs", TBS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import harness: {TBS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _enforce_probe_environment() -> None:
    os.environ["NEMOTRON_WARMUP_MS"] = "200"
    os.environ["NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE"] = "1"
    torch.backends.cudnn.benchmark = False


def _load_model(tbs: Any) -> Any:
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf

    _enforce_probe_environment()
    model = nemo_asr.models.ASRModel.restore_from(tbs.EN_NEMO, map_location="cuda")
    model.encoder.set_default_att_context_size([70, 1])
    decoding_cfg = OmegaConf.create(
        {
            "strategy": "greedy",
            "greedy": {
                "max_symbols": 10,
                "loop_labels": False,
                "use_cuda_graph_decoder": False,
            },
        }
    )
    try:
        model.change_decoding_strategy(decoding_cfg=decoding_cfg, verbose=False)
    except TypeError:
        model.change_decoding_strategy(decoding_cfg=decoding_cfg)
    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    return model


def _select_clip_ids(tbs: Any, *, max_b: int) -> list[str]:
    con = sqlite3.connect(tbs.DB)
    try:
        rows = con.execute(
            "SELECT sample_id, audio_path, dataset_index FROM samples "
            "WHERE language='eng' ORDER BY dataset_index"
        ).fetchall()
    finally:
        con.close()

    selected: list[tuple[int, str]] = []
    for sample_id, audio_path, dataset_index in rows:
        audio_file = REPO / "stt-benchmark" / str(audio_path)
        if audio_file.exists():
            selected.append((int(dataset_index), str(sample_id)))
        if len(selected) >= max_b:
            break
    if len(selected) < max_b:
        raise RuntimeError(f"only found {len(selected)} local English clips; need {max_b}")
    return [sample_id for _idx, sample_id in sorted(selected)[:max_b]]


def _advance_raw_ring(geom: Any, raw_ring: np.ndarray, consumed_audio: np.ndarray) -> np.ndarray:
    if len(consumed_audio) >= geom.raw_audio_ring_samples:
        return consumed_audio[-geom.raw_audio_ring_samples :].copy()
    if len(consumed_audio) > 0:
        keep = geom.raw_audio_ring_samples - len(consumed_audio)
        return np.concatenate([raw_ring[-keep:], consumed_audio]).astype(np.float32, copy=False)
    return raw_ring


def _build_finalize_chunk_mel(
    model: Any,
    tbs: Any,
    geom: Any,
    state: Any,
    *,
    time_steps: int,
) -> torch.Tensor:
    if state.mel_frame_ring is None:
        raise RuntimeError("finalize test expects a non-first state with a mel pre-encode ring")
    prefix_frames = int(state.mel_frame_ring.shape[-1])
    remaining_frames = int(time_steps) - prefix_frames
    if remaining_frames <= 0:
        raise ValueError(
            f"finalize T={time_steps} must exceed pre-encode prefix={prefix_frames}"
        )

    pending = np.zeros(max(0, remaining_frames - 1) * int(geom.hop_samples), dtype=np.float32)
    raw_ring = state.raw_audio_ring.copy()
    frames_collected = 0
    new_mels: list[torch.Tensor] = []
    while frames_collected < remaining_frames:
        frames_this_call = min(
            int(geom.shift_frames),
            remaining_frames - frames_collected,
        )
        needed_new_samples = min(len(pending), int(geom.preprocess_new_audio_samples))
        new_audio = pending[:needed_new_samples]
        fixed_audio, valid_samples = tbs.build_fixed_preprocess_audio(
            geom,
            raw_ring,
            new_audio,
        )
        mel, _mel_len = tbs.preprocess_fixed_audio(model, fixed_audio, valid_samples)
        start = int(geom.first_preprocess_mel_frame)
        new_mels.append(mel[:, :, start : start + frames_this_call])

        if frames_this_call == int(geom.shift_frames):
            consumed_samples = min(
                int(geom.shift_frames) * int(geom.hop_samples),
                len(pending),
            )
            consumed_audio = pending[:consumed_samples]
            raw_ring = _advance_raw_ring(geom, raw_ring, consumed_audio)
            pending = pending[consumed_samples:]
        frames_collected += frames_this_call

    new_mel = torch.cat(new_mels, dim=-1)
    chunk_mel = torch.cat((state.mel_frame_ring, new_mel), dim=-1)
    assert int(chunk_mel.shape[-1]) == int(time_steps)
    return chunk_mel


def _build_finalize_inputs(
    model: Any,
    tbs: Any,
    geom: Any,
    states: list[Any],
    *,
    batch_size: int,
    time_steps: int,
) -> EncoderGraphInputs:
    chunk_mels = [
        _build_finalize_chunk_mel(model, tbs, geom, state, time_steps=time_steps)
        for state in states[:batch_size]
    ]
    processed_signal, processed_signal_length = tbs.stack_processed(chunk_mels)
    cache_last_channel, cache_last_time, cache_last_channel_len = tbs.stack_caches(
        [
            (
                state.cache_last_channel,
                state.cache_last_time,
                state.cache_last_channel_len,
            )
            for state in states[:batch_size]
        ]
    )
    return EncoderGraphInputs(
        processed_signal=processed_signal,
        processed_signal_length=processed_signal_length,
        cache_last_channel=cache_last_channel,
        cache_last_time=cache_last_time,
        cache_last_channel_len=cache_last_channel_len,
    )


def _clone_inputs(inputs: EncoderGraphInputs) -> EncoderGraphInputs:
    return EncoderGraphInputs(
        processed_signal=inputs.processed_signal.detach().clone(),
        processed_signal_length=inputs.processed_signal_length.detach().clone(),
        cache_last_channel=inputs.cache_last_channel.detach().clone(),
        cache_last_time=inputs.cache_last_time.detach().clone(),
        cache_last_channel_len=inputs.cache_last_channel_len.detach().clone(),
    )


def _tensor_max_abs(expected: torch.Tensor, actual: torch.Tensor) -> float:
    if expected.numel() == 0 and actual.numel() == 0:
        return 0.0
    return float((expected.to(torch.float32) - actual.to(torch.float32)).abs().max().item())


def _compare_tensor(
    result: CompareResult,
    path: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    state_tensor: bool,
) -> None:
    result.tensor_count += 1
    if expected.shape != actual.shape:
        result.diffs.append(f"{path}: shape mismatch {tuple(expected.shape)} != {tuple(actual.shape)}")
        if state_tensor:
            result.state_bit_equal = False
        elif path == "encoded_len":
            result.encoded_len_bit_equal = False
        else:
            result.encoded_bit_equal = False
        return
    if expected.dtype != actual.dtype:
        result.diffs.append(f"{path}: dtype mismatch {expected.dtype} != {actual.dtype}")
        if state_tensor:
            result.state_bit_equal = False
        elif path == "encoded_len":
            result.encoded_len_bit_equal = False
        else:
            result.encoded_bit_equal = False
        return

    bit_equal = bool(torch.equal(expected, actual))
    max_abs = _tensor_max_abs(expected, actual)
    if state_tensor:
        result.state_bit_equal = result.state_bit_equal and bit_equal
        if max_abs > result.state_max_abs:
            result.state_max_abs = max_abs
            result.max_path = path
    elif path == "encoded_len":
        result.encoded_len_bit_equal = result.encoded_len_bit_equal and bit_equal
        if max_abs > result.encoded_max_abs:
            result.encoded_max_abs = max_abs
            result.max_path = path
    else:
        result.encoded_bit_equal = result.encoded_bit_equal and bit_equal
        if max_abs > result.encoded_max_abs:
            result.encoded_max_abs = max_abs
            result.max_path = path

    if not bit_equal:
        result.diffs.append(f"{path}: bit mismatch max_abs={max_abs:.6e}")


def _compare_encoder_outputs(
    expected: tuple[torch.Tensor, ...],
    actual: tuple[torch.Tensor, ...],
) -> CompareResult:
    result = CompareResult()
    names = (
        "encoded",
        "encoded_len",
        "cache_last_channel",
        "cache_last_time",
        "cache_last_channel_len",
    )
    for idx, name in enumerate(names):
        _compare_tensor(
            result,
            name,
            expected[idx],
            actual[idx],
            state_tensor=idx >= 2,
        )
    return result


@pytest.fixture(scope="module")
def cuda_finalize_context() -> tuple[Any, Any, Any, list[Any]]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for finalize CUDA graph encoder tests")
    if DEFAULT_K < 1:
        pytest.skip("TEST_CUDAGRAPH_FINALIZE_MAX_B must be >= 1")
    if not FINALIZE_T_VALUES:
        pytest.skip("TEST_CUDAGRAPH_FINALIZE_T did not provide any T buckets")

    tbs = _import_harness()
    model = _load_model(tbs)
    geom = tbs.build_geometry(model)
    min_t = min(FINALIZE_T_VALUES)
    if min_t <= int(geom.pre_encode_cache_size):
        raise RuntimeError(
            f"finalize T buckets {FINALIZE_T_VALUES} must exceed pre_cache={geom.pre_encode_cache_size}"
        )
    selected_ids = _select_clip_ids(tbs, max_b=DEFAULT_K)
    clips = {sample_id: tbs.load_clip(sample_id) for sample_id in selected_ids}
    states = [
        tbs.init_state(model, geom, sample_id, clips[sample_id])
        for sample_id in selected_ids
    ]
    assert all(state.mel_frame_ring is not None for state in states)
    print(
        "CUDA_GRAPH_FINALIZE_TEST_CONTEXT "
        f"K={DEFAULT_K} T={FINALIZE_T_VALUES} "
        f"pre_cache={geom.pre_encode_cache_size} drop_extra={geom.drop_extra} "
        f"sample_ids={','.join(selected_ids)}"
    )
    return model, tbs, geom, states


def test_finalize_cuda_graph_encoder_byte_exact_per_captured_bucket(
    cuda_finalize_context: tuple[Any, ...],
) -> None:
    model, tbs, geom, states = cuda_finalize_context
    requested_keys = [
        FinalizeEncoderGraphKey(
            batch_size=batch_size,
            time_steps=time_steps,
            drop_extra=int(geom.drop_extra),
            keep_all_outputs=True,
        )
        for batch_size in range(1, DEFAULT_K + 1)
        for time_steps in FINALIZE_T_VALUES
    ]

    disabled_manager = BucketedCudaGraphEncoder.warmup_finalize(
        model,
        requested_keys[:1],
        warmup_iters=0,
        enable_finalize=False,
    )
    disabled_inputs = _build_finalize_inputs(
        model,
        tbs,
        geom,
        states,
        batch_size=1,
        time_steps=requested_keys[0].time_steps,
    )
    assert disabled_manager.captured_finalize_keys == ()
    assert disabled_manager.replay_finalize(requested_keys[0], disabled_inputs) is None
    assert "not enabled" in (disabled_manager.finalize_capture_error(requested_keys[0]) or "")

    manager = BucketedCudaGraphEncoder.warmup_finalize(
        model,
        requested_keys,
        warmup_iters=CAPTURE_WARMUP_ITERS,
        enable_finalize=True,
    )
    print(
        "CUDA_GRAPH_FINALIZE_CAPTURED "
        f"captured={manager.captured_finalize_keys} "
        f"uncaptured={manager.uncaptured_finalize_keys} "
        f"capture_ms={[manager.finalize_capture_ms(key) for key in manager.captured_finalize_keys]} "
        f"memory={manager.finalize_capture_memory_bytes()}"
    )
    assert tuple(sorted(requested_keys)) == manager.captured_finalize_keys, (
        "not all requested finalize buckets captured: "
        f"captured={manager.captured_finalize_keys} "
        f"uncaptured={manager.uncaptured_finalize_keys} "
        f"errors={[manager.finalize_capture_error(key) for key in manager.uncaptured_finalize_keys]}"
    )

    for key in manager.captured_finalize_keys:
        inputs = _build_finalize_inputs(
            model,
            tbs,
            geom,
            states,
            batch_size=key.batch_size,
            time_steps=key.time_steps,
        )
        eager_inputs = _clone_inputs(inputs)
        graph_inputs = _clone_inputs(inputs)

        with torch.inference_mode():
            eager_outputs = encoder_stream_step_restoring_drop_extra(
                model,
                processed_signal=eager_inputs.processed_signal,
                processed_signal_length=eager_inputs.processed_signal_length,
                cache_last_channel=eager_inputs.cache_last_channel,
                cache_last_time=eager_inputs.cache_last_time,
                cache_last_channel_len=eager_inputs.cache_last_channel_len,
                keep_all_outputs=True,
                drop_extra_pre_encoded=key.drop_extra,
            )
            graph_static_outputs = manager.replay_finalize(key, graph_inputs)
        assert graph_static_outputs is not None, (
            f"captured finalize key={key} unexpectedly returned eager fallback: "
            f"{manager.finalize_replay_error(key)}"
        )
        graph_outputs = tuple(tensor.detach().clone() for tensor in graph_static_outputs)
        result = _compare_encoder_outputs(eager_outputs, graph_outputs)
        print(
            "CUDA_GRAPH_FINALIZE_BYTE_EXACT "
            f"B={key.batch_size} T={key.time_steps} drop_extra={key.drop_extra} "
            f"encoded_max_abs={result.encoded_max_abs:.6e} "
            f"state_max_abs={result.state_max_abs:.6e} "
            f"tensor_count={result.tensor_count} replays={manager.finalize_replays(key)} "
            f"mem={manager.finalize_capture_memory_bytes_for_key(key)}"
        )
        assert result.hard_pass, (
            f"finalize graph replay differed from eager for key={key}: "
            f"encoded_bit_equal={result.encoded_bit_equal} "
            f"encoded_len_bit_equal={result.encoded_len_bit_equal} "
            f"state_bit_equal={result.state_bit_equal} "
            f"encoded_max_abs={result.encoded_max_abs:.6e} "
            f"state_max_abs={result.state_max_abs:.6e} "
            f"max_path={result.max_path or 'n/a'} "
            f"diffs={result.diffs[:8]}"
        )

    fallback_t = max(FINALIZE_T_VALUES) + 1
    fallback_key = FinalizeEncoderGraphKey(
        batch_size=1,
        time_steps=fallback_t,
        drop_extra=int(geom.drop_extra),
        keep_all_outputs=True,
    )
    fallback_inputs = _build_finalize_inputs(
        model,
        tbs,
        geom,
        states,
        batch_size=1,
        time_steps=fallback_t,
    )
    assert manager.finalize_captured(fallback_key) is False
    assert manager.replay_finalize(fallback_key, fallback_inputs) is None
    print(
        "CUDA_GRAPH_FINALIZE_FALLBACK "
        f"B={fallback_key.batch_size} T={fallback_key.time_steps} replay=None captured=False"
    )
