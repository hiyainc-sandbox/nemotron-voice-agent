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
    encoder_stream_step_restoring_drop_extra,
)


DEFAULT_K = int(os.environ.get("TEST_CUDAGRAPH_MAX_B", "16"))
CAPTURE_WARMUP_ITERS = int(os.environ.get("TEST_CUDAGRAPH_WARMUPS", "5"))
MIN_STEADY_CHUNKS = int(os.environ.get("TEST_CUDAGRAPH_MIN_STEADY_CHUNKS", "60"))
FORCED_CHUNK_COUNT = os.environ.get("TEST_CUDAGRAPH_CHUNK_COUNT")


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
    spec = importlib.util.spec_from_file_location("cudagraph_encoder_tbs", TBS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import harness: {TBS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _enforce_probe_environment() -> None:
    os.environ["NEMOTRON_WARMUP_MS"] = "200"
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


def _normal_chunk_count(tbs: Any, geom: Any, total_audio_samples: int) -> int:
    emitted_frames = int(geom.warmup_frames or 0)
    synthetic_prefix_samples = emitted_frames * int(geom.hop_samples)
    pending_audio_len = int(total_audio_samples)
    count = 0
    while tbs.ready_predicate(
        synthetic_prefix_samples=synthetic_prefix_samples,
        total_audio_samples=int(total_audio_samples),
        emitted_frames=emitted_frames,
        shift_frames=int(geom.shift_frames),
        hop_samples=int(geom.hop_samples),
        pending_audio_len=pending_audio_len,
        preprocess_new_audio_samples=int(geom.preprocess_new_audio_samples),
    ):
        count += 1
        emitted_frames += int(geom.shift_frames)
        pending_audio_len = max(
            0,
            pending_audio_len - int(geom.shift_frames) * int(geom.hop_samples),
        )
    return count


def _select_clip_ids(tbs: Any, geom: Any, *, max_b: int) -> tuple[list[str], int]:
    con = sqlite3.connect(tbs.DB)
    try:
        rows = con.execute(
            "SELECT sample_id, audio_path, dataset_index FROM samples "
            "WHERE language='eng' ORDER BY dataset_index"
        ).fetchall()
    finally:
        con.close()

    groups: dict[int, list[tuple[int, str]]] = {}
    for sample_id, audio_path, dataset_index in rows:
        audio_file = REPO / "stt-benchmark" / str(audio_path)
        if not audio_file.exists():
            continue
        total_audio_samples = audio_file.stat().st_size // np.dtype(np.int16).itemsize
        chunks = _normal_chunk_count(tbs, geom, total_audio_samples)
        groups.setdefault(chunks, []).append((int(dataset_index), str(sample_id)))

    if FORCED_CHUNK_COUNT is not None:
        chunk_count = int(FORCED_CHUNK_COUNT)
        selected_group = groups.get(chunk_count, [])
        if len(selected_group) < max_b:
            raise RuntimeError(
                f"forced chunk count {chunk_count} has {len(selected_group)} clips; need {max_b}"
            )
    else:
        candidates = [
            (chunks, clips)
            for chunks, clips in groups.items()
            if chunks >= MIN_STEADY_CHUNKS and len(clips) >= max_b
        ]
        if not candidates:
            candidates = [
                (chunks, clips)
                for chunks, clips in groups.items()
                if len(clips) >= max_b
            ]
        if not candidates:
            best = max((len(clips), chunks) for chunks, clips in groups.items())
            raise RuntimeError(f"no chunk-count group has {max_b} clips; best={best}")
        chunk_count, selected_group = sorted(candidates, key=lambda item: item[0])[0]

    selected = [sample_id for _idx, sample_id in sorted(selected_group)[:max_b]]
    return selected, int(chunk_count)


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


def _cache_layout_diagnostic(
    tbs: Any,
    expected: tuple[torch.Tensor, ...],
    actual: tuple[torch.Tensor, ...],
    *,
    batch_size: int,
) -> str:
    lines = [
        "cache layout diagnostic:",
        "  expected stack_caches layout: channel/time concat on dim 1, len concat on dim 0",
        f"  eager cache_last_channel shape={tuple(expected[2].shape)} graph={tuple(actual[2].shape)}",
        f"  eager cache_last_time shape={tuple(expected[3].shape)} graph={tuple(actual[3].shape)}",
        f"  eager cache_last_channel_len shape={tuple(expected[4].shape)} graph={tuple(actual[4].shape)}",
    ]
    for row in range(batch_size):
        exp_row = tbs.scatter_cache_row(expected[2], expected[3], expected[4], row)
        act_row = tbs.scatter_cache_row(actual[2], actual[3], actual[4], row)
        row_parts = []
        for name, exp_tensor, act_tensor in zip(
            ("clc", "clt", "clcl"),
            exp_row,
            act_row,
        ):
            row_parts.append(f"{name}_max_abs={_tensor_max_abs(exp_tensor, act_tensor):.6e}")
        lines.append(f"  row={row}: " + " ".join(row_parts))
    return "\n".join(lines)


def _build_inputs(
    tbs: Any,
    states: list[Any],
    row_prepared: list[tuple[torch.Tensor, torch.Tensor, int]],
    *,
    batch_size: int,
) -> tuple[EncoderGraphInputs, int]:
    prepared = row_prepared[:batch_size]
    drop_values = [int(item[2]) for item in prepared]
    if len(set(drop_values)) != 1:
        raise RuntimeError(f"mixed drop_extra in test batch: {drop_values}")

    processed_signal, processed_signal_length = tbs.stack_processed([item[0] for item in prepared])
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
    return (
        EncoderGraphInputs(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        ),
        drop_values[0],
    )


@pytest.fixture(scope="module")
def cuda_encoder_context() -> tuple[Any, Any, Any, list[str], list[Any], list[tuple[torch.Tensor, torch.Tensor, int]]]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for CUDA graph encoder tests")

    tbs = _import_harness()
    model = _load_model(tbs)
    geom = tbs.build_geometry(model)
    selected_ids, selected_chunk_count = _select_clip_ids(tbs, geom, max_b=DEFAULT_K)
    clips = {sample_id: tbs.load_clip(sample_id) for sample_id in selected_ids}
    states = [
        tbs.init_state(model, geom, sample_id, clips[sample_id])
        for sample_id in selected_ids
    ]
    assert all(tbs.state_is_ready(geom, state) for state in states)
    row_prepared = [tbs.prepare_row(model, geom, state) for state in states]
    steady_T = int(geom.pre_encode_cache_size + geom.shift_frames)
    assert all(int(item[0].shape[-1]) == steady_T for item in row_prepared)
    assert all(int(item[2]) == int(geom.drop_extra) for item in row_prepared)
    print(
        "CUDA_GRAPH_TEST_CONTEXT "
        f"K={DEFAULT_K} selected_chunk_count={selected_chunk_count} "
        f"steady_T={steady_T} drop_extra={geom.drop_extra} "
        f"sample_ids={','.join(selected_ids)}"
    )
    return model, tbs, geom, selected_ids, states, row_prepared


def test_bucketed_cuda_graph_encoder_byte_exact_per_captured_b(cuda_encoder_context: tuple[Any, ...]) -> None:
    model, tbs, geom, _selected_ids, states, row_prepared = cuda_encoder_context
    manager = BucketedCudaGraphEncoder.warmup(
        model,
        DEFAULT_K,
        warmup_iters=CAPTURE_WARMUP_ITERS,
    )

    print(
        "CUDA_GRAPH_CAPTURED "
        f"captured={manager.captured_batch_sizes} "
        f"uncaptured={manager.uncaptured_batch_sizes} "
        f"capture_ms={[manager.capture_ms(B) for B in manager.captured_batch_sizes]}"
    )
    captured = list(manager.captured_batch_sizes)
    assert captured, "no CUDA graph encoder buckets captured"

    for batch_size in range(1, DEFAULT_K + 1):
        if not manager.captured(batch_size):
            print(
                "CUDA_GRAPH_UNCAPTURED "
                f"B={batch_size} reason={manager.capture_error(batch_size)}"
            )
            continue

        inputs, drop_extra = _build_inputs(
            tbs,
            states,
            row_prepared,
            batch_size=batch_size,
        )
        assert int(drop_extra) == int(geom.drop_extra)
        assert int(inputs.processed_signal.shape[-1]) == int(manager.steady_T)

        with torch.inference_mode():
            eager_outputs = encoder_stream_step_restoring_drop_extra(
                model,
                processed_signal=inputs.processed_signal,
                processed_signal_length=inputs.processed_signal_length,
                cache_last_channel=inputs.cache_last_channel,
                cache_last_time=inputs.cache_last_time,
                cache_last_channel_len=inputs.cache_last_channel_len,
                keep_all_outputs=False,
                drop_extra_pre_encoded=drop_extra,
            )
            graph_static_outputs = manager.replay(batch_size, inputs)
        assert graph_static_outputs is not None, (
            f"captured B={batch_size} unexpectedly returned eager fallback: "
            f"{manager.replay_error(batch_size)}"
        )
        graph_outputs = tuple(tensor.detach().clone() for tensor in graph_static_outputs)
        result = _compare_encoder_outputs(eager_outputs, graph_outputs)
        print(
            "CUDA_GRAPH_BYTE_EXACT "
            f"B={batch_size} encoded_max_abs={result.encoded_max_abs:.6e} "
            f"state_max_abs={result.state_max_abs:.6e} "
            f"tensor_count={result.tensor_count} replays={manager.replays(batch_size)}"
        )
        assert result.hard_pass, (
            f"B={batch_size} graph encoder replay differed from eager encoder: "
            f"encoded_bit_equal={result.encoded_bit_equal} "
            f"encoded_len_bit_equal={result.encoded_len_bit_equal} "
            f"state_bit_equal={result.state_bit_equal} "
            f"encoded_max_abs={result.encoded_max_abs:.6e} "
            f"state_max_abs={result.state_max_abs:.6e} "
            f"max_path={result.max_path or 'n/a'} "
            f"diffs={result.diffs[:8]}\n"
            + _cache_layout_diagnostic(
                tbs,
                eager_outputs,
                graph_outputs,
                batch_size=batch_size,
            )
        )

    fallback_inputs, _drop_extra = _build_inputs(
        tbs,
        states,
        row_prepared,
        batch_size=DEFAULT_K,
    )
    assert manager.captured(DEFAULT_K + 1) is False
    assert manager.replay(DEFAULT_K + 1, fallback_inputs) is None
    print(f"CUDA_GRAPH_FALLBACK B={DEFAULT_K + 1} replay=None captured=False")
