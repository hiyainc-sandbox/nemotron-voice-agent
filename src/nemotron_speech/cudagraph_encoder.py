"""Bucketed manual CUDA-graph capture for the steady streaming encoder.

This module only graphs ``model.encoder.cache_aware_stream_step``. RNNT/CTC
decode stays eager and remains the caller's responsibility.

The bucket manager captures one graph per exact batch size ``B`` in ``1..K``.
It does not pad smaller batches into larger buckets. Static buffer layouts match
``batch_primitives.stack_caches``:

* ``processed_signal``: ``[B, F, T]``
* ``cache_last_channel``: ``[layers, B, cache_T, d_model]``
* ``cache_last_time``: ``[layers, B, d_model, time_T]``
* ``cache_last_channel_len``: ``[B]``

``replay`` returns tensors owned by the static CUDA-graph output pool. Callers
that need to retain results after another replay must clone/detach them before
the next replay.
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import os
import time
from typing import Any, Iterable, Optional, Sequence

import torch


EncoderGraphOutputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclasses.dataclass(frozen=True)
class EncoderGraphInputs:
    """Input tensors for one exact steady encoder bucket."""

    processed_signal: torch.Tensor
    processed_signal_length: torch.Tensor
    cache_last_channel: torch.Tensor
    cache_last_time: torch.Tensor
    cache_last_channel_len: torch.Tensor


@dataclasses.dataclass(frozen=True, order=True)
class FinalizeEncoderGraphKey:
    """Exact finalize encoder CUDA-graph bucket key."""

    batch_size: int
    time_steps: int
    drop_extra: int
    keep_all_outputs: bool = True


def encoder_stream_step_restoring_drop_extra(model: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Call ``encoder.cache_aware_stream_step`` and restore NeMo's global drop setting."""

    streaming_cfg = model.encoder.streaming_cfg
    original_drop_extra = streaming_cfg.drop_extra_pre_encoded
    try:
        return model.encoder.cache_aware_stream_step(**kwargs)
    finally:
        streaming_cfg.drop_extra_pre_encoded = original_drop_extra


def _streaming_cfg_int(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return int(value[1])
    return int(value)


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def encoder_finalize_cudagraph_requested() -> bool:
    """Return whether finalize encoder CUDA graphs are requested by env flag."""

    return _env_flag("NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE", default=False)


def _coerce_inputs(inputs: EncoderGraphInputs | Sequence[torch.Tensor]) -> Optional[EncoderGraphInputs]:
    if isinstance(inputs, EncoderGraphInputs):
        return inputs
    try:
        values = tuple(inputs)
    except Exception:
        return None
    if len(values) != 5 or not all(torch.is_tensor(item) for item in values):
        return None
    return EncoderGraphInputs(
        processed_signal=values[0],
        processed_signal_length=values[1],
        cache_last_channel=values[2],
        cache_last_time=values[3],
        cache_last_channel_len=values[4],
    )


def _coerce_finalize_key(
    key_or_batch_size: FinalizeEncoderGraphKey | Sequence[Any] | Any,
    time_steps: Any = None,
    drop_extra: Any = None,
) -> Optional[FinalizeEncoderGraphKey]:
    if isinstance(key_or_batch_size, FinalizeEncoderGraphKey):
        key = key_or_batch_size
    elif time_steps is None and drop_extra is None:
        try:
            values = tuple(key_or_batch_size)
        except Exception:
            return None
        if len(values) == 3:
            key = FinalizeEncoderGraphKey(
                batch_size=int(values[0]),
                time_steps=int(values[1]),
                drop_extra=int(values[2]),
                keep_all_outputs=True,
            )
        elif len(values) == 4:
            key = FinalizeEncoderGraphKey(
                batch_size=int(values[0]),
                time_steps=int(values[1]),
                drop_extra=int(values[2]),
                keep_all_outputs=bool(values[3]),
            )
        else:
            return None
    else:
        try:
            key = FinalizeEncoderGraphKey(
                batch_size=int(key_or_batch_size),
                time_steps=int(time_steps),
                drop_extra=int(drop_extra),
                keep_all_outputs=True,
            )
        except Exception:
            return None

    if (
        int(key.batch_size) < 1
        or int(key.time_steps) < 1
        or not bool(key.keep_all_outputs)
    ):
        return None
    return FinalizeEncoderGraphKey(
        batch_size=int(key.batch_size),
        time_steps=int(key.time_steps),
        drop_extra=int(key.drop_extra),
        keep_all_outputs=True,
    )


class _CudaGraphEncoderBucket:
    """One captured graph and its static input/output buffers for an exact encoder bucket."""

    def __init__(
        self,
        model: Any,
        *,
        batch_size: int,
        steady_T: Optional[int] = None,
        time_steps: Optional[int] = None,
        drop_extra: int,
        keep_all_outputs: bool = False,
        warmup_iters: int,
    ) -> None:
        self.model = model
        self.batch_size = int(batch_size)
        if time_steps is None:
            if steady_T is None:
                raise ValueError("encoder CUDA graph bucket requires time_steps or steady_T")
            time_steps = int(steady_T)
        self.steady_T = int(time_steps)
        self.time_steps = int(time_steps)
        self.drop_extra = int(drop_extra)
        self.keep_all_outputs = bool(keep_all_outputs)
        self.replays = 0
        self.capture_ms = 0.0
        self.capture_allocated_bytes = 0
        self.capture_reserved_bytes = 0

        cache = model.encoder.get_initial_cache_state(batch_size=self.batch_size)
        self.device = cache[0].device
        if self.device.type != "cuda":
            raise RuntimeError(f"CUDA graph capture requires CUDA tensors, got {self.device}")

        feat = int(model.cfg.preprocessor.features)
        dtype = cache[0].dtype
        self.static_processed = torch.empty(
            (self.batch_size, feat, self.time_steps),
            device=self.device,
            dtype=dtype,
        )
        self.static_processed.zero_()
        self.static_len = torch.full(
            (self.batch_size,),
            self.time_steps,
            device=self.device,
            dtype=torch.long,
        )
        self.static_clc = torch.empty_like(cache[0])
        self.static_clt = torch.empty_like(cache[1])
        self.static_clcl = torch.empty_like(cache[2])
        self.static_clc.zero_()
        self.static_clt.zero_()
        self.static_clcl.zero_()

        self.graph = torch.cuda.CUDAGraph()
        self.static_outputs: Optional[EncoderGraphOutputs] = None

        torch.cuda.synchronize(device=self.device)
        mem_before_allocated = torch.cuda.memory_allocated(device=self.device)
        mem_before_reserved = torch.cuda.memory_reserved(device=self.device)
        start = time.perf_counter()
        self._capture(warmup_iters=warmup_iters)
        torch.cuda.synchronize(device=self.device)
        self.capture_ms = (time.perf_counter() - start) * 1000.0
        self.capture_allocated_bytes = max(
            0,
            int(torch.cuda.memory_allocated(device=self.device) - mem_before_allocated),
        )
        self.capture_reserved_bytes = max(
            0,
            int(torch.cuda.memory_reserved(device=self.device) - mem_before_reserved),
        )

    def _call_encoder(self) -> EncoderGraphOutputs:
        outputs = encoder_stream_step_restoring_drop_extra(
            self.model,
            processed_signal=self.static_processed,
            processed_signal_length=self.static_len,
            cache_last_channel=self.static_clc,
            cache_last_time=self.static_clt,
            cache_last_channel_len=self.static_clcl,
            keep_all_outputs=self.keep_all_outputs,
            drop_extra_pre_encoded=self.drop_extra,
        )
        if len(outputs) != 5 or not all(torch.is_tensor(item) for item in outputs):
            raise RuntimeError("encoder graph capture expected exactly 5 tensor outputs")
        return outputs  # type: ignore[return-value]

    def _capture(self, *, warmup_iters: int) -> None:
        with torch.cuda.device(self.device):
            side_stream = torch.cuda.Stream(device=self.device)
            side_stream.wait_stream(torch.cuda.current_stream(device=self.device))
            with torch.inference_mode(), torch.cuda.stream(side_stream):
                for _ in range(int(warmup_iters)):
                    self._call_encoder()

            torch.cuda.current_stream(device=self.device).wait_stream(side_stream)
            torch.cuda.synchronize(device=self.device)

            with torch.inference_mode(), torch.cuda.graph(self.graph):
                self.static_outputs = self._call_encoder()

        if self.static_outputs is None:
            raise RuntimeError("encoder graph capture did not produce static outputs")

    def input_shapes_match(self, inputs: EncoderGraphInputs) -> bool:
        return (
            inputs.processed_signal.shape == self.static_processed.shape
            and inputs.processed_signal_length.shape == self.static_len.shape
            and inputs.cache_last_channel.shape == self.static_clc.shape
            and inputs.cache_last_time.shape == self.static_clt.shape
            and inputs.cache_last_channel_len.shape == self.static_clcl.shape
        )

    def padded_time_input_shapes_match(self, inputs: EncoderGraphInputs) -> bool:
        if inputs.processed_signal.ndim != self.static_processed.ndim:
            return False
        if inputs.processed_signal.shape[:-1] != self.static_processed.shape[:-1]:
            return False
        if int(inputs.processed_signal.shape[-1]) > int(self.static_processed.shape[-1]):
            return False
        return (
            inputs.processed_signal_length.shape == self.static_len.shape
            and inputs.cache_last_channel.shape == self.static_clc.shape
            and inputs.cache_last_time.shape == self.static_clt.shape
            and inputs.cache_last_channel_len.shape == self.static_clcl.shape
        )

    def replay(self, inputs: EncoderGraphInputs) -> EncoderGraphOutputs:
        self.static_processed.copy_(inputs.processed_signal)
        self.static_len.copy_(inputs.processed_signal_length)
        self.static_clc.copy_(inputs.cache_last_channel)
        self.static_clt.copy_(inputs.cache_last_time)
        self.static_clcl.copy_(inputs.cache_last_channel_len)
        self.graph.replay()
        self.replays += 1
        assert self.static_outputs is not None
        return self.static_outputs

    def replay_padded_time(self, inputs: EncoderGraphInputs) -> EncoderGraphOutputs:
        real_time_steps = int(inputs.processed_signal.shape[-1])
        self.static_processed.zero_()
        self.static_processed[..., :real_time_steps].copy_(inputs.processed_signal)
        self.static_len.copy_(inputs.processed_signal_length)
        self.static_clc.copy_(inputs.cache_last_channel)
        self.static_clt.copy_(inputs.cache_last_time)
        self.static_clcl.copy_(inputs.cache_last_channel_len)
        self.graph.replay()
        self.replays += 1
        assert self.static_outputs is not None
        return self.static_outputs


class BucketedCudaGraphEncoder:
    """Fail-closed bucket manager for steady streaming encoder CUDA graphs.

    Use ``BucketedCudaGraphEncoder.warmup(model, K)`` to capture exact buckets
    ``B=1..K``. ``captured(B)`` reports whether a bucket can be replayed.
    ``replay(B, inputs)`` returns static graph outputs for captured, shape-
    matching buckets and returns ``None`` as the use-eager signal for uncaptured
    buckets, ``B > K``, non-positive/invalid B, or mismatched inputs.

    Finalize encoder buckets are optional and default off via
    ``NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE``. They are keyed by exact
    ``(B, T, drop_extra, keep_all_outputs=True)``. The normal
    ``replay_finalize`` path is exact-shape; ``replay_finalize_padded`` pads a
    shorter finalize input into the captured T bucket before replay. Use
    ``warmup_finalize`` or ``capture_finalize`` to capture buckets explicitly.
    """

    def __init__(
        self,
        model: Any,
        *,
        max_batch_size: int,
        warmup_iters: int = 5,
        logger: Optional[logging.Logger] = None,
        enable_finalize: Optional[bool] = None,
    ) -> None:
        self.model = model
        self.max_batch_size = max(0, int(max_batch_size))
        self.warmup_iters = max(0, int(warmup_iters))
        self.logger = logger or logging.getLogger(__name__)
        self.finalize_cudagraph_requested = (
            encoder_finalize_cudagraph_requested()
            if enable_finalize is None
            else bool(enable_finalize)
        )

        streaming_cfg = model.encoder.streaming_cfg
        self.shift_frames = _streaming_cfg_int(streaming_cfg.shift_size)
        self.pre_encode_cache_size = _streaming_cfg_int(streaming_cfg.pre_encode_cache_size)
        self.steady_T = int(self.pre_encode_cache_size + self.shift_frames)
        self.drop_extra = int(streaming_cfg.drop_extra_pre_encoded)

        self._buckets: dict[int, _CudaGraphEncoderBucket] = {}
        self._capture_errors: dict[int, str] = {}
        self._replay_errors: dict[int, str] = {}
        self._warmed = False
        self._finalize_buckets: dict[FinalizeEncoderGraphKey, _CudaGraphEncoderBucket] = {}
        self._finalize_requested_keys: set[FinalizeEncoderGraphKey] = set()
        self._finalize_capture_errors: dict[FinalizeEncoderGraphKey, str] = {}
        self._finalize_replay_errors: dict[FinalizeEncoderGraphKey, str] = {}
        self._finalize_capture_mem_before: Optional[tuple[int, int]] = None
        self._finalize_capture_mem_after: Optional[tuple[int, int]] = None

    @classmethod
    def warmup(
        cls,
        model: Any,
        K: int,
        *,
        warmup_iters: int = 5,
        logger: Optional[logging.Logger] = None,
    ) -> "BucketedCudaGraphEncoder":
        """Capture buckets ``B=1..K`` and return the fail-closed manager."""

        manager = cls(
            model,
            max_batch_size=int(K),
            warmup_iters=warmup_iters,
            logger=logger,
        )
        manager.capture()
        return manager

    @classmethod
    def warmup_finalize(
        cls,
        model: Any,
        buckets: Iterable[FinalizeEncoderGraphKey | Sequence[Any]],
        *,
        warmup_iters: int = 5,
        logger: Optional[logging.Logger] = None,
        enable_finalize: Optional[bool] = None,
    ) -> "BucketedCudaGraphEncoder":
        """Capture exact finalize buckets and return the fail-closed manager."""

        keys = cls.normalize_finalize_keys(buckets)
        max_batch_size = max((key.batch_size for key in keys), default=0)
        manager = cls(
            model,
            max_batch_size=max_batch_size,
            warmup_iters=warmup_iters,
            logger=logger,
            enable_finalize=enable_finalize,
        )
        manager.capture_finalize(keys)
        return manager

    @staticmethod
    def normalize_finalize_keys(
        buckets: Iterable[FinalizeEncoderGraphKey | Sequence[Any]],
    ) -> tuple[FinalizeEncoderGraphKey, ...]:
        keys: set[FinalizeEncoderGraphKey] = set()
        for bucket in buckets:
            key = _coerce_finalize_key(bucket)
            if key is None:
                raise ValueError(f"invalid finalize CUDA graph bucket key: {bucket!r}")
            keys.add(key)
        return tuple(sorted(keys))

    def capture(self) -> None:
        """Capture all requested buckets, marking failures uncaptured."""

        if self._warmed:
            return
        self._warmed = True

        if not torch.cuda.is_available():
            for batch_size in range(1, self.max_batch_size + 1):
                self._capture_errors[batch_size] = "torch.cuda is unavailable"
            return

        for batch_size in range(1, self.max_batch_size + 1):
            try:
                self._buckets[batch_size] = _CudaGraphEncoderBucket(
                    self.model,
                    batch_size=batch_size,
                    steady_T=self.steady_T,
                    drop_extra=self.drop_extra,
                    warmup_iters=self.warmup_iters,
                )
            except Exception as exc:  # CUDA arch/OOM/capture failures must not escape.
                self._buckets.pop(batch_size, None)
                self._capture_errors[batch_size] = f"{type(exc).__name__}: {exc}"
                self.logger.warning(
                    "encoder_cuda_graph_capture_failed B=%s error=%s",
                    batch_size,
                    self._capture_errors[batch_size],
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def capture_finalize(
        self,
        buckets: Iterable[FinalizeEncoderGraphKey | Sequence[Any]],
    ) -> None:
        """Capture exact finalize buckets, marking failures uncaptured.

        This method is partial-capture aware: a failure for one key removes only
        that key. Later ``replay_finalize`` calls for missing or failed keys
        return ``None`` so the caller can run eager.
        """

        keys = self.normalize_finalize_keys(buckets)
        self._finalize_requested_keys.update(keys)

        if not self.finalize_cudagraph_requested:
            for key in keys:
                self._finalize_capture_errors[key] = (
                    "NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE is not enabled"
                )
            return

        if not torch.cuda.is_available():
            for key in keys:
                self._finalize_capture_errors[key] = "torch.cuda is unavailable"
            return

        device = self.model.encoder.get_initial_cache_state(batch_size=1)[0].device
        self._finalize_capture_mem_before = (
            int(torch.cuda.memory_allocated(device=device)),
            int(torch.cuda.memory_reserved(device=device)),
        )
        for key in keys:
            if key in self._finalize_buckets:
                continue
            try:
                self._finalize_buckets[key] = _CudaGraphEncoderBucket(
                    self.model,
                    batch_size=key.batch_size,
                    time_steps=key.time_steps,
                    drop_extra=key.drop_extra,
                    keep_all_outputs=True,
                    warmup_iters=self.warmup_iters,
                )
            except Exception as exc:  # CUDA arch/OOM/capture failures must not escape.
                self._finalize_buckets.pop(key, None)
                self._finalize_capture_errors[key] = f"{type(exc).__name__}: {exc}"
                self.logger.warning(
                    "encoder_finalize_cuda_graph_capture_failed key=%s error=%s",
                    key,
                    self._finalize_capture_errors[key],
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        self._finalize_capture_mem_after = (
            int(torch.cuda.memory_allocated(device=device)),
            int(torch.cuda.memory_reserved(device=device)),
        )

    def captured(self, batch_size: Any) -> bool:
        """Return ``False`` instead of raising for out-of-range or uncaptured buckets."""

        try:
            batch_size_int = int(batch_size)
        except Exception:
            return False
        if batch_size_int < 1 or batch_size_int > self.max_batch_size:
            return False
        return batch_size_int in self._buckets

    def finalize_captured(
        self,
        key_or_batch_size: FinalizeEncoderGraphKey | Sequence[Any] | Any,
        time_steps: Any = None,
        drop_extra: Any = None,
    ) -> bool:
        """Return ``False`` instead of raising for uncaptured finalize buckets."""

        key = _coerce_finalize_key(key_or_batch_size, time_steps, drop_extra)
        if key is None:
            return False
        return key in self._finalize_buckets

    def replay(
        self,
        batch_size: Any,
        inputs: EncoderGraphInputs | Sequence[torch.Tensor],
    ) -> Optional[EncoderGraphOutputs]:
        """Replay a captured bucket or return ``None`` to signal eager fallback.

        The returned tensors are static graph-owned buffers. Clone/detach them
        before another replay if the values must outlive the next graph call.
        """

        try:
            batch_size_int = int(batch_size)
        except Exception:
            return None
        bucket = self._buckets.get(batch_size_int)
        if bucket is None:
            return None

        coerced_inputs = _coerce_inputs(inputs)
        if coerced_inputs is None:
            self._replay_errors[batch_size_int] = "invalid input container"
            return None
        try:
            input_shapes_match = bucket.input_shapes_match(coerced_inputs)
        except Exception as exc:
            self._replay_errors[batch_size_int] = f"invalid input tensors: {type(exc).__name__}: {exc}"
            return None
        if not input_shapes_match:
            self._replay_errors[batch_size_int] = (
                "input shape mismatch: "
                f"processed={tuple(coerced_inputs.processed_signal.shape)} "
                f"expected={tuple(bucket.static_processed.shape)} "
                f"clc={tuple(coerced_inputs.cache_last_channel.shape)} "
                f"expected_clc={tuple(bucket.static_clc.shape)} "
                f"clt={tuple(coerced_inputs.cache_last_time.shape)} "
                f"expected_clt={tuple(bucket.static_clt.shape)} "
                f"clcl={tuple(coerced_inputs.cache_last_channel_len.shape)} "
                f"expected_clcl={tuple(bucket.static_clcl.shape)}"
            )
            return None

        try:
            with torch.inference_mode():
                return bucket.replay(coerced_inputs)
        except Exception as exc:  # Fail closed: future calls to this bucket use eager.
            self._buckets.pop(batch_size_int, None)
            self._replay_errors[batch_size_int] = f"{type(exc).__name__}: {exc}"
            self.logger.warning(
                "encoder_cuda_graph_replay_failed B=%s error=%s",
                batch_size_int,
                self._replay_errors[batch_size_int],
            )
            return None

    def replay_finalize(
        self,
        key_or_batch_size: FinalizeEncoderGraphKey | Sequence[Any] | Any,
        inputs: EncoderGraphInputs | Sequence[torch.Tensor],
        time_steps: Any = None,
        drop_extra: Any = None,
    ) -> Optional[EncoderGraphOutputs]:
        """Replay a captured finalize bucket or return ``None`` for eager fallback."""

        key = _coerce_finalize_key(key_or_batch_size, time_steps, drop_extra)
        if key is None:
            return None
        bucket = self._finalize_buckets.get(key)
        if bucket is None:
            return None

        coerced_inputs = _coerce_inputs(inputs)
        if coerced_inputs is None:
            self._finalize_replay_errors[key] = "invalid input container"
            return None
        try:
            input_shapes_match = bucket.input_shapes_match(coerced_inputs)
        except Exception as exc:
            self._finalize_replay_errors[key] = f"invalid input tensors: {type(exc).__name__}: {exc}"
            return None
        if not input_shapes_match:
            self._finalize_replay_errors[key] = (
                "input shape mismatch: "
                f"processed={tuple(coerced_inputs.processed_signal.shape)} "
                f"expected={tuple(bucket.static_processed.shape)} "
                f"clc={tuple(coerced_inputs.cache_last_channel.shape)} "
                f"expected_clc={tuple(bucket.static_clc.shape)} "
                f"clt={tuple(coerced_inputs.cache_last_time.shape)} "
                f"expected_clt={tuple(bucket.static_clt.shape)} "
                f"clcl={tuple(coerced_inputs.cache_last_channel_len.shape)} "
                f"expected_clcl={tuple(bucket.static_clcl.shape)}"
            )
            return None

        try:
            with torch.inference_mode():
                return bucket.replay(coerced_inputs)
        except Exception as exc:  # Fail closed: future calls to this key use eager.
            self._finalize_buckets.pop(key, None)
            self._finalize_replay_errors[key] = f"{type(exc).__name__}: {exc}"
            self.logger.warning(
                "encoder_finalize_cuda_graph_replay_failed key=%s error=%s",
                key,
                self._finalize_replay_errors[key],
            )
            return None

    def replay_finalize_padded(
        self,
        key_or_batch_size: FinalizeEncoderGraphKey | Sequence[Any] | Any,
        inputs: EncoderGraphInputs | Sequence[torch.Tensor],
        time_steps: Any = None,
        drop_extra: Any = None,
    ) -> Optional[EncoderGraphOutputs]:
        """Replay a finalize bucket with T padded to the captured static width."""

        key = _coerce_finalize_key(key_or_batch_size, time_steps, drop_extra)
        if key is None:
            return None
        bucket = self._finalize_buckets.get(key)
        if bucket is None:
            return None

        coerced_inputs = _coerce_inputs(inputs)
        if coerced_inputs is None:
            self._finalize_replay_errors[key] = "invalid input container"
            return None
        try:
            input_shapes_match = bucket.padded_time_input_shapes_match(coerced_inputs)
        except Exception as exc:
            self._finalize_replay_errors[key] = f"invalid input tensors: {type(exc).__name__}: {exc}"
            return None
        if not input_shapes_match:
            self._finalize_replay_errors[key] = (
                "padded input shape mismatch: "
                f"processed={tuple(coerced_inputs.processed_signal.shape)} "
                f"expected_prefix={tuple(bucket.static_processed.shape[:-1])} "
                f"expected_T_max={int(bucket.static_processed.shape[-1])} "
                f"clc={tuple(coerced_inputs.cache_last_channel.shape)} "
                f"expected_clc={tuple(bucket.static_clc.shape)} "
                f"clt={tuple(coerced_inputs.cache_last_time.shape)} "
                f"expected_clt={tuple(bucket.static_clt.shape)} "
                f"clcl={tuple(coerced_inputs.cache_last_channel_len.shape)} "
                f"expected_clcl={tuple(bucket.static_clcl.shape)}"
            )
            return None

        try:
            with torch.inference_mode():
                return bucket.replay_padded_time(coerced_inputs)
        except Exception as exc:  # Fail closed: future calls to this key use eager.
            self._finalize_buckets.pop(key, None)
            self._finalize_replay_errors[key] = f"{type(exc).__name__}: {exc}"
            self.logger.warning(
                "encoder_finalize_cuda_graph_replay_failed key=%s mode=padded_T error=%s",
                key,
                self._finalize_replay_errors[key],
            )
            return None

    def capture_error(self, batch_size: Any) -> Optional[str]:
        try:
            return self._capture_errors.get(int(batch_size))
        except Exception:
            return None

    def replay_error(self, batch_size: Any) -> Optional[str]:
        try:
            return self._replay_errors.get(int(batch_size))
        except Exception:
            return None

    def finalize_capture_error(
        self,
        key_or_batch_size: FinalizeEncoderGraphKey | Sequence[Any] | Any,
        time_steps: Any = None,
        drop_extra: Any = None,
    ) -> Optional[str]:
        key = _coerce_finalize_key(key_or_batch_size, time_steps, drop_extra)
        if key is None:
            return None
        return self._finalize_capture_errors.get(key)

    def finalize_replay_error(
        self,
        key_or_batch_size: FinalizeEncoderGraphKey | Sequence[Any] | Any,
        time_steps: Any = None,
        drop_extra: Any = None,
    ) -> Optional[str]:
        key = _coerce_finalize_key(key_or_batch_size, time_steps, drop_extra)
        if key is None:
            return None
        return self._finalize_replay_errors.get(key)

    @property
    def captured_batch_sizes(self) -> tuple[int, ...]:
        return tuple(sorted(self._buckets))

    @property
    def uncaptured_batch_sizes(self) -> tuple[int, ...]:
        return tuple(
            batch_size
            for batch_size in range(1, self.max_batch_size + 1)
            if batch_size not in self._buckets
        )

    @property
    def captured_finalize_keys(self) -> tuple[FinalizeEncoderGraphKey, ...]:
        return tuple(sorted(self._finalize_buckets))

    @property
    def uncaptured_finalize_keys(self) -> tuple[FinalizeEncoderGraphKey, ...]:
        return tuple(
            key
            for key in sorted(self._finalize_requested_keys)
            if key not in self._finalize_buckets
        )

    def finalize_capture_memory_bytes(self) -> dict[str, int]:
        """Return total finalize capture memory delta for this manager."""

        if self._finalize_capture_mem_before is None or self._finalize_capture_mem_after is None:
            return {
                "allocated_bytes": 0,
                "reserved_bytes": 0,
            }
        allocated_before, reserved_before = self._finalize_capture_mem_before
        allocated_after, reserved_after = self._finalize_capture_mem_after
        return {
            "allocated_bytes": max(0, int(allocated_after - allocated_before)),
            "reserved_bytes": max(0, int(reserved_after - reserved_before)),
        }

    def capture_ms(self, batch_size: Any) -> Optional[float]:
        try:
            bucket = self._buckets.get(int(batch_size))
        except Exception:
            return None
        return None if bucket is None else float(bucket.capture_ms)

    def finalize_capture_ms(
        self,
        key_or_batch_size: FinalizeEncoderGraphKey | Sequence[Any] | Any,
        time_steps: Any = None,
        drop_extra: Any = None,
    ) -> Optional[float]:
        key = _coerce_finalize_key(key_or_batch_size, time_steps, drop_extra)
        if key is None:
            return None
        bucket = self._finalize_buckets.get(key)
        return None if bucket is None else float(bucket.capture_ms)

    def finalize_capture_memory_bytes_for_key(
        self,
        key_or_batch_size: FinalizeEncoderGraphKey | Sequence[Any] | Any,
        time_steps: Any = None,
        drop_extra: Any = None,
    ) -> Optional[dict[str, int]]:
        key = _coerce_finalize_key(key_or_batch_size, time_steps, drop_extra)
        if key is None:
            return None
        bucket = self._finalize_buckets.get(key)
        if bucket is None:
            return None
        return {
            "allocated_bytes": int(bucket.capture_allocated_bytes),
            "reserved_bytes": int(bucket.capture_reserved_bytes),
        }

    def replays(self, batch_size: Any) -> int:
        try:
            bucket = self._buckets.get(int(batch_size))
        except Exception:
            return 0
        return 0 if bucket is None else int(bucket.replays)

    def finalize_replays(
        self,
        key_or_batch_size: FinalizeEncoderGraphKey | Sequence[Any] | Any,
        time_steps: Any = None,
        drop_extra: Any = None,
    ) -> int:
        key = _coerce_finalize_key(key_or_batch_size, time_steps, drop_extra)
        if key is None:
            return 0
        bucket = self._finalize_buckets.get(key)
        return 0 if bucket is None else int(bucket.replays)
