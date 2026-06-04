"""WebSocket ASR server for Nemotron-Speech with true incremental streaming."""

import asyncio
import argparse
import concurrent.futures
import copy
import contextlib
import dataclasses
import hashlib
import json
import logging
import os
from collections import Counter, deque
from pathlib import Path
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from aiohttp import ClientConnectionResetError, web, WSMsgType
from loguru import logger

if os.environ.get("NEMOTRON_FAULTHANDLER") == "1":  # debug: `kill -USR1 <pid>` dumps all thread stacks (hang diagnosis)
    import faulthandler as _faulthandler
    import signal as _signal

    _faulthandler.register(_signal.SIGUSR1, all_threads=True, chain=False)

if os.environ.get("NEMOTRON_GC_PROFILE") == "1":  # debug: log GC stop-the-world pauses (P99-stall hypothesis)
    import gc as _gc
    import time as _gctime

    from loguru import logger as _gclogger

    _gc_state: dict = {}

    def _gc_callback(phase, info):
        if phase == "start":
            _gc_state["t"] = _gctime.perf_counter()
        elif phase == "stop":
            dur_ms = (_gctime.perf_counter() - _gc_state.get("t", _gctime.perf_counter())) * 1000.0
            if dur_ms >= 20.0:  # only meaningful pauses
                _gclogger.warning(
                    f"gc_pause gen={info.get('generation')} collected={info.get('collected')} "
                    f"uncollectable={info.get('uncollectable')} dur_ms={dur_ms:.1f}"
                )

    _gc.callbacks.append(_gc_callback)

try:
    from nemotron_speech.batch_primitives import (
        batch_group_key,
        ready_predicate,
        scatter_cache_row,
        stack_caches,
        stack_hypotheses,
        stack_pred_out,
        stack_processed,
    )
    from nemotron_speech.cudagraph_encoder import (
        BucketedCudaGraphEncoder,
        EncoderGraphInputs,
        FinalizeEncoderGraphKey,
    )
except ImportError:  # Allows `python src/nemotron_speech/server.py`.
    from batch_primitives import (
        batch_group_key,
        ready_predicate,
        scatter_cache_row,
        stack_caches,
        stack_hypotheses,
        stack_pred_out,
        stack_processed,
    )
    from cudagraph_encoder import (
        BucketedCudaGraphEncoder,
        EncoderGraphInputs,
        FinalizeEncoderGraphKey,
    )

# Enable debug logging with DEBUG_ASR=1
DEBUG_ASR = os.environ.get("DEBUG_ASR", "0") == "1"

_DEFAULT_FINALIZE_SILENCE_MS = 150
_MAX_FINALIZE_SILENCE_MS = 10_000
_ADMISSION_MAX_BACKLOG_DEFAULT = 1_000_000_000
_ADMISSION_MAX_READY_AGE_MS_DEFAULT = 1_000_000_000_000.0
# Default sliding-window size for the /stats endpoint. ~2k records covers
# ~10 minutes at conc=10 / 1 utterance every 3s. Each sample is ~80 bytes so
# even 16k samples is ~1.3MB — bounded memory.
_STATS_WINDOW_DEFAULT = 2048


def _compute_quantile_summary(values: list[float]) -> dict[str, Any]:
    """Compute p50/p90/p95/p99/max/count for a list of floats.

    Pure function (no self) so it's testable without instantiating ASRServer.
    Empty list → all metrics set to None. Single value → all percentiles
    equal that value. Linear time for the sort (Python timsort); fast enough
    on a 2048-element window (<1ms in practice).
    """
    if not values:
        return {"p50": None, "p90": None, "p95": None, "p99": None, "max": None, "count": 0}
    sorted_values = sorted(values)
    n = len(sorted_values)

    def percentile(p: float) -> float:
        # Nearest-rank (per HAProxy convention; same shape as run_full1000_conc12.py).
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return float(sorted_values[idx])

    return {
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": float(sorted_values[-1]),
        "count": n,
    }

STREAMING = "STREAMING"
PENDING_FINALIZE = "PENDING_FINALIZE"
FINALIZED = "FINALIZED"
_TRUE_BOUNDARY_FINALIZE_REASONS = frozenset({"close", "end"})
# Finalize reasons where the client is actively awaiting a terminal (is_final && finalize)
# response. Such a turn must yield exactly one terminal event even when it produced no text
# (silence / VAD false-trigger); otherwise the client waits forever. "close"/"vad_start*" are
# drains/turn-starts, not client-awaited finalizes, so they never synthesize an empty terminal.
_CLIENT_FINALIZE_REASONS = frozenset({"reset_then_debounce", "debounce_expired", "reset", "end"})
_EOU_PROBE_LOCK = threading.Lock()
_EOU_SNAPSHOT_LOCK = threading.Lock()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got {value!r}") from e


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as e:
        raise ValueError(f"{name} must be a float, got {value!r}") from e


def _telemetry_run_tag() -> str | None:
    for env_name in (
        "NEMOTRON_RUN_TAG",
        "NEMOTRON_TELEMETRY_RUN_TAG",
        "STT_BENCHMARK_RUN_TAG",
    ):
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def _safe_tag_filename(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("._-") or "tag"


def _telemetry_dir() -> Path:
    configured = os.environ.get("NEMOTRON_TELEMETRY_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[2] / "stt-benchmark" / "stt_benchmark_data" / "client_telemetry"


def _hash_audio(audio: np.ndarray) -> str:
    """Get short hash of audio array for debugging."""
    if audio is None or len(audio) == 0:
        return "empty"
    return hashlib.md5(audio.tobytes()).hexdigest()[:8]


def tensor_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone()


def tensor_clone_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clone()


def clone_tree(obj: Any, memo: Optional[dict[int, Any]] = None) -> Any:
    """Tensor-aware deepcopy for disposable ASR fork state."""
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
        cloned_list: list[Any] = []
        memo[oid] = cloned_list
        cloned_list.extend(clone_tree(item, memo) for item in obj)
        return cloned_list
    if isinstance(obj, tuple):
        placeholder: list[Any] = []
        memo[oid] = placeholder
        cloned_tuple = tuple(clone_tree(item, memo) for item in obj)
        memo[oid] = cloned_tuple
        return cloned_tuple
    if isinstance(obj, dict):
        cloned_dict: dict[Any, Any] = {}
        memo[oid] = cloned_dict
        for key, value in obj.items():
            cloned_dict[clone_tree(key, memo)] = clone_tree(value, memo)
        return cloned_dict
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # Covers NeMo LabelLoopingStateItem decoder state under greedy_batch.
        cloned_obj = copy.copy(obj)
        memo[oid] = cloned_obj
        for field in dataclasses.fields(obj):
            setattr(cloned_obj, field.name, clone_tree(getattr(obj, field.name), memo))
        return cloned_obj

    # Some NeMo helper objects may live inside optional fields. Avoid copying
    # modules or other heavy objects; copy plain object state only when present.
    if hasattr(obj, "__dict__") and obj.__class__.__module__.startswith("nemo."):
        cloned_obj = copy.copy(obj)
        memo[oid] = cloned_obj
        for key, value in vars(obj).items():
            setattr(cloned_obj, key, clone_tree(value, memo))
        return cloned_obj

    try:
        return copy.deepcopy(obj, memo)
    except Exception:
        return obj


def clone_hypotheses_deep(previous_hypotheses: Any) -> Any:
    """Deep-copy each Hypothesis and recursively clone tensor decoder state."""
    if previous_hypotheses is None:
        return None
    return [clone_tree(hyp) for hyp in previous_hypotheses]


def snapshot_tree_cpu(obj: Any, memo: Optional[dict[int, Any]] = None) -> Any:
    """Deep-copy snapshot state with all tensors detached and moved to CPU."""
    if memo is None:
        memo = {}

    oid = id(obj)
    if oid in memo:
        return memo[oid]

    if torch.is_tensor(obj):
        return tensor_clone_cpu(obj)
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return obj
    if isinstance(obj, list):
        cloned_list: list[Any] = []
        memo[oid] = cloned_list
        cloned_list.extend(snapshot_tree_cpu(item, memo) for item in obj)
        return cloned_list
    if isinstance(obj, tuple):
        placeholder: list[Any] = []
        memo[oid] = placeholder
        cloned_tuple = tuple(snapshot_tree_cpu(item, memo) for item in obj)
        memo[oid] = cloned_tuple
        return cloned_tuple
    if isinstance(obj, dict):
        cloned_dict: dict[Any, Any] = {}
        memo[oid] = cloned_dict
        for key, value in obj.items():
            cloned_dict[snapshot_tree_cpu(key, memo)] = snapshot_tree_cpu(value, memo)
        return cloned_dict
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # Covers NeMo LabelLoopingStateItem decoder state under greedy_batch.
        cloned_obj = copy.copy(obj)
        memo[oid] = cloned_obj
        for field in dataclasses.fields(obj):
            setattr(cloned_obj, field.name, snapshot_tree_cpu(getattr(obj, field.name), memo))
        return cloned_obj

    if hasattr(obj, "__dict__") and obj.__class__.__module__.startswith("nemo."):
        cloned_obj = copy.copy(obj)
        memo[oid] = cloned_obj
        for key, value in vars(obj).items():
            setattr(cloned_obj, key, snapshot_tree_cpu(value, memo))
        return cloned_obj

    try:
        return copy.deepcopy(obj, memo)
    except Exception:
        return obj


def _tensor_assert_hash(tensor: torch.Tensor) -> str:
    cpu_tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.md5()
    digest.update(str(tuple(cpu_tensor.shape)).encode("utf-8"))
    digest.update(str(cpu_tensor.dtype).encode("utf-8"))
    digest.update(cpu_tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()[:12]


def _array_assert_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.md5()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()[:12]


def _assert_tree_equal(label: str, before: Any, after: Any) -> None:
    if torch.is_tensor(before) or torch.is_tensor(after):
        if not (torch.is_tensor(before) and torch.is_tensor(after)):
            raise AssertionError(f"{label}: tensor/non-tensor mismatch")
        same_meta = (
            tuple(before.shape) == tuple(after.shape)
            and before.dtype == after.dtype
            and before.device == after.device
        )
        same_bytes = same_meta and torch.equal(before, after)
        if not same_bytes:
            raise AssertionError(
                f"{label}: tensor changed "
                f"before(shape={tuple(before.shape)}, dtype={before.dtype}, "
                f"device={before.device}, hash={_tensor_assert_hash(before)}) "
                f"after(shape={tuple(after.shape)}, dtype={after.dtype}, "
                f"device={after.device}, hash={_tensor_assert_hash(after)})"
            )
        return

    if isinstance(before, np.ndarray) or isinstance(after, np.ndarray):
        if not (isinstance(before, np.ndarray) and isinstance(after, np.ndarray)):
            raise AssertionError(f"{label}: ndarray/non-ndarray mismatch")
        if (
            before.shape != after.shape
            or before.dtype != after.dtype
            or not np.array_equal(before, after)
        ):
            raise AssertionError(
                f"{label}: ndarray changed "
                f"before(shape={before.shape}, dtype={before.dtype}, "
                f"hash={_array_assert_hash(before)}) "
                f"after(shape={after.shape}, dtype={after.dtype}, "
                f"hash={_array_assert_hash(after)})"
            )
        return

    if before is None or after is None or isinstance(before, (str, bytes, int, float, bool)):
        if before != after:
            raise AssertionError(f"{label}: value changed from {before!r} to {after!r}")
        return

    if isinstance(before, list) or isinstance(after, list):
        if not (isinstance(before, list) and isinstance(after, list)):
            raise AssertionError(f"{label}: list/non-list mismatch")
        if len(before) != len(after):
            raise AssertionError(f"{label}: list length changed {len(before)} -> {len(after)}")
        for index, (before_item, after_item) in enumerate(zip(before, after)):
            _assert_tree_equal(f"{label}[{index}]", before_item, after_item)
        return

    if isinstance(before, tuple) or isinstance(after, tuple):
        if not (isinstance(before, tuple) and isinstance(after, tuple)):
            raise AssertionError(f"{label}: tuple/non-tuple mismatch")
        if len(before) != len(after):
            raise AssertionError(f"{label}: tuple length changed {len(before)} -> {len(after)}")
        for index, (before_item, after_item) in enumerate(zip(before, after)):
            _assert_tree_equal(f"{label}[{index}]", before_item, after_item)
        return

    if isinstance(before, dict) or isinstance(after, dict):
        if not (isinstance(before, dict) and isinstance(after, dict)):
            raise AssertionError(f"{label}: dict/non-dict mismatch")
        if before.keys() != after.keys():
            raise AssertionError(f"{label}: dict keys changed")
        for key in before:
            _assert_tree_equal(f"{label}[{key!r}]", before[key], after[key])
        return

    before_is_dataclass = dataclasses.is_dataclass(before) and not isinstance(before, type)
    after_is_dataclass = dataclasses.is_dataclass(after) and not isinstance(after, type)
    if before_is_dataclass or after_is_dataclass:
        if not (before_is_dataclass and after_is_dataclass and before.__class__ is after.__class__):
            raise AssertionError(f"{label}: dataclass type changed")
        # Covers NeMo LabelLoopingStateItem decoder state under greedy_batch.
        for field in dataclasses.fields(before):
            _assert_tree_equal(
                f"{label}.{field.name}",
                getattr(before, field.name),
                getattr(after, field.name),
            )
        return

    before_is_nemo_obj = hasattr(before, "__dict__") and before.__class__.__module__.startswith("nemo.")
    after_is_nemo_obj = hasattr(after, "__dict__") and after.__class__.__module__.startswith("nemo.")
    if before_is_nemo_obj or after_is_nemo_obj:
        if not (before_is_nemo_obj and after_is_nemo_obj and before.__class__ is after.__class__):
            raise AssertionError(f"{label}: NeMo object type changed")
        if vars(before).keys() != vars(after).keys():
            raise AssertionError(f"{label}: NeMo object fields changed")
        for key in vars(before):
            _assert_tree_equal(f"{label}.{key}", getattr(before, key), getattr(after, key))
        return

    try:
        equal = before == after
    except Exception:
        equal = repr(before) == repr(after)
    if not equal:
        raise AssertionError(f"{label}: object changed from {before!r} to {after!r}")


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
        # A non-prefix correction cannot be sent to the append-only collector.
        # Freeze the already-emitted token count and only append the new tail.
        delta_tokens = final_tokens[len(emitted_tokens):]
        max_overlap = min(len(emitted_tokens), len(delta_tokens))
        for overlap in range(max_overlap, 0, -1):
            if emitted_tokens[-overlap:] == delta_tokens[:overlap]:
                delta_tokens = delta_tokens[overlap:]
                break

    return " ".join(delta_tokens)


# Default model - HuggingFace model name (auto-downloads) or local .nemo path
DEFAULT_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"

# Right context options for att_context_size=[70, X]
RIGHT_CONTEXT_OPTIONS = {
    0: "~80ms ultra-low latency",
    1: "~160ms low latency (recommended)",
    6: "~560ms balanced",
    13: "~1.12s highest accuracy",
}

PROMPTED_FALLBACK_ATT_CONTEXT_SIZES = [[56, 0], [56, 3], [56, 6], [56, 13]]
PROMPTED_DEFAULT_RIGHT_CONTEXT = 3
PROMPTED_DEFAULT_TARGET_LANG = "auto"
LANG_TAG_RE = re.compile(r"\s*<[a-z]{2}-[A-Z]{2}>")
LANG_TAG_CAPTURE_RE = re.compile(r"<([a-z]{2}(?:-[A-Z]{2})?)>")
PARTIAL_LANG_TAG_RE = re.compile(r"\s*<[a-z]{0,2}(?:-[A-Z]{0,2})?$")


@dataclass
class TranscriptResult:
    text: str
    language: Optional[str] = None


@dataclass(frozen=True)
class StreamingPlan:
    route: str
    att_context_size: list[int]
    right_context: int
    hop_samples: int
    window_size_samples: int
    shift_frames: int
    pre_encode_cache_size: int
    drop_extra: int
    final_padding_frames: int
    preprocess_align_pad_samples: int
    raw_audio_ring_samples: int
    preprocess_new_audio_samples: int
    stream_preprocess_valid_samples: int
    first_preprocess_mel_frame: int
    constant_preprocess_frames: int
    constant_preprocess_samples: int
    overlap_samples: int


@dataclass
class ASRSession:
    """Per-connection session state with caches for true incremental streaming."""

    id: str
    websocket: Any
    target_lang: Optional[str] = None
    model_route: str = "default"
    last_language: Optional[str] = None
    stats_language: Optional[str] = None
    stats_model_route: Optional[str] = None
    vad_gated_audio: bool = False
    accepting_vad_audio: bool = True

    # Legacy/debug audio buffer name. Step 6b keeps this bounded to pending
    # audio only; the preprocessor must never see a growing full-stream buffer.
    accumulated_audio: Optional[np.ndarray] = None

    # Raw audio not yet advanced past `emitted_frames * hop_samples`.
    pending_audio: Optional[np.ndarray] = None

    # Total real audio samples received in this session, excluding synthetic
    # finalization padding.
    total_audio_samples: int = 0

    # Synthetic warm-up samples already emitted to the model. This is only a
    # timeline cursor offset; it is not real audio and is never accumulated.
    synthetic_prefix_samples: int = 0

    # STFT boundary state: trailing window_size - hop samples before pending.
    raw_audio_ring: Optional[np.ndarray] = None

    # Cache-aware chunker state: trailing pre_encode_cache_size mel frames.
    mel_frame_ring: Optional[torch.Tensor] = None

    # Number of mel frames already emitted to encoder
    emitted_frames: int = 0

    # Encoder cache state
    cache_last_channel: Optional[torch.Tensor] = None
    cache_last_time: Optional[torch.Tensor] = None
    cache_last_channel_len: Optional[torch.Tensor] = None

    # Decoder state
    previous_hypotheses: Any = None
    pred_out_stream: Any = None

    # Current transcription (model's cumulative output)
    current_text: str = ""

    # Last text emitted to client on hard reset (for server-side deduplication)
    # We only send the delta (new portion) to avoid downstream duplication
    last_emitted_text: str = ""

    # Continuous-context mode state. These fields are only used when
    # NEMOTRON_CONTINUOUS=1; the default hard-reset path ignores them.
    state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    continuous_state: str = STREAMING
    committed_text: str = ""
    continuous_emitted_text: str = ""
    continuous_event_queue: Optional[asyncio.Queue] = None
    continuous_worker_task: Optional[asyncio.Task] = None
    continuous_debounce_task: Optional[asyncio.Task] = None
    continuous_stop_seq: int = 0
    continuous_reset_seen: bool = False
    continuous_post_stop_audio: bytearray = field(default_factory=bytearray)
    continuous_vad_stop_ts: Optional[float] = None
    continuous_vad_stop_recv_ts: Optional[float] = None  # when the vad_stop MESSAGE arrived off the socket (I/O-gap probe)
    continuous_finalize_terminal_sent: bool = False  # a terminal (final) was emitted for the current finalize cycle
    continuous_debounce_expiry_ts: Optional[float] = None
    eou_probe_chunk_index: int = 0
    eou_snapshot_audio: bytearray = field(default_factory=bytearray)
    scheduler_generation: int = 0
    scheduler_inflight_generation: Optional[int] = None
    scheduler_ready_since: Optional[float] = None
    scheduler_closed: bool = False
    scheduler_last_audio_monotonic: Optional[float] = None
    scheduler_pending_barrier_event: Optional[tuple] = None
    scheduler_pending_barrier_queue: Optional[asyncio.Queue] = None
    scheduler_pending_barrier_drained: int = 0

    # Audio overlap buffer for mid-utterance reset continuity
    # This preserves the last N ms of audio to provide encoder left-context
    # when a new segment starts after a reset
    overlap_buffer: Optional[np.ndarray] = None


@dataclass
class SchedulerBatchRow:
    session: ASRSession
    generation: int
    chunk_mel: torch.Tensor
    valid_new_mel: torch.Tensor
    drop_extra: int
    eou_probe_snapshot: Optional[dict[str, Any]]


@dataclass
class SchedulerFinalizeEvent:
    session: ASRSession
    event: tuple
    queue: asyncio.Queue


@dataclass
class SchedulerFinalizeItem:
    session: ASRSession
    reason: str
    expected_generation: int
    timing: dict[str, Any]
    parent_snapshot: Optional[dict[str, Any]]
    fork: Optional[ASRSession]
    final_text: str
    should_flush: bool
    fork_clone_ms: float = 0.0
    finalize_profile: Optional[dict[str, Any]] = None


@dataclass
class SchedulerFinalizeBatchRow:
    item: SchedulerFinalizeItem
    chunk_mel: torch.Tensor
    drop_extra: int


@dataclass
class SchedulerFinalizePreprocessState:
    item: SchedulerFinalizeItem
    pending: np.ndarray
    raw_ring: np.ndarray
    remaining_frames: int
    frames_collected: int = 0
    new_mels: list[torch.Tensor] = field(default_factory=list)


class ASRServer:
    """WebSocket server for streaming ASR with true incremental processing."""

    def __init__(
        self,
        model: str,
        multilingual_model: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8080,
        right_context: Optional[int] = None,
    ):
        self.model_name_or_path = model
        self.multilingual_model_name_or_path = multilingual_model
        self.dual_model_enabled = bool(multilingual_model)
        self.host = host
        self.port = port
        self.right_context = right_context
        self.model = None
        self.models: dict[str, Any] = {}
        self.model_prompted: dict[str, bool] = {}
        self.model_prompt_dictionaries: dict[str, dict[str, Any]] = {}
        self.model_routes: dict[str, str] = {"default": model, "en": model}
        self.streaming_plans: dict[str, StreamingPlan] = {}
        if self.dual_model_enabled:
            self.model_routes["multilingual"] = multilingual_model or ""
        self._inference_tls = threading.local()
        self._decoding_cfg_for_lane_models = None
        self.prompted_model = False
        self.prompt_dictionary: dict[str, Any] = {}
        self.target_lang = os.environ.get("NEMOTRON_TARGET_LANG", "en-US").strip() or "en-US"
        self.sample_rate = 16000
        # ASR benchmark server REQUIRES CUDA — fail fast, never silently fall
        # back to CPU (a CPU/wrong-device run yields invalid benchmark numbers
        # while looking 'fine'). Baseline hardening 2026-05-18 (dual review).
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for the Nemotron ASR benchmark server "
                "(torch.cuda.is_available() is False); refusing to run on CPU."
            )

        # Inference lock
        self.inference_lock = asyncio.Lock()

        self.continuous_context = os.environ.get("NEMOTRON_CONTINUOUS", "") == "1"
        self.scheduler_b1_requested = os.environ.get("NEMOTRON_SCHEDULER_B1", "") == "1"
        self.scheduler_enabled = self.continuous_context and self.scheduler_b1_requested
        self.batch_requested = os.environ.get("NEMOTRON_BATCH_SCHED", "") == "1"
        self.batch_enabled = self.batch_requested
        self.batch_barrier_drain_requested = (
            os.environ.get("NEMOTRON_BATCH_BARRIER_DRAIN", "") == "1"
        )
        self.batch_finalize_requested = (
            os.environ.get("NEMOTRON_BATCH_FINALIZE", "") == "1"
        )
        self.batch_finalize_preproc_requested = (
            os.environ.get("NEMOTRON_BATCH_FINALIZE_PREPROC", "") == "1"
        )
        self.finalize_priority_enabled = (
            os.environ.get("NEMOTRON_FINALIZE_PRIORITY", "") == "1"
        )
        self.batch_fallback_reason: Optional[str] = None
        self.model_lanes_requested = _env_int("NEMOTRON_MODEL_LANES", 1)
        if self.model_lanes_requested <= 0:
            raise ValueError("NEMOTRON_MODEL_LANES must be >= 1")
        self.model_lanes = self.model_lanes_requested
        if self.dual_model_enabled and (
            self.scheduler_enabled or self.batch_enabled or self.model_lanes_requested > 1
        ):
            raise ValueError(
                "Dual-checkpoint language routing currently supports the serial websocket "
                "path only; disable scheduler/batch/model-lane options."
            )
        self.requested_decoder_strategy = (
            os.environ.get("NEMOTRON_DECODING", "greedy").strip().lower() or "greedy"
        )
        self.fork_assert_enabled = os.environ.get("NEMOTRON_FORK_ASSERT", "") == "1"
        self.eou_probe_enabled = os.environ.get("NEMOTRON_EOU_PROBE", "") == "1"
        self._scheduler_batch_fallback_counts: dict[str, int] = {}
        if self.batch_enabled and not self.scheduler_b1_requested:
            raise ValueError("NEMOTRON_BATCH_SCHED=1 requires NEMOTRON_SCHEDULER_B1=1")
        if self.batch_enabled and not self.scheduler_enabled:
            raise ValueError(
                "NEMOTRON_BATCH_SCHED=1 requires the continuous scheduler "
                "(NEMOTRON_CONTINUOUS=1 and NEMOTRON_SCHEDULER_B1=1)"
            )
        if self.batch_requested and self.requested_decoder_strategy not in ("", "greedy"):
            raise ValueError(
                "NEMOTRON_BATCH_SCHED=1 requires NEMOTRON_DECODING=greedy "
                f"(got {self.requested_decoder_strategy!r})"
            )
        if self.batch_enabled and self.eou_probe_enabled:
            self._disable_batching("eou_probe_preserve_alignments_unprobed")
        if self.model_lanes > 1 and not self.scheduler_enabled:
            logger.warning(
                "model_lanes_disabled "
                f"requested={self.model_lanes_requested} reason=scheduler_off "
                "fallback_to_lanes=1"
            )
            self.model_lanes = 1
        elif self.model_lanes > 1 and not self.batch_enabled:
            logger.warning(
                "model_lanes_disabled "
                f"requested={self.model_lanes_requested} reason=batch_scheduler_off "
                "fallback_to_lanes=1"
            )
            self.model_lanes = 1
        elif self.model_lanes > 1:
            logger.info(
                "model_lanes_enabled "
                f"requested={self.model_lanes_requested} effective={self.model_lanes} "
                "drop_extra_rule=steady_drop_extra_only_else_exclusive_session_lane"
            )
        if self.batch_enabled:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            logger.info(
                "batch_sched_tf32_disabled "
                "cuda.matmul.allow_tf32=False cudnn.allow_tf32=False"
            )
        self.scheduler_queue_maxsize = _env_int("NEMOTRON_SCHEDULER_QUEUE_MAXSIZE", 256)
        if self.scheduler_queue_maxsize <= 0:
            raise ValueError("NEMOTRON_SCHEDULER_QUEUE_MAXSIZE must be > 0")
        # Defaults from the max-parallelism sweep (proj-2026-05-21-0410/max-parallelism-sweep.md):
        # MAX_SIZE=32 + MAX_WAIT=8ms raised the local realtime knee 40->56 with N=1 latency unchanged
        # (~17ms p95). The Step-8 device-aware startup cap clamps MAX_SIZE down on smaller GPUs, so 32 is
        # safe as a ceiling. Only active when NEMOTRON_BATCH_SCHED=1 (batching is off by default).
        self.batch_max_wait_ms = _env_int("NEMOTRON_BATCH_MAX_WAIT_MS", 8)
        self.batch_max_size = _env_int("NEMOTRON_BATCH_MAX_SIZE", 32)
        if self.batch_max_wait_ms < 0:
            raise ValueError("NEMOTRON_BATCH_MAX_WAIT_MS must be >= 0")
        if self.batch_max_size <= 0:
            raise ValueError("NEMOTRON_BATCH_MAX_SIZE must be > 0")
        if self.batch_requested and not self.batch_enabled:
            self.batch_max_size = 1
        self.batch_memory_headroom_fraction = _env_float(
            "NEMOTRON_BATCH_MEMORY_HEADROOM_FRACTION",
            0.80,
        )
        if not (0.0 < self.batch_memory_headroom_fraction <= 1.0):
            raise ValueError(
                "NEMOTRON_BATCH_MEMORY_HEADROOM_FRACTION must be > 0 and <= 1"
            )
        self.batch_memory_row_floor_bytes = (
            _env_int("NEMOTRON_BATCH_MEMORY_ROW_FLOOR_MB", 512) * 1024 * 1024
        )
        if self.batch_memory_row_floor_bytes <= 0:
            raise ValueError("NEMOTRON_BATCH_MEMORY_ROW_FLOOR_MB must be > 0")
        self.batch_memory_telemetry_every = _env_int(
            "NEMOTRON_BATCH_MEMORY_TELEMETRY_EVERY",
            1,
        )
        if self.batch_memory_telemetry_every <= 0:
            raise ValueError("NEMOTRON_BATCH_MEMORY_TELEMETRY_EVERY must be > 0")
        self.scheduler_task: Optional[asyncio.Task] = None
        self._scheduler_wakeup: Optional[asyncio.Event] = None
        self._scheduler_ready: set[str] = set()
        self._scheduler_batch_first_ready: dict[tuple, float] = {}
        self._scheduler_batch_size_hist: dict[int, int] = {}
        self._scheduler_batch_queue_wait_ms_total = 0.0
        self._scheduler_batch_queue_wait_ms_max = 0.0
        self._scheduler_batch_queue_wait_count = 0
        self._scheduler_batches = 0
        self._scheduler_chunks = 0
        self._scheduler_finalize_batches = 0
        self._scheduler_finalize_rows = 0
        self._scheduler_finalize_batch_size_hist: dict[int, int] = {}
        self._scheduler_finalize_serial_fallback_calls = 0
        self._scheduler_finalize_fallback_counts: dict[str, int] = {}
        self._scheduler_priority_finalize_events: list[SchedulerFinalizeEvent] = []
        self._scheduler_lane_wait_ms_total = 0.0
        self._scheduler_lane_wait_ms_max = 0.0
        self._scheduler_model_lane_executors: list[concurrent.futures.ThreadPoolExecutor] = []
        self._scheduler_model_lane_streams: list[Any] = []
        self._scheduler_model_lane_models: list[Any] = []
        self._scheduler_model_lane_tls = threading.local()
        self._scheduler_model_lane_condition: Optional[asyncio.Condition] = None
        self._scheduler_model_lane_exclusive_active = False
        self._scheduler_model_lane_active_key: Optional[tuple[Any, int]] = None
        self._scheduler_available_model_lanes: set[int] = (
            set(range(self.model_lanes)) if self.model_lanes > 1 else set()
        )
        self._scheduler_inflight_model_lane_tasks: set[asyncio.Task] = set()
        self._scheduler_inflight_sessions: set[str] = set()
        self._scheduler_session_model_lane_affinity: dict[str, int] = {}
        self.decoder_strategy = (
            "greedy_batch" if self.batch_enabled else self.requested_decoder_strategy
        )
        # Per-chunk profiling (additive, flag-gated): time preprocess vs conformer_stream_step
        # to locate the single-thread bottleneck. Adds cuda.synchronize() so it perturbs
        # timing slightly — only enabled under NEMOTRON_PROFILE_CHUNK=1.
        self.profile_chunk = os.environ.get("NEMOTRON_PROFILE_CHUNK", "") == "1"
        self.finalize_profile_enabled = (
            os.environ.get("NEMOTRON_FINALIZE_PROFILE", "") == "1"
        )
        self.gil_attrib_enabled = os.environ.get("NEMOTRON_GIL_ATTRIB", "") == "1"
        self.sync_compress_enabled = os.environ.get("NEMOTRON_SYNC_COMPRESS", "") == "1"
        if self.sync_compress_enabled:
            logger.info(
                "sync_compress_enabled=True "
                "flag=NEMOTRON_SYNC_COMPRESS "
                "mode=skip_entry_telemetry_pre_syncs"
            )
        self._finalize_profile_records = 0
        self._finalize_profile_hist: dict[tuple[int, int, int | None, str], int] = {}
        self._finalize_profile_b_hist: dict[int, int] = {}
        self._finalize_profile_hist_every = 10
        self._gil_attrib_lock = threading.Lock()
        self._gil_attrib_tls = threading.local()
        self._gil_attrib_samples: dict[str, dict[str, list[float]]] = {}
        self._gil_attrib_batch_hist: dict[str, dict[int, int]] = {}
        self._gil_attrib_paths: dict[str, dict[str, int]] = {}
        self._gil_attrib_loop_lag_ms: list[float] = []
        self._gil_attrib_loop_task: Optional[asyncio.Task] = None
        self._gil_attrib_started_unix: Optional[float] = None
        self._gil_attrib_started_perf: Optional[float] = None
        self._gil_attrib_record_emitted = False
        if self.finalize_profile_enabled:
            logger.info(
                "finalize_profile_enabled=True "
                "flag=NEMOTRON_FINALIZE_PROFILE "
                "mode=read_only syncs=profile_only"
            )
        if self.gil_attrib_enabled:
            logger.info(
                "gil_attribution_enabled=True "
                "flag=NEMOTRON_GIL_ATTRIB "
                "mode=probe_only schema=_continuous_finalize_timing "
                "default_off=True"
            )
        if self.finalize_priority_enabled:
            logger.info(
                "finalize_priority_enabled=True "
                "flag=NEMOTRON_FINALIZE_PRIORITY "
                "mode=cross_session_submit_priority"
            )
        self._prof_n = 0
        self._prof_pre_ms = 0.0
        self._prof_step_ms = 0.0
        self._prof_enc_ms = 0.0
        self.encoder_compile_requested = os.environ.get("NEMOTRON_ENCODER_COMPILE", "") == "1"
        self.encoder_compile_enabled = False
        self._encoder_compile_startup_logged = False
        self._encoder_compiled_cache_aware_stream_step: Any = None
        self._encoder_compile_warmup_done = False
        self._encoder_compile_warmed_buckets: set[tuple[int, int]] = set()
        self._encoder_compile_calls = 0
        self._encoder_compile_recapture_events = 0
        self._encoder_compile_last_graph_count = 0
        self._encoder_compile_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._encoder_compile_thread_id: Optional[int] = None
        self.encoder_cudagraph_requested = (
            os.environ.get("NEMOTRON_ENCODER_CUDAGRAPH", "") == "1"
        )
        self.encoder_cudagraph_enabled = False
        self.encoder_cudagraph_finalize_requested = (
            os.environ.get("NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE", "") == "1"
        )
        self.encoder_cudagraph_finalize_padded_requested = (
            os.environ.get("NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_PADDED", "") == "1"
        )
        self.encoder_cudagraph_finalize_enabled = False
        self.encoder_cudagraph_finalize_t_min = 42
        self.encoder_cudagraph_finalize_t_max = 60
        self.encoder_cudagraph_finalize_drop_extra = 2
        self._encoder_cudagraph_finalize_config_error: Optional[str] = None
        self._encoder_cudagraph_finalize_padded_canary_failed: Optional[str] = None
        if self.dual_model_enabled and (
            self.encoder_compile_requested
            or self.encoder_cudagraph_requested
            or self.encoder_cudagraph_finalize_requested
            or self.encoder_cudagraph_finalize_padded_requested
        ):
            raise ValueError(
                "Dual-checkpoint language routing currently supports only plain "
                "serial inference; disable encoder compile/cudagraph options."
            )
        if self.encoder_cudagraph_finalize_requested:
            try:
                self.encoder_cudagraph_finalize_t_min = _env_int(
                    "NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_T_MIN",
                    42,
                )
                self.encoder_cudagraph_finalize_t_max = _env_int(
                    "NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_T_MAX",
                    60,
                )
                if self.encoder_cudagraph_finalize_t_min <= 0:
                    raise ValueError(
                        "NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_T_MIN must be > 0"
                    )
                if self.encoder_cudagraph_finalize_t_max < self.encoder_cudagraph_finalize_t_min:
                    raise ValueError(
                        "NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_T_MAX must be >= "
                        "NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_T_MIN"
                    )
            except Exception as exc:
                self._encoder_cudagraph_finalize_config_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                self.encoder_cudagraph_finalize_requested = False
        self.encoder_cudagraph_max_b_requested: Optional[int] = None
        self.encoder_cudagraph_max_b = 0
        if self.encoder_cudagraph_requested:
            self.encoder_cudagraph_max_b_requested = _env_int(
                "NEMOTRON_ENCODER_CUDAGRAPH_MAX_B",
                self.batch_max_size,
            )
            if self.encoder_cudagraph_max_b_requested <= 0:
                raise ValueError("NEMOTRON_ENCODER_CUDAGRAPH_MAX_B must be > 0")
            self.encoder_cudagraph_max_b = min(
                int(self.encoder_cudagraph_max_b_requested),
                int(self.batch_max_size),
                16,
            )
        self._encoder_cudagraph_startup_logged = False
        self._encoder_cudagraph_managers: dict[int, BucketedCudaGraphEncoder] = {}
        self._encoder_cudagraph_stream_managers: dict[
            tuple[int, int],
            BucketedCudaGraphEncoder,
        ] = {}
        self._encoder_cudagraph_manager_labels: dict[int | tuple[int, int], str] = {}
        self._encoder_cudagraph_finalize_keys: tuple[FinalizeEncoderGraphKey, ...] = ()
        self._encoder_cudagraph_replay_calls = 0
        self._encoder_cudagraph_eager_fallbacks = 0
        self._encoder_finalize_cudagraph_replay_calls = 0
        self._encoder_finalize_cudagraph_eager_fallbacks = 0
        self._encoder_finalize_cudagraph_replay_hist: dict[tuple[int, int, int, int], int] = {}
        self._encoder_finalize_cudagraph_fallback_hist: dict[tuple[int, int, int, int], int] = {}
        self._encoder_cudagraph_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._encoder_cudagraph_thread_id: Optional[int] = None
        self.eou_probe_tag = _telemetry_run_tag() or "eou_probe"
        self.eou_probe_path = (
            _telemetry_dir() / f"{_safe_tag_filename(self.eou_probe_tag)}.eou_probe.jsonl"
            if self.eou_probe_enabled
            else None
        )
        snapshot_dir = os.environ.get("NEMOTRON_EOU_SNAPSHOT_DIR", "")
        self.eou_snapshot_dir = (
            Path(snapshot_dir).expanduser()
            if self.eou_probe_enabled and snapshot_dir
            else None
        )
        self.eou_snapshot_every = 1
        if self.eou_snapshot_dir is not None:
            self.eou_snapshot_every = _env_int("NEMOTRON_EOU_SNAPSHOT_EVERY", 1)
            if self.eou_snapshot_every < 1:
                raise ValueError("NEMOTRON_EOU_SNAPSHOT_EVERY must be >= 1")
        self.session_warmup_ms = _env_int("NEMOTRON_WARMUP_MS", 0)
        if self.session_warmup_ms < 0:
            raise ValueError("NEMOTRON_WARMUP_MS must be >= 0")
        self.finalize_silence_ms = _DEFAULT_FINALIZE_SILENCE_MS
        if self.continuous_context:
            self.finalize_silence_ms = _env_int(
                "NEMOTRON_FINALIZE_SILENCE_MS", _DEFAULT_FINALIZE_SILENCE_MS
            )
            if not (0 <= self.finalize_silence_ms < _MAX_FINALIZE_SILENCE_MS):
                raise ValueError(
                    "NEMOTRON_FINALIZE_SILENCE_MS must be >= 0 and < "
                    f"{_MAX_FINALIZE_SILENCE_MS}"
                )
        self.finalize_silence_seconds = self.finalize_silence_ms / 1000

        self.admission_max_backlog = _env_int(
            "NEMOTRON_ADMISSION_MAX_BACKLOG",
            _ADMISSION_MAX_BACKLOG_DEFAULT,
        )
        if self.admission_max_backlog < 0:
            raise ValueError("NEMOTRON_ADMISSION_MAX_BACKLOG must be >= 0")
        self.admission_max_ready_age_ms = _env_float(
            "NEMOTRON_ADMISSION_MAX_READY_AGE_MS",
            _ADMISSION_MAX_READY_AGE_MS_DEFAULT,
        )
        if self.admission_max_ready_age_ms < 0:
            raise ValueError("NEMOTRON_ADMISSION_MAX_READY_AGE_MS must be >= 0")
        self.admission_enabled = (
            self.admission_max_backlog != _ADMISSION_MAX_BACKLOG_DEFAULT
            or self.admission_max_ready_age_ms != _ADMISSION_MAX_READY_AGE_MS_DEFAULT
        )
        self.admission_attempted = 0
        self.admission_admitted = 0
        self.admission_rejected = 0
        if self.admission_enabled:
            logger.info(
                "admission_backpressure_enabled "
                f"max_backlog={self.admission_max_backlog} "
                f"max_ready_age_ms={self.admission_max_ready_age_ms}"
            )

        # /stats sliding window of finalize-time signals — cheap, always-on
        # rolling latency telemetry. All signals are derived from the regular
        # timing dict that every finalize already populates, so the only cost
        # is a deque.append per finalize (O(1)). NEMOTRON_STATS_ENABLED=0
        # disables; NEMOTRON_STATS_WINDOW sets the sample cap.
        self.stats_enabled = os.environ.get("NEMOTRON_STATS_ENABLED", "1") != "0"
        self.stats_window_size = _env_int(
            "NEMOTRON_STATS_WINDOW",
            _STATS_WINDOW_DEFAULT,
        )
        if self.stats_window_size <= 0:
            raise ValueError("NEMOTRON_STATS_WINDOW must be > 0")
        self._stats_samples: deque = deque(maxlen=self.stats_window_size)
        self._stats_emitted_count = 0
        self._stats_suppressed_count = 0
        self._stats_request_language_counts: Counter[str] = Counter()
        self._stats_request_route_counts: Counter[str] = Counter()
        self._stats_active_language_counts: Counter[str] = Counter()
        self._stats_active_route_counts: Counter[str] = Counter()
        self._stats_routing_event_language_counts: Counter[str] = Counter()
        self._stats_routing_event_route_counts: Counter[str] = Counter()
        self._stats_request_count = 0
        self._stats_routing_event_count = 0
        if self.stats_enabled:
            logger.info(
                "stats_endpoint_enabled "
                f"window={self.stats_window_size}"
            )

        # Active sessions
        self.sessions: dict[str, ASRSession] = {}

        # Model loaded flag for health check
        self.model_loaded = False

        # Streaming parameters (calculated from model config)
        self.shift_frames = None
        self.pre_encode_cache_size = None
        self.hop_samples = None
        self.window_size_samples = None
        self.raw_audio_ring_samples = None
        self.preprocess_align_pad_samples = None
        self.preprocess_new_audio_samples = None
        self.constant_preprocess_frames = None
        self.constant_preprocess_samples = None
        self.stream_preprocess_valid_samples = None
        self.first_preprocess_mel_frame = None
        self.att_context_size = None
        self.drop_extra = None
        self.final_padding_frames = None

        # Audio overlap for mid-utterance reset continuity (calculated in load_model)
        self.overlap_samples = None

    @staticmethod
    def _to_plain(value: Any) -> Any:
        if value.__class__.__module__.startswith("omegaconf."):
            try:
                from omegaconf import OmegaConf

                return OmegaConf.to_container(value, resolve=True)
            except Exception:
                pass
        if hasattr(value, "to_container"):
            try:
                return value.to_container(resolve=True)
            except TypeError:
                return value.to_container()
        if isinstance(value, dict):
            return {key: ASRServer._to_plain(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [ASRServer._to_plain(item) for item in value]
        return value

    @staticmethod
    def _cfg_get(container: Any, *keys: str) -> Any:
        current = container
        for key in keys:
            if current is None:
                return None
            try:
                if hasattr(current, "get"):
                    current = current.get(key)
                else:
                    current = getattr(current, key)
            except Exception:
                return None
        return current

    def _record_batch_fallback(self, reason: str) -> None:
        self._scheduler_batch_fallback_counts[reason] = (
            self._scheduler_batch_fallback_counts.get(reason, 0) + 1
        )

    def _disable_batching(self, reason: str) -> None:
        if not self.batch_enabled and self.batch_fallback_reason == reason:
            return
        self.batch_enabled = False
        self.batch_fallback_reason = reason
        self.decoder_strategy = getattr(self, "requested_decoder_strategy", "greedy") or "greedy"
        if hasattr(self, "batch_max_size"):
            self.batch_max_size = 1
        self._record_batch_fallback(reason)
        logger.warning(
            "batch_sched_disabled "
            f"reason={reason} requested={self.batch_requested} fallback_to_B=1"
        )

    def _batch_model_rnnt_pure_status(self) -> tuple[bool, str]:
        model = self.model
        if model is None:
            return False, "model_not_loaded"
        if not hasattr(model, "conformer_stream_step"):
            return False, "model_missing_conformer_stream_step"
        if not hasattr(model, "joint"):
            return False, "model_missing_rnnt_joint"
        if not hasattr(model, "decoder"):
            return False, "model_missing_rnnt_decoder"
        if hasattr(model, "ctc_decoder"):
            return False, "hybrid_ctc_decoder_present"
        if hasattr(model, "ctc_loss"):
            return False, "ctc_loss_present"

        class_label = f"{model.__class__.__module__}.{model.__class__.__name__}".lower()
        if "ctc" in class_label and "rnnt" not in class_label:
            return False, "ctc_model_class"
        if "hybrid" in class_label and "ctc" in class_label:
            return False, "hybrid_ctc_model_class"

        cfg = self._to_plain(getattr(model, "cfg", None))
        if isinstance(cfg, dict):
            for key in ("ctc", "ctc_decoder", "ctc_loss"):
                if key in cfg and cfg.get(key) not in (None, False):
                    return False, f"cfg_{key}_present"
            decoder_cfg = cfg.get("decoder")
            if isinstance(decoder_cfg, dict):
                decoder_name = str(decoder_cfg.get("_target_", "")).lower()
                if "ctc" in decoder_name and "rnnt" not in decoder_name:
                    return False, "cfg_decoder_ctc_target"
            decoding_cfg = cfg.get("decoding")
            if isinstance(decoding_cfg, dict):
                strategy = str(decoding_cfg.get("strategy", "")).lower()
                if "ctc" in strategy and "rnnt" not in strategy:
                    return False, "cfg_decoding_ctc_strategy"
        return True, "rnnt_pure"

    def _assert_batch_decoder_blackwell_safe(self) -> None:
        if not self.batch_enabled:
            return
        decoding = getattr(self.model, "decoding", None)
        uses_cuda_graph = False
        for owner in (
            decoding,
            getattr(decoding, "decoding", None),
            getattr(decoding, "greedy", None),
        ):
            value = getattr(owner, "use_cuda_graph_decoder", None)
            if value is not None:
                uses_cuda_graph = uses_cuda_graph or bool(value)
        cfg = self._to_plain(getattr(self.model, "cfg", None))
        cfg_value = self._cfg_get(cfg, "decoding", "greedy", "use_cuda_graph_decoder")
        if cfg_value is not None:
            uses_cuda_graph = uses_cuda_graph or bool(cfg_value)
        if uses_cuda_graph:
            self._disable_batching("cuda_graph_decoder_enabled")
        else:
            logger.info(
                "batch_sched_decoder_assert "
                "use_cuda_graph_decoder=False status=ok"
            )

    @staticmethod
    def _tensor_storage_nbytes(tensor: torch.Tensor, seen: set[tuple]) -> int:
        if tensor.device.type != "cuda":
            return 0
        try:
            storage = tensor.untyped_storage()
            key = (tensor.device.type, tensor.device.index, int(storage.data_ptr()))
            if key in seen:
                return 0
            seen.add(key)
            return int(storage.nbytes())
        except Exception:
            key = (tensor.device.type, tensor.device.index, int(tensor.data_ptr()))
            if key in seen:
                return 0
            seen.add(key)
            return int(tensor.nelement() * tensor.element_size())

    @classmethod
    def _tensor_tree_storage_nbytes(
        cls,
        obj: Any,
        *,
        seen_tensors: Optional[set[int]] = None,
        seen_storages: Optional[set[tuple]] = None,
        seen_objects: Optional[set[int]] = None,
    ) -> int:
        if seen_tensors is None:
            seen_tensors = set()
        if seen_storages is None:
            seen_storages = set()
        if seen_objects is None:
            seen_objects = set()

        if torch.is_tensor(obj):
            oid = id(obj)
            if oid in seen_tensors:
                return 0
            seen_tensors.add(oid)
            return cls._tensor_storage_nbytes(obj, seen_storages)
        if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
            return 0

        oid = id(obj)
        if oid in seen_objects:
            return 0
        seen_objects.add(oid)

        if isinstance(obj, dict):
            return sum(
                cls._tensor_tree_storage_nbytes(
                    value,
                    seen_tensors=seen_tensors,
                    seen_storages=seen_storages,
                    seen_objects=seen_objects,
                )
                for value in obj.values()
            )
        if isinstance(obj, (list, tuple, set)):
            return sum(
                cls._tensor_tree_storage_nbytes(
                    value,
                    seen_tensors=seen_tensors,
                    seen_storages=seen_storages,
                    seen_objects=seen_objects,
                )
                for value in obj
            )
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return sum(
                cls._tensor_tree_storage_nbytes(
                    getattr(obj, field.name),
                    seen_tensors=seen_tensors,
                    seen_storages=seen_storages,
                    seen_objects=seen_objects,
                )
                for field in dataclasses.fields(obj)
            )
        if hasattr(obj, "__dict__") and obj.__class__.__module__.startswith("nemo."):
            return sum(
                cls._tensor_tree_storage_nbytes(
                    value,
                    seen_tensors=seen_tensors,
                    seen_storages=seen_storages,
                    seen_objects=seen_objects,
                )
                for value in vars(obj).values()
            )
        return 0

    def _session_cache_storage_bytes(self, session: ASRSession) -> int:
        seen_tensors: set[int] = set()
        seen_storages: set[tuple] = set()
        seen_objects: set[int] = set()
        total = 0
        for value in (
            session.cache_last_channel,
            session.cache_last_time,
            session.cache_last_channel_len,
            session.mel_frame_ring,
            session.previous_hypotheses,
            session.pred_out_stream,
        ):
            total += self._tensor_tree_storage_nbytes(
                value,
                seen_tensors=seen_tensors,
                seen_storages=seen_storages,
                seen_objects=seen_objects,
            )
        return total

    def _retained_session_cache_bytes(self) -> int:
        return sum(
            self._session_cache_storage_bytes(session)
            for session in list(self.sessions.values())
            if not session.scheduler_closed
        )

    def _cuda_memory_snapshot(self) -> dict[str, int]:
        if not torch.cuda.is_available():
            return {
                "active_bytes": 0,
                "allocated_bytes": 0,
                "reserved_bytes": 0,
                "max_reserved_bytes": 0,
                "retained_session_cache_bytes": self._retained_session_cache_bytes(),
                "num_alloc_retries": 0,
                "num_ooms": 0,
            }
        stats = torch.cuda.memory_stats()
        active = int(stats.get("active_bytes.all.current", torch.cuda.memory_allocated()))
        return {
            "active_bytes": active,
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "retained_session_cache_bytes": self._retained_session_cache_bytes(),
            # cudaMalloc-stall signal: num_alloc_retries increments when a malloc fails -> cache-free -> retry
            # (a synchronous, device-syncing stall). num_ooms = hard OOMs. Both CUMULATIVE.
            "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
            "num_ooms": int(stats.get("num_ooms", 0)),
        }

    def _log_retained_cache_telemetry(self, reason: str) -> None:
        if not self.batch_requested:
            return
        mem = self._cuda_memory_snapshot()
        logger.info(
            "scheduler_batch_retained_memory "
            f"reason={reason} "
            f"active_sessions={len(self.sessions)} "
            f"retained_session_cache_bytes={mem['retained_session_cache_bytes']} "
            f"cuda_active_bytes={mem['active_bytes']} "
            f"cuda_allocated_bytes={mem['allocated_bytes']} "
            f"cuda_reserved_bytes={mem['reserved_bytes']} "
            f"cuda_max_reserved_bytes={mem['max_reserved_bytes']}"
        )

    def _estimate_batch_extra_row_bytes(self) -> int:
        cache = self._current_inference_model().encoder.get_initial_cache_state(batch_size=1)
        cache_bytes = sum(
            self._tensor_tree_storage_nbytes(tensor)
            for tensor in cache
        )
        mel_bytes = 0
        if self.constant_preprocess_frames is not None:
            mel_bytes = int(self.constant_preprocess_frames) * 128 * 4
        return max(
            int(self.batch_memory_row_floor_bytes),
            int(cache_bytes * 4),
            int(mel_bytes * 16),
        )

    def _configure_batch_memory_cap(self) -> None:
        if not self.batch_enabled:
            if self.batch_requested:
                logger.info(
                    "batch_memory_startup_cap "
                    f"requested_max={self.batch_max_size} effective_max=1 "
                    f"reason={self.batch_fallback_reason or 'batch_disabled'}"
                )
            return
        torch.cuda.synchronize()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        reserved_bytes = int(torch.cuda.memory_reserved())
        allocated_bytes = int(torch.cuda.memory_allocated())
        max_reserved_bytes = int(torch.cuda.max_memory_reserved())
        headroom_bytes = int(total_bytes * self.batch_memory_headroom_fraction)
        extra_row_bytes = self._estimate_batch_extra_row_bytes()
        if reserved_bytes >= headroom_bytes:
            device_cap = 1
        else:
            device_cap = 1 + ((headroom_bytes - reserved_bytes) // extra_row_bytes)
            device_cap = max(1, int(device_cap))
        requested_max = int(self.batch_max_size)
        effective_max = max(1, min(requested_max, device_cap))
        if effective_max < requested_max:
            self.batch_max_size = effective_max
            self._record_batch_fallback("memory_cap_clamped")
        logger.info(
            "batch_memory_startup_cap "
            f"requested_max={requested_max} effective_max={self.batch_max_size} "
            f"device_cap={device_cap} "
            f"headroom_fraction={self.batch_memory_headroom_fraction:.2f} "
            f"total_bytes={int(total_bytes)} free_bytes={int(free_bytes)} "
            f"reserved_bytes={reserved_bytes} allocated_bytes={allocated_bytes} "
            f"max_reserved_bytes={max_reserved_bytes} "
            f"estimated_extra_row_bytes={extra_row_bytes} "
            f"fallback_counts={dict(sorted(self._scheduler_batch_fallback_counts.items()))}"
        )

    @staticmethod
    def _is_int_like(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return False
        return isinstance(value, (int, np.integer))

    @classmethod
    def _normalize_att_context_sizes(cls, value: Any) -> list[list[int]]:
        value = cls._to_plain(value)
        if value is None:
            return []
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list) or not value:
            return []
        if all(cls._is_int_like(item) for item in value):
            return [[int(item) for item in value]]

        sizes: list[list[int]] = []
        for item in value:
            normalized = cls._normalize_att_context_sizes(item)
            sizes.extend(normalized)
        return sizes

    def _supported_att_context_sizes(self) -> list[list[int]]:
        cfg_contexts = self._normalize_att_context_sizes(
            self._cfg_get(getattr(self.model, "cfg", None), "encoder", "att_context_size")
        )
        if cfg_contexts:
            return cfg_contexts

        encoder_contexts = self._normalize_att_context_sizes(
            getattr(getattr(self.model, "encoder", None), "att_context_size_all", None)
        )
        if encoder_contexts:
            return encoder_contexts

        if self.prompted_model:
            return [context.copy() for context in PROMPTED_FALLBACK_ATT_CONTEXT_SIZES]
        return [[70, right_context] for right_context in RIGHT_CONTEXT_OPTIONS]

    def _select_att_context_size(self) -> list[int]:
        if not self.prompted_model:
            if self.right_context is None:
                self.right_context = 1
            if self.right_context not in RIGHT_CONTEXT_OPTIONS:
                raise ValueError(
                    "English model right context must be one of "
                    f"{sorted(RIGHT_CONTEXT_OPTIONS)}, got {self.right_context}"
                )
            return [70, self.right_context]

        supported_contexts = self._supported_att_context_sizes()
        requested_right_context = (
            PROMPTED_DEFAULT_RIGHT_CONTEXT
            if self.right_context is None
            else self.right_context
        )
        for context in supported_contexts:
            if len(context) >= 2 and context[-1] == requested_right_context:
                self.right_context = requested_right_context
                return context.copy()

        supported_right_contexts = sorted(
            {context[-1] for context in supported_contexts if len(context) >= 2}
        )
        raise ValueError(
            "Prompted model right context must be one of "
            f"{supported_right_contexts}, got {requested_right_context}"
        )

    def _ensure_session_target_lang(self, session: ASRSession) -> str:
        if session.target_lang is None:
            session.target_lang = self.target_lang
        return session.target_lang

    def _apply_inference_prompt(self, session: ASRSession) -> None:
        route = getattr(self._inference_tls, "model_route", None) or session.model_route
        if self.model_prompted.get(route, self.prompted_model):
            prompt = self._ensure_session_target_lang(session)
            prompt_dict = self.model_prompt_dictionaries.get(route) or self.prompt_dictionary
            if prompt not in prompt_dict and "auto" in prompt_dict:
                prompt = "auto"
            self._current_inference_model().set_inference_prompt(prompt)

    def _read_prompt_dictionary(self) -> dict[str, Any]:
        prompt_dictionary_paths = (
            ("model_defaults", "prompt_dictionary"),
            ("train_ds", "prompt_dictionary"),
            ("validation_ds", "prompt_dictionary"),
            ("test_ds", "prompt_dictionary"),
        )
        model = self._current_inference_model()
        for cfg in (getattr(model, "cfg", None), getattr(model, "_cfg", None)):
            for path in prompt_dictionary_paths:
                prompt_dict = self._to_plain(self._cfg_get(cfg, *path))
                if isinstance(prompt_dict, dict):
                    return {str(key): value for key, value in prompt_dict.items()}
        return {}

    @staticmethod
    def _format_supported_languages(prompt_dictionary: dict[str, Any]) -> str:
        return ", ".join(sorted(prompt_dictionary))

    def _model_identity_aliases(self) -> set[str]:
        aliases: set[str] = set()
        configured = os.environ.get("NEMOTRON_MODEL_NAME", "").strip()
        if configured:
            aliases.add(configured)

        if self.model_name_or_path:
            aliases.add(self.model_name_or_path)
            model_path = Path(self.model_name_or_path)
            aliases.add(model_path.name)
            aliases.add(model_path.stem)

        if self.prompted_model:
            aliases.update({"multilingual", "ml"})
        else:
            aliases.update({"english", "en"})

        return {alias for alias in aliases if alias}

    def _validate_model_query_param(self, requested_model: Optional[str]) -> None:
        if not requested_model:
            return

        aliases = self._model_identity_aliases()
        if not aliases:
            logger.info(
                "Client requested model={} but no server model identity is configured; accepting",
                requested_model,
            )
            return

        normalized_aliases = {alias.lower() for alias in aliases}
        if requested_model.lower() not in normalized_aliases:
            raise ValueError(
                "model mismatch: requested "
                f"{requested_model}; server accepts: {', '.join(sorted(aliases))}"
            )

    def _validate_session_target_lang(self, requested_language: Optional[str]) -> str:
        if self.dual_model_enabled:
            return (requested_language or self.target_lang or "en").strip() or "en"

        if requested_language:
            if not self.prompted_model:
                raise ValueError("this model does not accept a language argument")

            if requested_language not in self.prompt_dictionary:
                raise ValueError(
                    f"unsupported language {requested_language}; supported: "
                    f"{self._format_supported_languages(self.prompt_dictionary)}"
                )
            return requested_language

        if self.prompted_model:
            if PROMPTED_DEFAULT_TARGET_LANG not in self.prompt_dictionary:
                raise ValueError(
                    f"unsupported language {PROMPTED_DEFAULT_TARGET_LANG}; supported: "
                    f"{self._format_supported_languages(self.prompt_dictionary)}"
                )
            return PROMPTED_DEFAULT_TARGET_LANG

        return self.target_lang

    def _validate_connection_query(self, query: Any) -> str:
        requested_language = (query.get("language") or "").strip()
        requested_model = (query.get("model") or "").strip()

        self._validate_model_query_param(requested_model or None)
        return self._validate_session_target_lang(requested_language or None)

    def _strip_lang_tags(self, text: Any) -> str:
        stripped = LANG_TAG_RE.sub(" ", str(text))
        # A cumulative streaming hypothesis may end with the beginning of the
        # next language tag token. Keep that fragment out of emitted state; the
        # complete tag is stripped when it appears in a later cumulative hyp.
        stripped = PARTIAL_LANG_TAG_RE.sub("", stripped)
        return re.sub(r"\s+", " ", stripped).strip()

    def _extract_hypothesis_result(self, hyp: Any) -> TranscriptResult:
        if hasattr(hyp, 'text'):
            text = hyp.text
        elif isinstance(hyp, str):
            text = hyp
        else:
            text = str(hyp)

        raw_text = text
        matches = LANG_TAG_CAPTURE_RE.findall(raw_text)
        language = matches[-1] if matches else None
        text = self._strip_lang_tags(text)
        if os.environ.get("NEMOTRON_DEBUG_RAW_HYP_TEXT", "") == "1":
            logger.info(
                "debug_raw_hyp_text "
                f"target_lang={self.target_lang!r} "
                f"language={language!r} "
                f"raw={raw_text!r} "
                f"stripped={text!r}"
            )
        return TranscriptResult(text=text, language=language)

    def _extract_hypothesis_text(self, hyp: Any) -> str:
        return self._extract_hypothesis_result(hyp).text

    def _transcript_payload(
        self,
        session: ASRSession,
        *,
        text: str,
        is_final: bool,
        finalize: Optional[bool] = None,
        language: Optional[str] = None,
    ) -> dict[str, Any]:
        if language:
            session.last_language = language
        language_tag = language or session.last_language or session.target_lang
        payload: dict[str, Any] = {
            "type": "transcript",
            "text": text,
            "is_final": is_final,
            "language": language_tag,
            "language_tag": language_tag,
            "model_route": session.model_route,
        }
        if finalize is not None:
            payload["finalize"] = finalize
        return payload

    def _model_key_for_language(self, language: Optional[str]) -> str:
        if not self.dual_model_enabled:
            return "default"
        lang = (language or self.target_lang or "").strip().lower()
        if lang == "en" or lang.startswith("en-"):
            return "en"
        return "multilingual"

    def _set_session_language_route(self, session: ASRSession, language: Optional[str]) -> None:
        if language:
            session.target_lang = language
        self._ensure_session_target_lang(session)
        session.model_route = self._model_key_for_language(session.target_lang)

    def _restore_asr_model(self, model_name_or_path: str, *, label: str):
        import nemo.collections.asr as nemo_asr

        is_local_file = (
            model_name_or_path.endswith('.nemo') or
            os.path.exists(model_name_or_path)
        )
        if is_local_file:
            logger.info(f"Loading {label} model from local file: {model_name_or_path}")
            return nemo_asr.models.ASRModel.restore_from(
                model_name_or_path, map_location='cpu'
            )
        logger.info(f"Loading {label} model from HuggingFace: {model_name_or_path}")
        return nemo_asr.models.ASRModel.from_pretrained(
            model_name_or_path, map_location='cpu'
        )

    def _register_model_route(self, route: str, model: Any) -> None:
        self.models[route] = model
        prompted = hasattr(model, "set_inference_prompt")
        self.model_prompted[route] = prompted
        previous_route = getattr(self._inference_tls, "model_route", None)
        self._inference_tls.model_route = route
        try:
            self.model_prompt_dictionaries[route] = (
                self._read_prompt_dictionary() if prompted else {}
            )
        finally:
            self._inference_tls.model_route = previous_route

    def _build_streaming_plan(
        self,
        route: str,
        model: Any,
        att_context_size: list[int],
    ) -> StreamingPlan:
        scfg = model.encoder.streaming_cfg
        preprocessor_cfg = model.cfg.preprocessor
        hop_length_sec = preprocessor_cfg.get('window_stride', 0.01)
        window_size_sec = preprocessor_cfg.get('window_size', 0.025)
        featurizer = model.preprocessor.featurizer
        hop_samples = int(
            getattr(featurizer, "hop_length", int(hop_length_sec * self.sample_rate))
        )
        window_size_samples = int(
            getattr(featurizer, "win_length", int(window_size_sec * self.sample_rate))
        )
        shift_frames = scfg.shift_size[1] if isinstance(scfg.shift_size, list) else scfg.shift_size
        pre_cache = scfg.pre_encode_cache_size
        pre_encode_cache_size = pre_cache[1] if isinstance(pre_cache, list) else pre_cache
        drop_extra = scfg.drop_extra_pre_encoded
        right_context = int(att_context_size[-1]) if att_context_size else int(self.right_context or 1)
        final_padding_frames = (right_context + 1) * shift_frames
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
            raise RuntimeError(
                "Constant preprocessor prefix must align to mel frame hops: "
                f"prefix={prefix_samples}, hop={hop_samples}"
            )
        first_preprocess_mel_frame = prefix_samples // hop_samples
        min_plan_frames = (
            first_preprocess_mel_frame
            + pre_encode_cache_size
            + shift_frames
            + final_padding_frames
            + 1
        )
        constant_preprocess_frames = 1 << (min_plan_frames - 1).bit_length()
        constant_preprocess_samples = (constant_preprocess_frames - 1) * hop_samples
        overlap_samples = pre_encode_cache_size * hop_samples
        return StreamingPlan(
            route=route,
            att_context_size=att_context_size.copy(),
            right_context=right_context,
            hop_samples=int(hop_samples),
            window_size_samples=int(window_size_samples),
            shift_frames=int(shift_frames),
            pre_encode_cache_size=int(pre_encode_cache_size),
            drop_extra=int(drop_extra),
            final_padding_frames=int(final_padding_frames),
            preprocess_align_pad_samples=int(preprocess_align_pad_samples),
            raw_audio_ring_samples=int(raw_audio_ring_samples),
            preprocess_new_audio_samples=int(preprocess_new_audio_samples),
            stream_preprocess_valid_samples=int(stream_preprocess_valid_samples),
            first_preprocess_mel_frame=int(first_preprocess_mel_frame),
            constant_preprocess_frames=int(constant_preprocess_frames),
            constant_preprocess_samples=int(constant_preprocess_samples),
            overlap_samples=int(overlap_samples),
        )

    def _apply_default_streaming_plan(self, plan: StreamingPlan) -> None:
        self.att_context_size = plan.att_context_size.copy()
        self.shift_frames = plan.shift_frames
        self.pre_encode_cache_size = plan.pre_encode_cache_size
        self.hop_samples = plan.hop_samples
        self.window_size_samples = plan.window_size_samples
        self.raw_audio_ring_samples = plan.raw_audio_ring_samples
        self.preprocess_align_pad_samples = plan.preprocess_align_pad_samples
        self.preprocess_new_audio_samples = plan.preprocess_new_audio_samples
        self.stream_preprocess_valid_samples = plan.stream_preprocess_valid_samples
        self.constant_preprocess_frames = plan.constant_preprocess_frames
        self.constant_preprocess_samples = plan.constant_preprocess_samples
        self.first_preprocess_mel_frame = plan.first_preprocess_mel_frame
        self.drop_extra = plan.drop_extra
        self.final_padding_frames = plan.final_padding_frames
        self.overlap_samples = plan.overlap_samples

    def _log_streaming_plan(self, plan: StreamingPlan, model: Any) -> None:
        scfg = model.encoder.streaming_cfg
        hop_length_sec = model.cfg.preprocessor.get('window_stride', 0.01)
        shift_ms = plan.shift_frames * hop_length_sec * 1000
        padding_ms = plan.final_padding_frames * hop_length_sec * 1000
        overlap_ms = plan.overlap_samples * 1000 / self.sample_rate
        logger.info(
            f"Streaming plan[{plan.route}]: chunk_size={scfg.chunk_size}, "
            f"shift_size={scfg.shift_size}, att_context={plan.att_context_size}"
        )
        logger.info(
            f"Streaming plan[{plan.route}]: shift={shift_ms:.0f}ms "
            f"({plan.shift_frames} frames), pre_encode_cache={plan.pre_encode_cache_size} "
            f"frames, drop_extra={plan.drop_extra}"
        )
        logger.info(
            f"Streaming plan[{plan.route}]: K={plan.constant_preprocess_samples} samples "
            f"({plan.constant_preprocess_frames} STFT frames), "
            f"align={plan.preprocess_align_pad_samples}, "
            f"raw_ring={plan.raw_audio_ring_samples}, "
            f"new_audio={plan.preprocess_new_audio_samples}, "
            f"final_padding={padding_ms:.0f}ms ({plan.final_padding_frames} frames), "
            f"overlap={overlap_ms:.0f}ms ({plan.overlap_samples} samples)"
        )

    def _plan_for_route(self, route: str) -> StreamingPlan:
        return self.streaming_plans.get(route) or self.streaming_plans["default"]

    def _plan_for_session(self, session: ASRSession) -> StreamingPlan:
        return self._plan_for_route(session.model_route)

    def load_model(self):
        """Load the NeMo ASR model with streaming configuration."""
        import nemo.collections.asr as nemo_asr
        from omegaconf import OmegaConf

        self.model = self._restore_asr_model(self.model_name_or_path, label="primary")
        self.model = self.model.cuda()
        logger.info("ASR model loaded on CUDA")
        self.prompted_model = hasattr(self.model, "set_inference_prompt")
        self.prompt_dictionary = (
            self._read_prompt_dictionary() if self.prompted_model else {}
        )
        if self.batch_enabled:
            rnnt_ok, rnnt_reason = self._batch_model_rnnt_pure_status()
            if not rnnt_ok:
                self._disable_batching(rnnt_reason)
            else:
                logger.info(
                    "batch_sched_model_assert "
                    f"rnnt_pure=True reason={rnnt_reason}"
                )

        # Configure attention context for streaming
        self.att_context_size = self._select_att_context_size()
        if self.prompted_model:
            logger.info(
                "Prompted model detected; setting "
                f"att_context_size={self.att_context_size} "
                f"(NEMOTRON_TARGET_LANG={self.target_lang!r})"
            )
        else:
            logger.info(f"Setting att_context_size=[70, {self.right_context}] ({RIGHT_CONTEXT_OPTIONS.get(self.right_context, 'custom')})")
        self.model.encoder.set_default_att_context_size(self.att_context_size)

        # Decoding strategy: greedy (default, Blackwell-safe) or beam via
        # NEMOTRON_DECODING=beam (experimental; may be slower / unsupported on
        # some GPUs).
        _decoding = self.decoder_strategy
        if _decoding == "beam":
            logger.info("Configuring BEAM (maes) decoding...")
            _decoding_cfg = OmegaConf.create({
                'strategy': 'beam',
                'beam': {
                    'beam_size': 4,
                    'search_type': 'maes',
                    'maes_num_steps': 2,
                    'return_best_hypothesis': True,
                }
            })
        elif _decoding == "greedy_batch":
            logger.info(
                "Configuring greedy_batch decoding "
                "(loop_labels=True, use_cuda_graph_decoder=False)..."
            )
            _decoding_cfg = OmegaConf.create({
                'strategy': 'greedy_batch',
                'greedy': {
                    'max_symbols': 10,
                    'loop_labels': True,
                    'use_cuda_graph_decoder': False,
                }
            })
            if self.eou_probe_enabled:
                _decoding_cfg.greedy.preserve_alignments = True
                _decoding_cfg.greedy.preserve_frame_confidence = True
                _decoding_cfg.greedy.confidence_method_cfg = {
                    'name': 'entropy',
                    'entropy_type': 'tsallis',
                    'alpha': 0.5,
                    'entropy_norm': 'exp',
                }
        else:
            logger.info("Configuring greedy decoding for Blackwell compatibility...")
            _decoding_cfg = OmegaConf.create({
                'strategy': 'greedy',
                'greedy': {
                    'max_symbols': 10,
                    'loop_labels': False,
                    'use_cuda_graph_decoder': False,
                }
            })
            if self.eou_probe_enabled:
                _decoding_cfg.greedy.preserve_alignments = True
                _decoding_cfg.greedy.preserve_frame_confidence = True
                _decoding_cfg.greedy.confidence_method_cfg = {
                    'name': 'entropy',
                    'entropy_type': 'tsallis',
                    'alpha': 0.5,
                    'entropy_norm': 'exp',
                }
        self._decoding_cfg_for_lane_models = _decoding_cfg
        self.model.change_decoding_strategy(decoding_cfg=_decoding_cfg)
        self.model.eval()
        self._assert_batch_decoder_blackwell_safe()

        # Disable dither for deterministic preprocessing
        self.model.preprocessor.featurizer.dither = 0.0
        self._register_model_route("default", self.model)
        self.models["en"] = self.model
        self.model_prompted["en"] = self.model_prompted["default"]
        self.model_prompt_dictionaries["en"] = self.model_prompt_dictionaries["default"]
        default_plan = self._build_streaming_plan("default", self.model, self.att_context_size)
        self.streaming_plans["default"] = default_plan
        self.streaming_plans["en"] = default_plan
        self._apply_default_streaming_plan(default_plan)

        if self.dual_model_enabled:
            multilingual_model = self._restore_asr_model(
                self.multilingual_model_name_or_path or "",
                label="multilingual",
            ).cuda()
            multilingual_model.change_decoding_strategy(decoding_cfg=copy.deepcopy(_decoding_cfg))
            multilingual_model.eval()
            try:
                multilingual_model.preprocessor.featurizer.dither = 0.0
            except Exception:
                pass
            multilingual_att_context_size = [56, PROMPTED_DEFAULT_RIGHT_CONTEXT]
            if hasattr(multilingual_model, "encoder"):
                multilingual_model.encoder.set_default_att_context_size(
                    multilingual_att_context_size
                )
            self._register_model_route("multilingual", multilingual_model)
            self.streaming_plans["multilingual"] = self._build_streaming_plan(
                "multilingual",
                multilingual_model,
                multilingual_att_context_size,
            )
            logger.info(
                "dual_model_routing_enabled "
                f"en={self.model_name_or_path} "
                f"multilingual={self.multilingual_model_name_or_path}"
            )

        logger.info(f"Model loaded: {type(self.model).__name__}")
        for route, plan in self.streaming_plans.items():
            if route == "en":
                continue
            self._log_streaming_plan(plan, self.models[route])

        self._configure_encoder_compile()
        logger.info(
            "startup_flags "
            f"scheduler_enabled={self.scheduler_enabled} "
            f"batch_requested={self.batch_requested} "
            f"batch_enabled={self.batch_enabled} "
            f"batch_finalize_requested={self.batch_finalize_requested} "
            f"batch_finalize={self._scheduler_batch_finalize_active()} "
            f"batch_finalize_preproc_requested={self.batch_finalize_preproc_requested} "
            f"batch_finalize_preproc={self._scheduler_batch_finalize_preproc_active()} "
            f"finalize_priority={self._scheduler_finalize_priority_active()} "
            f"batch_fallback_reason={self.batch_fallback_reason or 'none'} "
            f"decoder_strategy={self.decoder_strategy} "
            f"encoder_compile_enabled={self.encoder_compile_enabled} "
            f"encoder_cudagraph_requested={self.encoder_cudagraph_requested} "
            f"encoder_cudagraph_max_B={self.encoder_cudagraph_max_b} "
            f"encoder_cudagraph_finalize_requested={self.encoder_cudagraph_finalize_requested} "
            f"encoder_cudagraph_finalize_padded_requested={self.encoder_cudagraph_finalize_padded_requested} "
            f"encoder_cudagraph_finalize_T={self.encoder_cudagraph_finalize_t_min}..{self.encoder_cudagraph_finalize_t_max} "
            f"model_lanes_requested={self.model_lanes_requested} "
            f"model_lanes={self.model_lanes} "
            f"batch_max_size={self.batch_max_size}"
        )

        # Warmup inference to ensure model is fully loaded on GPU
        # This prevents GPU memory issues when LLM starts later
        self._warmup()
        if self.model_lanes > 1:
            self._ensure_scheduler_model_lane_resources()
        self._configure_encoder_cudagraph()
        self._configure_batch_memory_cap()

    def _configure_encoder_compile(self) -> None:
        """Configure optional B=1 static-shape encoder compilation."""
        if self._encoder_compile_startup_logged:
            return

        if not self.encoder_compile_requested:
            logger.info("encoder_compile_enabled=False requested=False")
            self._encoder_compile_startup_logged = True
            return

        if self.encoder_cudagraph_requested:
            logger.info(
                "encoder_compile_enabled=False requested=True "
                "reason=encoder_cudagraph_supersedes_compile"
            )
            self._encoder_compile_startup_logged = True
            return

        if self.prompted_model:
            logger.warning(
                "encoder_compile_enabled=False requested=True "
                "reason=prompted_model_static_shapes_unvalidated"
            )
            self._encoder_compile_startup_logged = True
            return

        if not hasattr(torch, "compile"):
            raise RuntimeError("NEMOTRON_ENCODER_COMPILE=1 requires torch.compile")

        self._encoder_compiled_cache_aware_stream_step = torch.compile(
            self.model.encoder.cache_aware_stream_step,
            mode="reduce-overhead",
        )
        self._encoder_compile_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="nemotron-encoder-compile",
        )
        self.encoder_compile_enabled = True
        self._encoder_compile_last_graph_count = self._encoder_compile_counter_snapshot()
        logger.info(
            "encoder_compile_enabled=True requested=True mode=reduce-overhead "
            "scope=B1_static_buckets warmup=enabled first=enabled steady=enabled "
            "final_uncompiled=True"
        )
        self._encoder_compile_startup_logged = True

    def _capture_encoder_cudagraph_manager_sync(
        self,
        label: str,
        record_default_thread: bool = False,
    ) -> BucketedCudaGraphEncoder:
        if record_default_thread:
            self._encoder_cudagraph_thread_id = threading.get_ident()
        manager = BucketedCudaGraphEncoder.warmup(
            self._current_inference_model(),
            self.encoder_cudagraph_max_b,
            logger=logging.getLogger("nemotron_speech.cudagraph_encoder"),
        )
        self._capture_encoder_cudagraph_finalize_buckets(manager, label=label)
        self._run_encoder_finalize_padded_canary(manager, label=label)
        return manager

    def _encoder_finalize_cudagraph_padded_key(self) -> FinalizeEncoderGraphKey:
        return FinalizeEncoderGraphKey(
            batch_size=1,
            time_steps=int(self.encoder_cudagraph_finalize_t_max),
            drop_extra=int(self.encoder_cudagraph_finalize_drop_extra),
            keep_all_outputs=True,
        )

    @staticmethod
    def _hypothesis_token_tuple(hyp: Any) -> tuple[int, ...]:
        if hyp is None:
            return ()
        sequence = getattr(hyp, "y_sequence", None)
        if sequence is None:
            return ()
        try:
            if torch.is_tensor(sequence):
                values = sequence.detach().cpu().reshape(-1).tolist()
            elif isinstance(sequence, np.ndarray):
                values = sequence.reshape(-1).tolist()
            else:
                values = list(sequence)
            return tuple(int(value) for value in values)
        except Exception:
            return ()

    def _finalize_canary_signature(self, result: Any) -> tuple[tuple[int, ...], str]:
        hyps = None
        if isinstance(result, (tuple, list)) and len(result) > 5:
            hyps = result[5]
        hyp = hyps[0] if isinstance(hyps, (tuple, list)) and hyps else hyps
        if hyp is None:
            return (), ""
        return self._hypothesis_token_tuple(hyp), self._extract_hypothesis_text(hyp)

    def _disable_encoder_finalize_padded_fail_closed(self, *, label: str, reason: str) -> None:
        if self._encoder_cudagraph_finalize_padded_canary_failed is not None:
            return
        self._encoder_cudagraph_finalize_padded_canary_failed = reason
        self.encoder_cudagraph_finalize_padded_requested = False
        self.encoder_cudagraph_finalize_requested = False
        self.encoder_cudagraph_finalize_enabled = False
        self._encoder_cudagraph_finalize_keys = ()
        logger.warning(
            "encoder_finalize_cuda_graph_padded_disabled_fail_closed "
            f"label={label} reason={reason} fallback=eager"
        )

    def _run_encoder_finalize_padded_canary(
        self,
        manager: BucketedCudaGraphEncoder,
        *,
        label: str,
    ) -> None:
        if not (
            self.encoder_cudagraph_finalize_requested
            and self.encoder_cudagraph_finalize_padded_requested
        ):
            return
        key = self._encoder_finalize_cudagraph_padded_key()
        if not manager.finalize_captured(key):
            self._disable_encoder_finalize_padded_fail_closed(
                label=label,
                reason="canary_bucket_not_captured",
            )
            return

        real_t = int(self.encoder_cudagraph_finalize_t_min)
        real_t = max(1, min(real_t, int(self.encoder_cudagraph_finalize_t_max)))
        model = self._current_inference_model()
        try:
            cache = model.encoder.get_initial_cache_state(batch_size=1)
            device = cache[0].device
            dtype = cache[0].dtype
            features = int(model.cfg.preprocessor.features)
            processed_signal = torch.zeros(
                (1, features, real_t),
                device=device,
                dtype=dtype,
            )
            processed_signal_length = torch.full(
                (1,),
                real_t,
                device=device,
                dtype=torch.long,
            )

            def clone_cache() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                return (
                    cache[0].detach().clone(),
                    cache[1].detach().clone(),
                    cache[2].detach().clone(),
                )

            eager_cache = clone_cache()
            graph_cache = clone_cache()
            common_kwargs = {
                "processed_signal": processed_signal,
                "processed_signal_length": processed_signal_length,
                "keep_all_outputs": True,
                "previous_hypotheses": None,
                "previous_pred_out": None,
                "drop_extra_pre_encoded": int(key.drop_extra),
                "return_transcription": True,
            }
            with torch.inference_mode():
                eager_result = model.conformer_stream_step(
                    **common_kwargs,
                    cache_last_channel=eager_cache[0],
                    cache_last_time=eager_cache[1],
                    cache_last_channel_len=eager_cache[2],
                )
                with self._finalize_cudagraph_encoder_cache_step_installed(
                    model,
                    manager,
                    key,
                    None,
                ):
                    graph_result = model.conformer_stream_step(
                        **common_kwargs,
                        cache_last_channel=graph_cache[0],
                        cache_last_time=graph_cache[1],
                        cache_last_channel_len=graph_cache[2],
                    )
            self._cuda_synchronize_for_current_model_lane()
            eager_tokens, eager_text = self._finalize_canary_signature(eager_result)
            graph_tokens, graph_text = self._finalize_canary_signature(graph_result)
            if eager_tokens != graph_tokens or eager_text != graph_text:
                self._disable_encoder_finalize_padded_fail_closed(
                    label=label,
                    reason=(
                        "canary_tokens_or_text_mismatch "
                        f"real_T={real_t} graph_T={key.time_steps} "
                        f"eager_tokens={list(eager_tokens)} graph_tokens={list(graph_tokens)} "
                        f"eager_text={eager_text!r} graph_text={graph_text!r}"
                    ),
                )
                return
            logger.info(
                "encoder_finalize_cuda_graph_padded_canary_ok "
                f"label={label} B=1 real_T={real_t} graph_T={key.time_steps} "
                f"drop_extra={key.drop_extra} tokens={len(eager_tokens)}"
            )
        except Exception as exc:
            self._disable_encoder_finalize_padded_fail_closed(
                label=label,
                reason=f"canary_exception {type(exc).__name__}: {exc}",
            )

    def _encoder_cudagraph_finalize_requested_keys(
        self,
    ) -> tuple[FinalizeEncoderGraphKey, ...]:
        if not self.encoder_cudagraph_finalize_requested:
            return ()
        if self._encoder_cudagraph_finalize_keys:
            return self._encoder_cudagraph_finalize_keys
        if self.encoder_cudagraph_finalize_padded_requested:
            self._encoder_cudagraph_finalize_keys = (
                self._encoder_finalize_cudagraph_padded_key(),
            )
            return self._encoder_cudagraph_finalize_keys
        self._encoder_cudagraph_finalize_keys = tuple(
            FinalizeEncoderGraphKey(
                batch_size=1,
                time_steps=time_steps,
                drop_extra=self.encoder_cudagraph_finalize_drop_extra,
                keep_all_outputs=True,
            )
            for time_steps in range(
                int(self.encoder_cudagraph_finalize_t_min),
                int(self.encoder_cudagraph_finalize_t_max) + 1,
            )
        )
        return self._encoder_cudagraph_finalize_keys

    def _capture_encoder_cudagraph_finalize_buckets(
        self,
        manager: BucketedCudaGraphEncoder,
        *,
        label: str,
    ) -> None:
        keys = self._encoder_cudagraph_finalize_requested_keys()
        if not keys:
            return
        try:
            manager.capture_finalize(keys)
        except Exception as exc:
            logger.warning(
                "encoder_finalize_cuda_graph_manager_disabled "
                f"label={label} reason=capture_exception_fail_closed "
                f"error={type(exc).__name__}: {exc}"
            )

    def _encoder_cudagraph_manager_is_complete(
        self,
        manager: BucketedCudaGraphEncoder,
        *,
        label: str,
    ) -> bool:
        captured = manager.captured_batch_sizes
        uncaptured = manager.uncaptured_batch_sizes
        if not captured:
            logger.warning(
                "encoder_cuda_graph_manager_disabled "
                f"label={label} reason=no_captured_buckets "
                f"requested_max_B={self.encoder_cudagraph_max_b}"
            )
            return False
        if uncaptured:
            logger.warning(
                "encoder_cuda_graph_manager_disabled "
                f"label={label} reason=capture_incomplete_fail_closed "
                f"captured_B={list(captured)} uncaptured_B={list(uncaptured)}"
            )
            return False
        return True

    def _store_encoder_cudagraph_manager(
        self,
        manager: BucketedCudaGraphEncoder,
        *,
        model: Any,
        label: str,
        stream: Any = None,
    ) -> None:
        captured = manager.captured_batch_sizes
        capture_ms: list[float] = []
        for batch_size in captured:
            value = manager.capture_ms(batch_size)
            if value is not None:
                capture_ms.append(float(value))
        total_capture_ms = sum(float(value) for value in capture_ms)

        if stream is None:
            manager_key: int | tuple[int, int] = id(model)
            self._encoder_cudagraph_managers[id(model)] = manager
            stream_label = "default"
        else:
            manager_key = (id(model), id(stream))
            self._encoder_cudagraph_stream_managers[manager_key] = manager
            stream_label = f"lane_stream:{id(stream)}"
        self._encoder_cudagraph_manager_labels[manager_key] = label

        logger.info(
            "encoder_cuda_graph_manager_captured "
            f"label={label} stream={stream_label} "
            f"captured_B={list(captured)} "
            f"capture_ms_total={total_capture_ms:.1f}"
        )

        if self.encoder_cudagraph_finalize_requested:
            finalize_captured = manager.captured_finalize_keys
            finalize_uncaptured = manager.uncaptured_finalize_keys
            finalize_capture_ms = [
                float(value)
                for key in finalize_captured
                for value in [manager.finalize_capture_ms(key)]
                if value is not None
            ]
            finalize_memory = manager.finalize_capture_memory_bytes()
            padded_mode = bool(self.encoder_cudagraph_finalize_padded_requested)
            per_t_absent = (
                not padded_mode
                or all(
                    int(key.batch_size) == 1
                    and int(key.time_steps) == int(self.encoder_cudagraph_finalize_t_max)
                    for key in finalize_captured
                )
            )
            logger.info(
                "encoder_finalize_cuda_graph_manager_captured "
                f"label={label} stream={stream_label} "
                f"mode={'padded_T_max' if padded_mode else 'per_T'} "
                f"captured_keys={[str(key) for key in finalize_captured]} "
                f"uncaptured_keys={[str(key) for key in finalize_uncaptured]} "
                f"padded_per_T_absent={per_t_absent} "
                f"capture_ms_total={sum(finalize_capture_ms):.1f} "
                f"allocated_bytes={finalize_memory.get('allocated_bytes', 0)} "
                f"reserved_bytes={finalize_memory.get('reserved_bytes', 0)}"
            )
            if padded_mode and not per_t_absent:
                logger.warning(
                    "encoder_finalize_cuda_graph_padded_disabled_fail_closed "
                    f"label={label} reason=per_T_finalize_buckets_present "
                    f"captured_keys={[str(key) for key in finalize_captured]} "
                    "fallback=eager"
                )
                self.encoder_cudagraph_finalize_padded_requested = False
                self.encoder_cudagraph_finalize_requested = False
                self.encoder_cudagraph_finalize_enabled = False
                self._encoder_cudagraph_finalize_keys = ()
            if finalize_uncaptured:
                logger.warning(
                    "encoder_finalize_cuda_graph_manager_incomplete_fail_closed "
                    f"label={label} "
                    f"errors={{"
                    + ", ".join(
                        f"{key}: {manager.finalize_capture_error(key)}"
                        for key in finalize_uncaptured
                    )
                    + "}}"
                )

    def _configure_encoder_cudagraph(self) -> None:
        """Configure optional per-B manual CUDA graphs for the streaming encoder."""
        if self._encoder_cudagraph_startup_logged:
            return

        if self._encoder_cudagraph_finalize_config_error is not None:
            logger.warning(
                "encoder_finalize_cuda_graph_enabled=False requested=True "
                "reason=invalid_finalize_bucket_config_fail_closed "
                f"error={self._encoder_cudagraph_finalize_config_error}"
            )
        if (
            self.encoder_cudagraph_finalize_padded_requested
            and not self.encoder_cudagraph_finalize_requested
        ):
            logger.warning(
                "encoder_finalize_cuda_graph_padded_enabled=False requested=True "
                "reason=finalize_cudagraph_disabled_fail_closed"
            )

        if not self.encoder_cudagraph_requested:
            logger.info("encoder_cuda_graph_enabled=False requested=False")
            if self.encoder_cudagraph_finalize_requested:
                logger.warning(
                    "encoder_finalize_cuda_graph_enabled=False requested=True "
                    "reason=encoder_cudagraph_disabled_fail_closed"
                )
            self._encoder_cudagraph_startup_logged = True
            return

        if self.prompted_model:
            logger.warning(
                "encoder_cuda_graph_enabled=False requested=True "
                "reason=prompted_model_static_shapes_unvalidated"
            )
            if self.encoder_cudagraph_finalize_requested:
                logger.warning(
                    "encoder_finalize_cuda_graph_enabled=False requested=True "
                    "reason=prompted_model_static_shapes_unvalidated"
                )
            self._encoder_cudagraph_startup_logged = True
            return

        if not torch.cuda.is_available():
            logger.warning(
                "encoder_cuda_graph_enabled=False requested=True "
                "reason=torch_cuda_unavailable"
            )
            if self.encoder_cudagraph_finalize_requested:
                logger.warning(
                    "encoder_finalize_cuda_graph_enabled=False requested=True "
                    "reason=torch_cuda_unavailable"
                )
            self._encoder_cudagraph_startup_logged = True
            return

        if self.encoder_compile_requested:
            logger.info(
                "encoder_cuda_graph_supersedes_compile "
                "NEMOTRON_ENCODER_COMPILE ignored while cudagraph is enabled"
            )

        if self.encoder_cudagraph_finalize_requested:
            if int(self.drop_extra) != int(self.encoder_cudagraph_finalize_drop_extra):
                logger.warning(
                    "encoder_finalize_cuda_graph_drop_extra_mismatch_fail_closed "
                    f"model_drop_extra={self.drop_extra} "
                    f"captured_drop_extra={self.encoder_cudagraph_finalize_drop_extra}"
                )
            if self.encoder_cudagraph_finalize_padded_requested:
                logger.info(
                    "encoder_finalize_cuda_graph_capture_requested "
                    "mode=padded_T_max capture_set=B1 "
                    f"real_T={self.encoder_cudagraph_finalize_t_min}..{self.encoder_cudagraph_finalize_t_max} "
                    f"graph_T={self.encoder_cudagraph_finalize_t_max} "
                    f"drop_extra={self.encoder_cudagraph_finalize_drop_extra} "
                    "keep_all_outputs=True"
                )
            else:
                logger.info(
                    "encoder_finalize_cuda_graph_capture_requested "
                    "mode=per_T capture_set=B1 "
                    f"T={self.encoder_cudagraph_finalize_t_min}..{self.encoder_cudagraph_finalize_t_max} "
                    f"drop_extra={self.encoder_cudagraph_finalize_drop_extra} "
                    "keep_all_outputs=True"
                )

        default_manager_stored = False
        self._encoder_cudagraph_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="nemotron-encoder-cudagraph",
        )
        try:
            default_manager = self._encoder_cudagraph_executor.submit(
                self._capture_encoder_cudagraph_manager_sync,
                "self.model",
                True,
            ).result()
            if self._encoder_cudagraph_manager_is_complete(
                default_manager,
                label="self.model",
            ):
                self._store_encoder_cudagraph_manager(
                    default_manager,
                    model=self.model,
                    label="self.model",
                )
                default_manager_stored = True
        except Exception as exc:
            logger.warning(
                "encoder_cuda_graph_manager_disabled "
                f"label=self.model reason=capture_exception_fail_closed "
                f"error={type(exc).__name__}: {exc}"
            )

        if not default_manager_stored and self._encoder_cudagraph_executor is not None:
            self._encoder_cudagraph_executor.shutdown(wait=False)
            self._encoder_cudagraph_executor = None
            self._encoder_cudagraph_thread_id = None

        if self.model_lanes > 1:
            self._ensure_scheduler_model_lane_resources()
            for lane_id, lane_model in enumerate(self._scheduler_model_lane_models):
                stream = self._scheduler_model_lane_streams[lane_id]
                if stream is None:
                    logger.warning(
                        "encoder_cuda_graph_lane_disabled "
                        f"lane={lane_id} reason=missing_cuda_stream"
                    )
                    continue
                label = f"lane:{lane_id}"
                try:
                    lane_manager = self._scheduler_model_lane_executors[lane_id].submit(
                        self._run_scheduler_model_lane_call_sync,
                        lane_id,
                        self._capture_encoder_cudagraph_manager_sync,
                        (label, False),
                    ).result()
                    if self._encoder_cudagraph_manager_is_complete(
                        lane_manager,
                        label=label,
                    ):
                        self._store_encoder_cudagraph_manager(
                            lane_manager,
                            model=lane_model,
                            label=label,
                            stream=stream,
                        )
                except Exception as exc:
                    logger.warning(
                        "encoder_cuda_graph_lane_disabled "
                        f"lane={lane_id} reason=capture_exception_fail_closed "
                        f"error={type(exc).__name__}: {exc}"
                    )

        manager_count = (
            len(self._encoder_cudagraph_managers)
            + len(self._encoder_cudagraph_stream_managers)
        )
        self.encoder_cudagraph_enabled = manager_count > 0
        all_managers = list(self._encoder_cudagraph_managers.values()) + list(
            self._encoder_cudagraph_stream_managers.values()
        )
        finalize_manager_count = sum(
            1 for manager in all_managers if manager.captured_finalize_keys
        )
        finalize_captured_key_count = sum(
            len(manager.captured_finalize_keys) for manager in all_managers
        )
        finalize_padded_per_t_absent = (
            not self.encoder_cudagraph_finalize_padded_requested
            or all(
                all(
                    int(key.batch_size) == 1
                    and int(key.time_steps) == int(self.encoder_cudagraph_finalize_t_max)
                    for key in manager.captured_finalize_keys
                )
                for manager in all_managers
            )
        )
        self.encoder_cudagraph_finalize_enabled = (
            self.encoder_cudagraph_enabled
            and self.encoder_cudagraph_finalize_requested
            and finalize_manager_count > 0
            and finalize_padded_per_t_absent
        )
        logger.info(
            "encoder_cuda_graph_enabled="
            f"{self.encoder_cudagraph_enabled} requested=True "
            f"max_B={self.encoder_cudagraph_max_b} "
            f"managers={manager_count} "
            f"default_managers={len(self._encoder_cudagraph_managers)} "
            f"lane_stream_managers={len(self._encoder_cudagraph_stream_managers)}"
        )
        logger.info(
            "encoder_finalize_cuda_graph_enabled="
            f"{self.encoder_cudagraph_finalize_enabled} "
            f"requested={self.encoder_cudagraph_finalize_requested} "
            f"padded_requested={self.encoder_cudagraph_finalize_padded_requested} "
            f"padded_canary_failed={self._encoder_cudagraph_finalize_padded_canary_failed is not None} "
            f"T={self.encoder_cudagraph_finalize_t_min}..{self.encoder_cudagraph_finalize_t_max} "
            f"graph_T={self.encoder_cudagraph_finalize_t_max if self.encoder_cudagraph_finalize_padded_requested else 'per_T'} "
            f"drop_extra={self.encoder_cudagraph_finalize_drop_extra} "
            f"captured_managers={finalize_manager_count} "
            f"captured_key_count={finalize_captured_key_count} "
            f"padded_per_T_absent={finalize_padded_per_t_absent}"
        )
        self._encoder_cudagraph_startup_logged = True

    def _session_warmup_frames(self) -> Optional[int]:
        if self.session_warmup_ms <= 0:
            return None
        target_samples = int(round(self.sample_rate * self.session_warmup_ms / 1000))
        return max(
            self.pre_encode_cache_size,
            int(round(target_samples / self.hop_samples)),
        )

    @staticmethod
    def _encoder_compile_counter_snapshot() -> int:
        try:
            from torch._dynamo.utils import counters
        except Exception:
            return 0
        try:
            return int(counters.get("stats", {}).get("unique_graphs", 0))
        except Exception:
            return 0

    def _encoder_compile_bucket_for_call(self, kwargs: dict[str, Any]) -> Optional[tuple[int, int]]:
        if not self.encoder_compile_enabled:
            return None
        if kwargs.get("keep_all_outputs", True):
            return None
        if kwargs.get("bypass_pre_encode", False):
            return None

        processed_signal = kwargs.get("processed_signal")
        if not torch.is_tensor(processed_signal) or processed_signal.ndim != 3:
            return None
        if int(processed_signal.shape[0]) != 1:
            return None

        try:
            chunk_frames = int(processed_signal.shape[-1])
            drop_extra = int(kwargs.get("drop_extra_pre_encoded"))
        except Exception:
            return None

        static_buckets: set[tuple[int, int]] = {
            (int(self.shift_frames), 0),
            (int(self.pre_encode_cache_size + self.shift_frames), int(self.drop_extra)),
        }
        warmup_frames = self._session_warmup_frames()
        if warmup_frames is not None:
            static_buckets.add((int(warmup_frames), 0))

        bucket = (chunk_frames, drop_extra)
        if bucket in static_buckets:
            return bucket
        return None

    def _encoder_cudagraph_manager_for_model(
        self,
        model: Any,
    ) -> Optional[BucketedCudaGraphEncoder]:
        if not self.encoder_cudagraph_enabled:
            return None
        stream = getattr(self._scheduler_model_lane_tls, "stream", None)
        if stream is not None:
            return self._encoder_cudagraph_stream_managers.get((id(model), id(stream)))
        return self._encoder_cudagraph_managers.get(id(model))

    def _encoder_cudagraph_bucket_for_call(
        self,
        kwargs: dict[str, Any],
        manager: Optional[BucketedCudaGraphEncoder],
    ) -> Optional[int]:
        if not self.encoder_cudagraph_enabled or manager is None:
            return None
        if kwargs.get("keep_all_outputs", True):
            return None
        if kwargs.get("bypass_pre_encode", False):
            return None

        processed_signal = kwargs.get("processed_signal")
        if not torch.is_tensor(processed_signal) or processed_signal.ndim != 3:
            return None

        try:
            batch_size = int(processed_signal.shape[0])
            chunk_frames = int(processed_signal.shape[-1])
            drop_extra = int(kwargs.get("drop_extra_pre_encoded"))
            steady_T = int(self.pre_encode_cache_size + self.shift_frames)
        except Exception:
            return None

        if chunk_frames != steady_T:
            return None
        if drop_extra != int(self.drop_extra):
            return None
        if batch_size < 1 or batch_size > int(manager.max_batch_size):
            return None
        if not manager.captured(batch_size):
            return None
        return batch_size

    def _record_encoder_finalize_cudagraph_replay(
        self,
        key: FinalizeEncoderGraphKey,
        *,
        real_time_steps: Optional[int] = None,
    ) -> None:
        graph_t = int(key.time_steps)
        real_t = graph_t if real_time_steps is None else int(real_time_steps)
        hist_key = (int(key.batch_size), graph_t, real_t, int(key.drop_extra))
        self._encoder_finalize_cudagraph_replay_calls += 1
        self._encoder_finalize_cudagraph_replay_hist[hist_key] = (
            self._encoder_finalize_cudagraph_replay_hist.get(hist_key, 0) + 1
        )
        if (
            self._encoder_finalize_cudagraph_replay_calls <= 5
            or self._encoder_finalize_cudagraph_replay_calls % 50 == 0
        ):
            logger.info(
                "encoder_finalize_cuda_graph_status "
                f"replays={self._encoder_finalize_cudagraph_replay_calls} "
                f"fallbacks={self._encoder_finalize_cudagraph_eager_fallbacks} "
                f"B={key.batch_size} graph_T={graph_t} real_T={real_t} "
                f"drop_extra={key.drop_extra} "
                f"mode={'padded_T_max' if self.encoder_cudagraph_finalize_padded_requested else 'per_T'}"
            )

    def _record_encoder_finalize_cudagraph_fallback(
        self,
        key: Optional[FinalizeEncoderGraphKey],
        *,
        reason: str,
        real_time_steps: Optional[int] = None,
    ) -> None:
        if key is not None:
            graph_t = int(key.time_steps)
            real_t = graph_t if real_time_steps is None else int(real_time_steps)
            hist_key = (int(key.batch_size), graph_t, real_t, int(key.drop_extra))
            self._encoder_finalize_cudagraph_fallback_hist[hist_key] = (
                self._encoder_finalize_cudagraph_fallback_hist.get(hist_key, 0) + 1
            )
        self._encoder_finalize_cudagraph_eager_fallbacks += 1
        if (
            self._encoder_finalize_cudagraph_eager_fallbacks <= 5
            or self._encoder_finalize_cudagraph_eager_fallbacks % 50 == 0
        ):
            if key is None:
                key_text = "B=? graph_T=? real_T=? drop_extra=?"
            else:
                graph_t = int(key.time_steps)
                real_t = graph_t if real_time_steps is None else int(real_time_steps)
                key_text = (
                    f"B={key.batch_size} graph_T={graph_t} real_T={real_t} "
                    f"drop_extra={key.drop_extra}"
                )
            logger.info(
                "encoder_finalize_cuda_graph_fallback "
                f"replays={self._encoder_finalize_cudagraph_replay_calls} "
                f"fallbacks={self._encoder_finalize_cudagraph_eager_fallbacks} "
                f"{key_text} "
                f"mode={'padded_T_max' if self.encoder_cudagraph_finalize_padded_requested else 'per_T'} "
                f"reason={reason}"
            )

    def _encoder_finalize_cudagraph_key_for_call(
        self,
        kwargs: dict[str, Any],
        manager: Optional[BucketedCudaGraphEncoder],
    ) -> Optional[FinalizeEncoderGraphKey]:
        if not self.encoder_cudagraph_finalize_requested:
            return None
        if not kwargs.get("keep_all_outputs", False):
            return None

        processed_signal = kwargs.get("processed_signal")
        if not torch.is_tensor(processed_signal) or processed_signal.ndim != 3:
            self._record_encoder_finalize_cudagraph_fallback(
                None,
                reason="invalid_processed_signal",
            )
            return None

        try:
            batch_size = int(processed_signal.shape[0])
            real_time_steps = int(processed_signal.shape[-1])
            graph_time_steps = (
                int(self.encoder_cudagraph_finalize_t_max)
                if self.encoder_cudagraph_finalize_padded_requested
                else real_time_steps
            )
            key = FinalizeEncoderGraphKey(
                batch_size=batch_size,
                time_steps=graph_time_steps,
                drop_extra=int(kwargs.get("drop_extra_pre_encoded")),
                keep_all_outputs=True,
            )
        except Exception:
            self._record_encoder_finalize_cudagraph_fallback(
                None,
                reason="invalid_bucket_key",
            )
            return None

        if kwargs.get("bypass_pre_encode", False):
            self._record_encoder_finalize_cudagraph_fallback(
                key,
                reason="bypass_pre_encode",
                real_time_steps=real_time_steps,
            )
            return None
        if not self.encoder_cudagraph_finalize_enabled or manager is None:
            self._record_encoder_finalize_cudagraph_fallback(
                key,
                reason="manager_unavailable",
                real_time_steps=real_time_steps,
            )
            return None
        if int(key.batch_size) != 1:
            self._record_encoder_finalize_cudagraph_fallback(
                key,
                reason="uncaptured_batch_size",
                real_time_steps=real_time_steps,
            )
            return None
        if int(key.drop_extra) != int(self.encoder_cudagraph_finalize_drop_extra):
            self._record_encoder_finalize_cudagraph_fallback(
                key,
                reason="uncaptured_drop_extra",
                real_time_steps=real_time_steps,
            )
            return None
        if not (
            int(self.encoder_cudagraph_finalize_t_min)
            <= int(real_time_steps)
            <= int(self.encoder_cudagraph_finalize_t_max)
        ):
            self._record_encoder_finalize_cudagraph_fallback(
                key,
                reason="uncaptured_time_steps",
                real_time_steps=real_time_steps,
            )
            return None
        if not manager.finalize_captured(key):
            self._record_encoder_finalize_cudagraph_fallback(
                key,
                reason="bucket_not_captured",
                real_time_steps=real_time_steps,
            )
            return None
        return key

    @contextlib.contextmanager
    def _compiled_encoder_cache_step_installed(self):
        encoder = self.model.encoder
        attr_name = "cache_aware_stream_step"
        had_instance_attr = attr_name in vars(encoder)
        original_instance_attr = vars(encoder).get(attr_name)
        compiled_callable = self._encoder_compiled_cache_aware_stream_step

        def cache_aware_stream_step_wrapper(*args, **call_kwargs):
            return self._gil_attrib_timed_call(
                "dispatch_ms",
                compiled_callable,
                *args,
                cuda_event=True,
                **call_kwargs,
            )

        object.__setattr__(encoder, attr_name, cache_aware_stream_step_wrapper)
        try:
            yield
        finally:
            if had_instance_attr:
                object.__setattr__(encoder, attr_name, original_instance_attr)
            else:
                object.__delattr__(encoder, attr_name)

    @contextlib.contextmanager
    def _cudagraph_encoder_cache_step_installed(
        self,
        model: Any,
        manager: BucketedCudaGraphEncoder,
        batch_size: int,
    ):
        encoder = model.encoder
        attr_name = "cache_aware_stream_step"
        had_instance_attr = attr_name in vars(encoder)
        original_instance_attr = vars(encoder).get(attr_name)
        original_callable = getattr(encoder, attr_name)

        def cache_aware_stream_step_wrapper(*args, **call_kwargs):
            if args:
                self._encoder_cudagraph_eager_fallbacks += 1
                return self._gil_attrib_timed_call(
                    "dispatch_ms",
                    original_callable,
                    *args,
                    cuda_event=True,
                    **call_kwargs,
                )
            try:
                inputs = EncoderGraphInputs(
                    processed_signal=call_kwargs["processed_signal"],
                    processed_signal_length=call_kwargs["processed_signal_length"],
                    cache_last_channel=call_kwargs["cache_last_channel"],
                    cache_last_time=call_kwargs["cache_last_time"],
                    cache_last_channel_len=call_kwargs["cache_last_channel_len"],
                )
            except Exception:
                self._encoder_cudagraph_eager_fallbacks += 1
                return self._gil_attrib_timed_call(
                    "dispatch_ms",
                    original_callable,
                    cuda_event=True,
                    **call_kwargs,
                )

            replay_outputs = self._gil_attrib_timed_call(
                "dispatch_ms",
                manager.replay,
                batch_size,
                inputs,
                cuda_event=True,
            )
            if replay_outputs is None:
                self._encoder_cudagraph_eager_fallbacks += 1
                return self._gil_attrib_timed_call(
                    "dispatch_ms",
                    original_callable,
                    cuda_event=True,
                    **call_kwargs,
                )

            self._encoder_cudagraph_replay_calls += 1
            if (
                self._encoder_cudagraph_replay_calls <= 5
                or self._encoder_cudagraph_replay_calls % 50 == 0
            ):
                logger.info(
                    "encoder_cuda_graph_status "
                    f"replays={self._encoder_cudagraph_replay_calls} "
                    f"fallbacks={self._encoder_cudagraph_eager_fallbacks} "
                    f"B={batch_size}"
                )
            return replay_outputs

        object.__setattr__(encoder, attr_name, cache_aware_stream_step_wrapper)
        try:
            yield
        finally:
            if had_instance_attr:
                object.__setattr__(encoder, attr_name, original_instance_attr)
            else:
                object.__delattr__(encoder, attr_name)

    @contextlib.contextmanager
    def _finalize_cudagraph_encoder_cache_step_installed(
        self,
        model: Any,
        manager: BucketedCudaGraphEncoder,
        key: FinalizeEncoderGraphKey,
        profile: Optional[dict[str, Any]],
    ):
        encoder = model.encoder
        attr_name = "cache_aware_stream_step"
        had_instance_attr = attr_name in vars(encoder)
        original_instance_attr = vars(encoder).get(attr_name)
        original_callable = getattr(encoder, attr_name)
        state = {"replayed": False}

        def cache_aware_stream_step_wrapper(*args, **call_kwargs):
            self._finalize_profile_cuda_synchronize(profile)
            start_event = None
            end_event = None
            if torch.cuda.is_available():
                try:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                except Exception:
                    start_event = None
                    end_event = None

            start = time.perf_counter()
            if start_event is not None:
                start_event.record()

            replay_outputs = None
            if args:
                if profile is not None:
                    profile["encoder_finalize_cudagraph"] = "fallback"
                    profile["encoder_finalize_cudagraph_reason"] = "positional_args"
                self._record_encoder_finalize_cudagraph_fallback(
                    key,
                    reason="positional_args",
                )
                result = self._gil_attrib_timed_call(
                    "dispatch_ms",
                    original_callable,
                    *args,
                    cuda_event=True,
                    **call_kwargs,
                )
            else:
                real_time_steps = None
                try:
                    processed_signal = call_kwargs["processed_signal"]
                    if torch.is_tensor(processed_signal) and processed_signal.ndim >= 1:
                        real_time_steps = int(processed_signal.shape[-1])
                    inputs = EncoderGraphInputs(
                        processed_signal=processed_signal,
                        processed_signal_length=call_kwargs["processed_signal_length"],
                        cache_last_channel=call_kwargs["cache_last_channel"],
                        cache_last_time=call_kwargs["cache_last_time"],
                        cache_last_channel_len=call_kwargs["cache_last_channel_len"],
                    )
                except Exception:
                    if profile is not None:
                        profile["encoder_finalize_cudagraph"] = "fallback"
                        profile["encoder_finalize_cudagraph_reason"] = "invalid_inputs"
                    self._record_encoder_finalize_cudagraph_fallback(
                        key,
                        reason="invalid_inputs",
                        real_time_steps=real_time_steps,
                    )
                    result = self._gil_attrib_timed_call(
                        "dispatch_ms",
                        original_callable,
                        cuda_event=True,
                        **call_kwargs,
                    )
                else:
                    if self.encoder_cudagraph_finalize_padded_requested:
                        replay_outputs = self._gil_attrib_timed_call(
                            "dispatch_ms",
                            manager.replay_finalize_padded,
                            key,
                            inputs,
                            cuda_event=True,
                        )
                    else:
                        replay_outputs = self._gil_attrib_timed_call(
                            "dispatch_ms",
                            manager.replay_finalize,
                            key,
                            inputs,
                            cuda_event=True,
                        )
                    if replay_outputs is None:
                        replay_error = manager.finalize_replay_error(key)
                        if profile is not None:
                            profile["encoder_finalize_cudagraph"] = "fallback"
                            profile["encoder_finalize_cudagraph_reason"] = (
                                replay_error or "replay_returned_none"
                            )
                        self._record_encoder_finalize_cudagraph_fallback(
                            key,
                            reason=replay_error or "replay_returned_none",
                            real_time_steps=real_time_steps,
                        )
                        result = self._gil_attrib_timed_call(
                            "dispatch_ms",
                            original_callable,
                            cuda_event=True,
                            **call_kwargs,
                        )
                    else:
                        state["replayed"] = True
                        if profile is not None:
                            profile["encoder_finalize_cudagraph"] = "replay"
                            profile["real_T"] = (
                                int(real_time_steps)
                                if real_time_steps is not None
                                else int(key.time_steps)
                            )
                            profile["graph_T"] = int(key.time_steps)
                            profile["encoder_finalize_cudagraph_key"] = {
                                "B": int(key.batch_size),
                                "graph_T": int(key.time_steps),
                                "real_T": (
                                    int(real_time_steps)
                                    if real_time_steps is not None
                                    else int(key.time_steps)
                                ),
                                "drop_extra": int(key.drop_extra),
                                "mode": (
                                    "padded_T_max"
                                    if self.encoder_cudagraph_finalize_padded_requested
                                    else "per_T"
                                ),
                            }
                        self._record_encoder_finalize_cudagraph_replay(
                            key,
                            real_time_steps=real_time_steps,
                        )
                        result = replay_outputs

            if end_event is not None:
                end_event.record()
            self._finalize_profile_cuda_synchronize(profile)
            wall_ms = (time.perf_counter() - start) * 1000
            if profile is not None:
                profile["encoder_wall_ms"] = float(
                    profile.get("encoder_wall_ms") or 0.0
                ) + wall_ms
                profile["encoder_invocations"] = int(
                    profile.get("encoder_invocations") or 0
                ) + 1
                if start_event is not None and end_event is not None:
                    try:
                        profile["encoder_cuda_event_ms"] = float(
                            profile.get("encoder_cuda_event_ms") or 0.0
                        ) + float(start_event.elapsed_time(end_event))
                    except Exception:
                        pass
                self._finalize_profile_record_encoder_outputs(profile, result)
            return result

        object.__setattr__(encoder, attr_name, cache_aware_stream_step_wrapper)
        try:
            yield state
        finally:
            if had_instance_attr:
                object.__setattr__(encoder, attr_name, original_instance_attr)
            else:
                object.__delattr__(encoder, attr_name)

    @staticmethod
    def _finalize_profile_shape(value: Any) -> Optional[list[int]]:
        if not torch.is_tensor(value):
            return None
        return [int(dim) for dim in value.shape]

    @staticmethod
    def _finalize_profile_first_key(first: Any) -> str:
        if first is True:
            return "first"
        if first is False:
            return "non_first"
        return "unknown"

    @staticmethod
    def _finalize_profile_tensor_values(
        value: Any,
        *,
        index: Optional[int] = None,
    ) -> Optional[list[int | float]]:
        if not torch.is_tensor(value):
            return None
        try:
            tensor = value.detach()
            if index is not None and tensor.ndim > 0 and int(tensor.shape[0]) > index:
                tensor = tensor[index : index + 1]
            flat = tensor.reshape(-1).cpu().tolist()
        except Exception:
            return None
        values: list[int | float] = []
        for item in flat:
            number = float(item)
            if number.is_integer():
                values.append(int(number))
            else:
                values.append(number)
        return values

    @classmethod
    def _finalize_profile_tensor_value(
        cls,
        value: Any,
        *,
        index: Optional[int] = None,
    ) -> Any:
        values = cls._finalize_profile_tensor_values(value, index=index)
        if values is None:
            return None
        if len(values) == 1:
            return values[0]
        return values

    @staticmethod
    def _finalize_profile_event_queued_perf(event: tuple) -> Optional[float]:
        if len(event) <= 2:
            return None
        value = event[2]
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _new_finalize_profile(
        self,
        session: ASRSession,
        *,
        reason: str,
        path: str,
        debounce_event_queued_perf: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        if not self.finalize_profile_enabled:
            return None

        start_perf = time.perf_counter()
        queue_wait_ms = None
        if debounce_event_queued_perf is not None:
            queue_wait_ms = max(0.0, (start_perf - debounce_event_queued_perf) * 1000)

        return {
            "_start_perf": start_perf,
            "session_id": session.id,
            "reason": reason,
            "path": path,
            "start_unix": time.time(),
            "queue_wait_ms": queue_wait_ms,
            "debounce_wait_ms": None,
            "lock_wait_ms": None,
            "fork_flush_wall_ms": None,
            # vad_stop -> final_sent TRIGGER-path decomposition (the half FINALIZE_PROFILE
            # previously left dark; all from the time.time() stamps in _continuous_finalize_timing).
            "vad_stop_to_finalize_start_ms": None,   # vad_stop -> fork_flush_start (the trigger latency)
            "debounce_to_finalize_start_ms": None,   # debounce_expiry -> fork_flush_start (post-debounce pickup)
            "finalize_done_to_sent_ms": None,        # fork_flush_done -> final_sent (emit)
            "vad_stop_to_sent_ms": None,             # vad_stop -> final_sent (total server-side path)
            "vad_stop_recv_to_process_ms": None,     # vad_stop RECEIVED off socket -> scheduler-PROCESSED (end-of-stream backlog)
            # Last-stage finalize gather (the "final gather / token Python" suspect): Python
            # stack+clone NOT otherwise broken out of fork_flush_wall; cost scales w/ hyp/token length.
            "final_gather_ms": None,                 # stack inputs + clone hyps/pred + stack (6792-6814)
            "clone_hyp_flush_ms": None,              # just clone_hypotheses_deep + clone_tree (per-row)
            "fork_clone_ms": 0.0,
            "fork_clone_audio_ms": 0.0,
            "fork_clone_cache_ms": 0.0,
            "fork_clone_hyps_ms": 0.0,
            "fork_clone_pred_ms": 0.0,
            "fork_clone_other_ms": 0.0,
            "preproc_wall_ms": 0.0,
            "preproc_invocations": 0,
            "model_wall_ms": None,
            "encoder_wall_ms": None,
            "encoder_cuda_event_ms": None,
            "encoder_invocations": 0,
            "encoder_finalize_cudagraph": None,
            "encoder_finalize_cudagraph_key": None,
            "encoder_finalize_cudagraph_reason": None,
            "real_T": None,
            "graph_T": None,
            "decode_wall_ms": None,
            "cuda_sync_ms": 0.0,
            "cuda_sync_invocations": 0,
            "B": None,
            "T": None,
            "processed_signal_length": None,
            "drop_extra": None,
            "first": None,
            "encoded_shape": None,
            "encoded_len": None,
            "cache_present": None,
            "cache_last_channel_shape": None,
            "cache_last_channel_len": None,
            "cache_last_channel_out_shape": None,
            "cache_last_channel_len_out": None,
            "att_context": list(self.att_context_size)
            if self.att_context_size is not None
            else None,
            "scheduler_enabled": self.scheduler_enabled,
            "batch_finalize": self._scheduler_batch_finalize_active(),
            "batch_finalize_preproc": self._scheduler_batch_finalize_preproc_active(),
            "model_lanes": self.model_lanes,
            "gil_attrib_enabled": self.gil_attrib_enabled,
        }

    def _finalize_profile_cuda_synchronize(
        self,
        profile: Optional[dict[str, Any]],
    ) -> float:
        if profile is None or not torch.cuda.is_available():
            return 0.0
        start = time.perf_counter()
        self._cuda_synchronize_for_current_model_lane()
        elapsed_ms = (time.perf_counter() - start) * 1000
        profile["cuda_sync_ms"] = float(profile.get("cuda_sync_ms") or 0.0) + elapsed_ms
        profile["cuda_sync_invocations"] = int(
            profile.get("cuda_sync_invocations") or 0
        ) + 1
        return elapsed_ms

    @staticmethod
    def _finalize_profile_add_cuda_sync_observed(
        profile: Optional[dict[str, Any]],
        elapsed_ms: float,
    ) -> None:
        if profile is None:
            return
        profile["cuda_sync_ms"] = float(profile.get("cuda_sync_ms") or 0.0) + elapsed_ms
        profile["cuda_sync_invocations"] = int(
            profile.get("cuda_sync_invocations") or 0
        ) + 1

    def _finalize_profile_cuda_synchronize_many(
        self,
        profiles: list[Optional[dict[str, Any]]],
    ) -> None:
        live_profiles = [profile for profile in profiles if profile is not None]
        if not live_profiles:
            self._cuda_synchronize_for_current_model_lane()
            return
        sync_ms = self._finalize_profile_cuda_synchronize(live_profiles[0])
        for profile in live_profiles[1:]:
            self._finalize_profile_add_cuda_sync_observed(profile, sync_ms)

    def _finalize_profile_add_preproc(
        self,
        profile: Optional[dict[str, Any]],
        wall_ms: float,
        *,
        invocations: int = 1,
    ) -> None:
        if profile is None:
            return
        profile["preproc_wall_ms"] = float(profile.get("preproc_wall_ms") or 0.0) + wall_ms
        profile["preproc_invocations"] = int(
            profile.get("preproc_invocations") or 0
        ) + invocations

    def _finalize_profile_set_model_inputs(
        self,
        profile: Optional[dict[str, Any]],
        *,
        processed_signal: Any,
        processed_signal_length: Any,
        cache_last_channel: Any,
        cache_last_channel_len: Any,
        drop_extra: int,
        first: bool,
        row_index: Optional[int] = None,
    ) -> None:
        if profile is None:
            return
        if torch.is_tensor(processed_signal) and processed_signal.ndim >= 3:
            profile["B"] = int(processed_signal.shape[0])
            profile["T"] = int(processed_signal.shape[-1])
            profile["real_T"] = int(processed_signal.shape[-1])
        profile["processed_signal_length"] = self._finalize_profile_tensor_value(
            processed_signal_length,
            index=row_index,
        )
        profile["drop_extra"] = int(drop_extra)
        profile["first"] = bool(first)
        profile["cache_present"] = cache_last_channel is not None
        cache_shape = self._finalize_profile_shape(cache_last_channel)
        if row_index is not None and isinstance(cache_shape, list):
            profile["cache_last_channel_batch_shape"] = cache_shape
            profile["cache_last_channel_shape"] = (
                [1, *cache_shape[1:]] if cache_shape else cache_shape
            )
        else:
            profile["cache_last_channel_shape"] = cache_shape
        profile["cache_last_channel_len"] = self._finalize_profile_tensor_value(
            cache_last_channel_len,
            index=row_index,
        )
        profile["att_context"] = (
            list(self.att_context_size) if self.att_context_size is not None else None
        )

    def _finalize_profile_record_encoder_outputs(
        self,
        profile: Optional[dict[str, Any]],
        result: Any,
    ) -> None:
        if profile is None:
            return
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            return
        encoded = result[0]
        encoded_len = result[1]
        profile["encoded_shape"] = self._finalize_profile_shape(encoded)
        encoded_len_values = self._finalize_profile_tensor_values(encoded_len)
        if encoded_len_values is not None:
            profile["encoded_len_values"] = encoded_len_values
            profile["encoded_len"] = (
                encoded_len_values[0]
                if len(encoded_len_values) == 1
                else encoded_len_values
            )
        if len(result) >= 5:
            profile["cache_last_channel_out_shape"] = self._finalize_profile_shape(
                result[2]
            )
            profile["cache_last_channel_len_out"] = self._finalize_profile_tensor_value(
                result[4]
            )

    def _maybe_finalize_torch_profile_begin(self):
        # Kernel-level finalize profile (NEMOTRON_FINALIZE_TORCH_PROFILE=N profiles the first N finalize model
        # calls). Observation-only (does NOT change outputs) -> byte-exact when off (returns None -> no-op).
        n = int(os.environ.get("NEMOTRON_FINALIZE_TORCH_PROFILE", "0"))
        if n <= 0 or getattr(self, "_finalize_torch_profile_count", 0) >= n:
            return None
        try:
            from torch.profiler import profile as _tp, ProfilerActivity as _pa
            prof = _tp(activities=[_pa.CPU, _pa.CUDA])
            prof.__enter__()
            return prof
        except Exception as e:  # noqa: BLE001
            logger.warning(f"finalize_torch_profile begin failed: {e}")
            return None

    def _maybe_finalize_torch_profile_end(self, prof, *, b, t) -> None:
        if prof is None:
            return
        try:
            prof.__exit__(None, None, None)
            self._finalize_torch_profile_count = getattr(self, "_finalize_torch_profile_count", 0) + 1
            ka = prof.key_averages()
            cuda_us = sum(e.self_device_time_total for e in ka)  # torch 2.11: cuda->device rename
            cpu_us = sum(e.self_cpu_time_total for e in ka)
            kernels = sum(e.count for e in ka if e.self_device_time_total > 0)
            low = lambda s: s.lower()  # noqa: E731
            hit = lambda name, subs: any(s in low(name) for s in subs)  # noqa: E731
            memcpy = sum(e.count for e in ka if hit(e.key, ("memcpy", "copy_", "item", "_local_scalar", "nonzero", "_to_copy")))
            sync = sum(e.count for e in ka if hit(e.key, ("synchron",)))
            logger.info(
                f"finalize_torch_profile #{self._finalize_torch_profile_count} B={b} T={t}: "
                f"kernel_launches={kernels} self_cuda_us={cuda_us:.0f} self_cpu_us={cpu_us:.0f} "
                f"cpu/cuda={cpu_us / max(cuda_us, 1e-9):.2f} (>>1 = launch-bound) "
                f"memcpy_d2h_ops={memcpy} sync_ops={sync}"
            )
            logger.info("finalize_torch_profile TOP-BY-CUDA:\n" + ka.table(sort_by="self_cuda_time_total", row_limit=12))
            logger.info("finalize_torch_profile TOP-BY-COUNT:\n" + ka.table(sort_by="count", row_limit=12))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"finalize_torch_profile end failed: {e}")

    def _finalize_profile_set_model_wall(
        self,
        profile: Optional[dict[str, Any]],
        wall_ms: float,
    ) -> None:
        if profile is None:
            return
        profile["model_wall_ms"] = wall_ms
        encoder_ms = profile.get("encoder_wall_ms")
        if isinstance(encoder_ms, (int, float)):
            profile["decode_wall_ms"] = max(0.0, wall_ms - float(encoder_ms))

    def _finalize_profile_copy_model_profile(
        self,
        dst: Optional[dict[str, Any]],
        src: Optional[dict[str, Any]],
        *,
        row_index: int,
    ) -> None:
        if dst is None or src is None:
            return
        for key in (
            "model_wall_ms",
            "encoder_wall_ms",
            "encoder_cuda_event_ms",
            "encoder_invocations",
            "encoder_finalize_cudagraph",
            "encoder_finalize_cudagraph_key",
            "encoder_finalize_cudagraph_reason",
            "real_T",
            "graph_T",
            "decode_wall_ms",
            "cuda_sync_ms",
            "cuda_sync_invocations",
        ):
            value = src.get(key)
            if key in ("real_T", "graph_T") and value is None:
                continue
            dst[key] = value

        encoded_shape = src.get("encoded_shape")
        if isinstance(encoded_shape, list):
            dst["encoded_batch_shape"] = encoded_shape
            dst["encoded_shape"] = [1, *encoded_shape[1:]] if encoded_shape else encoded_shape
        values = src.get("encoded_len_values")
        if isinstance(values, list) and row_index < len(values):
            dst["encoded_len"] = values[row_index]

    def _emit_finalize_profile_record(
        self,
        profile: Optional[dict[str, Any]],
        *,
        timing: Optional[dict[str, Any]],
        final_text: Optional[str],
        delta_text: Optional[str],
        emitted_to_client: bool,
        suppressed_reason: Optional[str],
        should_flush: bool,
    ) -> None:
        # Always-on stats sample (cheap, no profile required) — populates the
        # /stats sliding window even when NEMOTRON_FINALIZE_PROFILE is off.
        self._record_stats_sample(timing, emitted_to_client=emitted_to_client)

        if profile is None:
            return

        if timing is not None:
            profile["lock_wait_ms"] = timing.get("inference_lock_acquire_wait_ms")
            vad_stop = timing.get("vad_stop")
            debounce_expiry = timing.get("debounce_expiry")
            if vad_stop is not None and debounce_expiry is not None:
                profile["debounce_wait_ms"] = max(
                    0.0,
                    (float(debounce_expiry) - float(vad_stop)) * 1000,
                )
            fork_start = timing.get("fork_flush_start")
            fork_done = timing.get("fork_flush_done")
            if fork_start is not None and fork_done is not None:
                profile["fork_flush_wall_ms"] = max(
                    0.0,
                    (float(fork_done) - float(fork_start)) * 1000,
                )
            # Trigger-path decomposition (vad_stop -> finalize-start -> sent). These localize the
            # dark ~half of client TTFS that finalize_wall (fork_flush) does NOT cover.
            final_sent = timing.get("final_sent")
            if vad_stop is not None and fork_start is not None:
                profile["vad_stop_to_finalize_start_ms"] = max(
                    0.0, (float(fork_start) - float(vad_stop)) * 1000
                )
            if debounce_expiry is not None and fork_start is not None:
                profile["debounce_to_finalize_start_ms"] = max(
                    0.0, (float(fork_start) - float(debounce_expiry)) * 1000
                )
            if fork_done is not None and final_sent is not None:
                profile["finalize_done_to_sent_ms"] = max(
                    0.0, (float(final_sent) - float(fork_done)) * 1000
                )
            if vad_stop is not None and final_sent is not None:
                profile["vad_stop_to_sent_ms"] = max(
                    0.0, (float(final_sent) - float(vad_stop)) * 1000
                )
            vad_stop_recv = timing.get("vad_stop_recv")
            if vad_stop_recv is not None and vad_stop is not None:
                profile["vad_stop_recv_to_process_ms"] = max(  # end-of-stream backlog: socket-recv -> scheduler-process
                    0.0, (float(vad_stop) - float(vad_stop_recv)) * 1000
                )

        start_perf = profile.get("_start_perf")
        if isinstance(start_perf, (int, float)):
            profile["finalize_wall_ms"] = (time.perf_counter() - float(start_perf)) * 1000

        profile["final_text_chars"] = len(final_text or "")
        profile["delta_text_chars"] = len(delta_text or "")
        profile["emitted_to_client"] = bool(emitted_to_client)
        profile["suppressed_reason"] = suppressed_reason
        profile["should_flush"] = bool(should_flush)
        profile["bucket_key"] = {
            "B": int(profile.get("B") or 0),
            "T": int(profile.get("T") or 0),
            "real_T": int(profile.get("real_T") or profile.get("T") or 0),
            "graph_T": profile.get("graph_T"),
            "drop_extra": profile.get("drop_extra"),
            "first": self._finalize_profile_first_key(profile.get("first")),
        }

        self._finalize_profile_records += 1
        profile["record_index"] = self._finalize_profile_records
        bucket = profile["bucket_key"]
        hist_key = (
            int(bucket["B"]),
            int(bucket["T"]),
            bucket["drop_extra"],
            str(bucket["first"]),
        )
        self._finalize_profile_hist[hist_key] = (
            self._finalize_profile_hist.get(hist_key, 0) + 1
        )
        b_value = int(bucket["B"])
        self._finalize_profile_b_hist[b_value] = (
            self._finalize_profile_b_hist.get(b_value, 0) + 1
        )

        public_record = {
            key: value
            for key, value in profile.items()
            if not key.startswith("_") and key != "encoded_len_values"
        }
        logger.info(
            "finalize_profile_record "
            + json.dumps(public_record, sort_keys=True, default=str)
        )

        if self._finalize_profile_records % self._finalize_profile_hist_every == 0:
            self._log_finalize_profile_histogram(reason="periodic")

    def _log_finalize_profile_histogram(self, *, reason: str) -> None:
        if not self.finalize_profile_enabled:
            return
        bucket_hist = [
            {
                "B": key[0],
                "T": key[1],
                "drop_extra": key[2],
                "first": key[3],
                "count": count,
            }
            for key, count in sorted(
                self._finalize_profile_hist.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        payload = {
            "reason": reason,
            "records": self._finalize_profile_records,
            "bucket_hist": bucket_hist,
            "B_hist": {
                str(key): value
                for key, value in sorted(self._finalize_profile_b_hist.items())
            },
        }
        logger.info(
            "finalize_profile_histogram "
            + json.dumps(payload, sort_keys=True, default=str)
        )

    def _gil_attrib_current_sample(self) -> Optional[dict[str, Any]]:
        if not self.gil_attrib_enabled:
            return None
        sample = getattr(self._gil_attrib_tls, "sample", None)
        return sample if isinstance(sample, dict) else None

    def _gil_attrib_begin_sample(
        self,
        kind: str,
        *,
        path: str,
        batch_size: int,
        inference_lock_wait_ms: float = 0.0,
    ) -> Optional[dict[str, Any]]:
        if not self.gil_attrib_enabled:
            return None
        if self._gil_attrib_current_sample() is not None:
            return None
        sample: dict[str, Any] = {
            "kind": kind,
            "path": path,
            "batch_size": int(batch_size),
            "start_perf": time.perf_counter(),
            "decode_ms": 0.0,
            "dispatch_ms": 0.0,
            "scatter_gather_ms": 0.0,
            "host_sync_ms": 0.0,
            "inference_lock_wait_ms": max(0.0, float(inference_lock_wait_ms or 0.0)),
            "gpu_cuda_event_ms": 0.0,
            "cuda_events": [],
            "ended": False,
        }
        self._gil_attrib_tls.sample = sample
        return sample

    def _gil_attrib_add_ms(self, bucket: str, elapsed_ms: float) -> None:
        sample = self._gil_attrib_current_sample()
        if sample is None:
            return
        sample[bucket] = float(sample.get(bucket) or 0.0) + max(
            0.0,
            float(elapsed_ms),
        )

    def _gil_attrib_cuda_event_start(self) -> Optional[tuple[Any, Any]]:
        if self._gil_attrib_current_sample() is None or not torch.cuda.is_available():
            return None
        try:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            return start_event, end_event
        except Exception:
            return None

    def _gil_attrib_cuda_event_end(self, token: Optional[tuple[Any, Any]]) -> None:
        sample = self._gil_attrib_current_sample()
        if sample is None or token is None:
            return
        try:
            _start_event, end_event = token
            end_event.record()
            sample["cuda_events"].append(token)
        except Exception:
            pass

    def _gil_attrib_timed_call(self, bucket: str, fn, *args, cuda_event: bool = True, **kwargs):
        sample = self._gil_attrib_current_sample()
        if sample is None:
            return fn(*args, **kwargs)
        token = self._gil_attrib_cuda_event_start() if cuda_event else None
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            self._gil_attrib_cuda_event_end(token)
            self._gil_attrib_add_ms(bucket, (time.perf_counter() - start) * 1000.0)

    def _gil_attrib_finish_sample(self, sample: Optional[dict[str, Any]]) -> None:
        if not self.gil_attrib_enabled or sample is None:
            return
        if sample.get("observed"):
            return

        wall_ms = max(0.0, (time.perf_counter() - float(sample["start_perf"])) * 1000.0)
        inference_lock_wait_ms = max(
            0.0,
            float(sample.get("inference_lock_wait_ms") or 0.0),
        )
        thread_busy_ms = wall_ms + inference_lock_wait_ms

        cuda_event_ms = float(sample.get("gpu_cuda_event_ms") or 0.0)
        for start_event, end_event in sample.get("cuda_events") or []:
            try:
                cuda_event_ms += float(start_event.elapsed_time(end_event))
            except Exception:
                pass

        decode_ms = float(sample.get("decode_ms") or 0.0)
        dispatch_ms = float(sample.get("dispatch_ms") or 0.0)
        scatter_gather_ms = float(sample.get("scatter_gather_ms") or 0.0)
        host_sync_ms = float(sample.get("host_sync_ms") or 0.0)
        scheduling_socket_io_ms = (
            thread_busy_ms
            - decode_ms
            - dispatch_ms
            - scatter_gather_ms
            - host_sync_ms
            - inference_lock_wait_ms
        )
        glue_ms = (
            scatter_gather_ms
            + host_sync_ms
            + inference_lock_wait_ms
            + scheduling_socket_io_ms
        )
        bucket_sum_ms = decode_ms + dispatch_ms + glue_ms
        sanity_delta_ms = thread_busy_ms - bucket_sum_ms
        gpu_idle_pct = None
        if thread_busy_ms > 0.0 and cuda_event_ms > 0.0:
            gpu_idle_pct = max(0.0, min(100.0, (1.0 - cuda_event_ms / thread_busy_ms) * 100.0))

        metrics = {
            "thread_busy_ms": thread_busy_ms,
            "body_wall_ms": wall_ms,
            "decode_ms": decode_ms,
            "dispatch_ms": dispatch_ms,
            "glue_ms": glue_ms,
            "glue_scatter_gather_ms": scatter_gather_ms,
            "glue_host_sync_ms": host_sync_ms,
            "glue_inference_lock_wait_ms": inference_lock_wait_ms,
            "glue_scheduling_socket_io_ms": scheduling_socket_io_ms,
            "bucket_sum_ms": bucket_sum_ms,
            "bucket_sum_delta_ms": sanity_delta_ms,
            "gpu_cuda_event_ms": cuda_event_ms,
        }
        if gpu_idle_pct is not None:
            metrics["gpu_idle_pct_while_thread_busy"] = gpu_idle_pct
        if thread_busy_ms > 0.0:
            for key in (
                "decode_ms",
                "dispatch_ms",
                "glue_ms",
                "glue_scatter_gather_ms",
                "glue_host_sync_ms",
                "glue_inference_lock_wait_ms",
                "glue_scheduling_socket_io_ms",
            ):
                metrics[key.replace("_ms", "_pct_thread_busy")] = (
                    metrics[key] / thread_busy_ms
                ) * 100.0

        kind = str(sample.get("kind") or "unknown")
        path = str(sample.get("path") or "unknown")
        batch_size = int(sample.get("batch_size") or 0)
        with self._gil_attrib_lock:
            kind_metrics = self._gil_attrib_samples.setdefault(kind, {})
            for key, value in metrics.items():
                kind_metrics.setdefault(key, []).append(float(value))
            self._gil_attrib_batch_hist.setdefault(kind, {})[batch_size] = (
                self._gil_attrib_batch_hist.setdefault(kind, {}).get(batch_size, 0) + 1
            )
            self._gil_attrib_paths.setdefault(kind, {})[path] = (
                self._gil_attrib_paths.setdefault(kind, {}).get(path, 0) + 1
            )

        sample["observed"] = True
        if getattr(self._gil_attrib_tls, "sample", None) is sample:
            self._gil_attrib_tls.sample = None
        if getattr(self._gil_attrib_tls, "pending_sample", None) is sample:
            self._gil_attrib_tls.pending_sample = None

    def _gil_attrib_end_sample(self, sample: Optional[dict[str, Any]]) -> None:
        if sample is None or not self.gil_attrib_enabled:
            return
        if sample.get("ended"):
            return
        sample["ended"] = True
        if getattr(self._scheduler_model_lane_tls, "stream", None) is not None:
            self._gil_attrib_tls.pending_sample = sample
            return
        self._gil_attrib_finish_sample(sample)

    def _gil_attrib_finish_deferred_sample(self) -> None:
        sample = getattr(self._gil_attrib_tls, "pending_sample", None)
        if isinstance(sample, dict):
            self._gil_attrib_finish_sample(sample)

    def _gil_attrib_record_loop_lag(self, lag_ms: float) -> None:
        if not self.gil_attrib_enabled:
            return
        with self._gil_attrib_lock:
            self._gil_attrib_loop_lag_ms.append(max(0.0, float(lag_ms)))

    async def _gil_attrib_loop_lag_probe(self) -> None:
        interval_ms = _env_float("NEMOTRON_GIL_ATTRIB_LOOP_INTERVAL_MS", 5.0)
        interval = max(0.001, interval_ms / 1000.0)
        loop = asyncio.get_running_loop()
        expected = loop.time() + interval
        while True:
            await asyncio.sleep(max(0.0, expected - loop.time()))
            now = loop.time()
            self._gil_attrib_record_loop_lag((now - expected) * 1000.0)
            expected += interval
            if now - expected > interval:
                expected = now + interval

    @staticmethod
    def _gil_attrib_percentiles(values: list[float]) -> Optional[dict[str, float]]:
        if not values:
            return None
        ordered = sorted(float(value) for value in values)
        count = len(ordered)

        def percentile(pct: float) -> float:
            if count == 1:
                return ordered[0]
            rank = (count - 1) * (pct / 100.0)
            lower = int(rank)
            upper = min(lower + 1, count - 1)
            weight = rank - lower
            return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

        return {
            "p50": percentile(50.0),
            "p95": percentile(95.0),
            "p99": percentile(99.0),
        }

    def _gil_attrib_emit_record(self, *, reason: str) -> None:
        if not self.gil_attrib_enabled:
            return
        with self._gil_attrib_lock:
            samples_snapshot = {
                kind: {metric: list(values) for metric, values in metrics.items()}
                for kind, metrics in self._gil_attrib_samples.items()
            }
            batch_hist_snapshot = {
                kind: dict(hist) for kind, hist in self._gil_attrib_batch_hist.items()
            }
            paths_snapshot = {
                kind: dict(paths) for kind, paths in self._gil_attrib_paths.items()
            }
            loop_lag_snapshot = list(self._gil_attrib_loop_lag_ms)

        operations: dict[str, Any] = {}
        for kind, metrics in sorted(samples_snapshot.items()):
            metric_summary = {
                metric: self._gil_attrib_percentiles(values)
                for metric, values in sorted(metrics.items())
            }
            sample_count = len(next(iter(metrics.values()))) if metrics else 0
            operations[kind] = {
                "samples": sample_count,
                "batch_size_hist": {
                    str(key): value
                    for key, value in sorted(batch_hist_snapshot.get(kind, {}).items())
                },
                "paths": dict(sorted(paths_snapshot.get(kind, {}).items())),
                "metrics": metric_summary,
            }

        duration_ms = None
        if self._gil_attrib_started_perf is not None:
            duration_ms = (time.perf_counter() - self._gil_attrib_started_perf) * 1000.0
        payload = {
            "schema": "nemotron_gil_attribution_v1",
            "reason": reason,
            "flag": "NEMOTRON_GIL_ATTRIB",
            "enabled": True,
            "probe_only": True,
            "source_schema": "_continuous_finalize_timing + scheduler batch/finalize telemetry",
            "started_unix": self._gil_attrib_started_unix,
            "duration_ms": duration_ms,
            "config": {
                "model_lanes": self.model_lanes,
                "scheduler_enabled": self.scheduler_enabled,
                "batch_enabled": self.batch_enabled,
                "batch_finalize": self._scheduler_batch_finalize_active(),
                "batch_finalize_preproc": self._scheduler_batch_finalize_preproc_active(),
                "encoder_cudagraph": self.encoder_cudagraph_enabled,
                "encoder_cudagraph_finalize": self.encoder_cudagraph_finalize_enabled,
                "sync_compress": self.sync_compress_enabled,
                "finalize_priority": self.finalize_priority_enabled,
            },
            "operations": operations,
            "gil_wait_proxy": {
                "source": "asyncio_event_loop_tick_lag",
                "interval_ms": _env_float("NEMOTRON_GIL_ATTRIB_LOOP_INTERVAL_MS", 5.0),
                "samples": len(loop_lag_snapshot),
                "lag_ms": self._gil_attrib_percentiles(loop_lag_snapshot),
            },
        }
        logger.info(
            "gil_attribution_record "
            + json.dumps(payload, sort_keys=True, default=str)
        )
        self._gil_attrib_record_emitted = True

    @contextlib.contextmanager
    def _gil_attrib_decode_installed(self, model: Any):
        if not self.gil_attrib_enabled:
            yield
            return
        decoding = getattr(model, "decoding", None)
        if decoding is None or not hasattr(decoding, "rnnt_decoder_predictions_tensor"):
            yield
            return
        attr_name = "rnnt_decoder_predictions_tensor"
        had_instance_attr = attr_name in vars(decoding)
        original_instance_attr = vars(decoding).get(attr_name)
        original_callable = getattr(decoding, attr_name)

        def rnnt_decode_wrapper(*args, **kwargs):
            return self._gil_attrib_timed_call(
                "decode_ms",
                original_callable,
                *args,
                cuda_event=True,
                **kwargs,
            )

        object.__setattr__(decoding, attr_name, rnnt_decode_wrapper)
        try:
            yield
        finally:
            if had_instance_attr:
                object.__setattr__(decoding, attr_name, original_instance_attr)
            else:
                object.__delattr__(decoding, attr_name)

    @contextlib.contextmanager
    def _gil_attrib_encoder_cache_step_installed(self, model: Any):
        if not self.gil_attrib_enabled:
            yield
            return
        encoder = getattr(model, "encoder", None)
        if encoder is None or not hasattr(encoder, "cache_aware_stream_step"):
            yield
            return
        attr_name = "cache_aware_stream_step"
        had_instance_attr = attr_name in vars(encoder)
        original_instance_attr = vars(encoder).get(attr_name)
        original_callable = getattr(encoder, attr_name)

        def cache_aware_stream_step_wrapper(*args, **kwargs):
            return self._gil_attrib_timed_call(
                "dispatch_ms",
                original_callable,
                *args,
                cuda_event=True,
                **kwargs,
            )

        object.__setattr__(encoder, attr_name, cache_aware_stream_step_wrapper)
        try:
            yield
        finally:
            if had_instance_attr:
                object.__setattr__(encoder, attr_name, original_instance_attr)
            else:
                object.__delattr__(encoder, attr_name)

    @contextlib.contextmanager
    def _finalize_profile_encoder_cache_step_installed(
        self,
        model: Any,
        profile: Optional[dict[str, Any]],
    ):
        encoder = model.encoder
        attr_name = "cache_aware_stream_step"
        had_instance_attr = attr_name in vars(encoder)
        original_instance_attr = vars(encoder).get(attr_name)
        original_callable = getattr(encoder, attr_name)

        def cache_aware_stream_step_wrapper(*args, **call_kwargs):
            self._finalize_profile_cuda_synchronize(profile)
            start_event = None
            end_event = None
            if torch.cuda.is_available():
                try:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                except Exception:
                    start_event = None
                    end_event = None

            start = time.perf_counter()
            if start_event is not None:
                start_event.record()
            dispatch_start = time.perf_counter()
            dispatch_token = self._gil_attrib_cuda_event_start()
            result = original_callable(*args, **call_kwargs)
            self._gil_attrib_cuda_event_end(dispatch_token)
            self._gil_attrib_add_ms(
                "dispatch_ms",
                (time.perf_counter() - dispatch_start) * 1000,
            )
            if end_event is not None:
                end_event.record()
            self._finalize_profile_cuda_synchronize(profile)
            wall_ms = (time.perf_counter() - start) * 1000
            profile["encoder_wall_ms"] = float(
                profile.get("encoder_wall_ms") or 0.0
            ) + wall_ms
            profile["encoder_invocations"] = int(
                profile.get("encoder_invocations") or 0
            ) + 1
            if start_event is not None and end_event is not None:
                try:
                    profile["encoder_cuda_event_ms"] = float(
                        profile.get("encoder_cuda_event_ms") or 0.0
                    ) + float(start_event.elapsed_time(end_event))
                except Exception:
                    pass
            self._finalize_profile_record_encoder_outputs(profile, result)
            return result

        object.__setattr__(encoder, attr_name, cache_aware_stream_step_wrapper)
        try:
            yield
        finally:
            if had_instance_attr:
                object.__setattr__(encoder, attr_name, original_instance_attr)
            else:
                object.__delattr__(encoder, attr_name)

    def _conformer_stream_step(self, finalize_profile: Optional[dict[str, Any]] = None, **kwargs):
        """Call NeMo's stream step with optional encoder graph/compile swap."""
        model = self._current_inference_model()
        streaming_cfg = model.encoder.streaming_cfg
        original_drop_extra = streaming_cfg.drop_extra_pre_encoded
        cudagraph_manager = self._encoder_cudagraph_manager_for_model(model)
        cudagraph_bucket = self._encoder_cudagraph_bucket_for_call(
            kwargs,
            cudagraph_manager,
        )
        finalize_cudagraph_key = self._encoder_finalize_cudagraph_key_for_call(
            kwargs,
            cudagraph_manager,
        )
        if (
            finalize_cudagraph_key is None
            and finalize_profile is not None
            and self.encoder_cudagraph_finalize_requested
            and kwargs.get("keep_all_outputs", False)
        ):
            finalize_profile["encoder_finalize_cudagraph"] = "fallback"
            finalize_profile["encoder_finalize_cudagraph_reason"] = "gate_or_uncaptured"
        bucket = None
        if cudagraph_bucket is None and finalize_cudagraph_key is None:
            bucket = (
                self._encoder_compile_bucket_for_call(kwargs)
                if model is self.model
                else None
            )
        saw_unwarmed_bucket = False

        def conformer_step_call():
            with self._gil_attrib_decode_installed(model):
                return model.conformer_stream_step(**kwargs)

        try:
            if finalize_cudagraph_key is not None and cudagraph_manager is not None:
                if (
                    self._encoder_cudagraph_thread_id is not None
                    and getattr(self._scheduler_model_lane_tls, "stream", None) is None
                    and threading.get_ident() != self._encoder_cudagraph_thread_id
                ):
                    logger.warning(
                        "encoder_finalize_cuda_graph_call_on_different_thread "
                        f"capture_thread={self._encoder_cudagraph_thread_id} "
                        f"call_thread={threading.get_ident()}"
                    )
                with self._finalize_cudagraph_encoder_cache_step_installed(
                    model,
                    cudagraph_manager,
                    finalize_cudagraph_key,
                    finalize_profile,
                ) as finalize_cudagraph_state:
                    result = conformer_step_call()
                if finalize_cudagraph_state.get("replayed"):
                    result_list = list(result)
                    for result_index in (2, 3, 4):
                        if result_index < len(result_list) and torch.is_tensor(result_list[result_index]):
                            result_list[result_index] = result_list[result_index].detach().clone()
                    return tuple(result_list)
                return result

            if finalize_profile is not None and kwargs.get("keep_all_outputs", False):
                with self._finalize_profile_encoder_cache_step_installed(
                    model,
                    finalize_profile,
                ):
                    return conformer_step_call()

            if cudagraph_bucket is not None and cudagraph_manager is not None:
                if (
                    self._encoder_cudagraph_thread_id is not None
                    and getattr(self._scheduler_model_lane_tls, "stream", None) is None
                    and threading.get_ident() != self._encoder_cudagraph_thread_id
                ):
                    logger.warning(
                        "encoder_cuda_graph_call_on_different_thread "
                        f"capture_thread={self._encoder_cudagraph_thread_id} "
                        f"call_thread={threading.get_ident()}"
                    )
                with self._cudagraph_encoder_cache_step_installed(
                    model,
                    cudagraph_manager,
                    cudagraph_bucket,
                ):
                    result = conformer_step_call()
                result_list = list(result)
                for result_index in (2, 3, 4):
                    if result_index < len(result_list) and torch.is_tensor(result_list[result_index]):
                        result_list[result_index] = result_list[result_index].detach().clone()
                return tuple(result_list)

            if bucket is None:
                with self._gil_attrib_encoder_cache_step_installed(model):
                    return conformer_step_call()

            self._encoder_compile_calls += 1
            if (
                self._encoder_compile_thread_id is not None
                and threading.get_ident() != self._encoder_compile_thread_id
            ):
                logger.warning(
                    "encoder_compile_call_on_different_thread "
                    f"warm_thread={self._encoder_compile_thread_id} "
                    f"call_thread={threading.get_ident()}"
                )
            saw_unwarmed_bucket = (
                self._encoder_compile_warmup_done
                and bucket not in self._encoder_compile_warmed_buckets
            )
            if saw_unwarmed_bucket:
                logger.warning(
                    "encoder_compile_unwarmed_static_bucket_after_warmup "
                    f"bucket_T={bucket[0]} drop_extra={bucket[1]}"
                )

            mark_step_begin = getattr(
                getattr(torch, "compiler", None),
                "cudagraph_mark_step_begin",
                None,
            )
            if mark_step_begin is not None:
                mark_step_begin()
            with self._compiled_encoder_cache_step_installed():
                result = conformer_step_call()
            result_list = list(result)
            for result_index in (2, 3, 4):
                if result_index < len(result_list) and torch.is_tensor(result_list[result_index]):
                    result_list[result_index] = result_list[result_index].detach().clone()
            return tuple(result_list)
        finally:
            streaming_cfg.drop_extra_pre_encoded = original_drop_extra

            if bucket is not None:
                graph_count = self._encoder_compile_counter_snapshot()
                if self._encoder_compile_warmup_done:
                    if graph_count > self._encoder_compile_last_graph_count:
                        delta = graph_count - self._encoder_compile_last_graph_count
                        self._encoder_compile_recapture_events += delta
                        logger.warning(
                            "encoder_compile_recapture_after_warmup "
                            f"delta={delta} total={self._encoder_compile_recapture_events} "
                            f"bucket_T={bucket[0]} drop_extra={bucket[1]}"
                        )
                    elif saw_unwarmed_bucket:
                        self._encoder_compile_recapture_events += 1
                        logger.warning(
                            "encoder_compile_recapture_after_warmup "
                            f"delta=unknown total={self._encoder_compile_recapture_events} "
                            f"bucket_T={bucket[0]} drop_extra={bucket[1]}"
                        )
                self._encoder_compile_last_graph_count = max(
                    self._encoder_compile_last_graph_count,
                    graph_count,
                )
                if self._encoder_compile_calls % 50 == 0:
                    logger.info(
                        "encoder_compile_status "
                        f"compiled_calls={self._encoder_compile_calls} "
                        f"recapture_counter={self._encoder_compile_recapture_events} "
                        f"warmed_buckets={sorted(self._encoder_compile_warmed_buckets)}"
                    )

    async def _run_inference_call(self, fn, *args):
        executor = (
            self._encoder_cudagraph_executor
            if self.encoder_cudagraph_enabled and self._encoder_cudagraph_executor is not None
            else self._encoder_compile_executor
            if self.encoder_compile_enabled
            else None
        )
        return await asyncio.get_event_loop().run_in_executor(
            executor, self._run_inference_call_sync, fn, args
        )

    def _run_inference_call_sync(self, fn, args: tuple):
        previous_route = getattr(self._inference_tls, "model_route", None)
        route = None
        if args and isinstance(args[0], ASRSession):
            route = args[0].model_route
        self._inference_tls.model_route = route or "default"
        try:
            return fn(*args)
        finally:
            self._inference_tls.model_route = previous_route

    def _current_inference_model(self):
        if self.model_lanes > 1:
            lane_model = getattr(self._scheduler_model_lane_tls, "model", None)
            if lane_model is not None:
                return lane_model
        route = getattr(self._inference_tls, "model_route", None)
        if route and route in self.models:
            return self.models[route]
        return self.model

    def _current_model_is_prompted(self) -> bool:
        route = getattr(self._inference_tls, "model_route", None)
        if route and route in self.model_prompted:
            return self.model_prompted[route]
        return self.prompted_model

    def _scheduler_model_lane_condition_obj(self) -> asyncio.Condition:
        if self._scheduler_model_lane_condition is None:
            self._scheduler_model_lane_condition = asyncio.Condition()
        return self._scheduler_model_lane_condition

    def _load_scheduler_model_lane_model(self, lane_id: int):
        import nemo.collections.asr as nemo_asr

        is_local_file = (
            self.model_name_or_path.endswith(".nemo")
            or os.path.exists(self.model_name_or_path)
        )
        if is_local_file:
            logger.info(
                f"model_lane_restore_start lane={lane_id} source=local_file"
            )
            lane_model = nemo_asr.models.ASRModel.restore_from(
                self.model_name_or_path,
                map_location="cpu",
            )
        else:
            logger.info(f"model_lane_restore_start lane={lane_id} source=hf")
            lane_model = nemo_asr.models.ASRModel.from_pretrained(
                self.model_name_or_path,
                map_location="cpu",
            )
        lane_model = lane_model.cuda()
        if hasattr(lane_model, "encoder"):
            lane_model.encoder.set_default_att_context_size(self.att_context_size)
        if self._decoding_cfg_for_lane_models is not None:
            lane_model.change_decoding_strategy(
                decoding_cfg=copy.deepcopy(self._decoding_cfg_for_lane_models)
            )
        lane_model.eval()
        try:
            lane_model.preprocessor.featurizer.dither = 0.0
        except Exception:
            pass
        logger.info(f"model_lane_restore_complete lane={lane_id}")
        return lane_model

    def _ensure_scheduler_model_lane_resources(self) -> None:
        if self.model_lanes <= 1:
            return
        while len(self._scheduler_model_lane_executors) < self.model_lanes:
            lane_id = len(self._scheduler_model_lane_executors)
            if lane_id == 0:
                lane_model = self.model
            else:
                lane_model = self._load_scheduler_model_lane_model(lane_id)
            self._scheduler_model_lane_models.append(lane_model)
            self._scheduler_model_lane_executors.append(
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix=f"nemotron-model-lane-{lane_id}",
                )
            )
            self._scheduler_model_lane_streams.append(
                torch.cuda.Stream() if torch.cuda.is_available() else None
            )

    def _run_scheduler_model_lane_call_sync(self, lane_id: int, fn, args: tuple):
        stream = self._scheduler_model_lane_streams[lane_id]
        model = self._scheduler_model_lane_models[lane_id]
        previous_stream = getattr(self._scheduler_model_lane_tls, "stream", None)
        previous_model = getattr(self._scheduler_model_lane_tls, "model", None)
        self._scheduler_model_lane_tls.stream = stream
        self._scheduler_model_lane_tls.model = model
        try:
            if stream is None:
                return fn(*args)
            with torch.cuda.stream(stream):
                result = fn(*args)
            sync_start = time.perf_counter() if self._gil_attrib_current_sample() is not None else None
            stream.synchronize()
            if sync_start is not None:
                self._gil_attrib_add_ms(
                    "host_sync_ms",
                    (time.perf_counter() - sync_start) * 1000.0,
                )
            return result
        finally:
            self._gil_attrib_finish_deferred_sample()
            self._scheduler_model_lane_tls.stream = previous_stream
            self._scheduler_model_lane_tls.model = previous_model

    async def _run_scheduler_model_lane_call(self, lane_id: int, fn, *args):
        self._ensure_scheduler_model_lane_resources()
        executor = self._scheduler_model_lane_executors[lane_id]
        return await asyncio.get_event_loop().run_in_executor(
            executor,
            self._run_scheduler_model_lane_call_sync,
            lane_id,
            fn,
            args,
        )

    async def _run_scheduler_exclusive_inference_call(
        self,
        fn,
        *args,
        lane_id: int = 0,
    ):
        if self.model_lanes <= 1:
            return await self._run_inference_call(fn, *args)
        return await self._run_scheduler_model_lane_call(lane_id, fn, *args)

    def _cuda_synchronize_for_current_model_lane(self) -> None:
        if not torch.cuda.is_available():
            return
        sync_start = time.perf_counter() if self._gil_attrib_current_sample() is not None else None
        stream = getattr(self._scheduler_model_lane_tls, "stream", None)
        if stream is None:
            torch.cuda.synchronize()
        else:
            stream.synchronize()
        if sync_start is not None:
            self._gil_attrib_add_ms(
                "host_sync_ms",
                (time.perf_counter() - sync_start) * 1000.0,
            )

    @contextlib.asynccontextmanager
    async def _scheduler_exclusive_model_path(self, reason: str):
        if self.model_lanes <= 1:
            yield
            return

        condition = self._scheduler_model_lane_condition_obj()
        async with condition:
            while (
                self._scheduler_inflight_model_lane_tasks
                or self._scheduler_model_lane_exclusive_active
            ):
                await condition.wait()
            self._scheduler_model_lane_exclusive_active = True
        try:
            yield
        finally:
            async with condition:
                self._scheduler_model_lane_exclusive_active = False
                condition.notify_all()
            self._wake_scheduler()

    @staticmethod
    def _scheduler_batch_key_drop_extra(key: tuple) -> Optional[int]:
        try:
            return int(key[2])
        except Exception:
            return None

    def _scheduler_batch_key_parallel_lane_key(self, key: tuple) -> Optional[tuple[Any, int]]:
        """Return the compatibility key for concurrent steady batches.

        Only steady normal chunks may share lanes. First chunks, legacy B=1
        barrier drains, finalization, and any other drop/chunk geometry use the
        session's pinned lane exclusively after all in-flight lanes have
        completed. Flag-gated batched barrier drains enter through the normal
        ready-batch path, so steady barrier chunks can share lanes here.
        """
        drop_extra = self._scheduler_batch_key_drop_extra(key)
        if drop_extra is None:
            return None
        keep_all_outputs = bool(key[1]) if len(key) > 1 else True
        chunk_t = int(key[3]) if len(key) > 3 else -1
        steady_t = int(self.pre_encode_cache_size + self.shift_frames)
        if (
            not keep_all_outputs
            and chunk_t == steady_t
            and drop_extra == int(self.drop_extra)
        ):
            prompt_key = key[0] if self.prompted_model else None
            return (prompt_key, drop_extra)
        return None

    def _scheduler_model_lane_key_can_dispatch(
        self,
        key: tuple,
        *,
        excluded_lanes: Optional[set[int]] = None,
    ) -> bool:
        if self.model_lanes <= 1:
            return True
        if self._scheduler_model_lane_exclusive_active:
            return False
        available_lanes = self._scheduler_available_model_lanes
        if excluded_lanes:
            available_lanes = available_lanes.difference(excluded_lanes)
        parallel_key = self._scheduler_batch_key_parallel_lane_key(key)
        if parallel_key is None:
            return (
                not self._scheduler_inflight_model_lane_tasks
                and bool(available_lanes)
            )
        return (
            bool(available_lanes)
            and (
                self._scheduler_model_lane_active_key is None
                or self._scheduler_model_lane_active_key == parallel_key
            )
        )

    def _scheduler_session_affinity_allows_dispatch(
        self,
        session: ASRSession,
        key: tuple,
        *,
        excluded_lanes: Optional[set[int]] = None,
    ) -> bool:
        if self.model_lanes <= 1:
            return True
        lane_id = self._scheduler_session_model_lane_affinity.get(session.id)
        available_lanes = self._scheduler_available_model_lanes
        if excluded_lanes:
            available_lanes = available_lanes.difference(excluded_lanes)
        return lane_id is None or lane_id in available_lanes

    def _scheduler_assign_session_model_lane(self, session: ASRSession) -> int:
        if self.model_lanes <= 1:
            return 0
        pinned_lane = self._scheduler_session_model_lane_affinity.get(session.id)
        if pinned_lane is not None:
            return pinned_lane

        counts = {lane_id: 0 for lane_id in range(self.model_lanes)}
        for session_id, lane_id in self._scheduler_session_model_lane_affinity.items():
            if session_id in self.sessions and lane_id in counts:
                counts[lane_id] += 1
        lane_id = min(counts, key=lambda candidate: (counts[candidate], candidate))
        self._scheduler_session_model_lane_affinity[session.id] = lane_id
        return lane_id

    def _scheduler_ready_has_model_lane_compatible_key(self) -> bool:
        if self.model_lanes <= 1:
            return True
        excluded_lanes = self._scheduler_finalize_priority_pending_lanes()
        for session_id in list(self._scheduler_ready):
            if session_id in self._scheduler_inflight_sessions:
                continue
            session = self.sessions.get(session_id)
            if session is None:
                continue
            key = self._scheduler_batch_group_key_for_session(session)
            if (
                self._scheduler_model_lane_key_can_dispatch(
                    key,
                    excluded_lanes=excluded_lanes,
                )
                and self._scheduler_session_affinity_allows_dispatch(
                    session,
                    key,
                    excluded_lanes=excluded_lanes,
                )
            ):
                return True
        return False

    def _scheduler_preferred_lane_for_sessions(
        self,
        sessions: list[ASRSession],
    ) -> Optional[int]:
        lanes = {
            self._scheduler_session_model_lane_affinity[session.id]
            for session in sessions
            if session.id in self._scheduler_session_model_lane_affinity
        }
        if len(lanes) == 1:
            return next(iter(lanes))
        return None

    def _scheduler_select_lane_affine_sessions(
        self,
        key: tuple,
        sessions: list[ASRSession],
        *,
        excluded_lanes: Optional[set[int]] = None,
    ) -> list[ASRSession]:
        if self.model_lanes <= 1:
            return sessions
        available = self._scheduler_available_model_lanes
        if excluded_lanes:
            available = available.difference(excluded_lanes)
        if not available:
            return []

        pinned: dict[int, list[ASRSession]] = {}
        unpinned: list[ASRSession] = []
        for session in sessions:
            lane_id = self._scheduler_session_model_lane_affinity.get(session.id)
            if lane_id is None:
                unpinned.append(session)
            elif lane_id in available:
                pinned.setdefault(lane_id, []).append(session)

        if pinned:
            lane_id = max(
                sorted(pinned),
                key=lambda candidate: len(pinned[candidate]),
            )
            selected = pinned[lane_id] + unpinned
        else:
            selected = unpinned
        return selected[: self.batch_max_size]

    def _scheduler_reserve_model_lane_for_key(
        self,
        key: tuple,
        *,
        preferred_lane: Optional[int] = None,
        excluded_lanes: Optional[set[int]] = None,
    ) -> Optional[tuple[int, bool]]:
        if self.model_lanes <= 1:
            return None
        if not self._scheduler_model_lane_key_can_dispatch(
            key,
            excluded_lanes=excluded_lanes,
        ):
            return None

        available_lanes = self._scheduler_available_model_lanes
        if excluded_lanes:
            available_lanes = available_lanes.difference(excluded_lanes)
        parallel_key = self._scheduler_batch_key_parallel_lane_key(key)
        if parallel_key is None:
            if preferred_lane is not None:
                if preferred_lane not in available_lanes:
                    return None
                lane_id = preferred_lane
            else:
                lane_id = min(available_lanes)
            self._scheduler_available_model_lanes.remove(lane_id)
            self._scheduler_model_lane_exclusive_active = True
            return lane_id, True

        if preferred_lane is not None:
            if preferred_lane not in available_lanes:
                return None
            lane_id = preferred_lane
        else:
            lane_id = min(available_lanes)
        self._scheduler_available_model_lanes.remove(lane_id)
        self._scheduler_model_lane_active_key = parallel_key
        return lane_id, False

    async def _scheduler_release_model_lane(
        self,
        *,
        lane_id: int,
        exclusive_lane: bool,
        task: Optional[asyncio.Task],
        session_ids: set[str],
    ) -> None:
        condition = self._scheduler_model_lane_condition_obj()
        async with condition:
            self._scheduler_available_model_lanes.add(lane_id)
            if task is not None:
                self._scheduler_inflight_model_lane_tasks.discard(task)
            self._scheduler_inflight_sessions.difference_update(session_ids)
            if exclusive_lane:
                self._scheduler_model_lane_exclusive_active = False
            if not self._scheduler_inflight_model_lane_tasks:
                self._scheduler_model_lane_active_key = None
            condition.notify_all()
        self._wake_scheduler()

    def _warm_encoder_compile_static_buckets(
        self,
        _base_mel: torch.Tensor,
        on_compile_executor: bool = False,
    ) -> None:
        if not self.encoder_compile_enabled:
            return
        if self._encoder_compile_executor is not None and not on_compile_executor:
            self._encoder_compile_executor.submit(
                self._warm_encoder_compile_static_buckets,
                _base_mel,
                True,
            ).result()
            return

        self._encoder_compile_thread_id = threading.get_ident()

        warmup_frames = self._session_warmup_frames()
        first_bucket = (int(self.shift_frames), 0)
        steady_bucket = (
            int(self.pre_encode_cache_size + self.shift_frames),
            int(self.drop_extra),
        )
        warm_repeats = 2
        profile_chunk = self.profile_chunk
        self.profile_chunk = False
        try:
            for repeat in range(warm_repeats):
                first_session = ASRSession(
                    id=f"encoder_compile_first_{repeat}",
                    websocket=None,
                    target_lang=self.target_lang,
                )
                self._init_session_without_synthetic_warmup(first_session)
                self._queue_silent_compile_chunk(first_session)
                if self._process_chunk(first_session) is None:
                    raise RuntimeError("encoder compile first-bucket warmup failed")

                steady_session = ASRSession(
                    id=f"encoder_compile_steady_{repeat}",
                    websocket=None,
                    target_lang=self.target_lang,
                )
                if warmup_frames is not None:
                    self._init_session(steady_session)
                else:
                    self._init_session_without_synthetic_warmup(steady_session)
                    self._queue_silent_compile_chunk(steady_session)
                    if self._process_chunk(steady_session) is None:
                        raise RuntimeError("encoder compile pre-steady warmup failed")
                self._queue_silent_compile_chunk(steady_session)
                if self._process_chunk(steady_session) is None:
                    raise RuntimeError("encoder compile steady-bucket warmup failed")
        finally:
            self.profile_chunk = profile_chunk

        warmed_labels = [
            f"first:T={first_bucket[0]}:drop={first_bucket[1]}:repeats={warm_repeats}",
            f"steady:T={steady_bucket[0]}:drop={steady_bucket[1]}:repeats={warm_repeats}",
        ]
        self._encoder_compile_warmed_buckets.add(first_bucket)
        self._encoder_compile_warmed_buckets.add(steady_bucket)
        if warmup_frames is not None:
            warmup_bucket = (int(warmup_frames), 0)
            self._encoder_compile_warmed_buckets.add(warmup_bucket)
            warmed_labels.insert(
                0,
                f"warmup:T={warmup_bucket[0]}:drop={warmup_bucket[1]}:repeats={warm_repeats}",
            )

        self._encoder_compile_warmup_done = True
        self._encoder_compile_last_graph_count = self._encoder_compile_counter_snapshot()
        logger.info(
            "encoder_compile_warmup_complete "
            f"buckets={warmed_labels} "
            f"unique_graphs={self._encoder_compile_last_graph_count} "
            f"recapture_counter={self._encoder_compile_recapture_events}"
        )

    def _init_session_without_synthetic_warmup(self, session: ASRSession) -> None:
        self._set_session_language_route(session, session.target_lang)
        previous_route = getattr(self._inference_tls, "model_route", None)
        self._inference_tls.model_route = session.model_route
        try:
            plan = self._plan_for_session(session)
            cache = self._current_inference_model().encoder.get_initial_cache_state(batch_size=1)
            session.cache_last_channel = cache[0]
            session.cache_last_time = cache[1]
            session.cache_last_channel_len = cache[2]
            session.pending_audio = np.array([], dtype=np.float32)
            session.accumulated_audio = session.pending_audio
            session.total_audio_samples = 0
            session.raw_audio_ring = np.zeros(plan.raw_audio_ring_samples, dtype=np.float32)
            session.mel_frame_ring = None
            session.emitted_frames = 0
            session.previous_hypotheses = None
            session.pred_out_stream = None
            session.current_text = ""
            session.eou_probe_chunk_index = 0
            session.synthetic_prefix_samples = 0
        finally:
            self._inference_tls.model_route = previous_route

    def _queue_silent_compile_chunk(self, session: ASRSession) -> None:
        session.pending_audio = np.zeros(self.preprocess_new_audio_samples, dtype=np.float32)
        session.accumulated_audio = session.pending_audio
        session.total_audio_samples += len(session.pending_audio)

    def _warmup(self):
        """Run warmup inference using streaming API to claim GPU memory.

        IMPORTANT: We use the streaming API (conformer_stream_step) for warmup,
        NOT the batch API (model.transcribe). The batch API corrupts internal
        model state and causes subsequent streaming inference to become
        non-deterministic. See docs/asr-determinism-investigation.md.
        """
        import time

        logger.info("Running warmup inference (streaming API) to claim GPU memory...")
        start = time.perf_counter()

        # Keep warmup on the same fixed preprocessor plan as live streaming.
        warmup_audio = np.zeros(self.constant_preprocess_samples, dtype=np.float32)

        # Run streaming inference to force all CUDA kernels to compile
        with torch.inference_mode():
            mel, mel_len = self._preprocess_fixed_audio(warmup_audio, len(warmup_audio))

            # Get initial cache
            cache = self._current_inference_model().encoder.get_initial_cache_state(batch_size=1)
            warmup_session = ASRSession(
                id="warmup",
                websocket=None,
                target_lang=self.target_lang,
            )

            # Run streaming step (processes entire mel as one chunk)
            if self._current_model_is_prompted():
                self._apply_inference_prompt(warmup_session)
            _ = self._conformer_stream_step(
                processed_signal=mel,
                processed_signal_length=mel_len,
                cache_last_channel=cache[0],
                cache_last_time=cache[1],
                cache_last_channel_len=cache[2],
                keep_all_outputs=True,
                previous_hypotheses=None,
                previous_pred_out=None,
                drop_extra_pre_encoded=0,
                return_transcription=True,
            )
            self._warm_encoder_compile_static_buckets(mel)

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Warmup complete in {elapsed:.0f}ms - GPU memory claimed")

    def _init_session(self, session: ASRSession):
        """Initialize a fresh session.

        If an overlap_buffer is present from a previous segment, it will be
        prepended to the accumulated audio to provide encoder left-context.
        This enables seamless transcription across mid-utterance resets.
        """
        self._set_session_language_route(session, session.target_lang)
        previous_route = getattr(self._inference_tls, "model_route", None)
        self._inference_tls.model_route = session.model_route
        try:
            plan = self._plan_for_session(session)

            # Initialize encoder cache
            cache = self._current_inference_model().encoder.get_initial_cache_state(batch_size=1)
            session.cache_last_channel = cache[0]
            session.cache_last_time = cache[1]
            session.cache_last_channel_len = cache[2]

            # Reset audio buffer and frame counter
            # If overlap buffer exists, use it as the starting audio
            if session.overlap_buffer is not None and len(session.overlap_buffer) > 0:
                session.pending_audio = session.overlap_buffer.copy()
                session.accumulated_audio = session.pending_audio
                session.total_audio_samples = len(session.pending_audio)
                overlap_ms = len(session.overlap_buffer) * 1000 / self.sample_rate
                logger.debug(
                    f"Session {session.id}: prepending {len(session.overlap_buffer)} samples "
                    f"({overlap_ms:.0f}ms) of overlap audio"
                )
                session.overlap_buffer = None  # Clear after use
            else:
                session.pending_audio = np.array([], dtype=np.float32)
                session.accumulated_audio = session.pending_audio
                session.total_audio_samples = 0

            session.raw_audio_ring = np.zeros(plan.raw_audio_ring_samples, dtype=np.float32)
            session.mel_frame_ring = None

            # (Removed 2026-05-18 baseline hardening: the NEMOTRON_ONSET_WARMUP_MS
            # buffer-prepend was an ineffective + buggy onset warm-up. PLAN Step 8
            # implements the correct conformer_stream_step warm-up from scratch.)
            session.emitted_frames = 0

            # Reset decoder state
            session.previous_hypotheses = None
            session.pred_out_stream = None
            session.current_text = ""
            session.eou_probe_chunk_index = 0

            session.synthetic_prefix_samples = 0
            if self.session_warmup_ms > 0:
                self._run_session_warmup(session)
        finally:
            self._inference_tls.model_route = previous_route

    def _run_session_warmup(self, session: ASRSession) -> None:
        """Prime one fresh session with synthetic silence without seeding text."""
        warmup_frames = self._session_warmup_frames()
        if warmup_frames is None:
            return
        plan = self._plan_for_session(session)
        warmup_samples = warmup_frames * plan.hop_samples
        preprocess_samples = warmup_samples + plan.hop_samples

        warmup_audio = np.zeros(preprocess_samples, dtype=np.float32)
        fixed_audio, valid_samples = self._build_fixed_preprocess_audio(
            session.raw_audio_ring,
            warmup_audio,
            plan,
        )

        with torch.inference_mode():
            mel, _mel_len = self._preprocess_fixed_audio(fixed_audio, valid_samples, plan)
            warmup_mel = mel[
                :,
                :,
                plan.first_preprocess_mel_frame : plan.first_preprocess_mel_frame + warmup_frames,
            ]
            chunk_len = torch.tensor([warmup_mel.shape[-1]], device='cuda')

            if self._current_model_is_prompted():
                self._apply_inference_prompt(session)
            (
                session.pred_out_stream,
                discarded_transcribed_texts,
                session.cache_last_channel,
                session.cache_last_time,
                session.cache_last_channel_len,
                session.previous_hypotheses,
            ) = self._conformer_stream_step(
                processed_signal=warmup_mel,
                processed_signal_length=chunk_len,
                cache_last_channel=session.cache_last_channel,
                cache_last_time=session.cache_last_time,
                cache_last_channel_len=session.cache_last_channel_len,
                keep_all_outputs=False,
                previous_hypotheses=None,
                previous_pred_out=None,
                drop_extra_pre_encoded=0,
                return_transcription=True,
            )

        consumed_audio = warmup_audio[:warmup_samples]
        if len(consumed_audio) >= plan.raw_audio_ring_samples:
            session.raw_audio_ring = consumed_audio[-plan.raw_audio_ring_samples :].copy()
        else:
            keep = plan.raw_audio_ring_samples - len(consumed_audio)
            session.raw_audio_ring = np.concatenate(
                [session.raw_audio_ring[-keep:], consumed_audio]
            ).astype(np.float32, copy=False)
        self._update_mel_frame_ring(session, warmup_mel, plan)
        session.emitted_frames = warmup_frames
        session.synthetic_prefix_samples = warmup_samples

        discarded_text_present = bool(discarded_transcribed_texts and discarded_transcribed_texts[0])
        logger.info(
            f"Session {session.id}: per-session warm-up ran once at init "
            f"(NEMOTRON_WARMUP_MS={self.session_warmup_ms}, "
            f"frames={warmup_frames}, samples={warmup_samples}, "
            f"discarded_returned_text={discarded_text_present}, "
            f"current_text_chars={len(session.current_text)}, "
            f"last_emitted_text_chars={len(session.last_emitted_text)}, "
            f"mel_ring_frames={int(session.mel_frame_ring.shape[-1]) if session.mel_frame_ring is not None else 0}, "
            f"emitted_frames={session.emitted_frames})"
        )

    def _session_timeline_samples(self, session: ASRSession) -> int:
        return session.synthetic_prefix_samples + session.total_audio_samples

    def _preprocess_fixed_audio(
        self,
        audio: np.ndarray,
        valid_samples: int,
        plan: Optional[StreamingPlan] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the preprocessor with an invariant input shape.

        `valid_samples` may be shorter for final partial chunks; the tensor
        shape stays fixed so CUDA uses the same STFT/cuFFT plan every call.
        """
        plan = plan or self._plan_for_route(getattr(self._inference_tls, "model_route", None) or "default")
        if len(audio) != plan.constant_preprocess_samples:
            raise ValueError(
                f"Expected fixed preprocessor input of {plan.constant_preprocess_samples} samples, "
                f"got {len(audio)}"
            )
        audio_tensor = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0).cuda()
        audio_len = torch.tensor([valid_samples], device='cuda', dtype=torch.long)
        token = self._gil_attrib_cuda_event_start()
        try:
            return self._current_inference_model().preprocessor(
                input_signal=audio_tensor,
                length=audio_len,
            )
        finally:
            self._gil_attrib_cuda_event_end(token)

    def _build_fixed_preprocess_audio(
        self,
        raw_audio_ring: np.ndarray,
        new_audio: np.ndarray,
        plan: Optional[StreamingPlan] = None,
    ) -> tuple[np.ndarray, int]:
        """Assemble align-pad + raw ring + new audio and zero-pad to K."""
        plan = plan or self._plan_for_route(getattr(self._inference_tls, "model_route", None) or "default")
        if len(raw_audio_ring) != plan.raw_audio_ring_samples:
            raise ValueError(
                f"Expected raw ring of {plan.raw_audio_ring_samples} samples, "
                f"got {len(raw_audio_ring)}"
            )
        prefix_len = plan.preprocess_align_pad_samples + plan.raw_audio_ring_samples
        valid_samples = prefix_len + len(new_audio)
        if valid_samples > plan.constant_preprocess_samples:
            raise ValueError(
                f"Fixed preprocessor valid span {valid_samples} exceeds K={plan.constant_preprocess_samples}"
            )

        audio = np.zeros(plan.constant_preprocess_samples, dtype=np.float32)
        cursor = plan.preprocess_align_pad_samples
        audio[cursor : cursor + plan.raw_audio_ring_samples] = raw_audio_ring
        cursor += plan.raw_audio_ring_samples
        audio[cursor : cursor + len(new_audio)] = new_audio
        return audio, valid_samples

    def _update_mel_frame_ring(
        self,
        session: ASRSession,
        new_mel: torch.Tensor,
        plan: Optional[StreamingPlan] = None,
    ) -> None:
        """Retain the mel pre-encode cache separately from raw STFT context."""
        plan = plan or self._plan_for_session(session)
        if session.mel_frame_ring is None:
            combined = new_mel.detach()
        else:
            combined = torch.cat((session.mel_frame_ring, new_mel.detach()), dim=-1)
        session.mel_frame_ring = combined[:, :, -plan.pre_encode_cache_size :].detach()

    @staticmethod
    def _probe_scalar(value: Any) -> Any:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            value = value.detach().cpu().reshape(-1)[0].item()
        if isinstance(value, np.generic):
            value = value.item()
        return value

    @classmethod
    def _probe_int(cls, value: Any) -> Optional[int]:
        value = cls._probe_scalar(value)
        if value is None:
            return None
        return int(value)

    @classmethod
    def _probe_float(cls, value: Any) -> Optional[float]:
        value = cls._probe_scalar(value)
        if value is None:
            return None
        return float(value)

    @classmethod
    def _probe_int_list(cls, value: Any) -> list[int]:
        if value is None:
            return []
        if torch.is_tensor(value):
            return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
        return [int(cls._probe_scalar(item)) for item in value]

    def _probe_blank_id(self) -> int:
        decoding = getattr(self.model, "decoding", None)
        blank_id = getattr(decoding, "blank_id", None)
        if blank_id is not None:
            return int(self._probe_scalar(blank_id))
        joint = getattr(self.model, "joint", None)
        return int(getattr(joint, "num_classes_with_blank") - 1)

    def _probe_token_strings(self, token_ids: list[int]) -> list[str]:
        if not token_ids:
            return []
        decoding = getattr(self.model, "decoding", None)
        if decoding is not None and hasattr(decoding, "decode_ids_to_tokens"):
            try:
                token_strings = [str(token) for token in decoding.decode_ids_to_tokens(token_ids)]
                if len(token_strings) == len(token_ids):
                    return token_strings
            except Exception:
                pass
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "ids_to_tokens"):
            try:
                token_strings = [str(token) for token in tokenizer.ids_to_tokens(token_ids)]
                if len(token_strings) == len(token_ids):
                    return token_strings
            except Exception:
                pass
        return [str(token_id) for token_id in token_ids]

    @staticmethod
    def _probe_token_starts_word(token: str) -> bool:
        return token.startswith(("\u2581", "\u0120")) or token[:1].isspace()

    @classmethod
    def _probe_word_boundary_state(cls, tokens: list[str], index: int) -> dict[str, Any]:
        token = tokens[index] if index < len(tokens) else ""
        next_token = tokens[index + 1] if index + 1 < len(tokens) else None
        stripped = token.replace("\u2581", "").replace("\u0120", "").strip()
        starts_word = index == 0 or cls._probe_token_starts_word(token)
        next_starts_word = next_token is not None and cls._probe_token_starts_word(next_token)
        punctuation_boundary = stripped in {".", ",", "?", "!", ";", ":"}
        return {
            "starts_word": starts_word,
            "extends_word": not starts_word,
            "completes_word": bool(next_starts_word or punctuation_boundary),
            "completion_observed": next_token is not None or punctuation_boundary,
        }

    @classmethod
    def _probe_hyp_timestamps(cls, hyp: Any) -> list[int]:
        timestamp = getattr(hyp, "timestamp", None)
        if isinstance(timestamp, dict):
            timestamp = timestamp.get("timestep", [])
        return cls._probe_int_list(timestamp)

    @classmethod
    def _probe_alignment_label(cls, item: Any) -> Optional[int]:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            item = item[-1]
        return cls._probe_int(item)

    @classmethod
    def _probe_alignment_labels_by_frame(cls, alignments: Any) -> list[list[int]]:
        if not alignments:
            return []
        labels_by_frame: list[list[int]] = []
        for frame in alignments:
            if frame is None or (isinstance(frame, (list, tuple)) and len(frame) == 0):
                labels_by_frame.append([])
                continue
            if isinstance(frame, (list, tuple)) and len(frame) > 0 and isinstance(frame[0], (list, tuple)):
                items = frame
            else:
                items = [frame]
            labels = []
            for item in items:
                label = cls._probe_alignment_label(item)
                if label is not None:
                    labels.append(label)
            labels_by_frame.append(labels)
        return labels_by_frame

    def _probe_frame_alignment(
        self,
        labels_by_frame: list[list[int]],
        *,
        blank_id: int,
        chunk_model_frame_start: int,
    ) -> list[dict[str, Any]]:
        frame_alignment = []
        for frame_offset, labels in enumerate(labels_by_frame):
            frame_alignment.append(
                {
                    "frame_offset": frame_offset,
                    "model_frame_index": chunk_model_frame_start + frame_offset,
                    "labels": labels,
                    "has_non_blank": any(label != blank_id for label in labels),
                    "all_blank": bool(labels) and all(label == blank_id for label in labels),
                }
            )
        return frame_alignment

    @classmethod
    def _probe_frame_confidence(cls, frame_confidence: Any) -> list[list[float]]:
        if not frame_confidence:
            return []
        confidence_rows: list[list[float]] = []
        for frame in frame_confidence:
            if isinstance(frame, (list, tuple)):
                confidence_rows.append(
                    [
                        confidence
                        for confidence in (cls._probe_float(item) for item in frame)
                        if confidence is not None
                    ]
                )
            else:
                confidence = cls._probe_float(frame)
                confidence_rows.append([] if confidence is None else [confidence])
        return confidence_rows

    @staticmethod
    def _probe_changed_positions(prev_y: list[int], y_sequence: list[int]) -> list[int]:
        changed_positions = []
        for index in range(max(len(prev_y), len(y_sequence))):
            prev_token = prev_y[index] if index < len(prev_y) else None
            token = y_sequence[index] if index < len(y_sequence) else None
            if prev_token != token:
                changed_positions.append(index)
        return changed_positions

    @staticmethod
    def _probe_token_frame_from_alignments(
        labels_by_frame: list[list[int]],
        *,
        blank_id: int,
        token_offset: int,
    ) -> Optional[int]:
        non_blank_frames = []
        for frame_offset, labels in enumerate(labels_by_frame):
            non_blank_frames.extend(frame_offset for label in labels if label != blank_id)
        if token_offset < len(non_blank_frames):
            return non_blank_frames[token_offset]
        return None

    @staticmethod
    def _probe_token_frame_events_from_alignments(
        labels_by_frame: list[list[int]],
        *,
        blank_id: int,
    ) -> list[tuple[int, int]]:
        token_events = []
        for frame_offset, labels in enumerate(labels_by_frame):
            frame_token_offset = 0
            for label in labels:
                if label == blank_id:
                    continue
                token_events.append((frame_offset, frame_token_offset))
                frame_token_offset += 1
        return token_events

    def _eou_probe_snapshot(self, session: ASRSession) -> Optional[dict[str, Any]]:
        if not self.eou_probe_enabled:
            return None
        prev_hyp = session.previous_hypotheses[0] if session.previous_hypotheses else None
        prev_y = self._probe_int_list(getattr(prev_hyp, "y_sequence", [])) if prev_hyp is not None else []
        return {
            "chunk_index": session.eou_probe_chunk_index,
            "chunk_model_frame_start": session.emitted_frames,
            "prev_y": prev_y,
            "prev_y_len": len(prev_y),
            "monotonic_start": time.monotonic(),
            "wall_time_start": time.time(),
        }

    def _write_eou_probe_chunk(
        self,
        session: ASRSession,
        snapshot: Optional[dict[str, Any]],
    ) -> None:
        if not self.eou_probe_enabled or snapshot is None or self.eou_probe_path is None:
            return
        try:
            hyp = session.previous_hypotheses[0] if session.previous_hypotheses else None
            chunk_index = session.eou_probe_chunk_index
            session.eou_probe_chunk_index += 1
            chunk_model_frame_start = int(snapshot["chunk_model_frame_start"])
            prev_y = snapshot["prev_y"]
            blank_id = self._probe_blank_id()
            y_sequence = self._probe_int_list(getattr(hyp, "y_sequence", [])) if hyp is not None else []
            token_strings = self._probe_token_strings(y_sequence)
            timestamps = self._probe_hyp_timestamps(hyp) if hyp is not None else []
            alignments = getattr(hyp, "alignments", None) if hyp is not None else None
            labels_by_frame = self._probe_alignment_labels_by_frame(alignments)
            token_frame_events = self._probe_token_frame_events_from_alignments(
                labels_by_frame,
                blank_id=blank_id,
            )
            frame_confidence = self._probe_frame_confidence(
                getattr(hyp, "frame_confidence", None) if hyp is not None else None
            )

            new_tokens = []
            prev_y_len = len(prev_y)
            frame_subindex_counts: dict[int, int] = {}
            for token_index in range(prev_y_len, len(y_sequence)):
                token_offset = token_index - prev_y_len
                alignment_frame_index = None
                alignment_subindex = None
                if token_offset < len(token_frame_events):
                    alignment_frame_index, alignment_subindex = token_frame_events[token_offset]
                chunk_frame_index = (
                    timestamps[token_offset]
                    if token_offset < len(timestamps)
                    else alignment_frame_index
                )
                if chunk_frame_index is None:
                    chunk_frame_index = self._probe_token_frame_from_alignments(
                        labels_by_frame,
                        blank_id=blank_id,
                        token_offset=token_offset,
                    )
                model_frame_index = (
                    chunk_model_frame_start + chunk_frame_index
                    if chunk_frame_index is not None
                    else None
                )
                model_frame_subindex = None
                model_frame_event_index = None
                if chunk_frame_index is not None:
                    fallback_subindex = frame_subindex_counts.get(chunk_frame_index, 0)
                    model_frame_subindex = (
                        alignment_subindex
                        if alignment_subindex is not None and alignment_frame_index == chunk_frame_index
                        else fallback_subindex
                    )
                    frame_subindex_counts[chunk_frame_index] = max(
                        frame_subindex_counts.get(chunk_frame_index, 0),
                        model_frame_subindex + 1,
                    )
                if model_frame_index is not None and model_frame_subindex is not None:
                    model_frame_event_index = model_frame_index * 1024 + model_frame_subindex
                token_string = token_strings[token_index] if token_index < len(token_strings) else str(y_sequence[token_index])
                new_tokens.append(
                    {
                        "token_index": token_index,
                        "token_id": y_sequence[token_index],
                        "token": token_string,
                        "chunk_frame_index": chunk_frame_index,
                        "model_frame_index": model_frame_index,
                        "model_frame_subindex": model_frame_subindex,
                        "model_frame_event_index": model_frame_event_index,
                        "word_boundary": self._probe_word_boundary_state(token_strings, token_index),
                    }
                )

            payload = {
                "type": "eou_probe_chunk",
                "run_tag": self.eou_probe_tag,
                "session_id": session.id,
                "chunk_index": chunk_index,
                "chunk_model_frame_start": chunk_model_frame_start,
                "prev_y_len": snapshot["prev_y_len"],
                "emitted_frames_after": session.emitted_frames,
                "shift_frames": self.shift_frames,
                "right_context_chunks": self.right_context,
                "right_context_frames": self.right_context * self.shift_frames,
                "R": self.right_context * self.shift_frames,
                "blank_id": blank_id,
                "monotonic_start": snapshot["monotonic_start"],
                "monotonic_done": time.monotonic(),
                "wall_time_start": snapshot["wall_time_start"],
                "wall_time_done": time.time(),
                "real_audio_cursor_samples": session.total_audio_samples,
                "real_audio_cursor_seconds": session.total_audio_samples / self.sample_rate,
                "timeline_cursor_samples": self._session_timeline_samples(session),
                "hyp_score": self._probe_float(getattr(hyp, "score", None)) if hyp is not None else None,
                "y_sequence": y_sequence,
                "changed_positions": self._probe_changed_positions(prev_y, y_sequence),
                "new_tokens": new_tokens,
                "frame_alignment": self._probe_frame_alignment(
                    labels_by_frame,
                    blank_id=blank_id,
                    chunk_model_frame_start=chunk_model_frame_start,
                ),
                "frame_confidence": frame_confidence,
            }
            self.eou_probe_path.parent.mkdir(parents=True, exist_ok=True)
            with _EOU_PROBE_LOCK:
                with self.eou_probe_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception as e:
            logger.error(f"Session {session.id} EOU probe write error: {e}")

    def _eou_snapshot_file_stem(self, session: ASRSession) -> str:
        return f"{_safe_tag_filename(self.eou_probe_tag)}_{_safe_tag_filename(session.id)}"

    def _capture_eou_snapshot_audio(self, session: ASRSession, audio_bytes: bytes) -> None:
        if self.eou_snapshot_dir is None or not audio_bytes:
            return
        session.eou_snapshot_audio.extend(audio_bytes)

    def _flush_eou_snapshot_audio(self, session: ASRSession) -> None:
        if self.eou_snapshot_dir is None or not session.eou_snapshot_audio:
            return
        try:
            audio_path = self.eou_snapshot_dir / f"{self._eou_snapshot_file_stem(session)}_audio.bin"
            with _EOU_SNAPSHOT_LOCK:
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                with audio_path.open("wb") as f:
                    f.write(session.eou_snapshot_audio)
        except Exception as e:
            logger.error(f"Session {session.id} EOU snapshot audio write error: {e}")

    def _write_eou_snapshot_chunk(
        self,
        session: ASRSession,
        snapshot: Optional[dict[str, Any]],
    ) -> None:
        if self.eou_snapshot_dir is None or snapshot is None:
            return

        chunk_index = int(snapshot["chunk_index"])
        if chunk_index % self.eou_snapshot_every != 0:
            return

        try:
            previous_hypotheses = clone_hypotheses_deep(session.previous_hypotheses)
            payload = {
                "cache_last_channel": snapshot_tree_cpu(session.cache_last_channel),
                "cache_last_time": snapshot_tree_cpu(session.cache_last_time),
                "cache_last_channel_len": snapshot_tree_cpu(session.cache_last_channel_len),
                "previous_hypotheses": snapshot_tree_cpu(previous_hypotheses),
                "pred_out_stream": snapshot_tree_cpu(session.pred_out_stream),
                "pending_audio": (
                    session.pending_audio.copy()
                    if session.pending_audio is not None
                    else np.array([], dtype=np.float32)
                ),
                "raw_audio_ring": (
                    session.raw_audio_ring.copy()
                    if session.raw_audio_ring is not None
                    else np.array([], dtype=np.float32)
                ),
                "mel_frame_ring": snapshot_tree_cpu(session.mel_frame_ring),
                "emitted_frames": int(session.emitted_frames),
                "synthetic_prefix_samples": int(session.synthetic_prefix_samples),
                "total_audio_samples": int(session.total_audio_samples),
                "chunk_index": chunk_index,
                "monotonic_time": time.monotonic(),
                "run_tag": self.eou_probe_tag,
                "session_id": session.id,
                "real_audio_cursor_samples": int(session.total_audio_samples),
                "timeline_cursor_samples": int(self._session_timeline_samples(session)),
            }
            snapshot_path = (
                self.eou_snapshot_dir
                / f"{self._eou_snapshot_file_stem(session)}_chunk{chunk_index:06d}.pt"
            )
            with _EOU_SNAPSHOT_LOCK:
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, snapshot_path)
        except Exception as e:
            logger.error(f"Session {session.id} EOU snapshot chunk write error: {e}")

    def _admission_backlog_signal(self) -> dict[str, Any]:
        queued_events = 0
        for session in list(self.sessions.values()):
            queue = session.continuous_event_queue
            if queue is not None:
                queued_events += queue.qsize()

        now = time.monotonic()
        oldest_ready_age_ms = 0.0
        oldest_ready_session_id = None
        for session_id in list(self._scheduler_ready):
            session = self.sessions.get(session_id)
            ready_since = (
                session.scheduler_ready_since
                if session is not None
                else None
            )
            if ready_since is None:
                continue
            ready_age_ms = max(0.0, (now - ready_since) * 1000.0)
            if ready_age_ms > oldest_ready_age_ms:
                oldest_ready_age_ms = ready_age_ms
                oldest_ready_session_id = session_id

        ready_count = len(self._scheduler_ready)
        return {
            "queued_events": queued_events,
            "ready_count": ready_count,
            "backlog_count": queued_events + ready_count,
            "oldest_ready_age_ms": oldest_ready_age_ms,
            "oldest_ready_session_id": oldest_ready_session_id,
        }

    def _admission_reject_reason(
        self,
        signal: dict[str, Any],
    ) -> Optional[str]:
        if signal["backlog_count"] > self.admission_max_backlog:
            return "backlog_count"
        if signal["oldest_ready_age_ms"] > self.admission_max_ready_age_ms:
            return "oldest_ready_age_ms"
        return None

    def _admission_status_snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.admission_enabled,
            "attempted": self.admission_attempted,
            "admitted": self.admission_admitted,
            "rejected": self.admission_rejected,
            "max_backlog": self.admission_max_backlog,
            "max_ready_age_ms": self.admission_max_ready_age_ms,
            "signal": self._admission_backlog_signal(),
        }

    @staticmethod
    def _stats_key(value: Optional[str]) -> str:
        normalized = (value or "unknown").strip().lower()
        return normalized or "unknown"

    @staticmethod
    def _stats_counter_snapshot(counter: Counter[str], total: Optional[int] = None) -> dict[str, Any]:
        denominator = sum(counter.values()) if total is None else total
        values = {
            key: {
                "count": count,
                "pct": (float(count) / float(denominator) * 100.0) if denominator else 0.0,
            }
            for key, count in sorted(counter.items())
            if count
        }
        return {
            "total": denominator,
            "values": values,
        }

    @staticmethod
    def _stats_counter_decrement(counter: Counter[str], key: Optional[str]) -> None:
        if not key:
            return
        counter[key] -= 1
        if counter[key] <= 0:
            del counter[key]

    def _record_stats_session_start(self, session: ASRSession) -> None:
        if not self.stats_enabled:
            return
        language = self._stats_key(session.target_lang)
        route = self._stats_key(session.model_route)
        session.stats_language = language
        session.stats_model_route = route
        self._stats_request_count += 1
        self._stats_request_language_counts[language] += 1
        self._stats_request_route_counts[route] += 1
        self._stats_active_language_counts[language] += 1
        self._stats_active_route_counts[route] += 1

    def _record_stats_session_route_update(self, session: ASRSession) -> None:
        if not self.stats_enabled:
            return
        old_language = session.stats_language
        old_route = session.stats_model_route
        new_language = self._stats_key(session.target_lang)
        new_route = self._stats_key(session.model_route)
        if old_language != new_language:
            self._stats_counter_decrement(self._stats_active_language_counts, old_language)
            self._stats_active_language_counts[new_language] += 1
            session.stats_language = new_language
        if old_route != new_route:
            self._stats_counter_decrement(self._stats_active_route_counts, old_route)
            self._stats_active_route_counts[new_route] += 1
            session.stats_model_route = new_route
        self._stats_routing_event_count += 1
        self._stats_routing_event_language_counts[new_language] += 1
        self._stats_routing_event_route_counts[new_route] += 1

    def _record_stats_session_end(self, session: ASRSession) -> None:
        if not self.stats_enabled:
            return
        self._stats_counter_decrement(
            self._stats_active_language_counts,
            session.stats_language,
        )
        self._stats_counter_decrement(
            self._stats_active_route_counts,
            session.stats_model_route,
        )
        session.stats_language = None
        session.stats_model_route = None

    def _record_stats_sample(
        self,
        timing: Optional[dict[str, Any]],
        *,
        emitted_to_client: bool,
    ) -> None:
        """Append one finalize sample to the /stats sliding window.

        Reads ONLY from the always-on timing dict (no FINALIZE_PROFILE
        cuda_synchronize, no extra GPU work). Each sample is a tuple of
        floats sized for cheap storage. Concurrent appends from the
        scheduler thread are safe (CPython deque.append is atomic).
        """
        if not self.stats_enabled or timing is None:
            return
        try:
            vad_stop = timing.get("vad_stop")
            final_sent = timing.get("final_sent")
            if vad_stop is None or final_sent is None:
                # Not a complete finalize (suppressed/stale) — count separately.
                if emitted_to_client:
                    self._stats_emitted_count += 1
                else:
                    self._stats_suppressed_count += 1
                return
            fork_start = timing.get("fork_flush_start")
            fork_done = timing.get("fork_flush_done")
            vad_stop_recv = timing.get("vad_stop_recv")
            lock_wait = timing.get("inference_lock_acquire_wait_ms")

            def _ms(end: Any, start: Any) -> Optional[float]:
                if end is None or start is None:
                    return None
                return max(0.0, (float(end) - float(start)) * 1000.0)

            sample = (
                float(final_sent),                                       # ts_unix
                _ms(final_sent, vad_stop),                               # vad_stop_to_sent_ms (server TTFS)
                _ms(fork_done, fork_start),                              # fork_flush_wall_ms
                _ms(vad_stop, vad_stop_recv),                            # vad_stop_recv_to_process_ms (intake lag)
                float(lock_wait) if lock_wait is not None else None,     # lock_wait_ms
                _ms(fork_start, vad_stop),                               # vad_stop_to_finalize_start_ms
                len(self.sessions),                                      # active_sessions_at_emit
                bool(emitted_to_client),                                 # emitted
            )
            self._stats_samples.append(sample)
            if emitted_to_client:
                self._stats_emitted_count += 1
            else:
                self._stats_suppressed_count += 1
        except Exception:
            # /stats is a best-effort telemetry path; never crash a finalize.
            logger.exception("record_stats_sample failed (telemetry only — finalize continues)")

    def _stats_snapshot(self, last_n: Optional[int] = None) -> dict[str, Any]:
        """Build the /stats response from the current sliding window.

        last_n optionally narrows the snapshot to the most recent N samples
        (operator-friendly for short-horizon checks). Default = entire window.
        """
        samples = list(self._stats_samples)  # atomic snapshot of the deque
        if last_n is not None and last_n > 0:
            samples = samples[-last_n:]
        timestamps = [s[0] for s in samples]
        active_sess = [float(s[6]) for s in samples]
        emitted_in_window = sum(1 for s in samples if s[7])
        suppressed_in_window = len(samples) - emitted_in_window

        def collect(idx: int) -> list[float]:
            return [float(s[idx]) for s in samples if s[idx] is not None]

        return {
            "enabled": self.stats_enabled,
            "window_size": self.stats_window_size,
            "samples": len(samples),
            "since_unix": timestamps[0] if timestamps else None,
            "until_unix": timestamps[-1] if timestamps else None,
            "emitted_in_window": emitted_in_window,
            "suppressed_in_window": suppressed_in_window,
            "lifetime_emitted": self._stats_emitted_count,
            "lifetime_suppressed": self._stats_suppressed_count,
            "metrics": {
                "vad_stop_to_sent_ms": _compute_quantile_summary(collect(1)),
                "fork_flush_wall_ms": _compute_quantile_summary(collect(2)),
                "vad_stop_recv_to_process_ms": _compute_quantile_summary(collect(3)),
                "lock_wait_ms": _compute_quantile_summary(collect(4)),
                "vad_stop_to_finalize_start_ms": _compute_quantile_summary(collect(5)),
            },
            "active_sessions_at_emit": _compute_quantile_summary(active_sess),
            "requests": {
                "total": self._stats_request_count,
                "by_language": self._stats_counter_snapshot(
                    self._stats_request_language_counts,
                    self._stats_request_count,
                ),
                "by_model_route": self._stats_counter_snapshot(
                    self._stats_request_route_counts,
                    self._stats_request_count,
                ),
            },
            "active_sessions": {
                "total": len(self.sessions),
                "by_language": self._stats_counter_snapshot(
                    self._stats_active_language_counts,
                    len(self.sessions),
                ),
                "by_model_route": self._stats_counter_snapshot(
                    self._stats_active_route_counts,
                    len(self.sessions),
                ),
            },
            "routing_events": {
                "total": self._stats_routing_event_count,
                "by_language": self._stats_counter_snapshot(
                    self._stats_routing_event_language_counts,
                    self._stats_routing_event_count,
                ),
                "by_model_route": self._stats_counter_snapshot(
                    self._stats_routing_event_route_counts,
                    self._stats_routing_event_count,
                ),
            },
            "admission": self._admission_status_snapshot(),
        }

    async def stats_handler(self, request: web.Request) -> web.Response:
        """Rolling latency stats endpoint.

        Query params:
          last=N   limit summary to the most recent N samples (default: full window)
        """
        last_n: Optional[int] = None
        last_raw = request.query.get("last")
        if last_raw:
            try:
                last_n = int(last_raw)
                if last_n <= 0:
                    raise ValueError("last must be > 0")
            except ValueError:
                return web.json_response({"error": f"invalid 'last': {last_raw!r}"}, status=400)
        return web.json_response(self._stats_snapshot(last_n=last_n))

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Handle a WebSocket client connection."""
        import uuid

        ws = web.WebSocketResponse(max_msg_size=10 * 1024 * 1024)
        await ws.prepare(request)

        try:
            session_target_lang = self._validate_connection_query(request.query)
        except ValueError as e:
            logger.warning(f"Rejecting WebSocket connection: {e}")
            await ws.send_str(json.dumps({"type": "error", "message": str(e)}))
            await ws.close()
            return ws

        session_id = str(uuid.uuid4())[:8]
        self.admission_attempted += 1
        admission_signal = self._admission_backlog_signal()
        admission_reject_reason = self._admission_reject_reason(admission_signal)
        if admission_reject_reason is not None:
            self.admission_rejected += 1
            logger.warning(
                "admission_rejected "
                f"session={session_id} "
                f"reason={admission_reject_reason} "
                f"attempted={self.admission_attempted} "
                f"admitted={self.admission_admitted} "
                f"rejected={self.admission_rejected} "
                f"queued_events={admission_signal['queued_events']} "
                f"ready_count={admission_signal['ready_count']} "
                f"backlog_count={admission_signal['backlog_count']} "
                f"oldest_ready_age_ms={admission_signal['oldest_ready_age_ms']:.2f} "
                f"oldest_ready_session={admission_signal['oldest_ready_session_id']}"
            )
            await ws.close(code=1013, message=b"admission_backpressure")
            return ws

        session = ASRSession(
            id=session_id,
            websocket=ws,
            target_lang=session_target_lang,
        )
        self._set_session_language_route(session, session_target_lang)
        self.sessions[session_id] = session
        self.admission_admitted += 1
        self._record_stats_session_start(session)
        session_model_lane = self._scheduler_assign_session_model_lane(session)

        logger.info(f"Client {session_id} connected")

        try:
            if self.model_lanes > 1:
                async with self._scheduler_exclusive_model_path("session_init"):
                    async with self.inference_lock:
                        await self._run_scheduler_exclusive_inference_call(
                            self._init_session,
                            session,
                            lane_id=session_model_lane,
                        )
            else:
                async with self.inference_lock:
                    await self._run_inference_call(self._init_session, session)

            if self.continuous_context:
                self._start_continuous_session(session)

            await ws.send_str(json.dumps({"type": "ready"}))
            logger.debug(f"Client {session_id}: sent ready")

            async for msg in ws:
                if self.continuous_context:
                    if self.scheduler_enabled:
                        await self._queue_scheduler_ws_message(session, msg)
                    else:
                        await self._queue_continuous_ws_message(session, msg)
                    continue

                if msg.type == WSMsgType.BINARY:
                    await self._handle_audio(session, msg.data)
                elif msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")

                        if msg_type in ("settings", "set_language", "transcription_session.update"):
                            language = (
                                data.get("language")
                                or (data.get("settings") or {}).get("language")
                                or ((data.get("session") or {}).get("input_audio_transcription") or {}).get("language")
                            )
                            if language:
                                async with self.inference_lock:
                                    self._set_session_language_route(session, str(language))
                                    self._record_stats_session_route_update(session)
                                    await self._run_inference_call(self._init_session, session)
                                await ws.send_str(json.dumps({
                                    "type": "settings.updated",
                                    "language": session.target_lang,
                                    "model_route": session.model_route,
                                }))
                        elif msg_type == "reset" or msg_type == "end":
                            # finalize=True (default): hard reset with padding + keep_all_outputs
                            # finalize=False: soft reset, just return current text
                            finalize = data.get("finalize", True)
                            await self._reset_session(session, finalize=finalize)
                        elif msg_type == "vad_start" or msg_type == "vad_stop":
                            session.vad_gated_audio = True
                            if msg_type == "vad_start":
                                session.accepting_vad_audio = True
                            logger.debug(
                                f"Client {session_id}: received {msg_type} (no-op)"
                            )
                        else:
                            logger.warning(f"Client {session_id}: unknown message type: {msg_type}")

                    except json.JSONDecodeError:
                        logger.warning(f"Client {session_id}: invalid JSON")
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"Client {session_id} WebSocket error: {ws.exception()}")
                    break

            logger.info(f"Client {session_id} disconnected")

        except Exception as e:
            logger.error(f"Client {session_id} error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            try:
                await ws.send_str(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
            except:
                pass
        finally:
            if self.continuous_context:
                await self._close_continuous_session(session)
            self._flush_eou_snapshot_audio(session)
            self._record_stats_session_end(session)
            if session_id in self.sessions:
                del self.sessions[session_id]
            self._scheduler_session_model_lane_affinity.pop(session_id, None)
            self._log_retained_cache_telemetry("session_removed")

        return ws

    def _start_continuous_session(self, session: ASRSession) -> None:
        """Start the ordered per-session event worker for continuous mode."""
        if self.scheduler_enabled:
            self._start_scheduler_continuous_session(session)
            return

        session.continuous_event_queue = asyncio.Queue()
        session.continuous_worker_task = asyncio.create_task(
            self._continuous_session_worker(session),
            name=f"nemotron-continuous-session-{session.id}",
        )
        logger.info(
            f"Session {session.id}: continuous context enabled "
            f"(debounce={self.finalize_silence_ms}ms)"
        )

    async def _queue_continuous_ws_message(self, session: ASRSession, msg) -> None:
        """Queue raw WS events so continuous-mode control/audio is ordered."""
        queue = session.continuous_event_queue
        if queue is None:
            logger.warning(f"Session {session.id}: continuous event queue missing")
            return

        if msg.type == WSMsgType.BINARY:
            await queue.put(("audio", msg.data))
        elif msg.type == WSMsgType.TEXT:
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.warning(f"Client {session.id}: invalid JSON")
                return

            msg_type = data.get("type")
            if msg_type == "reset" or msg_type == "end":
                finalize = data.get("finalize", True)
                await queue.put(("reset", finalize, msg_type))
            elif msg_type == "vad_start" or msg_type == "vad_stop":
                if msg_type == "vad_stop" and self.finalize_profile_enabled:
                    session.continuous_vad_stop_recv_ts = time.time()  # I/O-gap probe: socket-receive vs scheduler-process
                await queue.put((msg_type,))
            else:
                logger.warning(f"Client {session.id}: unknown message type: {msg_type}")
        elif msg.type == WSMsgType.ERROR:
            logger.error(f"Client {session.id} WebSocket error: {session.websocket.exception()}")
        elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
            await queue.put(("close",))

    def _ensure_scheduler_task(self) -> None:
        if not self.scheduler_enabled:
            return
        if self._scheduler_wakeup is None:
            self._scheduler_wakeup = asyncio.Event()
        if self.scheduler_task is None or self.scheduler_task.done():
            self.scheduler_task = asyncio.create_task(
                self._scheduler_loop(),
                name="nemotron-scheduler-b1",
            )
            logger.info(
                "scheduler_b1_started "
                f"queue_maxsize={self.scheduler_queue_maxsize} "
                f"batch_enabled={self.batch_enabled} "
                f"batch_barrier_drain={self._scheduler_batch_barrier_drain_active()} "
                f"batch_finalize={self._scheduler_batch_finalize_active()} "
                f"batch_finalize_preproc={self._scheduler_batch_finalize_preproc_active()} "
                f"finalize_priority={self._scheduler_finalize_priority_active()} "
                f"model_lanes={self.model_lanes} "
                f"batch_max_wait_ms={self.batch_max_wait_ms} "
                f"batch_max_size={self.batch_max_size}"
            )

    def _wake_scheduler(self) -> None:
        if self._scheduler_wakeup is not None:
            self._scheduler_wakeup.set()

    def _start_scheduler_continuous_session(self, session: ASRSession) -> None:
        """Start scheduler-owned continuous mode without a per-session worker."""
        self._ensure_scheduler_task()
        session.continuous_event_queue = asyncio.Queue(
            maxsize=self.scheduler_queue_maxsize
        )
        session.continuous_worker_task = None
        session.scheduler_closed = False
        logger.info(
            f"Session {session.id}: continuous context enabled via scheduler_b1 "
            f"(debounce={self.finalize_silence_ms}ms)"
        )

    async def _scheduler_queue_event(self, session: ASRSession, event: tuple) -> None:
        queue = session.continuous_event_queue
        if queue is None:
            logger.warning(f"Session {session.id}: scheduler event queue missing")
            return
        await queue.put(event)
        self._wake_scheduler()

    async def _queue_scheduler_ws_message(self, session: ASRSession, msg) -> None:
        """Queue raw WS events for the central B=1 scheduler."""
        if msg.type == WSMsgType.BINARY:
            await self._scheduler_queue_event(session, ("audio", msg.data))
        elif msg.type == WSMsgType.TEXT:
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.warning(f"Client {session.id}: invalid JSON")
                return

            msg_type = data.get("type")
            if msg_type == "reset" or msg_type == "end":
                finalize = data.get("finalize", True)
                await self._scheduler_queue_event(session, ("reset", finalize, msg_type))
            elif msg_type == "vad_start" or msg_type == "vad_stop":
                if msg_type == "vad_stop" and self.finalize_profile_enabled:
                    session.continuous_vad_stop_recv_ts = time.time()  # I/O-gap probe: socket-receive vs scheduler-process
                await self._scheduler_queue_event(session, (msg_type,))
            else:
                logger.warning(f"Client {session.id}: unknown message type: {msg_type}")
        elif msg.type == WSMsgType.ERROR:
            logger.error(f"Client {session.id} WebSocket error: {session.websocket.exception()}")
        elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
            await self._scheduler_queue_event(session, ("close",))

    def _scheduler_next_batch_deadline(self) -> Optional[float]:
        if not self.batch_enabled or not self._scheduler_ready:
            return None
        if not self._scheduler_batch_first_ready:
            return None
        if self.model_lanes <= 1:
            return min(self._scheduler_batch_first_ready.values())
        excluded_lanes = self._scheduler_finalize_priority_pending_lanes()
        compatible_deadlines = []
        for key, deadline in self._scheduler_batch_first_ready.items():
            if not self._scheduler_model_lane_key_can_dispatch(
                key,
                excluded_lanes=excluded_lanes,
            ):
                continue
            for session_id in list(self._scheduler_ready):
                if session_id in self._scheduler_inflight_sessions:
                    continue
                session = self.sessions.get(session_id)
                if session is None:
                    continue
                if (
                    self._scheduler_batch_group_key_for_session(session) == key
                    and self._scheduler_session_affinity_allows_dispatch(
                        session,
                        key,
                        excluded_lanes=excluded_lanes,
                    )
                ):
                    compatible_deadlines.append(deadline)
                    break
        if not compatible_deadlines:
            return None
        return min(compatible_deadlines)

    def _scheduler_wait_timeout(self) -> Optional[float]:
        deadline = self._scheduler_next_batch_deadline()
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())

    def _scheduler_batch_barrier_drain_active(self) -> bool:
        return (
            self.batch_barrier_drain_requested
            and self.scheduler_enabled
            and self.batch_enabled
        )

    def _scheduler_batch_finalize_active(self) -> bool:
        return (
            self.batch_finalize_requested
            and self.scheduler_enabled
            and self.batch_enabled
        )

    def _scheduler_batch_finalize_preproc_active(self) -> bool:
        return (
            self.batch_finalize_preproc_requested
            and self._scheduler_batch_finalize_active()
        )

    def _scheduler_finalize_priority_active(self) -> bool:
        return (
            self.finalize_priority_enabled
            and self._scheduler_batch_finalize_active()
        )

    def _scheduler_finalize_priority_pending_lanes(self) -> set[int]:
        if (
            not self._scheduler_finalize_priority_active()
            or self.model_lanes <= 1
            or not self._scheduler_priority_finalize_events
        ):
            return set()

        lanes: set[int] = set()
        for finalize_event in self._scheduler_priority_finalize_events:
            session = finalize_event.session
            if session.scheduler_closed:
                continue
            lanes.add(self._scheduler_assign_session_model_lane(session))
        return lanes

    async def _scheduler_process_priority_finalize_events(self) -> bool:
        if (
            not self._scheduler_finalize_priority_active()
            or not self._scheduler_priority_finalize_events
        ):
            return False

        ready_to_finalize: list[SchedulerFinalizeEvent] = []
        still_pending: list[SchedulerFinalizeEvent] = []
        for finalize_event in self._scheduler_priority_finalize_events:
            session = finalize_event.session
            if session.id in self._scheduler_inflight_sessions:
                still_pending.append(finalize_event)
                continue

            async with session.state_lock:
                if self._scheduler_session_ready(session):
                    self._scheduler_mark_ready_if_ready_locked(session)
                    still_pending.append(finalize_event)
                    continue
                ready_to_finalize.append(finalize_event)

        self._scheduler_priority_finalize_events = still_pending
        if not ready_to_finalize:
            return False

        await self._scheduler_process_finalize_event_batch(ready_to_finalize)
        return True

    async def _scheduler_process_priority_finalize_ready_pass(self) -> bool:
        if (
            not self._scheduler_finalize_priority_active()
            or not self._scheduler_priority_finalize_events
        ):
            return False

        for finalize_event in list(self._scheduler_priority_finalize_events):
            session = finalize_event.session
            if session.id in self._scheduler_inflight_sessions:
                continue

            async with session.state_lock:
                if not self._scheduler_session_ready(session):
                    continue
                key = self._scheduler_batch_group_key_for_session(session)

            processed = await self._scheduler_process_ready_batch_locked_sessions(
                key,
                [session],
                reason="finalize_priority_ready",
                ready_count=1,
                eligible_count=1,
                excluded_lanes=set(),
            )
            if processed:
                return True

        return False

    async def _scheduler_process_priority_finalize_before_ready(self) -> bool:
        if not self._scheduler_finalize_priority_active():
            return False

        progressed = await self._scheduler_process_priority_finalize_events()
        if self._scheduler_priority_finalize_events:
            progressed = (
                await self._scheduler_process_priority_finalize_ready_pass()
                or progressed
            )
        if self._scheduler_priority_finalize_events:
            progressed = (
                await self._scheduler_process_priority_finalize_events()
                or progressed
            )
        return progressed

    async def _scheduler_stage_finalize_events_before_ready(
        self,
        finalize_events: list[SchedulerFinalizeEvent],
    ) -> bool:
        if not finalize_events:
            return False

        if not self._scheduler_finalize_priority_active():
            await self._scheduler_process_finalize_event_batch(finalize_events)
            return True

        self._scheduler_priority_finalize_events.extend(finalize_events)
        return await self._scheduler_process_priority_finalize_before_ready()

    def _scheduler_priority_finalize_has_runnable_work(self) -> bool:
        if (
            not self._scheduler_finalize_priority_active()
            or not self._scheduler_priority_finalize_events
        ):
            return False

        for finalize_event in self._scheduler_priority_finalize_events:
            session = finalize_event.session
            if session.scheduler_closed:
                continue
            if session.id in self._scheduler_inflight_sessions:
                continue
            if self.model_lanes <= 1:
                return True

            lane_id = self._scheduler_assign_session_model_lane(session)
            if (
                self._scheduler_model_lane_exclusive_active
                or lane_id not in self._scheduler_available_model_lanes
            ):
                continue
            if self._scheduler_session_ready(session):
                key = self._scheduler_batch_group_key_for_session(session)
                if self._scheduler_model_lane_key_can_dispatch(key):
                    return True
                continue
            return True
        return False

    def _scheduler_has_queued_events(self) -> bool:
        for session in list(self.sessions.values()):
            if (
                self.model_lanes > 1
                and session.id in self._scheduler_inflight_sessions
            ):
                continue
            if (
                self._scheduler_batch_barrier_drain_active()
                and session.scheduler_pending_barrier_event is not None
            ):
                return True
            queue = session.continuous_event_queue
            if queue is not None and not queue.empty():
                return True
        return False

    def _scheduler_has_work_or_due_timer(self) -> bool:
        if self._scheduler_priority_finalize_has_runnable_work():
            return True
        if self._scheduler_has_queued_events():
            return True
        if self._scheduler_ready:
            if not self.batch_enabled:
                return True
            if (
                self.model_lanes > 1
                and not self._scheduler_ready_has_model_lane_compatible_key()
            ):
                return False
            deadline = self._scheduler_next_batch_deadline()
            if deadline is None or time.monotonic() >= deadline:
                return True
        return False

    async def _scheduler_loop(self) -> None:
        logger.info(
            "scheduler_b1_loop_running "
            f"batch_enabled={self.batch_enabled} "
            f"batch_barrier_drain={self._scheduler_batch_barrier_drain_active()} "
            f"batch_finalize={self._scheduler_batch_finalize_active()} "
            f"batch_finalize_preproc={self._scheduler_batch_finalize_preproc_active()} "
            f"finalize_priority={self._scheduler_finalize_priority_active()} "
            f"model_lanes={self.model_lanes} "
            f"batch_max_wait_ms={self.batch_max_wait_ms} "
            f"batch_max_size={self.batch_max_size}"
        )
        # Debug A/B ONLY (default OFF => fix ON): NEMOTRON_SCHED_NO_YIELD=1 skips the cooperative yield below to
        # reproduce the pre-fix livelock for testing. Never set in production.
        sched_yield = os.environ.get("NEMOTRON_SCHED_NO_YIELD") != "1"
        while True:
            try:
                progressed = await self._scheduler_drain_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"scheduler_b1_loop_error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                progressed = False

            # Cooperative yield (REQUIRED for liveness). _scheduler_drain_once's awaits can ALL complete
            # synchronously — notably the batched-barrier path, where a barrier-pending session is perpetually
            # re-marked ready (_scheduler_maybe_complete_pending_barrier_event) and the ready-pass returns
            # progressed=True every pass. Without an explicit yield, `if progressed: continue` (and the
            # has_work `continue` below) then spin on in-memory work and NEVER return to the event loop to read
            # the socket → I/O is starved and the server freezes. Reproduced on the cloud under WAN load on BOTH
            # Python 3.11 AND 3.12 (stuck in _scheduler_drain_once_batched_barrier) — it is WAN-timing-triggered,
            # NOT a Python-version issue; it does not trigger on the fast local loopback. sleep(0) guarantees the
            # event loop services I/O + other tasks every iteration. Scheduling-timing only — outputs unchanged.
            if sched_yield:
                await asyncio.sleep(0)

            if progressed:
                continue

            if self._scheduler_wakeup is None:
                self._scheduler_wakeup = asyncio.Event()
            self._scheduler_wakeup.clear()
            if self._scheduler_has_work_or_due_timer():
                continue
            timeout = self._scheduler_wait_timeout()
            if timeout is None:
                await self._scheduler_wakeup.wait()
            else:
                try:
                    await asyncio.wait_for(self._scheduler_wakeup.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    pass

    async def _scheduler_drain_once(self) -> bool:
        if self._scheduler_batch_barrier_drain_active():
            return await self._scheduler_drain_once_batched_barrier()

        progressed = await self._scheduler_process_priority_finalize_before_ready()
        finalize_events: list[SchedulerFinalizeEvent] = []
        for session in list(self.sessions.values()):
            if session.id in self._scheduler_inflight_sessions:
                continue
            queue = session.continuous_event_queue
            if queue is None:
                continue
            try:
                event = queue.get_nowait()
            except asyncio.QueueEmpty:
                continue

            finalize_event = self._scheduler_finalize_event_if_batchable(
                session,
                event,
                queue,
            )
            if finalize_event is not None:
                finalize_events.append(finalize_event)
                progressed = True
                continue

            try:
                await self._scheduler_process_event(session, event)
            finally:
                queue.task_done()
            progressed = True

        if finalize_events:
            progressed = (
                await self._scheduler_stage_finalize_events_before_ready(finalize_events)
                or progressed
            )

        progressed = (
            await self._scheduler_process_priority_finalize_before_ready()
            or progressed
        )

        if (
            self._scheduler_ready
            and not (
                self._scheduler_finalize_priority_active()
                and self.model_lanes <= 1
                and self._scheduler_priority_finalize_events
            )
        ):
            progressed = await self._scheduler_process_ready_pass() or progressed

        return progressed

    async def _scheduler_drain_once_batched_barrier(self) -> bool:
        progressed = await self._scheduler_process_priority_finalize_before_ready()
        finalize_events: list[SchedulerFinalizeEvent] = []

        for session in list(self.sessions.values()):
            if session.id in self._scheduler_inflight_sessions:
                continue
            if session.scheduler_pending_barrier_event is None:
                continue
            completed = await self._scheduler_maybe_complete_pending_barrier_event(
                session
            )
            progressed = completed or progressed

        for session in list(self.sessions.values()):
            if session.id in self._scheduler_inflight_sessions:
                continue
            if session.scheduler_pending_barrier_event is not None:
                continue
            queue = session.continuous_event_queue
            if queue is None:
                continue
            try:
                event = queue.get_nowait()
            except asyncio.QueueEmpty:
                continue

            if event[0] == "audio":
                try:
                    await self._scheduler_process_event(session, event)
                finally:
                    queue.task_done()
                progressed = True
                continue

            finalize_event = self._scheduler_finalize_event_if_batchable(
                session,
                event,
                queue,
            )
            if finalize_event is not None:
                finalize_events.append(finalize_event)
                progressed = True
                continue

            task_done_now = False
            try:
                async with session.state_lock:
                    self._scheduler_ready.discard(session.id)
                    if self._scheduler_session_ready(session):
                        session.scheduler_pending_barrier_event = event
                        session.scheduler_pending_barrier_queue = queue
                        session.scheduler_pending_barrier_drained = 0
                        if session.scheduler_inflight_generation is not None:
                            logger.debug(
                                f"Session {session.id}: scheduler batched barrier "
                                f"waiting for in-flight "
                                f"gen={session.scheduler_inflight_generation} "
                                f"reason={event[0]}"
                            )
                        logger.debug(
                            f"Session {session.id}: scheduler batched barrier "
                            f"deferred {event[0]} until ready backlog drains"
                        )
                        self._scheduler_mark_ready_if_ready_locked(session)
                    else:
                        await self._scheduler_process_event_after_barrier_locked(
                            session,
                            event,
                        )
                        task_done_now = True
            finally:
                if task_done_now:
                    queue.task_done()
            progressed = True

        if finalize_events:
            progressed = (
                await self._scheduler_stage_finalize_events_before_ready(finalize_events)
                or progressed
            )

        progressed = (
            await self._scheduler_process_priority_finalize_before_ready()
            or progressed
        )

        if (
            self._scheduler_ready
            and not (
                self._scheduler_finalize_priority_active()
                and self.model_lanes <= 1
                and self._scheduler_priority_finalize_events
            )
        ):
            progressed = await self._scheduler_process_ready_pass() or progressed

        return progressed

    async def _scheduler_maybe_complete_pending_barrier_event(
        self,
        session: ASRSession,
    ) -> bool:
        async with session.state_lock:
            event = session.scheduler_pending_barrier_event
            if event is None:
                return False

            if self._scheduler_session_ready(session):
                self._scheduler_mark_ready_if_ready_locked(session)
                return False

            queue = session.scheduler_pending_barrier_queue
            drained = session.scheduler_pending_barrier_drained
            session.scheduler_pending_barrier_event = None
            session.scheduler_pending_barrier_queue = None
            session.scheduler_pending_barrier_drained = 0
            if drained:
                logger.debug(
                    f"Session {session.id}: scheduler_batch_barrier_drained "
                    f"chunks={drained} before {event[0]}"
                )
            await self._scheduler_process_event_after_barrier_locked(session, event)

        if queue is not None:
            queue.task_done()
        return True

    def _scheduler_finalize_event_if_batchable(
        self,
        session: ASRSession,
        event: tuple,
        queue: asyncio.Queue,
    ) -> Optional[SchedulerFinalizeEvent]:
        if not self._scheduler_batch_finalize_active():
            return None
        if not event or event[0] != "debounce_expired":
            return None
        return SchedulerFinalizeEvent(session=session, event=event, queue=queue)

    async def _scheduler_process_finalize_event_batch(
        self,
        events: list[SchedulerFinalizeEvent],
    ) -> None:
        if not events:
            return
        try:
            items: list[SchedulerFinalizeItem] = []
            async with contextlib.AsyncExitStack() as stack:
                for finalize_event in sorted(events, key=lambda item: item.session.id):
                    await stack.enter_async_context(finalize_event.session.state_lock)

                for finalize_event in events:
                    session = finalize_event.session
                    event = finalize_event.event
                    stop_seq = event[1] if len(event) > 1 else None
                    if (
                        session.continuous_state != PENDING_FINALIZE
                        or stop_seq != session.continuous_stop_seq
                    ):
                        logger.debug(
                            f"Session {session.id}: ignored stale batched debounce expiry "
                            f"seq={stop_seq} current={session.continuous_stop_seq} "
                            f"state={session.continuous_state}"
                        )
                        continue

                    session.continuous_debounce_task = None
                    reset_seen = session.continuous_reset_seen
                    session.continuous_reset_seen = False
                    session.continuous_state = FINALIZED
                    session.continuous_debounce_expiry_ts = time.time()
                    logger.debug(
                        f"Session {session.id}: debounce expired seq={stop_seq}; "
                        f"batched finalizing (reset_seen={reset_seen})"
                    )
                    reason = "reset_then_debounce" if reset_seen else "debounce_expired"
                    expected_generation = self._scheduler_invalidate_session_locked(
                        session,
                        reason=reason,
                    )
                    items.append(
                        self._continuous_prepare_finalize_item_locked(
                            session,
                            reason=reason,
                            expected_generation=expected_generation,
                            debounce_event_queued_perf=self._finalize_profile_event_queued_perf(
                                event
                            ),
                        )
                    )

                if items:
                    await self._continuous_flush_finalize_items_locked(items)
                    for item in items:
                        await self._continuous_emit_prepared_finalize_locked(item)
                        self._continuous_finish_speculative_finalize_locked(
                            item.session,
                            reason=item.reason,
                        )
        finally:
            for finalize_event in events:
                finalize_event.queue.task_done()

    def _scheduler_session_ready(self, session: ASRSession) -> bool:
        if session.scheduler_closed or session.continuous_event_queue is None:
            return False
        pending_len = len(session.pending_audio) if session.pending_audio is not None else 0
        return ready_predicate(
            synthetic_prefix_samples=session.synthetic_prefix_samples,
            total_audio_samples=session.total_audio_samples,
            emitted_frames=session.emitted_frames,
            shift_frames=self.shift_frames,
            hop_samples=self.hop_samples,
            pending_audio_len=pending_len,
            preprocess_new_audio_samples=self.preprocess_new_audio_samples,
        )

    def _scheduler_mark_ready_if_ready_locked(self, session: ASRSession) -> None:
        if session.id in self._scheduler_inflight_sessions:
            self._scheduler_ready.discard(session.id)
            return
        if self._scheduler_session_ready(session):
            if session.id not in self._scheduler_ready:
                session.scheduler_ready_since = time.monotonic()
            self._scheduler_ready.add(session.id)
        else:
            self._scheduler_ready.discard(session.id)
            session.scheduler_ready_since = None

    async def _scheduler_process_ready_pass(self) -> bool:
        if self.batch_enabled:
            return await self._scheduler_process_batched_ready_pass()

        progressed = False
        ready_ids = list(self._scheduler_ready)
        self._scheduler_ready.clear()
        for session_id in ready_ids:
            session = self.sessions.get(session_id)
            if session is None:
                continue
            async with session.state_lock:
                if not self._scheduler_session_ready(session):
                    session.scheduler_ready_since = None
                    continue
                processed = await self._scheduler_process_one_ready_chunk_locked(
                    session,
                    reason="ready",
                    requeue=True,
                )
                progressed = processed or progressed
        return progressed

    def _scheduler_batch_group_key_for_session(self, session: ASRSession) -> tuple:
        if session.emitted_frames == 0:
            chunk_t = self.shift_frames
            drop_extra = 0
        else:
            chunk_t = self.pre_encode_cache_size + self.shift_frames
            drop_extra = self.drop_extra
        target_lang = session.target_lang if session.target_lang is not None else self.target_lang
        base_key = batch_group_key(
            target_lang,
            False,
            drop_extra,
            chunk_t,
            self.decoder_strategy,
        )
        # RNNT batched state is validated only when decoder histories are
        # uniformly fresh or uniformly established. Include that in the live
        # grouping key so a newly joined stream cannot be coerced into an
        # established batch.
        return (
            *base_key,
            session.previous_hypotheses is None,
            session.pred_out_stream is None,
        )

    def _scheduler_batch_session_eligible_for_key(self, session: ASRSession, key: tuple) -> bool:
        if session.id in self._scheduler_inflight_sessions:
            return False
        if session.scheduler_closed or session.continuous_event_queue is None:
            return False
        if self._scheduler_batch_group_key_for_session(session) != key:
            return False
        pending_len = len(session.pending_audio) if session.pending_audio is not None else 0
        queue = session.continuous_event_queue
        queue_has_events = queue is not None and not queue.empty()
        return self._scheduler_session_ready(session) or pending_len > 0 or queue_has_events

    def _scheduler_batch_eligible_count(self, key: tuple) -> int:
        count = 0
        for session in list(self.sessions.values()):
            if self._scheduler_batch_session_eligible_for_key(session, key):
                count += 1
        return count

    async def _scheduler_collect_ready_groups(self) -> dict[tuple, list[ASRSession]]:
        ready_groups: dict[tuple, list[ASRSession]] = {}
        now = time.monotonic()
        for session_id in list(self._scheduler_ready):
            if session_id in self._scheduler_inflight_sessions:
                self._scheduler_ready.discard(session_id)
                continue
            session = self.sessions.get(session_id)
            if session is None:
                self._scheduler_ready.discard(session_id)
                continue
            async with session.state_lock:
                if not self._scheduler_session_ready(session):
                    self._scheduler_ready.discard(session_id)
                    session.scheduler_ready_since = None
                    continue
                if session.scheduler_ready_since is None:
                    session.scheduler_ready_since = now
                key = self._scheduler_batch_group_key_for_session(session)
                ready_groups.setdefault(key, []).append(session)

        for key in list(self._scheduler_batch_first_ready):
            if key not in ready_groups:
                self._scheduler_batch_first_ready.pop(key, None)
        return ready_groups

    async def _scheduler_process_batched_ready_pass(self) -> bool:
        ready_groups = await self._scheduler_collect_ready_groups()
        if not ready_groups:
            return False

        now = time.monotonic()
        excluded_lanes = self._scheduler_finalize_priority_pending_lanes()
        candidates: list[tuple[int, float, str, tuple, list[ASRSession], str, int, int]] = []
        for key, sessions in ready_groups.items():
            sessions.sort(key=lambda s: (s.scheduler_ready_since or now, s.id))
            ready_count = len(sessions)
            active_count = self._scheduler_batch_eligible_count(key)
            first_ready = min((s.scheduler_ready_since or now) for s in sessions)
            deadline = self._scheduler_batch_first_ready.get(key)
            if deadline is None:
                deadline = first_ready + (self.batch_max_wait_ms / 1000.0)
                self._scheduler_batch_first_ready[key] = deadline

            reason = ""
            if active_count <= 1:
                reason = "solo"
            elif ready_count >= self.batch_max_size:
                reason = "max_size"
            elif now >= deadline:
                reason = "timer"
            else:
                continue

            if (
                self.model_lanes > 1
                and not self._scheduler_model_lane_key_can_dispatch(
                    key,
                    excluded_lanes=excluded_lanes,
                )
            ):
                continue

            selected_sessions = sessions[: min(ready_count, self.batch_max_size)]
            if self.model_lanes > 1:
                selected_sessions = self._scheduler_select_lane_affine_sessions(
                    key,
                    selected_sessions,
                    excluded_lanes=excluded_lanes,
                )
                if not selected_sessions:
                    continue

            safe_size = len(selected_sessions)
            candidates.append(
                (
                    -safe_size,
                    deadline,
                    str(key),
                    key,
                    selected_sessions,
                    reason,
                    ready_count,
                    active_count,
                )
            )

        if not candidates:
            return False

        candidates.sort()
        (
            _neg_size,
            _deadline,
            _key_label,
            key,
            sessions,
            reason,
            ready_count,
            active_count,
        ) = candidates[0]
        return await self._scheduler_process_ready_batch_locked_sessions(
            key,
            sessions,
            reason=reason,
            ready_count=ready_count,
            eligible_count=active_count,
            excluded_lanes=excluded_lanes,
        )

    async def _scheduler_process_ready_batch_locked_sessions(
        self,
        key: tuple,
        sessions: list[ASRSession],
        *,
        reason: str,
        ready_count: int,
        eligible_count: int,
        excluded_lanes: Optional[set[int]] = None,
    ) -> bool:
        if not sessions:
            return False
        if self.model_lanes > 1:
            return await self._scheduler_dispatch_ready_batch_to_model_lane(
                key,
                sessions,
                reason=reason,
                ready_count=ready_count,
                eligible_count=eligible_count,
                excluded_lanes=excluded_lanes,
            )

        async with contextlib.AsyncExitStack() as stack:
            for session in sorted(sessions, key=lambda s: s.id):
                await stack.enter_async_context(session.state_lock)

            valid_sessions: list[ASRSession] = []
            for session in sessions:
                if (
                    not session.scheduler_closed
                    and self._scheduler_session_ready(session)
                    and self._scheduler_batch_group_key_for_session(session) == key
                ):
                    valid_sessions.append(session)
                else:
                    self._scheduler_ready.discard(session.id)
                    session.scheduler_ready_since = None

            if not valid_sessions:
                return False
            if len(valid_sessions) > self.batch_max_size:
                valid_sessions = valid_sessions[: self.batch_max_size]

            for session in valid_sessions:
                self._scheduler_ready.discard(session.id)
                session.scheduler_inflight_generation = session.scheduler_generation

            generations = {
                session.id: session.scheduler_generation for session in valid_sessions
            }
            dispatch_start = time.monotonic()
            lane_wait_start = time.perf_counter()
            lane_wait_ms = 0.0
            texts: dict[str, Optional[str]] = {}
            try:
                async with self.inference_lock:
                    lane_wait_ms = (time.perf_counter() - lane_wait_start) * 1000.0
                    live_sessions = [
                        session
                        for session in valid_sessions
                        if (
                            generations[session.id] == session.scheduler_generation
                            and not session.scheduler_closed
                        )
                    ]
                    if not live_sessions:
                        return False
                    texts = await self._run_inference_call(
                        self._process_ready_batch,
                        live_sessions,
                        lane_wait_ms,
                    )
            finally:
                for session in valid_sessions:
                    session.scheduler_inflight_generation = None

            progressed = False
            sent_count = 0
            for session in valid_sessions:
                generation = generations[session.id]
                if generation != session.scheduler_generation or session.scheduler_closed:
                    logger.debug(
                        f"Session {session.id}: suppressed stale scheduler batch output "
                        f"reason={reason} gen={generation} current={session.scheduler_generation}"
                    )
                    continue

                text = texts.get(session.id)
                if text is not None and text != session.current_text:
                    session.current_text = text
                    logger.debug(
                        f"Session {session.id} interim: "
                        f"{text[-50:] if len(text) > 50 else text}"
                    )
                    await self._send_json_locked(
                        session,
                        self._transcript_payload(session, text=text, is_final=False),
                        tolerate_closed=True,
                        description="scheduler batch interim transcript",
                    )
                    sent_count += 1

                queue_wait_ms = 0.0
                if session.scheduler_ready_since is not None:
                    queue_wait_ms = (
                        dispatch_start - session.scheduler_ready_since
                    ) * 1000.0
                session.scheduler_ready_since = None
                self._scheduler_record_batch_row_telemetry(
                    session,
                    batch_size=len(valid_sessions),
                    lane_wait_ms=lane_wait_ms,
                    queue_wait_ms=queue_wait_ms,
                    reason=reason,
                )
                if session.scheduler_pending_barrier_event is not None:
                    session.scheduler_pending_barrier_drained += 1
                self._scheduler_mark_ready_if_ready_locked(session)
                progressed = True

            self._scheduler_record_batch_telemetry(
                batch_size=len(valid_sessions),
                reason=reason,
                sent_count=sent_count,
                key=key,
                ready_count=ready_count,
                eligible_count=eligible_count,
            )
            return progressed

    def _scheduler_log_model_lane_task_failure(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(f"scheduler_model_lane_task_error: {exc}")
            import traceback
            logger.error("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

    async def _scheduler_dispatch_ready_batch_to_model_lane(
        self,
        key: tuple,
        sessions: list[ASRSession],
        *,
        reason: str,
        ready_count: int,
        eligible_count: int,
        excluded_lanes: Optional[set[int]] = None,
    ) -> bool:
        async with contextlib.AsyncExitStack() as stack:
            for session in sorted(sessions, key=lambda s: s.id):
                await stack.enter_async_context(session.state_lock)

            valid_sessions: list[ASRSession] = []
            for session in sessions:
                if (
                    session.id not in self._scheduler_inflight_sessions
                    and not session.scheduler_closed
                    and self._scheduler_session_ready(session)
                    and self._scheduler_batch_group_key_for_session(session) == key
                ):
                    valid_sessions.append(session)
                else:
                    self._scheduler_ready.discard(session.id)
                    session.scheduler_ready_since = None

            if not valid_sessions:
                return False
            if len(valid_sessions) > self.batch_max_size:
                valid_sessions = valid_sessions[: self.batch_max_size]

            preferred_lane = self._scheduler_preferred_lane_for_sessions(valid_sessions)
            reservation = self._scheduler_reserve_model_lane_for_key(
                key,
                preferred_lane=preferred_lane,
                excluded_lanes=excluded_lanes,
            )
            if reservation is None:
                return False
            lane_id, exclusive_lane = reservation

            try:
                for session in valid_sessions:
                    pinned_lane = self._scheduler_session_model_lane_affinity.get(
                        session.id
                    )
                    if pinned_lane is not None and pinned_lane != lane_id:
                        raise RuntimeError(
                            "scheduler_model_lane_affinity_violation "
                            f"session={session.id} pinned={pinned_lane} "
                            f"dispatch={lane_id}"
                        )
                    self._scheduler_session_model_lane_affinity[session.id] = lane_id

                for session in valid_sessions:
                    self._scheduler_ready.discard(session.id)
                    session.scheduler_inflight_generation = session.scheduler_generation
                    self._scheduler_inflight_sessions.add(session.id)

                generations = {
                    session.id: session.scheduler_generation for session in valid_sessions
                }
                dispatch_start = time.monotonic()
                lane_wait_start = time.perf_counter()
                task = asyncio.create_task(
                    self._scheduler_complete_model_lane_batch(
                        lane_id,
                        exclusive_lane,
                        key,
                        valid_sessions,
                        generations,
                        reason=reason,
                        ready_count=ready_count,
                        eligible_count=eligible_count,
                        dispatch_start=dispatch_start,
                        lane_wait_start=lane_wait_start,
                    ),
                    name=f"nemotron-model-lane-{lane_id}-batch",
                )
                self._scheduler_inflight_model_lane_tasks.add(task)
                task.add_done_callback(self._scheduler_log_model_lane_task_failure)
                return True
            except Exception:
                for session in valid_sessions:
                    session.scheduler_inflight_generation = None
                self._scheduler_inflight_sessions.difference_update(
                    {session.id for session in valid_sessions}
                )
                await self._scheduler_release_model_lane(
                    lane_id=lane_id,
                    exclusive_lane=exclusive_lane,
                    task=None,
                    session_ids=set(),
                )
                raise

    async def _scheduler_complete_model_lane_batch(
        self,
        lane_id: int,
        exclusive_lane: bool,
        key: tuple,
        valid_sessions: list[ASRSession],
        generations: dict[str, int],
        *,
        reason: str,
        ready_count: int,
        eligible_count: int,
        dispatch_start: float,
        lane_wait_start: float,
    ) -> None:
        task = asyncio.current_task()
        session_ids = {session.id for session in valid_sessions}
        lane_wait_ms = 0.0
        try:
            async with contextlib.AsyncExitStack() as stack:
                for session in sorted(valid_sessions, key=lambda s: s.id):
                    await stack.enter_async_context(session.state_lock)

                texts: dict[str, Optional[str]] = {}
                try:
                    lane_wait_ms = (time.perf_counter() - lane_wait_start) * 1000.0
                    live_sessions = [
                        session
                        for session in valid_sessions
                        if (
                            generations[session.id] == session.scheduler_generation
                            and not session.scheduler_closed
                        )
                    ]
                    if live_sessions:
                        texts = await self._run_scheduler_model_lane_call(
                            lane_id,
                            self._process_ready_batch,
                            live_sessions,
                            lane_wait_ms,
                        )
                finally:
                    for session in valid_sessions:
                        session.scheduler_inflight_generation = None

                progressed = False
                sent_count = 0
                ready_requeue_sessions: list[ASRSession] = []
                for session in valid_sessions:
                    generation = generations[session.id]
                    if generation != session.scheduler_generation or session.scheduler_closed:
                        logger.debug(
                            f"Session {session.id}: suppressed stale scheduler batch output "
                            f"reason={reason} gen={generation} current={session.scheduler_generation}"
                        )
                        continue

                    text = texts.get(session.id)
                    if text is not None and text != session.current_text:
                        session.current_text = text
                        logger.debug(
                            f"Session {session.id} interim: "
                            f"{text[-50:] if len(text) > 50 else text}"
                        )
                        await self._send_json_locked(
                            session,
                            self._transcript_payload(session, text=text, is_final=False),
                            tolerate_closed=True,
                            description="scheduler batch interim transcript",
                        )
                        sent_count += 1

                    queue_wait_ms = 0.0
                    if session.scheduler_ready_since is not None:
                        queue_wait_ms = (
                            dispatch_start - session.scheduler_ready_since
                        ) * 1000.0
                    session.scheduler_ready_since = None
                    self._scheduler_record_batch_row_telemetry(
                        session,
                        batch_size=len(valid_sessions),
                        lane_wait_ms=lane_wait_ms,
                        queue_wait_ms=queue_wait_ms,
                        reason=f"{reason}:lane{lane_id}",
                    )
                    if session.scheduler_pending_barrier_event is not None:
                        session.scheduler_pending_barrier_drained += 1
                    ready_requeue_sessions.append(session)
                    progressed = True

                self._scheduler_record_batch_telemetry(
                    batch_size=len(valid_sessions),
                    reason=f"{reason}:lane{lane_id}",
                    sent_count=sent_count,
                    key=key,
                    ready_count=ready_count,
                    eligible_count=eligible_count,
                )
                self._scheduler_inflight_sessions.difference_update(session_ids)
                for session in ready_requeue_sessions:
                    self._scheduler_mark_ready_if_ready_locked(session)
                if not progressed:
                    logger.debug(
                        f"scheduler_model_lane_no_progress lane={lane_id} "
                        f"sessions={','.join(sorted(session_ids))} reason={reason}"
                    )
        finally:
            await self._scheduler_release_model_lane(
                lane_id=lane_id,
                exclusive_lane=exclusive_lane,
                task=task,
                session_ids=session_ids,
            )

    async def _scheduler_process_one_ready_chunk_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
        requeue: bool,
    ) -> bool:
        if not self._scheduler_session_ready(session):
            self._scheduler_ready.discard(session.id)
            session.scheduler_ready_since = None
            return False

        self._scheduler_ready.discard(session.id)
        session.scheduler_ready_since = None
        generation = session.scheduler_generation
        session.scheduler_inflight_generation = generation
        lane_wait_start = time.perf_counter()
        lane_wait_ms = 0.0
        text: Optional[str] = None
        try:
            if self.model_lanes > 1:
                lane_id = self._scheduler_assign_session_model_lane(session)
                async with self._scheduler_exclusive_model_path(reason):
                    async with self.inference_lock:
                        lane_wait_ms = (time.perf_counter() - lane_wait_start) * 1000.0
                        if generation != session.scheduler_generation or session.scheduler_closed:
                            logger.debug(
                                f"Session {session.id}: skipped stale scheduler chunk "
                                f"reason={reason} gen={generation} current={session.scheduler_generation}"
                            )
                            return False
                        text = await self._run_scheduler_exclusive_inference_call(
                            self._process_chunk,
                            session,
                            lane_wait_ms,
                            lane_id=lane_id,
                        )
            else:
                async with self.inference_lock:
                    lane_wait_ms = (time.perf_counter() - lane_wait_start) * 1000.0
                    if generation != session.scheduler_generation or session.scheduler_closed:
                        logger.debug(
                            f"Session {session.id}: skipped stale scheduler chunk "
                            f"reason={reason} gen={generation} current={session.scheduler_generation}"
                        )
                        return False
                    text = await self._run_inference_call(
                        self._process_chunk,
                        session,
                        lane_wait_ms,
                    )
        finally:
            session.scheduler_inflight_generation = None

        if generation != session.scheduler_generation or session.scheduler_closed:
            logger.debug(
                f"Session {session.id}: suppressed stale scheduler chunk output "
                f"reason={reason} gen={generation} current={session.scheduler_generation}"
            )
            return False

        if text is not None and text != session.current_text:
            session.current_text = text
            logger.debug(
                f"Session {session.id} interim: "
                f"{text[-50:] if len(text) > 50 else text}"
            )
            if generation == session.scheduler_generation and not session.scheduler_closed:
                await self._send_json_locked(
                    session,
                    self._transcript_payload(session, text=text, is_final=False),
                    tolerate_closed=True,
                    description="scheduler interim transcript",
                )

        self._scheduler_record_chunk_telemetry(session, lane_wait_ms, reason)
        if requeue:
            self._scheduler_mark_ready_if_ready_locked(session)
        return True

    def _scheduler_record_batch_row_telemetry(
        self,
        session: ASRSession,
        *,
        batch_size: int,
        lane_wait_ms: float,
        queue_wait_ms: float,
        reason: str,
    ) -> None:
        self._scheduler_record_chunk_telemetry(
            session,
            lane_wait_ms,
            f"batch:{reason}:B{batch_size}",
        )
        self._scheduler_batch_queue_wait_ms_total += queue_wait_ms
        self._scheduler_batch_queue_wait_ms_max = max(
            self._scheduler_batch_queue_wait_ms_max,
            queue_wait_ms,
        )
        self._scheduler_batch_queue_wait_count += 1

    def _scheduler_record_batch_telemetry(
        self,
        *,
        batch_size: int,
        reason: str,
        sent_count: int,
        key: Optional[tuple] = None,
        ready_count: Optional[int] = None,
        eligible_count: Optional[int] = None,
    ) -> None:
        self._scheduler_batches += 1
        self._scheduler_batch_size_hist[batch_size] = (
            self._scheduler_batch_size_hist.get(batch_size, 0) + 1
        )
        if self._scheduler_batches % 25 == 0 or batch_size > 1:
            wait_avg = 0.0
            if self._scheduler_batch_queue_wait_count:
                wait_avg = (
                    self._scheduler_batch_queue_wait_ms_total
                    / self._scheduler_batch_queue_wait_count
                )
            logger.info(
                "scheduler_batch_telemetry "
                f"batches={self._scheduler_batches} "
                f"last_batch_size={batch_size} "
                f"last_reason={reason} "
                f"group_key={key} "
                f"prompt_lang={(key[0] if key else 'n/a')} "
                f"ready_count={(ready_count if ready_count is not None else 'n/a')} "
                f"eligible_ready_count={(eligible_count if eligible_count is not None else 'n/a')} "
                f"sent_count={sent_count} "
                f"effective_batch_hist={dict(sorted(self._scheduler_batch_size_hist.items()))} "
                f"fallback_counts={dict(sorted(self._scheduler_batch_fallback_counts.items()))} "
                f"last_fallback_reason={self.batch_fallback_reason or 'none'} "
                f"queue_wait_avg_ms={wait_avg:.2f} "
                f"queue_wait_max_ms={self._scheduler_batch_queue_wait_ms_max:.2f}"
            )

    def _scheduler_record_chunk_telemetry(
        self,
        session: ASRSession,
        lane_wait_ms: float,
        reason: str,
    ) -> None:
        self._scheduler_chunks += 1
        self._scheduler_lane_wait_ms_total += lane_wait_ms
        self._scheduler_lane_wait_ms_max = max(
            self._scheduler_lane_wait_ms_max,
            lane_wait_ms,
        )
        queue = session.continuous_event_queue
        queue_depth = queue.qsize() if queue is not None else 0
        lag_ms = None
        if session.scheduler_last_audio_monotonic is not None:
            lag_ms = (time.monotonic() - session.scheduler_last_audio_monotonic) * 1000.0
        lag_label = f"{lag_ms:.2f}" if lag_ms is not None else "n/a"
        if self._scheduler_chunks % 50 == 0:
            avg_wait = self._scheduler_lane_wait_ms_total / self._scheduler_chunks
            logger.info(
                "scheduler_b1_telemetry "
                f"chunks={self._scheduler_chunks} "
                f"model_lane_wait_avg_ms={avg_wait:.2f} "
                f"model_lane_wait_max_ms={self._scheduler_lane_wait_ms_max:.2f} "
                f"ready_set_size={len(self._scheduler_ready)} "
                f"queue_depth={queue_depth} "
                f"last_session={session.id} "
                f"last_session_lag_ms={lag_label}"
            )

    async def _scheduler_drain_ready_barrier_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        self._scheduler_ready.discard(session.id)
        if session.scheduler_inflight_generation is not None:
            logger.debug(
                f"Session {session.id}: scheduler barrier waiting for in-flight "
                f"gen={session.scheduler_inflight_generation} reason={reason}"
            )
        drained = 0
        while self._scheduler_session_ready(session):
            processed = await self._scheduler_process_one_ready_chunk_locked(
                session,
                reason=f"barrier:{reason}",
                requeue=False,
            )
            self._scheduler_ready.discard(session.id)
            if not processed:
                break
            drained += 1
        if drained:
            logger.debug(
                f"Session {session.id}: scheduler barrier drained {drained} "
                f"ready chunks before {reason}"
            )

    def _scheduler_invalidate_session_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> int:
        self._scheduler_ready.discard(session.id)
        session.scheduler_ready_since = None
        session.scheduler_generation += 1
        logger.debug(
            f"Session {session.id}: scheduler generation={session.scheduler_generation} "
            f"reason={reason}"
        )
        return session.scheduler_generation

    async def _scheduler_dispatch_event_locked(
        self,
        session: ASRSession,
        event: tuple,
    ) -> None:
        event_type = event[0]
        if event_type == "close":
            await self._scheduler_continuous_handle_close_locked(session)
        elif event_type == "audio":
            await self._scheduler_continuous_handle_audio_locked(session, event[1])
        elif event_type == "vad_start":
            await self._scheduler_continuous_handle_vad_start_locked(session)
        elif event_type == "vad_stop":
            await self._scheduler_continuous_handle_vad_stop_locked(session)
        elif event_type == "reset":
            await self._scheduler_continuous_handle_reset_locked(
                session,
                finalize=event[1],
                msg_type=event[2],
            )
        elif event_type == "debounce_expired":
            await self._scheduler_continuous_handle_debounce_expired_locked(
                session,
                stop_seq=event[1],
                debounce_event_queued_perf=self._finalize_profile_event_queued_perf(
                    event
                ),
            )
        else:
            logger.warning(
                f"Session {session.id}: unknown scheduler event {event_type}"
            )

    async def _scheduler_process_event_after_barrier_locked(
        self,
        session: ASRSession,
        event: tuple,
    ) -> None:
        event_type = event[0]
        close_future = event[1] if event_type == "close" and len(event) > 1 else None
        try:
            await self._scheduler_dispatch_event_locked(session, event)
        except Exception as e:
            logger.error(f"Session {session.id} scheduler worker error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            try:
                await session.websocket.send_str(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
            except Exception:
                pass
        finally:
            if close_future is not None and not close_future.done():
                close_future.set_result(True)

    async def _scheduler_process_event(self, session: ASRSession, event: tuple) -> None:
        event_type = event[0]
        close_future = event[1] if event_type == "close" and len(event) > 1 else None
        try:
            async with session.state_lock:
                if event_type != "audio":
                    self._scheduler_ready.discard(session.id)
                    await self._scheduler_drain_ready_barrier_locked(
                        session,
                        reason=event_type,
                    )
                await self._scheduler_dispatch_event_locked(session, event)
        except Exception as e:
            logger.error(f"Session {session.id} scheduler worker error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            try:
                await session.websocket.send_str(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
            except Exception:
                pass
        finally:
            if close_future is not None and not close_future.done():
                close_future.set_result(True)

    async def _scheduler_continuous_handle_audio_locked(
        self,
        session: ASRSession,
        audio_bytes: bytes,
    ) -> None:
        if session.continuous_state == PENDING_FINALIZE:
            session.continuous_post_stop_audio.extend(audio_bytes)
            if DEBUG_ASR:
                samples = len(session.continuous_post_stop_audio) // 2
                logger.debug(
                    f"Session {session.id}: held {len(audio_bytes)}B post-vad_stop "
                    f"audio ({samples} total samples) while pending finalize"
                )
            return

        if session.continuous_finalize_terminal_sent:
            # New streaming audio after a terminal = a new turn (covers clients that
            # finalize via reset/end without a vad_start/vad_stop frame).
            session.continuous_finalize_terminal_sent = False
        if session.continuous_post_stop_audio:
            await self._scheduler_flush_post_stop_audio_locked(
                session,
                reason="streaming_resume",
            )

        await self._scheduler_append_audio_locked(session, audio_bytes)

    async def _scheduler_append_audio_locked(
        self,
        session: ASRSession,
        audio_bytes: bytes,
    ) -> None:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if DEBUG_ASR:
            chunk_hash = hashlib.md5(audio_bytes).hexdigest()[:8]
            logger.debug(
                f"Session {session.id}: recv chunk {len(audio_bytes)}B hash={chunk_hash}"
            )

        self._capture_eou_snapshot_audio(session, audio_bytes)

        session.pending_audio = np.concatenate([session.pending_audio, audio_np])
        session.accumulated_audio = session.pending_audio
        session.total_audio_samples += len(audio_np)
        session.scheduler_last_audio_monotonic = time.monotonic()
        self._scheduler_mark_ready_if_ready_locked(session)

    async def _scheduler_flush_post_stop_audio_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        if not session.continuous_post_stop_audio:
            return

        audio_bytes = bytes(session.continuous_post_stop_audio)
        session.continuous_post_stop_audio.clear()
        samples = len(audio_bytes) // 2
        logger.debug(
            f"Session {session.id}: flushing {samples} post-vad_stop samples "
            f"for {reason}"
        )
        await self._scheduler_append_audio_locked(session, audio_bytes)

    def _scheduler_discard_post_stop_audio_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        if not session.continuous_post_stop_audio:
            return

        samples = len(session.continuous_post_stop_audio) // 2
        session.continuous_post_stop_audio.clear()
        logger.debug(
            f"Session {session.id}: discarded {samples} post-vad_stop samples "
            f"at true boundary ({reason})"
        )

    async def _scheduler_continuous_force_finalize_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
        include_post_stop_audio: bool,
    ) -> None:
        if reason not in _TRUE_BOUNDARY_FINALIZE_REASONS:
            raise RuntimeError(
                "continuous cold reset requested for non-boundary reason "
                f"{reason!r}"
            )

        await self._continuous_cancel_debounce_locked(session, invalidate=True)
        if include_post_stop_audio:
            await self._scheduler_flush_post_stop_audio_locked(session, reason=reason)
            await self._scheduler_drain_ready_barrier_locked(
                session,
                reason=f"{reason}:post_stop",
            )
        else:
            self._scheduler_discard_post_stop_audio_locked(session, reason=reason)

        if not self._continuous_has_audio_or_text(session):
            session.continuous_state = STREAMING
            session.continuous_reset_seen = False
            session.continuous_vad_stop_ts = None
            session.continuous_debounce_expiry_ts = None
            logger.debug(
                f"Session {session.id}: ignored empty forced continuous finalize "
                f"for {reason}"
            )
            return

        session.continuous_state = FINALIZED
        if session.continuous_debounce_expiry_ts is None:
            session.continuous_debounce_expiry_ts = time.time()
        self._scheduler_invalidate_session_locked(session, reason=reason)
        logger.debug(f"Session {session.id}: forced continuous finalize for {reason}")
        await self._scheduler_continuous_finalize_and_reset_locked(
            session,
            reason=reason,
        )

    async def _scheduler_continuous_handle_close_locked(self, session: ASRSession) -> None:
        if (
            session.continuous_state == PENDING_FINALIZE
            or self._continuous_has_audio_or_text(session)
            or session.continuous_post_stop_audio
        ):
            await self._scheduler_continuous_force_finalize_locked(
                session,
                reason="close",
                include_post_stop_audio=True,
            )
        else:
            logger.debug(
                f"Session {session.id}: continuous close with no pending final"
            )
        session.scheduler_closed = True
        self._scheduler_ready.discard(session.id)
        self._scheduler_session_model_lane_affinity.pop(session.id, None)
        self._scheduler_invalidate_session_locked(session, reason="close")

    async def _scheduler_continuous_handle_vad_start_locked(
        self,
        session: ASRSession,
    ) -> None:
        session.continuous_finalize_terminal_sent = False  # new turn begins (vad_start)
        if session.continuous_state == PENDING_FINALIZE:
            await self._continuous_cancel_debounce_locked(session, invalidate=True)
            session.continuous_state = STREAMING
            session.continuous_reset_seen = False
            session.continuous_vad_stop_ts = None
            session.continuous_debounce_expiry_ts = None
            logger.debug(
                f"Session {session.id}: vad_start canceled pending finalize; "
                "discarded speculative fork and continuing same ASR context"
            )
            await self._scheduler_flush_post_stop_audio_locked(
                session,
                reason="vad_start",
            )
        else:
            if session.continuous_post_stop_audio:
                await self._scheduler_flush_post_stop_audio_locked(
                    session,
                    reason="vad_start_after_speculative_finalize",
                )
            logger.debug(
                f"Session {session.id}: vad_start in state={session.continuous_state}"
            )

    async def _scheduler_continuous_handle_vad_stop_locked(
        self,
        session: ASRSession,
    ) -> None:
        self._scheduler_invalidate_session_locked(session, reason="vad_stop")
        await self._continuous_cancel_debounce_locked(session, invalidate=False)
        session.continuous_stop_seq += 1
        stop_seq = session.continuous_stop_seq
        session.continuous_state = PENDING_FINALIZE
        session.continuous_reset_seen = False
        session.continuous_vad_stop_ts = time.time()
        session.continuous_finalize_terminal_sent = False  # new finalize cycle begins
        session.continuous_debounce_expiry_ts = None
        session.continuous_debounce_task = asyncio.create_task(
            self._continuous_debounce_timer(session.id, stop_seq),
            name=f"nemotron-continuous-debounce-{session.id}-{stop_seq}",
        )
        logger.debug(
            f"Session {session.id}: vad_stop armed pending finalize seq={stop_seq} "
            f"({self.finalize_silence_ms}ms)"
        )

    async def _scheduler_continuous_handle_reset_locked(
        self,
        session: ASRSession,
        *,
        finalize: bool,
        msg_type: str,
    ) -> None:
        if not finalize:
            text = session.current_text
            await self._send_json_locked(
                session,
                self._transcript_payload(
                    session, text=text, is_final=True, finalize=False
                ),
                tolerate_closed=False,
                description="continuous soft reset",
            )
            logger.debug(
                f"Session {session.id}: continuous soft reset: "
                f"'{text[-50:] if len(text) > 50 else text}'"
            )
            return

        if msg_type == "end":
            if session.continuous_state == PENDING_FINALIZE:
                session.continuous_reset_seen = True
            if (
                session.continuous_state == PENDING_FINALIZE
                or self._continuous_has_audio_or_text(session)
                or session.continuous_post_stop_audio
            ):
                await self._scheduler_continuous_force_finalize_locked(
                    session,
                    reason=msg_type,
                    include_post_stop_audio=True,
                )
                return

            logger.debug(
                f"Session {session.id}: ignored empty continuous {msg_type} in "
                f"state={session.continuous_state}"
            )
            return

        if session.continuous_state == PENDING_FINALIZE:
            session.continuous_reset_seen = True
            logger.debug(
                f"Session {session.id}: delayed client {msg_type} while "
                "server debounce is pending"
            )
            return

        if self._continuous_has_audio_or_text(session):
            logger.debug(
                f"Session {session.id}: immediate continuous {msg_type} "
                "without pending VAD stop; speculative finalizing with "
                "context retained"
            )
            session.continuous_state = FINALIZED
            self._scheduler_invalidate_session_locked(session, reason=msg_type)
            await self._scheduler_continuous_finalize_emit_locked(
                session,
                reason=msg_type,
            )
            self._continuous_finish_speculative_finalize_locked(
                session,
                reason=msg_type,
            )
            return

        logger.debug(
            f"Session {session.id}: ignored empty continuous {msg_type} in "
            f"state={session.continuous_state}"
        )

    async def _scheduler_continuous_handle_debounce_expired_locked(
        self,
        session: ASRSession,
        *,
        stop_seq: int,
        debounce_event_queued_perf: Optional[float] = None,
    ) -> None:
        if (
            session.continuous_state != PENDING_FINALIZE
            or stop_seq != session.continuous_stop_seq
        ):
            logger.debug(
                f"Session {session.id}: ignored stale debounce expiry "
                f"seq={stop_seq} current={session.continuous_stop_seq} "
                f"state={session.continuous_state}"
            )
            return

        session.continuous_debounce_task = None
        reset_seen = session.continuous_reset_seen
        session.continuous_reset_seen = False
        session.continuous_state = FINALIZED
        session.continuous_debounce_expiry_ts = time.time()
        logger.debug(
            f"Session {session.id}: debounce expired seq={stop_seq}; "
            f"finalizing (reset_seen={reset_seen})"
        )
        reason = "reset_then_debounce" if reset_seen else "debounce_expired"
        self._scheduler_invalidate_session_locked(session, reason=reason)
        await self._scheduler_continuous_finalize_emit_locked(
            session,
            reason=reason,
            debounce_event_queued_perf=debounce_event_queued_perf,
        )
        self._continuous_finish_speculative_finalize_locked(
            session,
            reason=reason,
        )

    async def _scheduler_continuous_finalize_emit_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
        debounce_event_queued_perf: Optional[float] = None,
    ) -> None:
        await self._continuous_finalize_emit_locked(
            session,
            reason=reason,
            expected_generation=session.scheduler_generation,
            debounce_event_queued_perf=debounce_event_queued_perf,
        )

    async def _scheduler_continuous_finalize_and_reset_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        await self._scheduler_continuous_finalize_emit_locked(session, reason=reason)
        if reason == "close" and self._scheduler_batch_finalize_preproc_active():
            self._continuous_close_cleanup_after_finalize_locked(
                session,
                reason=reason,
            )
            self._scheduler_invalidate_session_locked(
                session,
                reason=f"{reason}:close_cleanup_complete",
            )
            return
        await self._continuous_cold_reset_after_finalize_locked(session, reason=reason)
        self._scheduler_invalidate_session_locked(
            session,
            reason=f"{reason}:cold_reset_complete",
        )

    async def _send_json_locked(
        self,
        session: ASRSession,
        payload: dict[str, Any],
        *,
        tolerate_closed: bool,
        description: str,
    ) -> bool:
        websocket = session.websocket
        if websocket is None or getattr(websocket, "closed", False):
            if tolerate_closed:
                logger.debug(
                    f"Session {session.id}: skipped {description} send; "
                    "websocket already closed"
                )
                return False
            raise ConnectionResetError(
                f"Session {session.id}: websocket closed before {description} send"
            )

        try:
            await websocket.send_str(json.dumps(payload))
            return True
        except (ClientConnectionResetError, ConnectionResetError) as e:
            if tolerate_closed:
                logger.debug(
                    f"Session {session.id}: skipped {description} send after "
                    f"connection close: {e}"
                )
                return False
            raise

    async def _continuous_session_worker(self, session: ASRSession) -> None:
        """Process continuous-mode events in arrival order."""
        queue = session.continuous_event_queue
        if queue is None:
            return

        while True:
            event = await queue.get()
            should_stop = event[0] == "close"
            try:
                event_type = event[0]

                async with session.state_lock:
                    if event_type == "close":
                        await self._continuous_handle_close_locked(session)
                    elif event_type == "audio":
                        await self._continuous_handle_audio_locked(session, event[1])
                    elif event_type == "vad_start":
                        await self._continuous_handle_vad_start_locked(session)
                    elif event_type == "vad_stop":
                        await self._continuous_handle_vad_stop_locked(session)
                    elif event_type == "reset":
                        await self._continuous_handle_reset_locked(
                            session,
                            finalize=event[1],
                            msg_type=event[2],
                        )
                    elif event_type == "debounce_expired":
                        await self._continuous_handle_debounce_expired_locked(
                            session,
                            stop_seq=event[1],
                            debounce_event_queued_perf=self._finalize_profile_event_queued_perf(
                                event
                            ),
                        )
                    else:
                        logger.warning(
                            f"Session {session.id}: unknown continuous event {event_type}"
                        )
            except Exception as e:
                logger.error(f"Session {session.id} continuous worker error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                try:
                    await session.websocket.send_str(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))
                except Exception:
                    pass
            finally:
                queue.task_done()
                if should_stop:
                    return

    async def _close_continuous_session(self, session: ASRSession) -> None:
        """Drain pending finalization through the worker and stop continuous mode."""
        if self.scheduler_enabled:
            await self._close_scheduler_continuous_session(session)
            return

        queue = session.continuous_event_queue
        worker = session.continuous_worker_task
        if queue is not None and worker is not None and not worker.done():
            await queue.put(("close",))
            with contextlib.suppress(asyncio.CancelledError):
                await worker

        task = session.continuous_debounce_task
        session.continuous_debounce_task = None
        session.continuous_stop_seq += 1
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        session.continuous_event_queue = None
        session.continuous_worker_task = None
        session.continuous_post_stop_audio.clear()

    async def _close_scheduler_continuous_session(self, session: ASRSession) -> None:
        queue = session.continuous_event_queue
        if queue is not None:
            close_future = asyncio.get_running_loop().create_future()
            await queue.put(("close", close_future))
            self._wake_scheduler()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await close_future

        task = session.continuous_debounce_task
        session.continuous_debounce_task = None
        session.continuous_stop_seq += 1
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self._scheduler_ready.discard(session.id)
        session.continuous_event_queue = None
        session.continuous_worker_task = None
        session.continuous_post_stop_audio.clear()
        session.scheduler_closed = True
        self._log_retained_cache_telemetry("scheduler_session_closed")

    async def _continuous_debounce_timer(self, session_id: str, stop_seq: int) -> None:
        """Wake after server-side silence and enqueue a finalize decision."""
        try:
            await asyncio.sleep(self.finalize_silence_seconds)
            session = self.sessions.get(session_id)
            if session is None or session.continuous_event_queue is None:
                return
            if self.finalize_profile_enabled:
                await session.continuous_event_queue.put(
                    ("debounce_expired", stop_seq, time.perf_counter())
                )
            else:
                await session.continuous_event_queue.put(("debounce_expired", stop_seq))
            self._wake_scheduler()
        except asyncio.CancelledError:
            pass

    async def _continuous_cancel_debounce_locked(
        self,
        session: ASRSession,
        *,
        invalidate: bool,
    ) -> None:
        task = session.continuous_debounce_task
        session.continuous_debounce_task = None
        if invalidate:
            session.continuous_stop_seq += 1

        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _continuous_handle_audio_locked(
        self,
        session: ASRSession,
        audio_bytes: bytes,
    ) -> None:
        if session.continuous_state == PENDING_FINALIZE:
            session.continuous_post_stop_audio.extend(audio_bytes)
            if DEBUG_ASR:
                samples = len(session.continuous_post_stop_audio) // 2
                logger.debug(
                    f"Session {session.id}: held {len(audio_bytes)}B post-vad_stop "
                    f"audio ({samples} total samples) while pending finalize"
                )
            return

        if session.continuous_finalize_terminal_sent:
            # New streaming audio after a terminal = a new turn (covers clients that
            # finalize via reset/end without a vad_start/vad_stop frame).
            session.continuous_finalize_terminal_sent = False
        if session.continuous_post_stop_audio:
            await self._continuous_flush_post_stop_audio_locked(
                session,
                reason="streaming_resume",
            )

        await self._handle_audio_locked(
            session,
            audio_bytes,
            tolerate_closed_send=True,
        )

    async def _continuous_flush_post_stop_audio_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        if not session.continuous_post_stop_audio:
            return

        audio_bytes = bytes(session.continuous_post_stop_audio)
        session.continuous_post_stop_audio.clear()
        samples = len(audio_bytes) // 2
        logger.debug(
            f"Session {session.id}: flushing {samples} post-vad_stop samples "
            f"for {reason}"
        )
        await self._handle_audio_locked(
            session,
            audio_bytes,
            tolerate_closed_send=(reason == "close"),
        )

    def _continuous_discard_post_stop_audio_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        if not session.continuous_post_stop_audio:
            return

        samples = len(session.continuous_post_stop_audio) // 2
        session.continuous_post_stop_audio.clear()
        logger.debug(
            f"Session {session.id}: discarded {samples} post-vad_stop samples "
            f"at true boundary ({reason})"
        )

    async def _continuous_force_finalize_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
        include_post_stop_audio: bool,
    ) -> None:
        if reason not in _TRUE_BOUNDARY_FINALIZE_REASONS:
            raise RuntimeError(
                "continuous cold reset requested for non-boundary reason "
                f"{reason!r}"
            )

        await self._continuous_cancel_debounce_locked(session, invalidate=True)
        if include_post_stop_audio:
            await self._continuous_flush_post_stop_audio_locked(session, reason=reason)
        else:
            self._continuous_discard_post_stop_audio_locked(session, reason=reason)

        if not self._continuous_has_audio_or_text(session):
            session.continuous_state = STREAMING
            session.continuous_reset_seen = False
            session.continuous_vad_stop_ts = None
            session.continuous_debounce_expiry_ts = None
            logger.debug(
                f"Session {session.id}: ignored empty forced continuous finalize "
                f"for {reason}"
            )
            return

        session.continuous_state = FINALIZED
        if session.continuous_debounce_expiry_ts is None:
            session.continuous_debounce_expiry_ts = time.time()
        logger.debug(f"Session {session.id}: forced continuous finalize for {reason}")
        await self._continuous_finalize_and_reset_locked(session, reason=reason)

    async def _continuous_handle_close_locked(self, session: ASRSession) -> None:
        if (
            session.continuous_state == PENDING_FINALIZE
            or self._continuous_has_audio_or_text(session)
            or session.continuous_post_stop_audio
        ):
            await self._continuous_force_finalize_locked(
                session,
                reason="close",
                include_post_stop_audio=True,
            )
            return

        logger.debug(
            f"Session {session.id}: continuous close with no pending final"
        )

    async def _continuous_handle_vad_start_locked(self, session: ASRSession) -> None:
        session.continuous_finalize_terminal_sent = False  # new turn begins (vad_start)
        if session.continuous_state == PENDING_FINALIZE:
            await self._continuous_cancel_debounce_locked(session, invalidate=True)
            session.continuous_state = STREAMING
            session.continuous_reset_seen = False
            session.continuous_vad_stop_ts = None
            session.continuous_debounce_expiry_ts = None
            logger.debug(
                f"Session {session.id}: vad_start canceled pending finalize; "
                "discarded speculative fork and continuing same ASR context"
            )
            await self._continuous_flush_post_stop_audio_locked(
                session,
                reason="vad_start",
            )
        else:
            if session.continuous_post_stop_audio:
                await self._continuous_flush_post_stop_audio_locked(
                    session,
                    reason="vad_start_after_speculative_finalize",
                )
            logger.debug(
                f"Session {session.id}: vad_start in state={session.continuous_state}"
            )

    async def _continuous_handle_vad_stop_locked(self, session: ASRSession) -> None:
        await self._continuous_cancel_debounce_locked(session, invalidate=False)
        session.continuous_stop_seq += 1
        stop_seq = session.continuous_stop_seq
        session.continuous_state = PENDING_FINALIZE
        session.continuous_reset_seen = False
        session.continuous_vad_stop_ts = time.time()
        session.continuous_finalize_terminal_sent = False  # new finalize cycle begins
        session.continuous_debounce_expiry_ts = None
        session.continuous_debounce_task = asyncio.create_task(
            self._continuous_debounce_timer(session.id, stop_seq),
            name=f"nemotron-continuous-debounce-{session.id}-{stop_seq}",
        )
        logger.debug(
            f"Session {session.id}: vad_stop armed pending finalize seq={stop_seq} "
            f"({self.finalize_silence_ms}ms)"
        )

    async def _continuous_handle_reset_locked(
        self,
        session: ASRSession,
        *,
        finalize: bool,
        msg_type: str,
    ) -> None:
        if not finalize:
            text = session.current_text
            await session.websocket.send_str(json.dumps(
                self._transcript_payload(
                    session, text=text, is_final=True, finalize=False
                )
            ))
            logger.debug(
                f"Session {session.id}: continuous soft reset: "
                f"'{text[-50:] if len(text) > 50 else text}'"
            )
            return

        if msg_type == "end":
            if session.continuous_state == PENDING_FINALIZE:
                session.continuous_reset_seen = True
            if (
                session.continuous_state == PENDING_FINALIZE
                or self._continuous_has_audio_or_text(session)
                or session.continuous_post_stop_audio
            ):
                await self._continuous_force_finalize_locked(
                    session,
                    reason=msg_type,
                    include_post_stop_audio=True,
                )
                return

            logger.debug(
                f"Session {session.id}: ignored empty continuous {msg_type} in "
                f"state={session.continuous_state}"
            )
            return

        if session.continuous_state == PENDING_FINALIZE:
            session.continuous_reset_seen = True
            logger.debug(
                f"Session {session.id}: delayed client {msg_type} while "
                "server debounce is pending"
            )
            return

        if self._continuous_has_audio_or_text(session):
            logger.debug(
                f"Session {session.id}: immediate continuous {msg_type} "
                "without pending VAD stop; speculative finalizing with "
                "context retained"
            )
            session.continuous_state = FINALIZED
            await self._continuous_finalize_emit_locked(session, reason=msg_type)
            self._continuous_finish_speculative_finalize_locked(
                session,
                reason=msg_type,
            )
            return

        logger.debug(
            f"Session {session.id}: ignored empty continuous {msg_type} in "
            f"state={session.continuous_state}"
        )

    async def _continuous_handle_debounce_expired_locked(
        self,
        session: ASRSession,
        *,
        stop_seq: int,
        debounce_event_queued_perf: Optional[float] = None,
    ) -> None:
        if (
            session.continuous_state != PENDING_FINALIZE
            or stop_seq != session.continuous_stop_seq
        ):
            logger.debug(
                f"Session {session.id}: ignored stale debounce expiry "
                f"seq={stop_seq} current={session.continuous_stop_seq} "
                f"state={session.continuous_state}"
            )
            return

        session.continuous_debounce_task = None
        reset_seen = session.continuous_reset_seen
        session.continuous_reset_seen = False
        session.continuous_state = FINALIZED
        session.continuous_debounce_expiry_ts = time.time()
        logger.debug(
            f"Session {session.id}: debounce expired seq={stop_seq}; "
            f"finalizing (reset_seen={reset_seen})"
        )
        reason = "reset_then_debounce" if reset_seen else "debounce_expired"
        await self._continuous_finalize_emit_locked(
            session,
            reason=reason,
            debounce_event_queued_perf=debounce_event_queued_perf,
        )
        self._continuous_finish_speculative_finalize_locked(
            session,
            reason=reason,
        )

    def _continuous_has_audio_or_text(self, session: ASRSession) -> bool:
        pending_len = len(session.pending_audio) if session.pending_audio is not None else 0
        return bool(session.current_text) or session.total_audio_samples > 0 or pending_len > 0

    def _build_continuous_finalize_fork(
        self,
        session: ASRSession,
        finalize_profile: Optional[dict[str, Any]] = None,
    ) -> ASRSession:
        """Create a disposable fork for final padding without touching parent state."""
        if finalize_profile is None:
            pending_audio = (
                session.pending_audio.copy()
                if session.pending_audio is not None
                else np.array([], dtype=np.float32)
            )
            padding_samples = 0
            if session.total_audio_samples > 0:
                padding_samples = self.final_padding_frames * self.hop_samples
                silence_padding = np.zeros(padding_samples, dtype=np.float32)
                pending_audio = np.concatenate([pending_audio, silence_padding])

            fork = ASRSession(
                id=f"{session.id}:fork",
                websocket=None,
                target_lang=session.target_lang,
            )
            fork.pending_audio = pending_audio
            fork.accumulated_audio = fork.pending_audio
            fork.total_audio_samples = session.total_audio_samples + padding_samples
            fork.synthetic_prefix_samples = session.synthetic_prefix_samples
            fork.raw_audio_ring = (
                session.raw_audio_ring.copy()
                if session.raw_audio_ring is not None
                else np.zeros(self.raw_audio_ring_samples, dtype=np.float32)
            )
            fork.mel_frame_ring = clone_tree(session.mel_frame_ring)
            fork.emitted_frames = session.emitted_frames
            fork.cache_last_channel = (
                tensor_clone(session.cache_last_channel)
                if session.cache_last_channel is not None
                else None
            )
            fork.cache_last_time = (
                tensor_clone(session.cache_last_time)
                if session.cache_last_time is not None
                else None
            )
            fork.cache_last_channel_len = (
                tensor_clone(session.cache_last_channel_len)
                if session.cache_last_channel_len is not None
                else None
            )
            fork.previous_hypotheses = clone_hypotheses_deep(session.previous_hypotheses)
            fork.pred_out_stream = clone_tree(session.pred_out_stream)
            fork.current_text = session.current_text
            fork.last_emitted_text = session.last_emitted_text
            fork.committed_text = session.committed_text
            fork.continuous_emitted_text = session.continuous_emitted_text
            return fork

        total_start = time.perf_counter()

        audio_start = time.perf_counter()
        pending_audio = (
            session.pending_audio.copy()
            if session.pending_audio is not None
            else np.array([], dtype=np.float32)
        )
        padding_samples = 0
        if session.total_audio_samples > 0:
            padding_samples = self.final_padding_frames * self.hop_samples
            silence_padding = np.zeros(padding_samples, dtype=np.float32)
            pending_audio = np.concatenate([pending_audio, silence_padding])
        audio_ms = (time.perf_counter() - audio_start) * 1000

        fork = ASRSession(
            id=f"{session.id}:fork",
            websocket=None,
            target_lang=session.target_lang,
        )
        fork.pending_audio = pending_audio
        fork.accumulated_audio = fork.pending_audio
        fork.total_audio_samples = session.total_audio_samples + padding_samples
        fork.synthetic_prefix_samples = session.synthetic_prefix_samples

        audio_start = time.perf_counter()
        fork.raw_audio_ring = (
            session.raw_audio_ring.copy()
            if session.raw_audio_ring is not None
            else np.zeros(self.raw_audio_ring_samples, dtype=np.float32)
        )
        audio_ms += (time.perf_counter() - audio_start) * 1000

        cache_start = time.perf_counter()
        fork.mel_frame_ring = clone_tree(session.mel_frame_ring)
        fork.emitted_frames = session.emitted_frames
        fork.cache_last_channel = (
            tensor_clone(session.cache_last_channel)
            if session.cache_last_channel is not None
            else None
        )
        fork.cache_last_time = (
            tensor_clone(session.cache_last_time)
            if session.cache_last_time is not None
            else None
        )
        fork.cache_last_channel_len = (
            tensor_clone(session.cache_last_channel_len)
            if session.cache_last_channel_len is not None
            else None
        )
        cache_ms = (time.perf_counter() - cache_start) * 1000

        hyps_start = time.perf_counter()
        fork.previous_hypotheses = clone_hypotheses_deep(session.previous_hypotheses)
        hyps_ms = (time.perf_counter() - hyps_start) * 1000

        pred_start = time.perf_counter()
        fork.pred_out_stream = clone_tree(session.pred_out_stream)
        pred_ms = (time.perf_counter() - pred_start) * 1000

        fork.current_text = session.current_text
        fork.last_emitted_text = session.last_emitted_text
        fork.committed_text = session.committed_text
        fork.continuous_emitted_text = session.continuous_emitted_text

        total_ms = (time.perf_counter() - total_start) * 1000
        finalize_profile["fork_clone_audio_ms"] = audio_ms
        finalize_profile["fork_clone_cache_ms"] = cache_ms
        finalize_profile["fork_clone_hyps_ms"] = hyps_ms
        finalize_profile["fork_clone_pred_ms"] = pred_ms
        finalize_profile["fork_clone_other_ms"] = max(
            0.0,
            total_ms - audio_ms - cache_ms - hyps_ms - pred_ms,
        )
        return fork

    def _snapshot_fork_assert_parent(self, session: ASRSession) -> dict[str, Any]:
        return {
            "cache_last_channel": (
                tensor_clone(session.cache_last_channel)
                if session.cache_last_channel is not None
                else None
            ),
            "cache_last_time": (
                tensor_clone(session.cache_last_time)
                if session.cache_last_time is not None
                else None
            ),
            "cache_last_channel_len": (
                tensor_clone(session.cache_last_channel_len)
                if session.cache_last_channel_len is not None
                else None
            ),
            "previous_hypotheses": clone_hypotheses_deep(session.previous_hypotheses),
            "pred_out_stream": clone_tree(session.pred_out_stream),
        }

    def _assert_fork_flush_parent_unchanged(
        self,
        session: ASRSession,
        snapshot: dict[str, Any],
    ) -> None:
        try:
            _assert_tree_equal(
                "cache_last_channel",
                snapshot["cache_last_channel"],
                session.cache_last_channel,
            )
            _assert_tree_equal(
                "cache_last_time",
                snapshot["cache_last_time"],
                session.cache_last_time,
            )
            _assert_tree_equal(
                "cache_last_channel_len",
                snapshot["cache_last_channel_len"],
                session.cache_last_channel_len,
            )
            _assert_tree_equal(
                "previous_hypotheses",
                snapshot["previous_hypotheses"],
                session.previous_hypotheses,
            )
            _assert_tree_equal(
                "pred_out_stream",
                snapshot["pred_out_stream"],
                session.pred_out_stream,
            )
        except AssertionError as e:
            logger.error(
                f"Session {session.id}: fork alias assertion FAILED: {e}"
            )
            raise

        logger.info(
            f"Session {session.id}: fork alias assertion PASSED "
            "(parent cache tensors + previous_hypotheses + pred_out_stream byte-identical)"
        )

    def _continuous_context_retention_summary(self, session: ASRSession) -> str:
        raw_ring_samples = (
            len(session.raw_audio_ring) if session.raw_audio_ring is not None else 0
        )
        mel_ring_frames = (
            int(session.mel_frame_ring.shape[-1])
            if session.mel_frame_ring is not None
            else 0
        )
        pending_samples = (
            len(session.pending_audio) if session.pending_audio is not None else 0
        )
        return (
            f"cache_last_channel={'set' if session.cache_last_channel is not None else 'None'}, "
            f"cache_last_time={'set' if session.cache_last_time is not None else 'None'}, "
            f"cache_last_channel_len={'set' if session.cache_last_channel_len is not None else 'None'}, "
            f"previous_hypotheses={'set' if session.previous_hypotheses is not None else 'None'}, "
            f"pred_out_stream={'set' if session.pred_out_stream is not None else 'None'}, "
            f"raw_audio_ring_samples={raw_ring_samples}, "
            f"mel_frame_ring_frames={mel_ring_frames}, "
            f"current_text_chars={len(session.current_text)}, "
            f"continuous_emitted_chars={len(session.continuous_emitted_text)}, "
            f"emitted_frames={session.emitted_frames}, "
            f"pending_samples={pending_samples}, "
            f"total_audio_samples={session.total_audio_samples}"
        )

    def _continuous_finalize_timing(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "reason": reason,
            "vad_stop": session.continuous_vad_stop_ts,
            "vad_stop_recv": session.continuous_vad_stop_recv_ts,
            "debounce_expiry": session.continuous_debounce_expiry_ts,
            "fork_flush_start": None,
            "fork_flush_done": None,
            "final_sent": None,
            "inference_lock_acquire_wait_ms": None,
            "gil_attrib_enabled": self.gil_attrib_enabled,
        }

    def _continuous_prepare_finalize_item_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
        expected_generation: int,
        debounce_event_queued_perf: Optional[float] = None,
    ) -> SchedulerFinalizeItem:
        audio_samples = session.total_audio_samples
        audio_duration_ms = (audio_samples * 1000) // self.sample_rate
        pending_len = len(session.pending_audio) if session.pending_audio is not None else 0
        held_len = len(session.continuous_post_stop_audio) // 2
        timing = self._continuous_finalize_timing(session, reason=reason)
        finalize_profile = self._new_finalize_profile(
            session,
            reason=reason,
            path="batch_finalize",
            debounce_event_queued_perf=debounce_event_queued_perf,
        )
        if finalize_profile is not None:
            finalize_profile["first"] = session.emitted_frames == 0
        logger.debug(
            f"Session {session.id} continuous finalize ({reason}): "
            f"audio={audio_samples} samples ({audio_duration_ms}ms), "
            f"pending={pending_len} samples, held_post_stop={held_len} samples, "
            f"emitted={session.emitted_frames} frames"
        )

        final_text = session.current_text
        should_flush = (
            session.total_audio_samples > 0
            or (session.pending_audio is not None and len(session.pending_audio) > 0)
        )
        parent_snapshot = None
        fork = None
        fork_clone_ms = 0.0
        if should_flush:
            timing["fork_flush_start"] = time.time()
            parent_snapshot = (
                self._snapshot_fork_assert_parent(session)
                if self.fork_assert_enabled
                else None
            )
            fork_clone_start = time.perf_counter()
            if finalize_profile is not None:
                fork = self._build_continuous_finalize_fork(
                    session,
                    finalize_profile=finalize_profile,
                )
            else:
                fork = self._build_continuous_finalize_fork(session)
            fork_clone_ms = (time.perf_counter() - fork_clone_start) * 1000
            if finalize_profile is not None:
                finalize_profile["fork_clone_ms"] = fork_clone_ms
            if self.scheduler_enabled:
                logger.info(
                    f"Session {session.id}: scheduler_b1 fork_clone_ms="
                    f"{fork_clone_ms:.2f} reason={reason}"
                )

        return SchedulerFinalizeItem(
            session=session,
            reason=reason,
            expected_generation=expected_generation,
            timing=timing,
            parent_snapshot=parent_snapshot,
            fork=fork,
            final_text=final_text,
            should_flush=should_flush,
            fork_clone_ms=fork_clone_ms,
            finalize_profile=finalize_profile,
        )

    @contextlib.asynccontextmanager
    async def _scheduler_pinned_model_lane_path(
        self,
        sessions: list[ASRSession],
        *,
        reason: str,
    ):
        if self.model_lanes <= 1:
            yield 0
            return

        if not sessions:
            yield 0
            return

        lane_ids = {
            self._scheduler_assign_session_model_lane(session)
            for session in sessions
        }
        if len(lane_ids) != 1:
            raise RuntimeError(
                "pinned model lane finalize batch received mixed lanes: "
                f"{sorted(lane_ids)}"
            )
        lane_id = next(iter(lane_ids))
        session_ids = {session.id for session in sessions}
        condition = self._scheduler_model_lane_condition_obj()
        async with condition:
            while (
                self._scheduler_model_lane_exclusive_active
                or lane_id not in self._scheduler_available_model_lanes
                or any(
                    session_id in self._scheduler_inflight_sessions
                    for session_id in session_ids
                )
            ):
                await condition.wait()
            self._scheduler_available_model_lanes.remove(lane_id)
            self._scheduler_inflight_sessions.update(session_ids)
        try:
            logger.debug(
                f"scheduler_finalize_pinned_lane_acquired lane={lane_id} "
                f"sessions={','.join(sorted(session_ids))} reason={reason}"
            )
            yield lane_id
        finally:
            async with condition:
                self._scheduler_available_model_lanes.add(lane_id)
                self._scheduler_inflight_sessions.difference_update(session_ids)
                if not self._scheduler_inflight_model_lane_tasks:
                    self._scheduler_model_lane_active_key = None
                condition.notify_all()
            self._wake_scheduler()

    async def _continuous_flush_finalize_items_locked(
        self,
        items: list[SchedulerFinalizeItem],
    ) -> None:
        flush_items = [
            item
            for item in items
            if (
                item.fork is not None
                and item.fork.pending_audio is not None
                and len(item.fork.pending_audio) > 0
            )
        ]
        if not flush_items:
            return

        lock_wait_start = time.perf_counter()
        if self.model_lanes > 1 and self._scheduler_batch_finalize_active():
            by_lane: dict[int, list[SchedulerFinalizeItem]] = {}
            for item in flush_items:
                lane_id = self._scheduler_assign_session_model_lane(item.session)
                by_lane.setdefault(lane_id, []).append(item)

            for lane_id, lane_items in sorted(by_lane.items()):
                async with self._scheduler_pinned_model_lane_path(
                    [item.session for item in lane_items],
                    reason="finalize_batch",
                ) as reserved_lane_id:
                    wait_ms = (time.perf_counter() - lock_wait_start) * 1000
                    for item in lane_items:
                        item.timing["inference_lock_acquire_wait_ms"] = wait_ms
                    texts = await self._run_scheduler_model_lane_call(
                        reserved_lane_id,
                        self._process_final_fork_groups,
                        lane_items,
                    )
                    for item in lane_items:
                        text = texts.get(item.session.id)
                        if text is not None:
                            item.final_text = text
        else:
            async with self.inference_lock:
                wait_ms = (time.perf_counter() - lock_wait_start) * 1000
                for item in flush_items:
                    item.timing["inference_lock_acquire_wait_ms"] = wait_ms
                texts = await self._run_inference_call(
                    self._process_final_fork_groups,
                    flush_items,
                )
                for item in flush_items:
                    text = texts.get(item.session.id)
                    if text is not None:
                        item.final_text = text

        for item in items:
            if item.parent_snapshot is not None:
                self._assert_fork_flush_parent_unchanged(
                    item.session,
                    item.parent_snapshot,
                )
            if item.should_flush:
                item.timing["fork_flush_done"] = time.time()
                logger.debug(
                    f"Session {item.session.id} continuous fork final chunk processed in "
                    f"{(item.timing['fork_flush_done'] - item.timing['fork_flush_start']) * 1000:.1f}ms: "
                    f"'{item.final_text[-50:] if len(item.final_text) > 50 else item.final_text}'"
                )

    async def _continuous_emit_prepared_finalize_locked(
        self,
        item: SchedulerFinalizeItem,
    ) -> None:
        session = item.session
        final_text = item.final_text
        timing = item.timing
        reason = item.reason
        expected_generation = item.expected_generation

        if not final_text.startswith(session.committed_text):
            logger.debug(
                f"Session {session.id}: continuous ASR correction detected, "
                f"committed='{session.committed_text[-30:]}', "
                f"new='{final_text[-30:]}'"
            )
        delta_text = _continuous_append_only_delta(
            final_text,
            session.continuous_emitted_text,
        )

        if expected_generation != session.scheduler_generation:
            logger.debug(
                f"Session {session.id}: suppressed stale continuous final "
                f"reason={reason} expected_gen={expected_generation} "
                f"current_gen={session.scheduler_generation}"
            )
            self._emit_finalize_profile_record(
                item.finalize_profile,
                timing=timing,
                final_text=final_text,
                delta_text="",
                emitted_to_client=False,
                suppressed_reason="stale_generation",
                should_flush=item.should_flush,
            )
            return

        session.committed_text = final_text
        session.last_emitted_text = final_text

        if delta_text:
            timing["final_sent"] = time.time()
            payload = self._transcript_payload(
                session, text=delta_text, is_final=True, finalize=True
            )
            payload["finalize_timing"] = timing
            sent = await self._send_json_locked(
                session,
                payload,
                tolerate_closed=(
                    reason == "close" or getattr(session.websocket, "closed", False)
                ),
                description="continuous final transcript",
            )
            if sent:
                session.continuous_finalize_terminal_sent = True
                session.continuous_emitted_text = (
                    session.continuous_emitted_text + " " + delta_text
                ).strip()
                logger.debug(
                    f"Session {session.id} continuous final: delta='{delta_text}' "
                    f"(cumulative='{final_text[-50:] if len(final_text) > 50 else final_text}', "
                    f"collector='{session.continuous_emitted_text[-50:]}')"
                )
            self._emit_finalize_profile_record(
                item.finalize_profile,
                timing=timing,
                final_text=final_text,
                delta_text=delta_text,
                emitted_to_client=sent,
                suppressed_reason=None if sent else "send_failed",
                should_flush=item.should_flush,
            )
        elif not await self._emit_empty_terminal_if_awaited(
            session,
            reason=reason,
            timing=timing,
            final_text=final_text,
            finalize_profile=item.finalize_profile,
            should_flush=item.should_flush,
        ):
            logger.debug(
                f"Session {session.id}: suppressed empty/duplicate continuous final "
                f"(cumulative='{final_text[-50:] if len(final_text) > 50 else final_text}', "
                f"collector='{session.continuous_emitted_text[-50:]}')"
            )
            self._emit_finalize_profile_record(
                item.finalize_profile,
                timing=timing,
                final_text=final_text,
                delta_text="",
                emitted_to_client=False,
                suppressed_reason="empty_or_duplicate",
                should_flush=item.should_flush,
            )

    def _prepare_final_fork_batch_row(
        self,
        item: SchedulerFinalizeItem,
    ) -> Optional[SchedulerFinalizeBatchRow]:
        session = item.fork
        if session is None or session.pending_audio is None or len(session.pending_audio) == 0:
            return None

        padded_total_samples = (
            session.emitted_frames * self.hop_samples + len(session.pending_audio)
        )
        total_mel_frames = (padded_total_samples // self.hop_samples) + 1
        remaining_frames = total_mel_frames - session.emitted_frames

        logger.debug(
            f"Session {session.id} final chunk: "
            f"total_mel={total_mel_frames}, emitted={session.emitted_frames}, "
            f"remaining={remaining_frames}"
        )

        if remaining_frames <= 0:
            logger.warning(f"Session {session.id}: No remaining frames to process!")
            if item.finalize_profile is not None:
                item.finalize_profile["model_skipped_reason"] = "no_remaining_frames"
            return None

        pending = session.pending_audio
        raw_ring = session.raw_audio_ring
        new_mels: list[torch.Tensor] = []
        frames_collected = 0
        while frames_collected < remaining_frames:
            frames_this_call = min(self.shift_frames, remaining_frames - frames_collected)
            needed_new_samples = min(
                len(pending),
                self.preprocess_new_audio_samples,
            )
            new_audio = pending[:needed_new_samples]
            fixed_audio, valid_samples = self._build_fixed_preprocess_audio(
                raw_ring,
                new_audio,
            )
            if item.finalize_profile is not None:
                self._finalize_profile_cuda_synchronize(item.finalize_profile)
                pre_start = time.perf_counter()
                mel, _mel_len = self._preprocess_fixed_audio(fixed_audio, valid_samples)
                self._finalize_profile_cuda_synchronize(item.finalize_profile)
                self._finalize_profile_add_preproc(
                    item.finalize_profile,
                    (time.perf_counter() - pre_start) * 1000,
                )
            else:
                mel, _mel_len = self._preprocess_fixed_audio(fixed_audio, valid_samples)
            start = self.first_preprocess_mel_frame
            new_mels.append(mel[:, :, start : start + frames_this_call])

            if frames_this_call == self.shift_frames:
                consumed_samples = min(self.shift_frames * self.hop_samples, len(pending))
                consumed_audio = pending[:consumed_samples]
                if len(consumed_audio) >= self.raw_audio_ring_samples:
                    raw_ring = consumed_audio[-self.raw_audio_ring_samples :].copy()
                elif len(consumed_audio) > 0:
                    keep = self.raw_audio_ring_samples - len(consumed_audio)
                    raw_ring = np.concatenate([raw_ring[-keep:], consumed_audio]).astype(
                        np.float32,
                        copy=False,
                    )
                pending = pending[consumed_samples:]
            frames_collected += frames_this_call

        new_mel = torch.cat(new_mels, dim=-1)
        if session.emitted_frames == 0:
            chunk_mel = new_mel
            drop_extra = 0
        else:
            chunk_mel = torch.cat((session.mel_frame_ring, new_mel), dim=-1)
            drop_extra = self.drop_extra

        return SchedulerFinalizeBatchRow(
            item=item,
            chunk_mel=chunk_mel,
            drop_extra=int(drop_extra),
        )

    def _preprocess_final_fixed_audio_batch(
        self,
        fixed_audios: list[np.ndarray],
        valid_samples: list[int],
        frames_this_call: int,
    ) -> Optional[list[torch.Tensor]]:
        if len(fixed_audios) < 2:
            return None
        if len(fixed_audios) != len(valid_samples):
            raise ValueError(
                f"fixed audio count {len(fixed_audios)} != length count {len(valid_samples)}"
            )
        if frames_this_call <= 0 or frames_this_call > self.shift_frames:
            raise ValueError(
                f"final preprocessor frames_this_call={frames_this_call} "
                f"outside 1..{self.shift_frames}"
            )

        audio_rows: list[np.ndarray] = []
        for audio, samples in zip(fixed_audios, valid_samples):
            if len(audio) != self.constant_preprocess_samples:
                return None
            if audio.dtype != np.float32:
                return None
            if samples < 0 or samples > self.constant_preprocess_samples:
                return None
            audio_rows.append(np.ascontiguousarray(audio))

        audio_batch = np.stack(audio_rows, axis=0)
        if audio_batch.shape != (len(audio_rows), self.constant_preprocess_samples):
            return None

        audio_tensor = torch.from_numpy(audio_batch).cuda()
        audio_len = torch.tensor(valid_samples, device='cuda', dtype=torch.long)
        token = self._gil_attrib_cuda_event_start()
        try:
            mel, _mel_len = self._current_inference_model().preprocessor(
                input_signal=audio_tensor,
                length=audio_len,
            )
        finally:
            self._gil_attrib_cuda_event_end(token)
        start = self.first_preprocess_mel_frame
        end = start + frames_this_call
        return [
            mel[index : index + 1, :, start:end].detach().clone()
            for index in range(len(audio_rows))
        ]

    def _advance_final_preprocess_state(
        self,
        state: SchedulerFinalizePreprocessState,
        frames_this_call: int,
    ) -> None:
        if frames_this_call == self.shift_frames:
            consumed_samples = min(
                self.shift_frames * self.hop_samples,
                len(state.pending),
            )
            consumed_audio = state.pending[:consumed_samples]
            if len(consumed_audio) >= self.raw_audio_ring_samples:
                state.raw_ring = consumed_audio[-self.raw_audio_ring_samples :].copy()
            elif len(consumed_audio) > 0:
                keep = self.raw_audio_ring_samples - len(consumed_audio)
                state.raw_ring = np.concatenate(
                    [state.raw_ring[-keep:], consumed_audio]
                ).astype(
                    np.float32,
                    copy=False,
                )
            state.pending = state.pending[consumed_samples:]
        state.frames_collected += frames_this_call

    def _prepare_final_fork_batch_rows_batched_preprocess(
        self,
        items: list[SchedulerFinalizeItem],
    ) -> list[SchedulerFinalizeBatchRow]:
        states: list[SchedulerFinalizePreprocessState] = []
        for item in items:
            session = item.fork
            if session is None or session.pending_audio is None or len(session.pending_audio) == 0:
                continue

            padded_total_samples = (
                session.emitted_frames * self.hop_samples + len(session.pending_audio)
            )
            total_mel_frames = (padded_total_samples // self.hop_samples) + 1
            remaining_frames = total_mel_frames - session.emitted_frames

            logger.debug(
                f"Session {session.id} final chunk: "
                f"total_mel={total_mel_frames}, emitted={session.emitted_frames}, "
                f"remaining={remaining_frames}"
            )

            if remaining_frames <= 0:
                logger.warning(f"Session {session.id}: No remaining frames to process!")
                if item.finalize_profile is not None:
                    item.finalize_profile["model_skipped_reason"] = "no_remaining_frames"
                continue

            states.append(
                SchedulerFinalizePreprocessState(
                    item=item,
                    pending=session.pending_audio,
                    raw_ring=session.raw_audio_ring,
                    remaining_frames=remaining_frames,
                )
            )

        while True:
            active_states = [
                state
                for state in states
                if state.frames_collected < state.remaining_frames
            ]
            if not active_states:
                break

            grouped: dict[
                tuple[int, int],
                list[tuple[SchedulerFinalizePreprocessState, np.ndarray, int, int]],
            ] = {}
            for state in active_states:
                frames_this_call = min(
                    self.shift_frames,
                    state.remaining_frames - state.frames_collected,
                )
                needed_new_samples = min(
                    len(state.pending),
                    self.preprocess_new_audio_samples,
                )
                new_audio = state.pending[:needed_new_samples]
                fixed_audio, valid_samples = self._build_fixed_preprocess_audio(
                    state.raw_ring,
                    new_audio,
                )
                grouped.setdefault((valid_samples, frames_this_call), []).append(
                    (state, fixed_audio, valid_samples, frames_this_call)
                )

            for (valid_sample_count, frames_this_call), group in grouped.items():
                for start_index in range(0, len(group), self.batch_max_size):
                    batch_group = group[start_index : start_index + self.batch_max_size]
                    mels: Optional[list[torch.Tensor]] = None
                    if len(batch_group) > 1:
                        fixed_audios = [entry[1] for entry in batch_group]
                        valid_samples = [entry[2] for entry in batch_group]
                        try:
                            profiles = [
                                entry[0].item.finalize_profile
                                for entry in batch_group
                                if entry[0].item.finalize_profile is not None
                            ]
                            if profiles:
                                sync_ms = self._finalize_profile_cuda_synchronize(
                                    profiles[0]
                                )
                                for profile in profiles[1:]:
                                    self._finalize_profile_add_cuda_sync_observed(
                                        profile,
                                        sync_ms,
                                    )
                                pre_start = time.perf_counter()
                            else:
                                pre_start = 0.0
                            mels = self._preprocess_final_fixed_audio_batch(
                                fixed_audios,
                                valid_samples,
                                frames_this_call,
                            )
                            if profiles:
                                sync_ms = self._finalize_profile_cuda_synchronize(
                                    profiles[0]
                                )
                                for profile in profiles[1:]:
                                    self._finalize_profile_add_cuda_sync_observed(
                                        profile,
                                        sync_ms,
                                    )
                                pre_ms = (time.perf_counter() - pre_start) * 1000
                                if mels is not None:
                                    for profile in profiles:
                                        self._finalize_profile_add_preproc(
                                            profile,
                                            pre_ms,
                                        )
                        except Exception as e:
                            logger.warning(
                                "scheduler_finalize_preproc_batch_fallback "
                                f"reason=batch_error group_size={len(batch_group)} "
                                f"valid_samples={valid_sample_count} "
                                f"frames={frames_this_call} "
                                f"error={type(e).__name__}: {e}"
                            )
                            mels = None

                    if mels is None:
                        mels = []
                        for _state, fixed_audio, valid_samples, _frames in batch_group:
                            profile = _state.item.finalize_profile
                            if profile is not None:
                                self._finalize_profile_cuda_synchronize(profile)
                                pre_start = time.perf_counter()
                            mel, _mel_len = self._preprocess_fixed_audio(
                                fixed_audio,
                                valid_samples,
                            )
                            if profile is not None:
                                self._finalize_profile_cuda_synchronize(profile)
                                self._finalize_profile_add_preproc(
                                    profile,
                                    (time.perf_counter() - pre_start) * 1000,
                                )
                            mel_start = self.first_preprocess_mel_frame
                            mels.append(
                                mel[:, :, mel_start : mel_start + frames_this_call]
                            )

                    for (state, _fixed_audio, _valid_samples, _frames), mel in zip(
                        batch_group,
                        mels,
                        strict=True,
                    ):
                        state.new_mels.append(mel)
                        self._advance_final_preprocess_state(state, frames_this_call)

        rows: list[SchedulerFinalizeBatchRow] = []
        for state in states:
            session = state.item.fork
            if session is None or not state.new_mels:
                continue
            new_mel = torch.cat(state.new_mels, dim=-1)
            if session.emitted_frames == 0:
                chunk_mel = new_mel
                drop_extra = 0
            else:
                chunk_mel = torch.cat((session.mel_frame_ring, new_mel), dim=-1)
                drop_extra = self.drop_extra

            rows.append(
                SchedulerFinalizeBatchRow(
                    item=state.item,
                    chunk_mel=chunk_mel,
                    drop_extra=int(drop_extra),
                )
            )
        return rows

    def _finalize_batch_group_key_for_row(
        self,
        row: SchedulerFinalizeBatchRow,
    ) -> tuple:
        fork = row.item.fork
        target_lang = (
            fork.target_lang
            if fork is not None and fork.target_lang is not None
            else self.target_lang
        )
        base_key = batch_group_key(
            target_lang,
            True,
            row.drop_extra,
            row.chunk_mel.shape[-1],
            self.decoder_strategy,
        )
        return (
            *base_key,
            fork.previous_hypotheses is None if fork is not None else True,
            fork.pred_out_stream is None if fork is not None else True,
        )

    def _record_finalize_batch_fallback(self, reason: str, count: int) -> None:
        self._scheduler_finalize_fallback_counts[reason] = (
            self._scheduler_finalize_fallback_counts.get(reason, 0) + count
        )
        self._scheduler_finalize_serial_fallback_calls += count

    def _record_finalize_batch_telemetry(
        self,
        *,
        batch_size: int,
        key: tuple,
        model_ms: float,
        scatter_ms: float,
    ) -> None:
        self._scheduler_finalize_batches += 1
        self._scheduler_finalize_rows += batch_size
        self._scheduler_finalize_batch_size_hist[batch_size] = (
            self._scheduler_finalize_batch_size_hist.get(batch_size, 0) + 1
        )
        avg_b = self._scheduler_finalize_rows / max(1, self._scheduler_finalize_batches)
        if self._scheduler_finalize_batches % 10 == 0 or batch_size > 1:
            logger.info(
                "scheduler_finalize_batch_telemetry "
                f"batches={self._scheduler_finalize_batches} "
                f"rows={self._scheduler_finalize_rows} "
                f"last_batch_size={batch_size} "
                f"avg_effective_B={avg_b:.2f} "
                f"group_key={key} "
                f"effective_batch_hist={dict(sorted(self._scheduler_finalize_batch_size_hist.items()))} "
                f"serial_fallback_calls={self._scheduler_finalize_serial_fallback_calls} "
                f"fallback_counts={dict(sorted(self._scheduler_finalize_fallback_counts.items()))} "
                f"model_batch_ms={model_ms:.2f} "
                f"scatter_postprocess_ms={scatter_ms:.2f}"
            )

    def _process_final_fork_groups(
        self,
        items: list[SchedulerFinalizeItem],
    ) -> dict[str, Optional[str]]:
        lock_wait_ms = 0.0
        for item in items:
            value = item.timing.get("inference_lock_acquire_wait_ms")
            if isinstance(value, (int, float)):
                lock_wait_ms = max(lock_wait_ms, float(value))
        gil_sample = self._gil_attrib_begin_sample(
            "finalize",
            path="batch_finalize",
            batch_size=len(items),
            inference_lock_wait_ms=lock_wait_ms,
        )
        try:
            if not items:
                return {}

            texts: dict[str, Optional[str]] = {
                item.session.id: item.final_text for item in items
            }
            with torch.inference_mode():
                if self.finalize_profile_enabled:
                    self._finalize_profile_cuda_synchronize_many(
                        [item.finalize_profile for item in items]
                    )
                elif not self.sync_compress_enabled:
                    self._cuda_synchronize_for_current_model_lane()
                live_items: list[SchedulerFinalizeItem] = []
                for item in items:
                    if item.expected_generation != item.session.scheduler_generation:
                        logger.debug(
                            f"Session {item.session.id}: skipped stale batched fork final chunk "
                            f"reason={item.reason} expected_gen={item.expected_generation} "
                            f"current_gen={item.session.scheduler_generation}"
                        )
                        continue
                    live_items.append(item)

                if self._scheduler_batch_finalize_preproc_active():
                    rows = self._prepare_final_fork_batch_rows_batched_preprocess(
                        live_items
                    )
                else:
                    rows = []
                    for item in live_items:
                        row = self._prepare_final_fork_batch_row(item)
                        if row is None:
                            continue
                        rows.append(row)

                grouped: dict[tuple, list[SchedulerFinalizeBatchRow]] = {}
                for row in rows:
                    grouped.setdefault(
                        self._finalize_batch_group_key_for_row(row),
                        [],
                    ).append(row)

                for key, group_rows in grouped.items():
                    for start in range(0, len(group_rows), self.batch_max_size):
                        batch_rows = group_rows[start : start + self.batch_max_size]
                        texts.update(self._process_final_batch_rows(batch_rows, key))

                if self.finalize_profile_enabled:
                    self._finalize_profile_cuda_synchronize_many(
                        [item.finalize_profile for item in live_items]
                    )
                else:
                    self._cuda_synchronize_for_current_model_lane()
                return texts
        except Exception as e:
            session_ids = ",".join(item.session.id for item in items)
            logger.error(f"scheduler finalize batch processing error sessions={session_ids}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {item.session.id: None for item in items}
        finally:
            self._gil_attrib_end_sample(gil_sample)

    def _process_final_batch_rows(
        self,
        rows: list[SchedulerFinalizeBatchRow],
        key: tuple,
    ) -> dict[str, Optional[str]]:
        gil_sample = self._gil_attrib_begin_sample(
            "finalize",
            path="finalize_batch_rows",
            batch_size=len(rows),
        )
        gather_start = time.perf_counter()
        final_gather_ms = None
        clone_hyp_flush_ms = None
        try:
            chunk_mels = [row.chunk_mel for row in rows]
            processed_signal, processed_signal_length = stack_processed(chunk_mels)
            cache_last_channel, cache_last_time, cache_last_channel_len = stack_caches(
                [
                    (
                        row.item.fork.cache_last_channel,
                        row.item.fork.cache_last_time,
                        row.item.fork.cache_last_channel_len,
                    )
                    for row in rows
                ]
            )
            clone_hyp_start = time.perf_counter()
            previous_hypotheses = [
                clone_hypotheses_deep(row.item.fork.previous_hypotheses)
                for row in rows
            ]
            previous_pred_out = [
                clone_tree(row.item.fork.pred_out_stream)
                for row in rows
            ]
            clone_hyp_flush_ms = (time.perf_counter() - clone_hyp_start) * 1000.0
            flat_hypotheses = stack_hypotheses(previous_hypotheses)
            flat_pred_out = stack_pred_out(previous_pred_out, rnnt=True)
            final_gather_ms = (time.perf_counter() - gather_start) * 1000.0
            self._gil_attrib_add_ms("scatter_gather_ms", final_gather_ms)
        except Exception as e:
            if len(rows) > 1:
                self._gil_attrib_end_sample(gil_sample)
                return self._process_final_batch_solo_fallback(
                    rows,
                    key=key,
                    reason="unsafe_stack",
                    error=e,
                )
            self._gil_attrib_end_sample(gil_sample)
            raise

        if self.prompted_model:
            self._apply_inference_prompt(rows[0].item.fork)

        model_profile: Optional[dict[str, Any]] = None
        if self.finalize_profile_enabled and any(
            row.item.finalize_profile is not None for row in rows
        ):
            model_profile = {
                "cuda_sync_ms": 0.0,
                "cuda_sync_invocations": 0,
                "encoder_invocations": 0,
            }
            self._finalize_profile_cuda_synchronize(model_profile)
        _ftp = self._maybe_finalize_torch_profile_begin()
        model_start = time.perf_counter()
        (
            pred_out_stream,
            transcribed_texts,
            batch_cache_last_channel,
            batch_cache_last_time,
            batch_cache_last_channel_len,
            batch_previous_hypotheses,
        ) = self._conformer_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=True,
            previous_hypotheses=flat_hypotheses,
            previous_pred_out=flat_pred_out,
            drop_extra_pre_encoded=rows[0].drop_extra,
            return_transcription=True,
            finalize_profile=model_profile,
        )
        if model_profile is not None:
            self._finalize_profile_cuda_synchronize(model_profile)
        else:
            self._cuda_synchronize_for_current_model_lane()
        model_ms = (time.perf_counter() - model_start) * 1000.0
        self._maybe_finalize_torch_profile_end(
            _ftp, b=len(rows), t=int(processed_signal.shape[-1]) if hasattr(processed_signal, "shape") else None
        )
        self._finalize_profile_set_model_wall(model_profile, model_ms)

        batch_size = len(rows)
        scatter_start = time.perf_counter()
        texts: dict[str, Optional[str]] = {}
        try:
            for index, row in enumerate(rows):
                fork = row.item.fork
                row_cache = scatter_cache_row(
                    batch_cache_last_channel,
                    batch_cache_last_time,
                    batch_cache_last_channel_len,
                    index,
                )
                fork.pred_out_stream = self._scatter_batch_list_item(
                    pred_out_stream,
                    index,
                    batch_size,
                )
                fork.previous_hypotheses = self._scatter_batch_list_item(
                    batch_previous_hypotheses,
                    index,
                    batch_size,
                )
                fork.cache_last_channel = row_cache[0]
                fork.cache_last_time = row_cache[1]
                fork.cache_last_channel_len = row_cache[2]
                if row.item.finalize_profile is not None:
                    self._finalize_profile_set_model_inputs(
                        row.item.finalize_profile,
                        processed_signal=processed_signal,
                        processed_signal_length=processed_signal_length,
                        cache_last_channel=cache_last_channel,
                        cache_last_channel_len=cache_last_channel_len,
                        drop_extra=rows[0].drop_extra,
                        first=(fork.emitted_frames == 0),
                        row_index=index,
                    )
                    self._finalize_profile_copy_model_profile(
                        row.item.finalize_profile,
                        model_profile,
                        row_index=index,
                    )
                    row.item.finalize_profile["cache_last_channel_out_shape"] = (
                        self._finalize_profile_shape(row_cache[0])
                    )
                    row.item.finalize_profile["cache_last_channel_len_out"] = (
                        self._finalize_profile_tensor_value(row_cache[2])
                    )
                    row.item.finalize_profile["final_gather_ms"] = final_gather_ms
                    row.item.finalize_profile["clone_hyp_flush_ms"] = clone_hyp_flush_ms
                text = row.item.final_text
                if (
                    transcribed_texts
                    and len(transcribed_texts) > index
                    and transcribed_texts[index]
                ):
                    text = self._extract_hypothesis_text(transcribed_texts[index])
                    logger.debug(
                        f"Session {fork.id} final chunk output: "
                        f"'{text[-50:] if len(text) > 50 else text}' "
                        f"(was: '{fork.current_text[-30:] if len(fork.current_text) > 30 else fork.current_text}')"
                    )
                else:
                    logger.debug(f"Session {fork.id} final chunk: no new text from model")
                texts[row.item.session.id] = text
        except Exception as e:
            if len(rows) > 1:
                self._gil_attrib_end_sample(gil_sample)
                return self._process_final_batch_solo_fallback(
                    rows,
                    key=key,
                    reason="unsafe_scatter",
                    error=e,
                )
            self._gil_attrib_end_sample(gil_sample)
            raise

        self._gil_attrib_add_ms(
            "scatter_gather_ms",
            (time.perf_counter() - scatter_start) * 1000.0,
        )
        if self.finalize_profile_enabled:
            self._finalize_profile_cuda_synchronize_many(
                [row.item.finalize_profile for row in rows]
            )
        else:
            self._cuda_synchronize_for_current_model_lane()
        scatter_ms = (time.perf_counter() - scatter_start) * 1000.0
        for row in rows:
            if row.item.finalize_profile is not None:
                row.item.finalize_profile["scatter_ms"] = scatter_ms
        self._record_finalize_batch_telemetry(
            batch_size=batch_size,
            key=key,
            model_ms=model_ms,
            scatter_ms=scatter_ms,
        )
        self._gil_attrib_end_sample(gil_sample)
        return texts

    def _process_final_batch_solo_fallback(
        self,
        rows: list[SchedulerFinalizeBatchRow],
        *,
        key: tuple,
        reason: str,
        error: Exception,
    ) -> dict[str, Optional[str]]:
        self._record_finalize_batch_fallback(reason, len(rows))
        logger.warning(
            "scheduler_finalize_batch_fallback "
            f"reason={reason} key={key} "
            f"sessions={','.join(row.item.session.id for row in rows)} "
            f"error={type(error).__name__}: {error}"
        )
        texts: dict[str, Optional[str]] = {}
        for row in rows:
            if row.item.finalize_profile is not None:
                text = self._process_final_chunk(
                    row.item.fork,
                    row.item.finalize_profile,
                )
            else:
                text = self._process_final_chunk(row.item.fork)
            texts[row.item.session.id] = text
        return texts

    async def _emit_empty_terminal_if_awaited(
        self,
        session: ASRSession,
        *,
        reason: str,
        timing: dict[str, Any],
        final_text: str,
        finalize_profile: Optional[dict[str, Any]],
        should_flush: bool,
    ) -> bool:
        """Guarantee one terminal per client-awaited finalize, even when the turn produced no
        text (silence / VAD false-trigger). Returns True if it emitted an empty terminal.

        Without this, a client that sent reset(finalize)/end and got an empty delta would wait
        forever for a final that is suppressed. Only fires for client-awaited finalize reasons
        and only once per finalize cycle (the flag is reset at vad_stop); "close"/"vad_start*"
        drains and post-terminal duplicates still suppress.
        """
        if reason not in _CLIENT_FINALIZE_REASONS or session.continuous_finalize_terminal_sent:
            return False
        timing["final_sent"] = time.time()
        payload = self._transcript_payload(session, text="", is_final=True, finalize=True)
        payload["finalize_timing"] = timing
        sent = await self._send_json_locked(
            session,
            payload,
            tolerate_closed=(reason == "close" or getattr(session.websocket, "closed", False)),
            description="continuous empty terminal final",
        )
        if sent:
            session.continuous_finalize_terminal_sent = True
        logger.debug(
            f"Session {session.id}: emitted EMPTY terminal final (reason={reason}, "
            f"no new text this finalize cycle)"
        )
        self._emit_finalize_profile_record(
            finalize_profile,
            timing=timing,
            final_text=final_text,
            delta_text="",
            emitted_to_client=sent,
            suppressed_reason=None if sent else "send_failed",
            should_flush=should_flush,
        )
        return True

    async def _continuous_finalize_emit_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
        expected_generation: Optional[int] = None,
        debounce_event_queued_perf: Optional[float] = None,
    ) -> None:
        """Finalize once on a disposable fork and emit one incremental delta."""
        finalize_profile = self._new_finalize_profile(
            session,
            reason=reason,
            path="serial_finalize",
            debounce_event_queued_perf=debounce_event_queued_perf,
        )
        if finalize_profile is not None:
            finalize_profile["first"] = session.emitted_frames == 0
        if (
            expected_generation is not None
            and expected_generation != session.scheduler_generation
        ):
            logger.debug(
                f"Session {session.id}: skipped stale continuous finalize "
                f"reason={reason} expected_gen={expected_generation} "
                f"current_gen={session.scheduler_generation}"
            )
            self._emit_finalize_profile_record(
                finalize_profile,
                timing=self._continuous_finalize_timing(session, reason=reason),
                final_text=session.current_text,
                delta_text="",
                emitted_to_client=False,
                suppressed_reason="stale_generation_pre",
                should_flush=False,
            )
            return

        audio_samples = session.total_audio_samples
        audio_duration_ms = (audio_samples * 1000) // self.sample_rate
        pending_len = len(session.pending_audio) if session.pending_audio is not None else 0
        held_len = len(session.continuous_post_stop_audio) // 2
        timing: dict[str, Any] = {
            "reason": reason,
            "vad_stop": session.continuous_vad_stop_ts,
            "vad_stop_recv": session.continuous_vad_stop_recv_ts,
            "debounce_expiry": session.continuous_debounce_expiry_ts,
            "fork_flush_start": None,
            "fork_flush_done": None,
            "final_sent": None,
            "inference_lock_acquire_wait_ms": None,
            "gil_attrib_enabled": self.gil_attrib_enabled,
        }
        logger.debug(
            f"Session {session.id} continuous finalize ({reason}): "
            f"audio={audio_samples} samples ({audio_duration_ms}ms), "
            f"pending={pending_len} samples, held_post_stop={held_len} samples, "
            f"emitted={session.emitted_frames} frames"
        )

        final_text = session.current_text
        should_flush = (
            session.total_audio_samples > 0
            or (session.pending_audio is not None and len(session.pending_audio) > 0)
        )
        if should_flush:
            timing["fork_flush_start"] = time.time()
            start_perf = time.perf_counter()
            parent_snapshot = (
                self._snapshot_fork_assert_parent(session)
                if self.fork_assert_enabled
                else None
            )
            fork_clone_start = time.perf_counter()
            if finalize_profile is not None:
                fork = self._build_continuous_finalize_fork(
                    session,
                    finalize_profile=finalize_profile,
                )
            else:
                fork = self._build_continuous_finalize_fork(session)
            fork_clone_ms = (time.perf_counter() - fork_clone_start) * 1000
            if finalize_profile is not None:
                finalize_profile["fork_clone_ms"] = fork_clone_ms
            if self.scheduler_enabled:
                logger.info(
                    f"Session {session.id}: scheduler_b1 fork_clone_ms="
                    f"{fork_clone_ms:.2f} reason={reason}"
                )
            if fork.pending_audio is not None and len(fork.pending_audio) > 0:
                lock_wait_start = time.perf_counter()
                if self.model_lanes > 1:
                    lane_id = self._scheduler_assign_session_model_lane(session)
                    if self._scheduler_batch_finalize_active():
                        async with self._scheduler_pinned_model_lane_path(
                            [session],
                            reason=f"finalize:{reason}",
                        ) as reserved_lane_id:
                            if reserved_lane_id != lane_id:
                                raise RuntimeError(
                                    "scheduler_finalize_lane_affinity_violation "
                                    f"session={session.id} pinned={lane_id} "
                                    f"dispatch={reserved_lane_id}"
                                )
                            if (
                                expected_generation is not None
                                and expected_generation != session.scheduler_generation
                            ):
                                logger.debug(
                                    f"Session {session.id}: skipped stale fork final chunk "
                                    f"reason={reason} expected_gen={expected_generation} "
                                    f"current_gen={session.scheduler_generation}"
                                )
                                self._emit_finalize_profile_record(
                                    finalize_profile,
                                    timing=timing,
                                    final_text=final_text,
                                    delta_text="",
                                    emitted_to_client=False,
                                    suppressed_reason="stale_generation_before_model",
                                    should_flush=should_flush,
                                )
                                return
                            timing["inference_lock_acquire_wait_ms"] = (
                                time.perf_counter() - lock_wait_start
                            ) * 1000
                            if finalize_profile is not None:
                                text = await self._run_scheduler_model_lane_call(
                                    reserved_lane_id,
                                    self._process_final_chunk,
                                    fork,
                                    finalize_profile,
                                    timing["inference_lock_acquire_wait_ms"] or 0.0,
                                )
                            else:
                                text = await self._run_scheduler_model_lane_call(
                                    reserved_lane_id,
                                    self._process_final_chunk,
                                    fork,
                                    None,
                                    timing["inference_lock_acquire_wait_ms"] or 0.0,
                                )
                            if text is not None:
                                final_text = text
                    else:
                        async with self._scheduler_exclusive_model_path(f"finalize:{reason}"):
                            async with self.inference_lock:
                                if (
                                    expected_generation is not None
                                    and expected_generation != session.scheduler_generation
                                ):
                                    logger.debug(
                                        f"Session {session.id}: skipped stale fork final chunk "
                                        f"reason={reason} expected_gen={expected_generation} "
                                        f"current_gen={session.scheduler_generation}"
                                    )
                                    self._emit_finalize_profile_record(
                                        finalize_profile,
                                        timing=timing,
                                        final_text=final_text,
                                        delta_text="",
                                        emitted_to_client=False,
                                        suppressed_reason="stale_generation_before_model",
                                        should_flush=should_flush,
                                    )
                                    return
                                timing["inference_lock_acquire_wait_ms"] = (
                                    time.perf_counter() - lock_wait_start
                                ) * 1000
                                if finalize_profile is not None:
                                    text = await self._run_scheduler_exclusive_inference_call(
                                        self._process_final_chunk,
                                        fork,
                                        finalize_profile,
                                        timing["inference_lock_acquire_wait_ms"] or 0.0,
                                        lane_id=lane_id,
                                    )
                                else:
                                    text = await self._run_scheduler_exclusive_inference_call(
                                        self._process_final_chunk,
                                        fork,
                                        None,
                                        timing["inference_lock_acquire_wait_ms"] or 0.0,
                                        lane_id=lane_id,
                                    )
                                if text is not None:
                                    final_text = text
                else:
                    async with self.inference_lock:
                        if (
                            expected_generation is not None
                            and expected_generation != session.scheduler_generation
                        ):
                            logger.debug(
                                f"Session {session.id}: skipped stale fork final chunk "
                                f"reason={reason} expected_gen={expected_generation} "
                                f"current_gen={session.scheduler_generation}"
                            )
                            self._emit_finalize_profile_record(
                                finalize_profile,
                                timing=timing,
                                final_text=final_text,
                                delta_text="",
                                emitted_to_client=False,
                                suppressed_reason="stale_generation_before_model",
                                should_flush=should_flush,
                            )
                            return
                        timing["inference_lock_acquire_wait_ms"] = (
                            time.perf_counter() - lock_wait_start
                        ) * 1000
                        if finalize_profile is not None:
                            text = await self._run_inference_call(
                                self._process_final_chunk,
                                fork,
                                finalize_profile,
                                timing["inference_lock_acquire_wait_ms"] or 0.0,
                            )
                        else:
                            text = await self._run_inference_call(
                                self._process_final_chunk,
                                fork,
                                None,
                                timing["inference_lock_acquire_wait_ms"] or 0.0,
                            )
                        if text is not None:
                            final_text = text
            if parent_snapshot is not None:
                self._assert_fork_flush_parent_unchanged(session, parent_snapshot)
            timing["fork_flush_done"] = time.time()
            elapsed_ms = (time.perf_counter() - start_perf) * 1000
            logger.debug(
                f"Session {session.id} continuous fork final chunk processed in "
                f"{elapsed_ms:.1f}ms: "
                f"'{final_text[-50:] if len(final_text) > 50 else final_text}'"
            )

        if not final_text.startswith(session.committed_text):
            logger.debug(
                f"Session {session.id}: continuous ASR correction detected, "
                f"committed='{session.committed_text[-30:]}', "
                f"new='{final_text[-30:]}'"
            )
        delta_text = _continuous_append_only_delta(
            final_text,
            session.continuous_emitted_text,
        )

        if (
            expected_generation is not None
            and expected_generation != session.scheduler_generation
        ):
            logger.debug(
                f"Session {session.id}: suppressed stale continuous final "
                f"reason={reason} expected_gen={expected_generation} "
                f"current_gen={session.scheduler_generation}"
            )
            self._emit_finalize_profile_record(
                finalize_profile,
                timing=timing,
                final_text=final_text,
                delta_text="",
                emitted_to_client=False,
                suppressed_reason="stale_generation",
                should_flush=should_flush,
            )
            return

        session.committed_text = final_text
        session.last_emitted_text = final_text

        if delta_text:
            timing["final_sent"] = time.time()
            payload = self._transcript_payload(
                session, text=delta_text, is_final=True, finalize=True
            )
            payload["finalize_timing"] = timing
            sent = await self._send_json_locked(
                session,
                payload,
                tolerate_closed=(
                    reason == "close" or getattr(session.websocket, "closed", False)
                ),
                description="continuous final transcript",
            )
            if sent:
                session.continuous_finalize_terminal_sent = True
                session.continuous_emitted_text = (
                    session.continuous_emitted_text + " " + delta_text
                ).strip()
                logger.debug(
                    f"Session {session.id} continuous final: delta='{delta_text}' "
                    f"(cumulative='{final_text[-50:] if len(final_text) > 50 else final_text}', "
                    f"collector='{session.continuous_emitted_text[-50:]}')"
                )
            self._emit_finalize_profile_record(
                finalize_profile,
                timing=timing,
                final_text=final_text,
                delta_text=delta_text,
                emitted_to_client=sent,
                suppressed_reason=None if sent else "send_failed",
                should_flush=should_flush,
            )
        elif not await self._emit_empty_terminal_if_awaited(
            session,
            reason=reason,
            timing=timing,
            final_text=final_text,
            finalize_profile=finalize_profile,
            should_flush=should_flush,
        ):
            logger.debug(
                f"Session {session.id}: suppressed empty/duplicate continuous final "
                f"(cumulative='{final_text[-50:] if len(final_text) > 50 else final_text}', "
                f"collector='{session.continuous_emitted_text[-50:]}')"
            )
            self._emit_finalize_profile_record(
                finalize_profile,
                timing=timing,
                final_text=final_text,
                delta_text="",
                emitted_to_client=False,
                suppressed_reason="empty_or_duplicate",
                should_flush=should_flush,
            )

    def _continuous_finish_speculative_finalize_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        """Clear debounce bookkeeping after a speculative emit; keep ASR state."""
        held_len = len(session.continuous_post_stop_audio) // 2
        session.continuous_state = STREAMING
        session.continuous_vad_stop_ts = None
        session.continuous_debounce_expiry_ts = None
        session.continuous_debounce_task = None
        session.continuous_reset_seen = False

        logger.info(
            f"Session {session.id}: continuous speculative finalize complete "
            f"(context retained; no cold reset): reason={reason}, "
            f"committed_chars={len(session.committed_text)}, "
            f"retained_post_stop_samples={held_len}, "
            f"{self._continuous_context_retention_summary(session)}"
        )
        self._log_retained_cache_telemetry(f"speculative_finalize:{reason}")

    def _continuous_close_cleanup_after_finalize_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        if reason != "close":
            raise RuntimeError(
                "close cleanup requested for non-close reason "
                f"{reason!r}"
            )

        session.current_text = ""
        session.committed_text = ""
        session.last_emitted_text = ""
        session.continuous_emitted_text = ""
        session.pending_audio = np.array([], dtype=np.float32)
        session.accumulated_audio = session.pending_audio
        session.total_audio_samples = 0
        session.synthetic_prefix_samples = 0
        session.raw_audio_ring = np.zeros(self.raw_audio_ring_samples, dtype=np.float32)
        session.mel_frame_ring = None
        session.emitted_frames = 0
        session.cache_last_channel = None
        session.cache_last_time = None
        session.cache_last_channel_len = None
        session.previous_hypotheses = None
        session.pred_out_stream = None
        session.overlap_buffer = None
        session.continuous_post_stop_audio.clear()
        session.continuous_reset_seen = False
        session.continuous_vad_stop_ts = None
        session.continuous_debounce_expiry_ts = None
        session.continuous_stop_seq += 1
        session.continuous_state = STREAMING

        logger.info(
            f"Session {session.id}: continuous close cleanup complete "
            "(state cleared without cold model reset)"
        )
        self._log_retained_cache_telemetry("close_cleanup")

    async def _continuous_cold_reset_after_finalize_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        if reason not in _TRUE_BOUNDARY_FINALIZE_REASONS:
            raise RuntimeError(
                "continuous cold reset requested for non-boundary reason "
                f"{reason!r}"
            )

        # True utterance boundary: now it is safe to cold-reset the ASR state.
        session.committed_text = ""
        session.last_emitted_text = ""
        session.continuous_emitted_text = ""
        session.overlap_buffer = None
        session.continuous_post_stop_audio.clear()
        session.continuous_reset_seen = False
        session.continuous_vad_stop_ts = None
        session.continuous_debounce_expiry_ts = None
        session.continuous_stop_seq += 1
        if self.model_lanes > 1:
            lane_id = self._scheduler_assign_session_model_lane(session)
            async with self._scheduler_exclusive_model_path(f"cold_reset:{reason}"):
                async with self.inference_lock:
                    await self._run_scheduler_exclusive_inference_call(
                        self._init_session,
                        session,
                        lane_id=lane_id,
                    )
        elif self.session_warmup_ms > 0:
            async with self.inference_lock:
                await self._run_inference_call(self._init_session, session)
        else:
            self._init_session(session)
        session.continuous_state = STREAMING

        logger.info(
            f"Session {session.id}: continuous true-boundary cold reset complete "
            f"(reason={reason})"
        )
        self._log_retained_cache_telemetry(f"cold_reset:{reason}")

    async def _continuous_finalize_and_reset_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        """Finalize on the shared fork path, then cold-reset at a true boundary."""
        await self._continuous_finalize_emit_locked(session, reason=reason)
        await self._continuous_cold_reset_after_finalize_locked(session, reason=reason)

    async def _handle_audio(self, session: ASRSession, audio_bytes: bytes):
        """Accumulate audio and process when enough frames available."""
        await self._handle_audio_locked(session, audio_bytes)

    async def _handle_audio_locked(
        self,
        session: ASRSession,
        audio_bytes: bytes,
        *,
        tolerate_closed_send: bool = False,
    ):
        """Accumulate audio and process when enough frames available."""
        if session.vad_gated_audio and not session.accepting_vad_audio:
            logger.debug(
                f"Session {session.id}: ignored {len(audio_bytes)}B post-final audio "
                "while waiting for next vad_start"
            )
            return

        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if DEBUG_ASR:
            chunk_hash = hashlib.md5(audio_bytes).hexdigest()[:8]
            logger.debug(f"Session {session.id}: recv chunk {len(audio_bytes)}B hash={chunk_hash}")

        self._capture_eou_snapshot_audio(session, audio_bytes)

        session.pending_audio = np.concatenate([session.pending_audio, audio_np])
        session.accumulated_audio = session.pending_audio
        session.total_audio_samples += len(audio_np)
        plan = self._plan_for_session(session)

        # Process if we have enough audio for new frames
        # We need shift_frames worth of new mel frames (after skipping edge frame)
        min_audio_for_chunk = (session.emitted_frames + plan.shift_frames + 1) * plan.hop_samples

        while self._session_timeline_samples(session) >= min_audio_for_chunk:
            async with self.inference_lock:
                text = await self._run_inference_call(self._process_chunk, session)

            if text is not None and text != session.current_text:
                session.current_text = text
                logger.debug(f"Session {session.id} interim: {text[-50:] if len(text) > 50 else text}")
                await self._send_json_locked(
                    session,
                    self._transcript_payload(session, text=text, is_final=False),
                    tolerate_closed=tolerate_closed_send,
                    description="interim transcript",
                )

            # Update minimum for next iteration
            min_audio_for_chunk = (session.emitted_frames + plan.shift_frames + 1) * plan.hop_samples

    def _prepare_scheduler_fixed_preprocess_audio(
        self,
        session: ASRSession,
    ) -> Optional[tuple[np.ndarray, int]]:
        if len(session.pending_audio) < self.preprocess_new_audio_samples:
            return None

        new_audio = session.pending_audio[: self.preprocess_new_audio_samples]
        return self._build_fixed_preprocess_audio(
            session.raw_audio_ring,
            new_audio,
        )

    def _preprocess_scheduler_fixed_audio_batch(
        self,
        fixed_audios: list[np.ndarray],
        valid_samples: list[int],
    ) -> Optional[list[torch.Tensor]]:
        if len(fixed_audios) < 2:
            return None
        if len(fixed_audios) != len(valid_samples):
            raise ValueError(
                f"fixed audio count {len(fixed_audios)} != length count {len(valid_samples)}"
            )

        audio_rows: list[np.ndarray] = []
        for audio, samples in zip(fixed_audios, valid_samples):
            if len(audio) != self.constant_preprocess_samples:
                return None
            if audio.dtype != np.float32:
                return None
            if samples < 0 or samples > self.constant_preprocess_samples:
                return None
            audio_rows.append(np.ascontiguousarray(audio))

        audio_batch = np.stack(audio_rows, axis=0)
        if audio_batch.shape != (len(audio_rows), self.constant_preprocess_samples):
            return None

        audio_tensor = torch.from_numpy(audio_batch).cuda()
        audio_len = torch.tensor(valid_samples, device='cuda', dtype=torch.long)
        token = self._gil_attrib_cuda_event_start()
        try:
            mel, _mel_len = self._current_inference_model().preprocessor(
                input_signal=audio_tensor,
                length=audio_len,
            )
        finally:
            self._gil_attrib_cuda_event_end(token)
        start = self.first_preprocess_mel_frame
        end = start + self.shift_frames
        return [
            mel[index : index + 1, :, start:end].detach().clone()
            for index in range(len(audio_rows))
        ]

    def _prepare_scheduler_batch_row(
        self,
        session: ASRSession,
        valid_new_mel: Optional[torch.Tensor] = None,
    ) -> Optional[SchedulerBatchRow]:
        if len(session.pending_audio) < self.preprocess_new_audio_samples:
            return None

        if valid_new_mel is None:
            fixed_input = self._prepare_scheduler_fixed_preprocess_audio(session)
            if fixed_input is None:
                return None
            fixed_audio, valid_samples = fixed_input
            mel, _mel_len = self._preprocess_fixed_audio(fixed_audio, valid_samples)
            start = self.first_preprocess_mel_frame
            valid_new_mel = mel[:, :, start : start + self.shift_frames]

        if session.emitted_frames == 0:
            chunk_mel = valid_new_mel
            drop_extra = 0
        else:
            chunk_mel = torch.cat((session.mel_frame_ring, valid_new_mel), dim=-1)
            drop_extra = self.drop_extra

        return SchedulerBatchRow(
            session=session,
            generation=session.scheduler_generation,
            chunk_mel=chunk_mel,
            valid_new_mel=valid_new_mel,
            drop_extra=int(drop_extra),
            eou_probe_snapshot=self._eou_probe_snapshot(session),
        )

    def _advance_session_after_normal_chunk(
        self,
        session: ASRSession,
        valid_new_mel: torch.Tensor,
    ) -> None:
        consumed_audio = session.pending_audio[: self.shift_frames * self.hop_samples]
        if len(consumed_audio) >= self.raw_audio_ring_samples:
            session.raw_audio_ring = consumed_audio[-self.raw_audio_ring_samples :].copy()
        else:
            keep = self.raw_audio_ring_samples - len(consumed_audio)
            session.raw_audio_ring = np.concatenate(
                [session.raw_audio_ring[-keep:], consumed_audio]
            ).astype(np.float32, copy=False)
        session.pending_audio = session.pending_audio[self.shift_frames * self.hop_samples :]
        session.accumulated_audio = session.pending_audio
        self._update_mel_frame_ring(session, valid_new_mel)
        session.emitted_frames += self.shift_frames

    @staticmethod
    def _scatter_batch_list_item(value: Any, index: int, batch_size: int) -> Any:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if len(value) != batch_size:
                raise RuntimeError(
                    f"batched decoder returned {len(value)} rows for B={batch_size}"
                )
            return [value[index]]
        if batch_size == 1:
            return [value]
        raise RuntimeError(
            f"batched decoder returned non-list {type(value).__name__} for B={batch_size}"
        )

    def _process_ready_batch_solo_fallback(
        self,
        sessions: list[ASRSession],
        *,
        reason: str,
        error: Exception,
    ) -> dict[str, Optional[str]]:
        self._record_batch_fallback(reason)
        logger.warning(
            "scheduler_batch_fallback "
            f"reason={reason} sessions={','.join(session.id for session in sessions)} "
            f"error={type(error).__name__}: {error}"
        )
        texts: dict[str, Optional[str]] = {}
        for session in sessions:
            if session.scheduler_closed:
                texts[session.id] = None
                continue
            texts[session.id] = self._process_chunk(session)
        return texts

    def _log_scheduler_batch_memory(
        self,
        *,
        rows: list[SchedulerBatchRow],
        preprocessor_ms: float,
        model_ms: float,
        scatter_ms: float,
        mem_before: dict[str, int],
        mem_after: dict[str, int],
    ) -> None:
        batch_size = len(rows)
        if batch_size <= 0:
            return
        if (
            batch_size == 1
            and model_ms < 200.0   # always log a STALL batch (find what coincides with num_alloc_retries)
            and self._scheduler_batches % self.batch_memory_telemetry_every != 0
        ):
            return
        prompt_lang = rows[0].session.target_lang or self.target_lang
        logger.info(
            "scheduler_batch_memory "
            f"batch_size={batch_size} "
            f"prompt_lang={prompt_lang} "
            f"drop_extra={rows[0].drop_extra} "
            f"preprocessor_batch_ms={preprocessor_ms:.2f} "
            f"model_batch_ms={model_ms:.2f} "
            f"scatter_postprocess_ms={scatter_ms:.2f} "
            f"cuda_active_before_bytes={mem_before['active_bytes']} "
            f"cuda_active_after_bytes={mem_after['active_bytes']} "
            f"cuda_allocated_before_bytes={mem_before['allocated_bytes']} "
            f"cuda_allocated_after_bytes={mem_after['allocated_bytes']} "
            f"cuda_reserved_before_bytes={mem_before['reserved_bytes']} "
            f"cuda_reserved_after_bytes={mem_after['reserved_bytes']} "
            f"cuda_max_reserved_bytes={mem_after['max_reserved_bytes']} "
            f"retained_session_cache_bytes={mem_after['retained_session_cache_bytes']} "
            f"num_alloc_retries={mem_after['num_alloc_retries']} "
            f"num_ooms={mem_after['num_ooms']}"
        )

    def _process_ready_batch(
        self,
        sessions: list[ASRSession],
        gil_lock_wait_ms: float = 0.0,
    ) -> dict[str, Optional[str]]:
        """Process one scheduler batch of same-group ready normal chunks."""
        gil_sample = self._gil_attrib_begin_sample(
            "chunk",
            path="scheduler_batch",
            batch_size=len(sessions),
            inference_lock_wait_ms=gil_lock_wait_ms,
        )
        try:
            if not sessions:
                return {}

            with torch.inference_mode():
                if not self.sync_compress_enabled:
                    self._cuda_synchronize_for_current_model_lane()
                mem_before = self._cuda_memory_snapshot()
                pre_start = time.perf_counter()
                preprocess_inputs: list[tuple[ASRSession, np.ndarray, int]] = []
                for session in sessions:
                    fixed_input = self._prepare_scheduler_fixed_preprocess_audio(session)
                    if fixed_input is None:
                        continue
                    fixed_audio, valid_samples = fixed_input
                    preprocess_inputs.append((session, fixed_audio, valid_samples))

                if not preprocess_inputs:
                    return {session.id: session.current_text for session in sessions}

                fixed_audios = [item[1] for item in preprocess_inputs]
                valid_samples = [item[2] for item in preprocess_inputs]
                batched_valid_new_mels = self._preprocess_scheduler_fixed_audio_batch(
                    fixed_audios,
                    valid_samples,
                )
                self._cuda_synchronize_for_current_model_lane()
                preprocessor_ms = (time.perf_counter() - pre_start) * 1000.0

                rows: list[SchedulerBatchRow] = []
                if batched_valid_new_mels is None:
                    for session, _fixed_audio, _valid_samples in preprocess_inputs:
                        row = self._prepare_scheduler_batch_row(session)
                        if row is None:
                            continue
                        rows.append(row)
                else:
                    for index, (session, _fixed_audio, _valid_samples) in enumerate(
                        preprocess_inputs
                    ):
                        row = self._prepare_scheduler_batch_row(
                            session,
                            batched_valid_new_mels[index],
                        )
                        if row is None:
                            continue
                        rows.append(row)

                if not rows:
                    return {session.id: session.current_text for session in sessions}

                stack_start = time.perf_counter()
                try:
                    drop_extras = {row.drop_extra for row in rows}
                    if len(drop_extras) != 1:
                        raise RuntimeError(
                            f"mixed drop_extra in scheduler batch: {sorted(drop_extras)}"
                        )

                    chunk_mels = [row.chunk_mel for row in rows]
                    processed_signal, processed_signal_length = stack_processed(chunk_mels)
                    cache_last_channel, cache_last_time, cache_last_channel_len = stack_caches(
                        [
                            (
                                row.session.cache_last_channel,
                                row.session.cache_last_time,
                                row.session.cache_last_channel_len,
                            )
                            for row in rows
                        ]
                    )
                    previous_hypotheses = [
                        clone_hypotheses_deep(row.session.previous_hypotheses)
                        for row in rows
                    ]
                    previous_pred_out = [
                        clone_tree(row.session.pred_out_stream)
                        for row in rows
                    ]
                    flat_hypotheses = stack_hypotheses(previous_hypotheses)
                    flat_pred_out = stack_pred_out(previous_pred_out, rnnt=True)
                    self._gil_attrib_add_ms(
                        "scatter_gather_ms",
                        (time.perf_counter() - stack_start) * 1000.0,
                    )
                except Exception as e:
                    if len(rows) > 1:
                        return self._process_ready_batch_solo_fallback(
                            [row.session for row in rows],
                            reason="unsafe_stack",
                            error=e,
                        )
                    raise

                if self._current_model_is_prompted():
                    self._apply_inference_prompt(rows[0].session)

                model_start = time.perf_counter()
                (
                    pred_out_stream,
                    transcribed_texts,
                    batch_cache_last_channel,
                    batch_cache_last_time,
                    batch_cache_last_channel_len,
                    batch_previous_hypotheses,
                ) = self._conformer_stream_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=False,
                    previous_hypotheses=flat_hypotheses,
                    previous_pred_out=flat_pred_out,
                    drop_extra_pre_encoded=rows[0].drop_extra,
                    return_transcription=True,
                )
                self._cuda_synchronize_for_current_model_lane()
                model_ms = (time.perf_counter() - model_start) * 1000.0

                batch_size = len(rows)
                scatter_start = time.perf_counter()
                scattered: list[tuple[SchedulerBatchRow, Any, Any, torch.Tensor, torch.Tensor, torch.Tensor, Optional[str]]] = []
                try:
                    for index, row in enumerate(rows):
                        row_cache = scatter_cache_row(
                            batch_cache_last_channel,
                            batch_cache_last_time,
                            batch_cache_last_channel_len,
                            index,
                        )
                        row_pred_out = self._scatter_batch_list_item(
                            pred_out_stream,
                            index,
                            batch_size,
                        )
                        row_hypotheses = self._scatter_batch_list_item(
                            batch_previous_hypotheses,
                            index,
                            batch_size,
                        )
                        text = row.session.current_text
                        if transcribed_texts and len(transcribed_texts) > index and transcribed_texts[index]:
                            text = self._extract_hypothesis_text(transcribed_texts[index])
                        scattered.append(
                            (
                                row,
                                row_pred_out,
                                row_hypotheses,
                                row_cache[0],
                                row_cache[1],
                                row_cache[2],
                                text,
                            )
                        )
                except Exception as e:
                    if len(rows) > 1:
                        return self._process_ready_batch_solo_fallback(
                            [row.session for row in rows],
                            reason="unsafe_scatter",
                            error=e,
                        )
                    raise

                texts: dict[str, Optional[str]] = {}
                for (
                    row,
                    row_pred_out,
                    row_hypotheses,
                    row_cache_last_channel,
                    row_cache_last_time,
                    row_cache_last_channel_len,
                    text,
                ) in scattered:
                    session = row.session
                    if row.generation != session.scheduler_generation or session.scheduler_closed:
                        continue
                    session.pred_out_stream = row_pred_out
                    session.previous_hypotheses = row_hypotheses
                    session.cache_last_channel = row_cache_last_channel
                    session.cache_last_time = row_cache_last_time
                    session.cache_last_channel_len = row_cache_last_channel_len
                    self._advance_session_after_normal_chunk(session, row.valid_new_mel)
                    self._write_eou_probe_chunk(session, row.eou_probe_snapshot)
                    self._write_eou_snapshot_chunk(session, row.eou_probe_snapshot)
                    texts[session.id] = text

                self._gil_attrib_add_ms(
                    "scatter_gather_ms",
                    (time.perf_counter() - scatter_start) * 1000.0,
                )
                self._cuda_synchronize_for_current_model_lane()
                scatter_ms = (time.perf_counter() - scatter_start) * 1000.0
                mem_after = self._cuda_memory_snapshot()
                self._log_scheduler_batch_memory(
                    rows=rows,
                    preprocessor_ms=preprocessor_ms,
                    model_ms=model_ms,
                    scatter_ms=scatter_ms,
                    mem_before=mem_before,
                    mem_after=mem_after,
                )
                return texts

        except Exception as e:
            session_ids = ",".join(session.id for session in sessions)
            oom = "out of memory" in str(e).lower()
            if oom and self.batch_max_size > 1:
                self.batch_max_size = 1
                self._record_batch_fallback("cuda_oom_clamped_to_B1")
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()
                logger.error(
                    "scheduler batch processing CUDA OOM; clamped batch_max_size=1 "
                    f"sessions={session_ids}: {e}"
                )
            else:
                logger.error(f"scheduler batch processing error sessions={session_ids}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {session.id: None for session in sessions}
        finally:
            self._gil_attrib_end_sample(gil_sample)

    def _process_chunk(
        self,
        session: ASRSession,
        gil_lock_wait_ms: float = 0.0,
    ) -> Optional[str]:
        """Process one fixed-plan audio window and run streaming inference."""
        gil_sample = self._gil_attrib_begin_sample(
            "chunk",
            path="serial_chunk",
            batch_size=1,
            inference_lock_wait_ms=gil_lock_wait_ms,
        )
        try:
            plan = self._plan_for_session(session)
            if len(session.pending_audio) < plan.preprocess_new_audio_samples:
                return session.current_text

            if DEBUG_ASR:
                audio_hash = _hash_audio(session.pending_audio)
                logger.debug(
                    f"Session {session.id}: process pending={len(session.pending_audio)} "
                    f"total={session.total_audio_samples} hash={audio_hash}"
                )

            new_audio = session.pending_audio[: plan.preprocess_new_audio_samples]
            fixed_audio, valid_samples = self._build_fixed_preprocess_audio(
                session.raw_audio_ring,
                new_audio,
                plan,
            )

            with torch.inference_mode():
                if self.profile_chunk:
                    torch.cuda.synchronize()
                    _prof_t0 = time.perf_counter()
                mel, mel_len = self._preprocess_fixed_audio(fixed_audio, valid_samples, plan)
                if self.profile_chunk:
                    torch.cuda.synchronize()
                    self._prof_pre_ms += (time.perf_counter() - _prof_t0) * 1000.0

                if DEBUG_ASR:
                    mel_hash = hashlib.md5(mel.cpu().numpy().tobytes()).hexdigest()[:8]
                    logger.debug(f"Session {session.id}: mel shape={mel.shape[-1]} hash={mel_hash}")

                valid_new_mel = mel[
                    :,
                    :,
                    plan.first_preprocess_mel_frame : plan.first_preprocess_mel_frame + plan.shift_frames,
                ]

                # Extract chunk with pre-encode cache
                if session.emitted_frames == 0:
                    # First chunk: just shift_frames, no mel cache
                    chunk_mel = valid_new_mel
                    drop_extra = 0
                else:
                    # Subsequent chunks: prepend retained mel pre-encode cache
                    chunk_mel = torch.cat((session.mel_frame_ring, valid_new_mel), dim=-1)
                    drop_extra = plan.drop_extra

                chunk_len = torch.tensor([chunk_mel.shape[-1]], device='cuda')
                eou_probe_snapshot = self._eou_probe_snapshot(session)

                # Run streaming inference
                if self._current_model_is_prompted():
                    self._apply_inference_prompt(session)
                if self.profile_chunk:
                    torch.cuda.synchronize()
                    _prof_t1 = time.perf_counter()
                (
                    session.pred_out_stream,
                    transcribed_texts,
                    session.cache_last_channel,
                    session.cache_last_time,
                    session.cache_last_channel_len,
                    session.previous_hypotheses,
                ) = self._conformer_stream_step(
                    processed_signal=chunk_mel,
                    processed_signal_length=chunk_len,
                    cache_last_channel=session.cache_last_channel,
                    cache_last_time=session.cache_last_time,
                    cache_last_channel_len=session.cache_last_channel_len,
                    keep_all_outputs=False,
                    previous_hypotheses=session.previous_hypotheses,
                    previous_pred_out=session.pred_out_stream,
                    drop_extra_pre_encoded=drop_extra,
                    return_transcription=True,
                )
                if self.profile_chunk:
                    torch.cuda.synchronize()
                    self._prof_step_ms += (time.perf_counter() - _prof_t1) * 1000.0
                    self._prof_n += 1
                    if self._prof_n % 25 == 0:
                        n = self._prof_n
                        logger.info(
                            f"[PROFILE] chunks={n} "
                            f"preprocess={self._prof_pre_ms / n:.2f}ms/chunk "
                            f"step(enc+dec)={self._prof_step_ms / n:.2f}ms/chunk "
                            f"total={(self._prof_pre_ms + self._prof_step_ms) / n:.2f}ms/chunk"
                        )

                # Update emitted frame count
                consumed_audio = session.pending_audio[: plan.shift_frames * plan.hop_samples]
                if len(consumed_audio) >= plan.raw_audio_ring_samples:
                    session.raw_audio_ring = consumed_audio[-plan.raw_audio_ring_samples :].copy()
                else:
                    keep = plan.raw_audio_ring_samples - len(consumed_audio)
                    session.raw_audio_ring = np.concatenate(
                        [session.raw_audio_ring[-keep:], consumed_audio]
                    ).astype(np.float32, copy=False)
                session.pending_audio = session.pending_audio[plan.shift_frames * plan.hop_samples :]
                session.accumulated_audio = session.pending_audio
                self._update_mel_frame_ring(session, valid_new_mel, plan)
                session.emitted_frames += plan.shift_frames
                self._write_eou_probe_chunk(session, eou_probe_snapshot)
                self._write_eou_snapshot_chunk(session, eou_probe_snapshot)

                # Extract text
                if transcribed_texts and transcribed_texts[0]:
                    result = self._extract_hypothesis_result(transcribed_texts[0])
                    if result.language:
                        session.last_language = result.language
                    return result.text

                return session.current_text

        except Exception as e:
            logger.error(f"Session {session.id} chunk processing error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        finally:
            self._gil_attrib_end_sample(gil_sample)

    async def _reset_session(self, session: ASRSession, finalize: bool = True):
        """Handle reset with soft or hard finalization.

        Args:
            finalize: If True (hard reset), add padding and use keep_all_outputs=True
                      to capture trailing words, then reset decoder state.
                      If False (soft reset), just return current cumulative text
                      without forcing decoder output.

        Soft reset (finalize=False):
        - Returns current_text as is_final (model's streaming output)
        - No audio processing, no decoder finalization
        - Decoder state preserved (no corruption)
        - Used on VADUserStoppedSpeakingFrame for fast response

        Hard reset (finalize=True):
        - Adds padding and processes with keep_all_outputs=True
        - Captures trailing words at segment boundaries
        - Resets decoder state to prevent corruption from multiple hard resets
        - Preserves encoder cache for acoustic context
        - Used on UserStoppedSpeakingFrame for complete transcription
        """
        import time

        # Log audio state at reset for diagnostics
        audio_samples = session.total_audio_samples
        audio_duration_ms = (audio_samples * 1000) // self.sample_rate
        logger.debug(
            f"Session {session.id} {'hard' if finalize else 'soft'} reset: "
            f"audio={audio_samples} samples ({audio_duration_ms}ms), "
            f"pending={len(session.pending_audio)} samples, "
            f"emitted={session.emitted_frames} frames"
        )

        if not finalize:
            # SOFT RESET: Return current text without processing
            # This is fast (~0ms) and doesn't corrupt decoder state.
            # The model's current_text is already cumulative (contains all text
            # from session start), so we just return it directly.
            # We don't concatenate with cumulative_text to avoid duplication.
            text = session.current_text

            await session.websocket.send_str(json.dumps(
                self._transcript_payload(
                    session, text=text, is_final=True, finalize=False
                )
            ))

            logger.debug(f"Session {session.id} soft reset: '{text[-50:] if len(text) > 50 else text}'")
            # Keep all state intact - decoder, encoder, audio buffer
            return

        # HARD RESET: Full finalization with padding
        # Save original audio length before adding padding
        original_audio_length = session.total_audio_samples
        plan = self._plan_for_session(session)

        # Pad with silence to ensure the model has enough trailing context
        # to finalize the last word. Padding = (right_context + 1) * shift_frames.
        if original_audio_length > 0:
            padding_samples = plan.final_padding_frames * plan.hop_samples
            silence_padding = np.zeros(padding_samples, dtype=np.float32)
            session.pending_audio = np.concatenate([session.pending_audio, silence_padding])
            session.accumulated_audio = session.pending_audio

        # Process all remaining audio with keep_all_outputs=True
        final_text = session.current_text
        if session.pending_audio is not None and len(session.pending_audio) > 0:
            start_time = time.perf_counter()
            async with self.inference_lock:
                text = await self._run_inference_call(self._process_final_chunk, session)
                if text is not None:
                    final_text = text
                    session.current_text = text  # Update current_text for next soft reset
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Session {session.id} final chunk processed in {elapsed_ms:.1f}ms: '{final_text[-50:] if len(final_text) > 50 else final_text}'")

        # Server-side deduplication: only send the delta (new portion)
        # This avoids downstream duplication when aggregators concatenate transcripts
        if final_text.startswith(session.last_emitted_text):
            delta_text = final_text[len(session.last_emitted_text):].lstrip()
        else:
            # ASR corrected earlier text - send full text
            # (This is rare but can happen with model corrections)
            delta_text = final_text
            logger.debug(
                f"Session {session.id}: ASR correction detected, "
                f"last='{session.last_emitted_text[-30:]}', new='{final_text[-30:]}'"
            )

        # Update tracking state before sending
        session.last_emitted_text = final_text

        # Send only the delta to client
        await session.websocket.send_str(json.dumps(
            self._transcript_payload(
                session, text=delta_text, is_final=True, finalize=True
            )
        ))

        logger.debug(
            f"Session {session.id} hard reset: delta='{delta_text}' "
            f"(cumulative='{final_text[-50:] if len(final_text) > 50 else final_text}')"
        )

        # MEMORY BOUNDING: Clear all state after hard reset
        # This prevents unbounded memory growth by resetting completely each turn:
        # - Audio buffer: cleared (no carryover between turns)
        # - Decoder state: reset fresh (no hypothesis accumulation)
        # - Encoder cache: re-initialized
        #
        # We considered keeping audio overlap for encoder context continuity,
        # but since we reset the encoder cache, overlap audio would just be
        # re-transcribed, causing duplicates. Clean reset avoids this.

        session.last_emitted_text = ""
        session.overlap_buffer = None
        if self.session_warmup_ms > 0:
            async with self.inference_lock:
                await self._run_inference_call(self._init_session, session)
        else:
            self._init_session(session)
        if session.vad_gated_audio:
            session.accepting_vad_audio = False

        logger.debug(
            f"Session {session.id} hard reset complete, state fully reset for next turn"
        )
        self._log_retained_cache_telemetry("hard_reset")

    def _process_final_chunk(
        self,
        session: ASRSession,
        finalize_profile: Optional[dict[str, Any]] = None,
        gil_lock_wait_ms: float = 0.0,
    ) -> Optional[str]:
        """Process remaining pending audio with fixed-plan preprocessing."""
        gil_sample = self._gil_attrib_begin_sample(
            "finalize",
            path="serial_finalize",
            batch_size=1,
            inference_lock_wait_ms=gil_lock_wait_ms,
        )
        try:
            if len(session.pending_audio) == 0:
                if finalize_profile is not None:
                    finalize_profile["model_skipped_reason"] = "no_pending_audio"
                return session.current_text
            plan = self._plan_for_session(session)

            with torch.inference_mode():
                # For final chunk, use ALL remaining frames (including edge)
                padded_total_samples = (
                    session.emitted_frames * plan.hop_samples + len(session.pending_audio)
                )
                total_mel_frames = (padded_total_samples // plan.hop_samples) + 1
                remaining_frames = total_mel_frames - session.emitted_frames

                logger.debug(
                    f"Session {session.id} final chunk: "
                    f"total_mel={total_mel_frames}, emitted={session.emitted_frames}, "
                    f"remaining={remaining_frames}"
                )

                if remaining_frames <= 0:
                    logger.warning(f"Session {session.id}: No remaining frames to process!")
                    if finalize_profile is not None:
                        finalize_profile["model_skipped_reason"] = "no_remaining_frames"
                    return session.current_text

                pending = session.pending_audio
                raw_ring = session.raw_audio_ring
                new_mels: list[torch.Tensor] = []
                frames_collected = 0
                while frames_collected < remaining_frames:
                    frames_this_call = min(plan.shift_frames, remaining_frames - frames_collected)
                    needed_new_samples = min(
                        len(pending),
                        plan.preprocess_new_audio_samples,
                    )
                    new_audio = pending[:needed_new_samples]
                    fixed_audio, valid_samples = self._build_fixed_preprocess_audio(
                        raw_ring,
                        new_audio,
                        plan,
                    )
                    if finalize_profile is not None:
                        self._finalize_profile_cuda_synchronize(finalize_profile)
                        pre_start = time.perf_counter()
                        mel, _mel_len = self._preprocess_fixed_audio(
                            fixed_audio,
                            valid_samples,
                            plan,
                        )
                        self._finalize_profile_cuda_synchronize(finalize_profile)
                        self._finalize_profile_add_preproc(
                            finalize_profile,
                            (time.perf_counter() - pre_start) * 1000,
                        )
                    else:
                        mel, _mel_len = self._preprocess_fixed_audio(
                            fixed_audio,
                            valid_samples,
                            plan,
                        )
                    start = plan.first_preprocess_mel_frame
                    new_mels.append(mel[:, :, start : start + frames_this_call])

                    if frames_this_call == plan.shift_frames:
                        consumed_samples = min(plan.shift_frames * plan.hop_samples, len(pending))
                        consumed_audio = pending[:consumed_samples]
                        if len(consumed_audio) >= plan.raw_audio_ring_samples:
                            raw_ring = consumed_audio[-plan.raw_audio_ring_samples :].copy()
                        elif len(consumed_audio) > 0:
                            keep = plan.raw_audio_ring_samples - len(consumed_audio)
                            raw_ring = np.concatenate([raw_ring[-keep:], consumed_audio]).astype(
                                np.float32,
                                copy=False,
                            )
                        pending = pending[consumed_samples:]
                    frames_collected += frames_this_call

                new_mel = torch.cat(new_mels, dim=-1)

                # Extract final chunk with pre-encode cache
                if session.emitted_frames == 0:
                    chunk_mel = new_mel
                    drop_extra = 0
                else:
                    chunk_mel = torch.cat((session.mel_frame_ring, new_mel), dim=-1)
                    drop_extra = plan.drop_extra

                chunk_len = torch.tensor([chunk_mel.shape[-1]], device='cuda')

                if self._current_model_is_prompted():
                    self._apply_inference_prompt(session)
                if finalize_profile is not None:
                    input_cache_last_channel = session.cache_last_channel
                    input_cache_last_time = session.cache_last_time
                    input_cache_last_channel_len = session.cache_last_channel_len
                    self._finalize_profile_set_model_inputs(
                        finalize_profile,
                        processed_signal=chunk_mel,
                        processed_signal_length=chunk_len,
                        cache_last_channel=input_cache_last_channel,
                        cache_last_channel_len=input_cache_last_channel_len,
                        drop_extra=drop_extra,
                        first=(session.emitted_frames == 0),
                    )
                    self._finalize_profile_cuda_synchronize(finalize_profile)
                    model_start = time.perf_counter()
                else:
                    input_cache_last_channel = session.cache_last_channel
                    input_cache_last_time = session.cache_last_time
                    input_cache_last_channel_len = session.cache_last_channel_len
                (
                    session.pred_out_stream,
                    transcribed_texts,
                    session.cache_last_channel,
                    session.cache_last_time,
                    session.cache_last_channel_len,
                    session.previous_hypotheses,
                ) = self._conformer_stream_step(
                    processed_signal=chunk_mel,
                    processed_signal_length=chunk_len,
                    cache_last_channel=input_cache_last_channel,
                    cache_last_time=input_cache_last_time,
                    cache_last_channel_len=input_cache_last_channel_len,
                    keep_all_outputs=True,  # Final chunk - output all remaining
                    previous_hypotheses=session.previous_hypotheses,
                    previous_pred_out=session.pred_out_stream,
                    drop_extra_pre_encoded=drop_extra,
                    return_transcription=True,
                    finalize_profile=finalize_profile,
                )
                if finalize_profile is not None:
                    self._finalize_profile_cuda_synchronize(finalize_profile)
                    self._finalize_profile_set_model_wall(
                        finalize_profile,
                        (time.perf_counter() - model_start) * 1000,
                    )
                    finalize_profile["cache_last_channel_out_shape"] = (
                        self._finalize_profile_shape(session.cache_last_channel)
                    )
                    finalize_profile["cache_last_channel_len_out"] = (
                        self._finalize_profile_tensor_value(
                            session.cache_last_channel_len
                        )
                    )

                if transcribed_texts and transcribed_texts[0]:
                    result = self._extract_hypothesis_result(transcribed_texts[0])
                    if result.language:
                        session.last_language = result.language
                    final_text = result.text
                    logger.debug(
                        f"Session {session.id} final chunk output: '{final_text[-50:] if len(final_text) > 50 else final_text}' "
                        f"(was: '{session.current_text[-30:] if len(session.current_text) > 30 else session.current_text}')"
                    )
                    return final_text

                logger.debug(f"Session {session.id} final chunk: no new text from model")
                return session.current_text

        except Exception as e:
            logger.error(f"Session {session.id} final chunk error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        finally:
            self._gil_attrib_end_sample(gil_sample)

    async def health_handler(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        payload = {
            "status": "healthy" if self.model_loaded else "loading",
            "model_loaded": self.model_loaded,
        }
        if self.admission_enabled:
            payload["admission"] = self._admission_status_snapshot()
        return web.json_response(payload)

    async def start(self):
        """Start the HTTP + WebSocket server."""
        self.load_model()
        self.model_loaded = True
        if self.scheduler_enabled:
            self._ensure_scheduler_task()
        if self.gil_attrib_enabled:
            self._gil_attrib_started_unix = time.time()
            self._gil_attrib_started_perf = time.perf_counter()
            self._gil_attrib_loop_task = asyncio.create_task(
                self._gil_attrib_loop_lag_probe(),
                name="nemotron-gil-attrib-loop-lag",
            )

        # GC tuning (NEMOTRON_GC_TUNE=1; byte-exact — changes WHEN garbage is collected, not outputs). The default
        # gen-2 collector does ~280ms stop-the-world pauses (it scans the ~74k long-lived model + CUDA-graph objects
        # plus the per-utterance churn) — a confirmed P99-stall contributor. gc.freeze() moves the now-loaded
        # startup heap (model, graphs) to the permanent generation so gen-2 only scans the runtime churn; the raised
        # thresholds make gen-2 rarer. Called after load_model() + graph capture so the long-lived objects exist.
        if os.environ.get("NEMOTRON_GC_TUNE", "") == "1":
            import gc as _gc

            _gc.collect()
            _gc.freeze()
            _gc.set_threshold(700, 100, 100)
            logger.info(f"gc_tuned frozen_objects={_gc.get_freeze_count()} thresholds={_gc.get_threshold()}")

        logger.info(f"Starting streaming ASR server on ws://{self.host}:{self.port}")

        app = web.Application()
        app.router.add_get("/health", self.health_handler)
        app.router.add_get("/stats", self.stats_handler)
        app.router.add_get("/", self.websocket_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        logger.info(f"ASR server listening on ws://{self.host}:{self.port}")
        logger.info(f"Health check available at http://{self.host}:{self.port}/health")
        try:
            await asyncio.Future()  # Run forever
        finally:
            if self._gil_attrib_loop_task is not None:
                self._gil_attrib_loop_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._gil_attrib_loop_task
            self._gil_attrib_emit_record(reason="shutdown")
            self._log_finalize_profile_histogram(reason="shutdown")


def main():
    parser = argparse.ArgumentParser(description="Nemotron Streaming ASR WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace model name or path to local .nemo file"
    )
    parser.add_argument(
        "--multilingual-model",
        default=None,
        help=(
            "Optional HuggingFace model name or local .nemo path for multilingual "
            "routing. When set, language=en uses --model and all other languages "
            "use this checkpoint."
        ),
    )
    parser.add_argument(
        "--right-context",
        type=int,
        default=None,
        choices=[0, 1, 3, 6, 13],
        help=(
            "Right context frames. Omit for model default "
            "(English=1, prompted multilingual=3)."
        )
    )
    args = parser.parse_args()

    server = ASRServer(
        model=args.model,
        multilingual_model=args.multilingual_model,
        host=args.host,
        port=args.port,
        right_context=args.right_context,
    )

    asyncio.run(server.start())


if __name__ == "__main__":
    main()
