#!/usr/bin/env python3
"""Side-by-side Python/C++ WebSocket server compatibility oracle.

The harness launches the shipping Python server and the C++ ws_server on
separate ports, drives the same utt0..utt7 PCM fixture through both, compares
canonical transcript JSON, checks protocol error cases, and gates the C++
WebSocket ttfs p95 against density_main's scheduler p95.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request

import numpy as np
import torch

try:  # Preferred by the Step 11 spec when present in the environment.
    import websockets  # type: ignore[import-not-found]
    from websockets.exceptions import InvalidStatus, InvalidStatusCode  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - exercised only when dependency is absent.
    websockets = None
    InvalidStatus = InvalidStatusCode = ()  # type: ignore[assignment]

try:  # Runtime uv-native venv currently has aiohttp but not websockets.
    import aiohttp
except Exception:  # pragma: no cover - import failure is diagnosed at runtime.
    aiohttp = None


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "runtime"
ARTIFACT_DIR = RUNTIME_DIR / "artifacts"
STEADY_BATCH_DIR = RUNTIME_DIR / "steady_b_artifacts"
LOG_ROOT = ARTIFACT_DIR / "logs"
PYTHON_EXE = RUNTIME_DIR / ".venv" / "bin" / "python"
PYTHON_SERVER = REPO_ROOT / "src" / "nemotron_speech" / "server.py"
CPP_SERVER = RUNTIME_DIR / "cpp" / "build" / "ws_server"
DENSITY_MAIN = RUNTIME_DIR / "cpp" / "build" / "density_main"
BUNDLE = ARTIFACT_DIR / "session_audio_bundle.ts"

DEFAULT_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
READY_FRAME = {"type": "ready"}
CHUNK_BYTES = 640
FINALIZE_TIMING_KEYS = (
    "reason",
    "vad_stop",
    "vad_stop_recv",
    "debounce_expiry",
    "fork_flush_start",
    "fork_flush_done",
    "final_sent",
    "inference_lock_acquire_wait_ms",
    "enc_first_lock_wait_ms",
    "lane_queue_wait_ms",
    "preproc_ms",
    "scheduler_enqueue_wait_ms",
    "scheduler_future_wait_ms",
    "scheduler_completion_wait_ms",
    "decode_ms",
    "gil_attrib_enabled",
)
OPTIONAL_FINALIZE_TIMING_KEYS = {
    "enc_first_lock_wait_ms",
    "lane_queue_wait_ms",
    "preproc_ms",
    "scheduler_enqueue_wait_ms",
    "scheduler_future_wait_ms",
    "scheduler_completion_wait_ms",  # native-only steady-scheduler telemetry
    "decode_ms",
}
TIMING_NUMERIC_REQUIRED = {
    "vad_stop",
    "debounce_expiry",
    "fork_flush_start",
    "fork_flush_done",
    "final_sent",
    "inference_lock_acquire_wait_ms",
}
TIMING_NUMERIC_NULLABLE = {
    "vad_stop_recv",
    "enc_first_lock_wait_ms",
    "lane_queue_wait_ms",
    "preproc_ms",
    "scheduler_enqueue_wait_ms",
    "scheduler_future_wait_ms",
    "scheduler_completion_wait_ms",
    "decode_ms",
}
VOLATILE_TOP_LEVEL_KEYS = {
    "finalize_seq",
    "pid",
    "process_label",
}
VOLATILE_STATS_KEYS = {
    "since_unix",
    "until_unix",
}
NATIVE_EXTENSION_KEYS = {
    "scheduler_telemetry",
    "native_scheduler",
    "stale_gen",
}


class CompatFailure(RuntimeError):
    """Raised for a compatibility assertion failure."""


@dataclasses.dataclass(frozen=True)
class Fixture:
    utt: int
    sample_index: int
    sample_id: str
    pcm: bytes
    audio_samples: int
    # Prompted (multilingual) model: per-connection ?language= query value
    # ("" = no query param, server default).
    language: str = ""


@dataclasses.dataclass
class TranscriptEvent:
    raw: dict[str, Any]
    canonical: dict[str, Any]
    canonical_json: str
    collector_text: str

    @property
    def signature(self) -> dict[str, Any]:
        sig = {
            "type": self.raw.get("type"),
            "text": self.raw.get("text"),
            "is_final": self.raw.get("is_final"),
        }
        if "finalize" in self.raw:
            sig["finalize"] = self.raw.get("finalize")
        if "language" in self.raw:
            sig["language"] = self.raw.get("language")
        return sig


@dataclasses.dataclass
class CaptureResult:
    server: str
    utt: int
    ready: dict[str, Any] | None
    events: list[TranscriptEvent]
    raw_messages: list[dict[str, Any]]
    final_collector_text: str
    ttfs_ms: float | None
    elapsed_s: float


@dataclasses.dataclass
class InvalidQueryResult:
    server: str
    kind: str
    status: int | None
    payload: dict[str, Any] | None
    message: str


@dataclasses.dataclass
class HttpResult:
    status: int
    body: str
    payload: Any


@dataclasses.dataclass
class ManagedProcess:
    name: str
    cmd: list[str]
    cwd: Path
    env: dict[str, str]
    log_path: Path
    proc: subprocess.Popen[bytes] | None = None
    _log_file: Any = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("wb")
        self.proc = subprocess.Popen(
            self.cmd,
            cwd=str(self.cwd),
            env=self.env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def poll(self) -> int | None:
        return None if self.proc is None else self.proc.poll()

    def terminate(self, timeout_s: float) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.proc.pid, signal.SIGTERM)
                try:
                    self.proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait(timeout=max(1.0, timeout_s))
        finally:
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None


class WsConnection:
    async def recv_text(self, timeout_s: float) -> str:
        raise NotImplementedError

    async def send_json(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    async def send_bytes(self, payload: bytes) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


class WebsocketsConnection(WsConnection):
    def __init__(self, ws: Any):
        self.ws = ws

    async def recv_text(self, timeout_s: float) -> str:
        message = await asyncio.wait_for(self.ws.recv(), timeout=timeout_s)
        if isinstance(message, bytes):
            raise CompatFailure(f"expected text WS frame, got {len(message)} binary bytes")
        return str(message)

    async def send_json(self, payload: dict[str, Any]) -> None:
        await self.ws.send(json.dumps(payload))

    async def send_bytes(self, payload: bytes) -> None:
        await self.ws.send(payload)

    async def close(self) -> None:
        await self.ws.close()


class AiohttpConnection(WsConnection):
    def __init__(self, session: Any, ws: Any):
        self.session = session
        self.ws = ws

    async def recv_text(self, timeout_s: float) -> str:
        msg = await self.ws.receive(timeout=timeout_s)
        if msg.type == aiohttp.WSMsgType.TEXT:
            return str(msg.data)
        if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
            raise CompatFailure("websocket closed before expected text frame")
        if msg.type == aiohttp.WSMsgType.ERROR:
            raise CompatFailure(f"websocket error: {self.ws.exception()}")
        if msg.type == aiohttp.WSMsgType.BINARY:
            raise CompatFailure(f"expected text WS frame, got {len(msg.data)} binary bytes")
        raise CompatFailure(f"unexpected WS message type: {msg.type}")

    async def send_json(self, payload: dict[str, Any]) -> None:
        await self.ws.send_str(json.dumps(payload))

    async def send_bytes(self, payload: bytes) -> None:
        await self.ws.send_bytes(payload)

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self.ws.close()
        with contextlib.suppress(Exception):
            await self.session.close()


async def ws_connect(url: str, timeout_s: float) -> WsConnection:
    if websockets is not None:
        ws = await websockets.connect(
            url,
            max_size=10 * 1024 * 1024,
            open_timeout=timeout_s,
            close_timeout=5.0,
            # Cold-start warmup (AOTI bucket loads) can stall the server event
            # loop past the default 20s keepalive; rely on final_timeout_s for
            # liveness instead of protocol pings.
            ping_interval=None,
        )
        return WebsocketsConnection(ws)
    if aiohttp is None:
        raise CompatFailure("neither websockets nor aiohttp is importable in this Python environment")
    session = aiohttp.ClientSession()
    try:
        ws = await session.ws_connect(
            url,
            max_msg_size=10 * 1024 * 1024,
            timeout=timeout_s,
        )
    except Exception:
        await session.close()
        raise
    return AiohttpConnection(session, ws)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _tensor_i64(bundle: torch.jit.ScriptModule, name: str) -> list[int]:
    tensor = getattr(bundle, name).detach().cpu().to(torch.int64).reshape(-1)
    return [int(v) for v in tensor.tolist()]


def _scalar_i64(bundle: torch.jit.ScriptModule, name: str) -> int:
    return int(getattr(bundle, name).detach().cpu().reshape(-1)[0].item())


def _tensor_f32(bundle: torch.jit.ScriptModule, name: str) -> np.ndarray:
    tensor = getattr(bundle, name).detach().cpu().to(torch.float32).contiguous()
    return tensor.numpy().copy()


def _unpack_utf8(bundle: torch.jit.ScriptModule, bytes_name: str, offsets_name: str) -> list[str]:
    data = bytes(int(v) for v in getattr(bundle, bytes_name).detach().cpu().reshape(-1).tolist())
    offsets = _tensor_i64(bundle, offsets_name)
    if not offsets or offsets[0] != 0 or offsets[-1] != len(data):
        raise CompatFailure(f"invalid UTF-8 offsets for {bytes_name}/{offsets_name}")
    return [
        data[offsets[i] : offsets[i + 1]].decode("utf-8")
        for i in range(len(offsets) - 1)
    ]


def _one_utf8(bundle: torch.jit.ScriptModule, prefix: str, name: str) -> str:
    values = _unpack_utf8(bundle, f"{prefix}_{name}_bytes", f"{prefix}_{name}_offsets")
    if len(values) != 1:
        raise CompatFailure(f"{prefix}_{name} expected one value, got {len(values)}")
    return values[0]


def _float_audio_to_pcm16(audio: np.ndarray) -> bytes:
    flat = np.asarray(audio, dtype=np.float32).reshape(-1)
    values = np.rint(np.clip(flat, -1.0, 1.0) * 32768.0)
    values = np.clip(values, -32768, 32767).astype("<i2")
    return values.tobytes()


def load_fixtures(bundle_path: Path, start: int, n: int, language: str = "") -> list[Fixture]:
    if not bundle_path.exists():
        raise FileNotFoundError(f"bundle not found: {bundle_path}")
    bundle = torch.jit.load(str(bundle_path), map_location="cpu")
    rows = _scalar_i64(bundle, "num_utts")
    if start < 0 or start >= rows:
        raise CompatFailure(f"--start {start} outside bundle rows={rows}")
    count = min(n, rows - start)
    fixtures: list[Fixture] = []
    for utt in range(start, start + count):
        prefix = f"utt{utt}"
        audio = _tensor_f32(bundle, f"{prefix}_audio")
        fixtures.append(
            Fixture(
                utt=utt,
                sample_index=_scalar_i64(bundle, f"{prefix}_sample_index"),
                sample_id=_one_utf8(bundle, prefix, "sample_id"),
                pcm=_float_audio_to_pcm16(audio),
                audio_samples=int(audio.reshape(-1).shape[0]),
                language=language,
            )
        )
    return fixtures


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_finalize_timing(timing: Any, *, label: str) -> None:
    if not isinstance(timing, dict):
        raise CompatFailure(f"{label}: finalize_timing missing or not an object")
    keys = set(timing)
    expected = set(FINALIZE_TIMING_KEYS)
    required = expected - OPTIONAL_FINALIZE_TIMING_KEYS
    if not required.issubset(keys) or not keys.issubset(expected):
        raise CompatFailure(
            f"{label}: finalize_timing keys mismatch got={sorted(keys)} expected={sorted(expected)}"
        )
    if not isinstance(timing["reason"], str) or not timing["reason"]:
        raise CompatFailure(f"{label}: finalize_timing.reason must be a non-empty string")
    if not isinstance(timing["gil_attrib_enabled"], bool):
        raise CompatFailure(f"{label}: finalize_timing.gil_attrib_enabled must be boolean")
    for key in TIMING_NUMERIC_REQUIRED:
        if not is_number(timing[key]):
            raise CompatFailure(f"{label}: finalize_timing.{key} must be numeric, got {timing[key]!r}")
    for key in TIMING_NUMERIC_NULLABLE:
        value = timing.get(key)
        if value is not None and not is_number(value):
            raise CompatFailure(
                f"{label}: finalize_timing.{key} must be numeric or null, got {value!r}"
            )


def canonicalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key in VOLATILE_TOP_LEVEL_KEYS or key in VOLATILE_STATS_KEYS:
                continue
            if key in NATIVE_EXTENSION_KEYS:
                continue
            if key == "finalize_timing" and isinstance(value, dict):
                out[key] = {
                    timing_key: canonical_timing_value(timing_key, value.get(timing_key))
                    for timing_key in sorted(FINALIZE_TIMING_KEYS)
                }
                continue
            if key == "admission" and isinstance(value, dict):
                out[key] = canonicalize_admission(value)
                continue
            out[key] = canonicalize(value)
        return out
    if isinstance(obj, list):
        return [canonicalize(item) for item in obj]
    return obj


def canonical_timing_value(key: str, value: Any) -> str:
    if key == "reason":
        return "<string>" if isinstance(value, str) else f"<{type(value).__name__}>"
    if key == "gil_attrib_enabled":
        return "<bool>" if isinstance(value, bool) else f"<{type(value).__name__}>"
    if key in OPTIONAL_FINALIZE_TIMING_KEYS and (value is None or is_number(value)):
        return "<number-or-null>"
    if value is None:
        return "<null>"
    if is_number(value):
        return "<number>"
    return f"<{type(value).__name__}>"


def canonicalize_admission(value: dict[str, Any]) -> dict[str, Any]:
    # Admission counters differ between Python and native implementations and
    # are not part of the transcript oracle.
    keep: dict[str, Any] = {}
    if "enabled" in value:
        keep["enabled"] = value["enabled"]
    return keep


def canonical_json(obj: Any) -> str:
    return json.dumps(canonicalize(obj), sort_keys=True, ensure_ascii=False)


def update_collector_text(event: dict[str, Any], collector_text: str) -> str:
    if event.get("type") != "transcript":
        return collector_text
    if event.get("is_final") is not True:
        return collector_text
    if event.get("finalize") is not True:
        return collector_text
    if "collector_text" in event and isinstance(event["collector_text"], str):
        return event["collector_text"]
    text = str(event.get("text") or "")
    if not text:
        return collector_text
    return text if not collector_text else f"{collector_text} {text}"


def assert_ready(server: str, payload: dict[str, Any], utt: int) -> None:
    if payload != READY_FRAME:
        raise CompatFailure(f"{server} utt{utt}: expected ready frame {READY_FRAME}, got {payload!r}")


async def capture_utterance(
    *,
    server: str,
    url: str,
    fixture: Fixture,
    final_timeout_s: float,
    close_timeout_s: float,
    chunk_bytes: int,
    realtime: bool,
    chunk_ms: int,
    measure_ttfs: bool,
    start_gate: asyncio.Event | None = None,
) -> CaptureResult:
    started = time.perf_counter()
    if fixture.language:
        url = f"{url}{'&' if '?' in url else '?'}language={fixture.language}"
    conn = await ws_connect(url, timeout_s=final_timeout_s)
    events: list[TranscriptEvent] = []
    raw_messages: list[dict[str, Any]] = []
    collector_text = ""
    final_seen = asyncio.Event()
    errors: list[str] = []
    ttfs_start: float | None = None
    ttfs_ms: float | None = None
    ready_payload: dict[str, Any] | None = None

    try:
        ready_text = await conn.recv_text(final_timeout_s)
        ready_payload = json.loads(ready_text)
        raw_messages.append(ready_payload)
        assert_ready(server, ready_payload, fixture.utt)

        if start_gate is not None:
            await start_gate.wait()

        async def recv_loop() -> None:
            nonlocal collector_text, ttfs_ms
            while not final_seen.is_set():
                try:
                    text = await conn.recv_text(final_timeout_s)
                    payload = json.loads(text)
                except asyncio.TimeoutError:
                    errors.append(f"timeout waiting for final transcript after {final_timeout_s}s")
                    final_seen.set()
                    return
                except Exception as exc:  # noqa: BLE001 - surfaced as assertion diagnostic.
                    errors.append(f"{type(exc).__name__}: {exc}")
                    final_seen.set()
                    return
                raw_messages.append(payload)
                if payload.get("type") == "transcript":
                    collector_text = update_collector_text(payload, collector_text)
                    event = TranscriptEvent(
                        raw=payload,
                        canonical=canonicalize(payload),
                        canonical_json=canonical_json(payload),
                        collector_text=collector_text,
                    )
                    events.append(event)
                    if payload.get("is_final") is True:
                        if measure_ttfs and ttfs_start is not None and ttfs_ms is None:
                            ttfs_ms = (time.perf_counter() - ttfs_start) * 1000.0
                        final_seen.set()
                elif payload.get("type") == "error":
                    errors.append(f"server error frame: {payload.get('message')}")
                    final_seen.set()

        recv_task = asyncio.create_task(recv_loop())
        try:
            await conn.send_json({"type": "vad_start"})
            sleep_s = chunk_ms / 1000.0
            for offset in range(0, len(fixture.pcm), chunk_bytes):
                await conn.send_bytes(fixture.pcm[offset : offset + chunk_bytes])
                if realtime:
                    await asyncio.sleep(sleep_s)
            ttfs_start = time.perf_counter()
            await conn.send_json({"type": "vad_stop"})
            await asyncio.wait_for(final_seen.wait(), timeout=final_timeout_s + 1.0)
        finally:
            await conn.close()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(recv_task, timeout=close_timeout_s)
            if not recv_task.done():
                recv_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await recv_task
    finally:
        with contextlib.suppress(Exception):
            await conn.close()

    if errors:
        raise CompatFailure(f"{server} utt{fixture.utt}: {'; '.join(errors)}")
    if not any(event.raw.get("is_final") is True for event in events):
        raise CompatFailure(f"{server} utt{fixture.utt}: no final transcript received")
    return CaptureResult(
        server=server,
        utt=fixture.utt,
        ready=ready_payload,
        events=events,
        raw_messages=raw_messages,
        final_collector_text=collector_text,
        ttfs_ms=ttfs_ms,
        elapsed_s=time.perf_counter() - started,
    )


def compare_capture(py: CaptureResult, cpp: CaptureResult) -> list[str]:
    failures: list[str] = []
    utt = py.utt
    if len(py.events) != len(cpp.events):
        failures.append(
            f"utt{utt}: event count mismatch python={len(py.events)} cpp={len(cpp.events)}"
        )
    for index, (py_event, cpp_event) in enumerate(zip(py.events, cpp.events)):
        py_sig = dict(py_event.signature)
        cpp_sig = dict(cpp_event.signature)
        # In target_lang=auto the servers may disagree on whether the trailing
        # <xx-XX> tag token landed inside the finalize decode window: one side
        # reports the detected locale, the other falls back to "auto". Treat
        # detected-vs-auto-fallback as compatible; genuine locale mismatches
        # (e.g. en-US vs es-ES) still fail.
        py_lang = py_sig.get("language")
        cpp_lang = cpp_sig.get("language")
        if (
            isinstance(py_lang, str)
            and isinstance(cpp_lang, str)
            and py_lang != cpp_lang
            and "auto" in (py_lang, cpp_lang)
        ):
            py_sig.pop("language", None)
            cpp_sig.pop("language", None)
        if py_sig != cpp_sig:
            failures.append(
                "utt{utt}: event {index} signature mismatch\n"
                "  python={py_sig}\n"
                "  cpp={cpp_sig}\n"
                "  python_canonical={py_json}\n"
                "  cpp_canonical={cpp_json}".format(
                    utt=utt,
                    index=index,
                    py_sig=py_event.signature,
                    cpp_sig=cpp_event.signature,
                    py_json=py_event.canonical_json,
                    cpp_json=cpp_event.canonical_json,
                )
            )
        if py_event.raw.get("is_final") is True:
            with contextlib.suppress(CompatFailure):
                validate_finalize_timing(py_event.raw.get("finalize_timing"), label=f"python utt{utt} event{index}")
        if cpp_event.raw.get("is_final") is True:
            with contextlib.suppress(CompatFailure):
                validate_finalize_timing(cpp_event.raw.get("finalize_timing"), label=f"cpp utt{utt} event{index}")

    for server_name, capture in (("python", py), ("cpp", cpp)):
        for index, event in enumerate(capture.events):
            if event.raw.get("is_final") is True:
                try:
                    validate_finalize_timing(
                        event.raw.get("finalize_timing"),
                        label=f"{server_name} utt{utt} event{index}",
                    )
                except CompatFailure as exc:
                    failures.append(str(exc))

    if py.final_collector_text != cpp.final_collector_text:
        failures.append(
            f"utt{utt}: final collector_text mismatch python={py.final_collector_text!r} "
            f"cpp={cpp.final_collector_text!r}"
        )
    return failures


def http_get(url: str, timeout_s: float) -> HttpResult:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = int(resp.status)
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        body = exc.read().decode("utf-8", errors="replace")
    payload: Any
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        payload = {"raw": body}
    return HttpResult(status=status, body=body, payload=payload)


async def wait_health(
    *,
    name: str,
    url: str,
    timeout_s: float,
    process: ManagedProcess,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error = "not probed"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise CompatFailure(
                f"{name} exited before /health became healthy; rc={process.poll()} log={process.log_path}"
            )
        try:
            result = await asyncio.to_thread(http_get, url, 2.0)
            if result.status == 200 and isinstance(result.payload, dict) and result.payload.get("model_loaded") is True:
                return result.payload
            last_error = f"status={result.status} body={result.body[:200]}"
        except Exception as exc:  # noqa: BLE001 - readiness probe.
            last_error = f"{type(exc).__name__}: {exc}"
        await asyncio.sleep(1.0)
    raise CompatFailure(
        f"{name} did not become healthy within {timeout_s:.1f}s; last={last_error}; log={process.log_path}"
    )


def assert_port_free(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise CompatFailure(f"port {host}:{port} is already in use: {exc}") from exc


def base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{existing_path}" if existing_path else str(REPO_ROOT / "src")
    env["HF_HUB_OFFLINE"] = "1"
    env["NEMOTRON_CONTINUOUS"] = "1"
    env["NEMOTRON_FINALIZE_SILENCE_MS"] = str(args.finalize_silence_ms)
    env["NEMOTRON_ARTIFACT_DIR"] = str(args.artifact_dir)
    env["NEMOTRON_DENSITY_BATCH_STEADY"] = "1"
    env["NEMOTRON_WS_SCHEDULER"] = "1"
    env["NEMOTRON_DENSITY_BATCH_MAX"] = str(args.batch_b_max)
    env["NEMOTRON_DENSITY_BATCH_WINDOW_MS"] = str(args.batch_window_ms)
    env["NEMOTRON_DENSITY_BATCH_LONE_TIMEOUT_MS"] = str(args.batch_lone_timeout_ms)
    env["NEMOTRON_DENSITY_ADMISSION_ACTIVE_CAP"] = str(args.admission_active_cap)
    ws_lanes = args.ws_lanes if args.ws_lanes is not None else args.admission_active_cap
    ws_finalize_runners = (
        args.ws_finalize_runners
        if args.ws_finalize_runners is not None
        else max(1, min(args.admission_active_cap, 2))
    )
    env["NEMOTRON_WS_LANES"] = str(ws_lanes)
    env["NEMOTRON_WS_FINALIZE_RUNNERS"] = str(ws_finalize_runners)
    env["NEMOTRON_DENSITY_FINALIZE_RUNNERS"] = str(ws_finalize_runners)
    prepend_env_path(env, "LD_LIBRARY_PATH", torch_lib_dir())
    return env


def torch_lib_dir() -> Path:
    candidates = sorted((RUNTIME_DIR / ".venv" / "lib").glob("python*/site-packages/torch/lib"))
    if not candidates:
        return RUNTIME_DIR / ".venv" / "lib" / "python3.12" / "site-packages" / "torch" / "lib"
    return candidates[0]


def prepend_env_path(env: dict[str, str], name: str, path: Path) -> None:
    if not path.exists():
        return
    current = env.get(name, "")
    env[name] = f"{path}:{current}" if current else str(path)


def python_env(args: argparse.Namespace) -> dict[str, str]:
    env = base_env(args)
    # Keep the shipped Python server's native experimental scheduler envs from
    # leaking into the compatibility baseline.
    for name in (
        "NEMOTRON_SCHEDULER_B1",
        "NEMOTRON_BATCH_SCHED",
        "NEMOTRON_BATCH_FINALIZE",
        "NEMOTRON_BATCH_FINALIZE_PREPROC",
        "NEMOTRON_FINALIZE_PRIORITY",
        "NEMOTRON_WS_STEADY_SHADOW",
    ):
        env.pop(name, None)
    return env


def cpp_env(args: argparse.Namespace) -> dict[str, str]:
    env = base_env(args)
    if args.ws_steady_shadow:
        env["NEMOTRON_WS_STEADY_SHADOW"] = "1"
    return env


def density_env(args: argparse.Namespace) -> dict[str, str]:
    env = base_env(args)
    env["NEMOTRON_DENSITY_BATCH_STEADY"] = "1"
    env.pop("NEMOTRON_WS_STEADY_SHADOW", None)
    return env


def python_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(args.python),
        str(PYTHON_SERVER),
        "--host",
        args.host,
        "--port",
        str(args.python_port),
        "--model",
        args.model,
    ]
    if args.right_context is not None:
        cmd.extend(["--right-context", str(args.right_context)])
    return cmd


def cpp_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.cpp_server),
        "--port",
        str(args.cpp_port),
        "--admission-active-cap",
        str(args.admission_active_cap),
        "--steady-batch-dir",
        str(args.steady_batch_dir),
    ]


def status_from_ws_exception(exc: BaseException) -> int | None:
    status = getattr(exc, "status", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    if status is None and getattr(exc, "response", None) is not None:
        status = getattr(exc.response, "status_code", None)
    return int(status) if status is not None else None


async def invalid_model_query(server: str, url: str, timeout_s: float) -> InvalidQueryResult:
    conn: WsConnection | None = None
    try:
        conn = await ws_connect(f"{url}/?model=bogus", timeout_s=timeout_s)
        text = await conn.recv_text(timeout_s)
        payload = json.loads(text)
        return InvalidQueryResult(
            server=server,
            kind="ws_error_frame",
            status=None,
            payload=payload,
            message=str(payload.get("message", "")),
        )
    except Exception as exc:  # noqa: BLE001 - converted to protocol result.
        status = status_from_ws_exception(exc)
        if status is not None:
            return InvalidQueryResult(
                server=server,
                kind="http_reject",
                status=status,
                payload=None,
                message=str(exc),
            )
        raise
    finally:
        if conn is not None:
            await conn.close()


def validate_invalid_model(result: InvalidQueryResult) -> None:
    if result.kind == "ws_error_frame":
        if not isinstance(result.payload, dict):
            raise CompatFailure(f"{result.server}: invalid model did not return JSON error frame")
        if result.payload.get("type") != "error":
            raise CompatFailure(f"{result.server}: invalid model expected error frame, got {result.payload!r}")
        message = str(result.payload.get("message", ""))
        if "model" not in message.lower():
            raise CompatFailure(f"{result.server}: invalid model error message missing model context: {message!r}")
        return
    if result.kind == "http_reject" and result.status is not None and result.status >= 400:
        return
    raise CompatFailure(f"{result.server}: invalid model query had unexpected result {result!r}")


def validate_invalid_stats(server: str, result: HttpResult) -> None:
    if result.status != 400:
        raise CompatFailure(f"{server}: /stats?last=bogus expected HTTP 400, got {result.status} body={result.body!r}")
    if not isinstance(result.payload, dict) or "error" not in result.payload:
        raise CompatFailure(f"{server}: /stats?last=bogus expected JSON error body, got {result.body!r}")
    if "invalid 'last'" not in str(result.payload["error"]):
        raise CompatFailure(f"{server}: /stats?last=bogus unexpected error body {result.payload!r}")


def percentile_p95(values: list[float]) -> float:
    if not values:
        raise CompatFailure("cannot compute p95 for an empty sample set")
    ordered = sorted(values)
    idx = int(math.ceil(0.95 * len(ordered)) - 1)
    idx = max(0, min(len(ordered) - 1, idx))
    return float(ordered[idx])


async def measure_ws_ttfs(
    *,
    url: str,
    fixtures: list[Fixture],
    args: argparse.Namespace,
) -> list[float]:
    selected = fixtures[: args.perf_n]
    if len(selected) != args.perf_n:
        raise CompatFailure(f"need {args.perf_n} fixtures for WS perf, got {len(selected)}")
    gate = asyncio.Event()
    tasks = [
        asyncio.create_task(
            capture_utterance(
                server="cpp-perf",
                url=url,
                fixture=fixture,
                final_timeout_s=args.final_timeout_s,
                close_timeout_s=args.close_timeout_s,
                chunk_bytes=args.chunk_bytes,
                # Pace the WS perf measurement (feed audio at realtime cadence) to match
                # density_main's paced N-stream baseline. A non-realtime blast piles the
                # whole utterance's streaming-encode work onto the serialized inference
                # path so finalize ttfs measures backlog drain, not WS-layer overhead —
                # not comparable to the cadenced density ttfs the gate comparisons against.
                realtime=True,
                chunk_ms=args.chunk_ms,
                measure_ttfs=True,
                start_gate=gate,
            )
        )
        for fixture in selected
    ]
    await asyncio.sleep(0)
    gate.set()
    captures = await asyncio.gather(*tasks)
    ttfs = [capture.ttfs_ms for capture in captures]
    if any(value is None for value in ttfs):
        raise CompatFailure("WS perf capture did not produce TTFS for every session")
    return [float(value) for value in ttfs if value is not None]


def run_density_scheduler(args: argparse.Namespace, log_path: Path) -> tuple[float, dict[str, Any]]:
    cmd = [
        str(args.density_main),
        str(args.artifact_dir),
        "--mode",
        "density-sweep",
        "--n-values",
        str(args.perf_n),
        "--target-n",
        str(args.perf_n),
        "--density-rows",
        str(args.density_rows),
        "--batch-steady",
        "on",
        "--steady-batch-dir",
        str(args.steady_batch_dir),
        "--admission-active-cap",
        str(args.admission_active_cap),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(RUNTIME_DIR),
            env=density_env(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.density_timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(cmd) + "\n")
            if exc.stdout:
                log.write(exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode("utf-8", errors="replace"))
        raise CompatFailure(f"density_main timed out after {args.density_timeout_s}s; log={log_path}") from exc

    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.write(completed.stdout or "")
    rows: list[dict[str, Any]] = []
    for line in (completed.stdout or "").splitlines():
        match = re.search(r"DENSITY_TELEMETRY path=.* json=(\{.*\})", line)
        if match:
            with contextlib.suppress(json.JSONDecodeError):
                rows.append(json.loads(match.group(1)))
    candidates = [
        row
        for row in rows
        if row.get("check") == "1a_density_sweep_full_session"
        and int(row.get("workers", -1)) == args.perf_n
    ]
    # density_main returns non-zero for its own Phase-2 sweep verdict (rc=1 NO_PASS_TO_1B,
    # rc=2 single-N can't establish a keep-up multiplier) — unrelated to the WS-overhead
    # gate, which only needs the N-row ttfs telemetry. Treat a MISSING row as the real
    # failure; tolerate rc!=0 when the telemetry row is present.
    if not candidates:
        raise CompatFailure(
            f"density_main did not emit N={args.perf_n} density telemetry (rc={completed.returncode}); log={log_path}"
        )
    if completed.returncode != 0:
        print(
            f"note: density_main rc={completed.returncode} (sweep verdict, not an error); "
            f"N={args.perf_n} telemetry present, continuing to gate",
            flush=True,
        )
    row = candidates[-1]
    try:
        p95 = float(row["ttfs"]["p95_ms"])
    except Exception as exc:
        raise CompatFailure(f"density telemetry missing ttfs.p95_ms: {row!r}") from exc
    return p95, row


def perf_gate(ws_p95: float, density_p95: float) -> tuple[bool, float, float]:
    overhead = ws_p95 - density_p95
    limit = max(2.0, 0.10 * density_p95)
    return overhead <= limit, overhead, limit


async def run_correctness(args: argparse.Namespace, fixtures: list[Fixture]) -> tuple[list[dict[str, Any]], list[str]]:
    py_url = f"ws://{args.host}:{args.python_port}"
    cpp_url = f"ws://{args.host}:{args.cpp_port}"
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for fixture in fixtures:
        try:
            py_capture = await capture_utterance(
                server="python",
                url=py_url,
                fixture=fixture,
                final_timeout_s=args.final_timeout_s,
                close_timeout_s=args.close_timeout_s,
                chunk_bytes=args.chunk_bytes,
                realtime=args.realtime,
                chunk_ms=args.chunk_ms,
                measure_ttfs=True,
            )
            cpp_capture = await capture_utterance(
                server="cpp",
                url=cpp_url,
                fixture=fixture,
                final_timeout_s=args.final_timeout_s,
                close_timeout_s=args.close_timeout_s,
                chunk_bytes=args.chunk_bytes,
                realtime=args.realtime,
                chunk_ms=args.chunk_ms,
                measure_ttfs=True,
            )
            row_failures = compare_capture(py_capture, cpp_capture)
            failures.extend(row_failures)
            ok = not row_failures
            print(
                f"{'PASS' if ok else 'FAIL'} utt={fixture.utt} sample={fixture.sample_index} "
                f"id={fixture.sample_id} events={len(py_capture.events)}/{len(cpp_capture.events)} "
                f"final_equal={py_capture.final_collector_text == cpp_capture.final_collector_text} "
                f"ttfs_ms(py/cpp)={py_capture.ttfs_ms:.3f}/{cpp_capture.ttfs_ms:.3f}",
                flush=True,
            )
            rows.append(
                {
                    "utt": fixture.utt,
                    "sample_index": fixture.sample_index,
                    "sample_id": fixture.sample_id,
                    "pass": ok,
                    "failures": row_failures,
                    "python": capture_to_json(py_capture),
                    "cpp": capture_to_json(cpp_capture),
                }
            )
        except Exception as exc:  # noqa: BLE001 - continue to produce per-utt diagnostics.
            message = f"utt{fixture.utt}: {type(exc).__name__}: {exc}"
            failures.append(message)
            print(f"FAIL {message}", flush=True)
            rows.append(
                {
                    "utt": fixture.utt,
                    "sample_index": fixture.sample_index,
                    "sample_id": fixture.sample_id,
                    "pass": False,
                    "failures": [message],
                }
            )
    return rows, failures


def capture_to_json(capture: CaptureResult) -> dict[str, Any]:
    return {
        "server": capture.server,
        "utt": capture.utt,
        "ready": capture.ready,
        "event_count": len(capture.events),
        "final_collector_text": capture.final_collector_text,
        "ttfs_ms": capture.ttfs_ms,
        "elapsed_s": capture.elapsed_s,
        "events": [
            {
                "signature": event.signature,
                "collector_text": event.collector_text,
                "canonical": event.canonical,
                "raw": event.raw,
            }
            for event in capture.events
        ],
    }


async def run_invalid_query_checks(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    py_base = f"ws://{args.host}:{args.python_port}"
    cpp_base = f"ws://{args.host}:{args.cpp_port}"
    results: dict[str, Any] = {}
    for server, base in (("python", py_base), ("cpp", cpp_base)):
        try:
            invalid = await invalid_model_query(server, base, args.final_timeout_s)
            validate_invalid_model(invalid)
            print(f"PASS invalid-model {server} kind={invalid.kind} status={invalid.status}", flush=True)
            results[f"{server}_invalid_model"] = dataclasses.asdict(invalid)
        except Exception as exc:  # noqa: BLE001
            message = f"{server} invalid ?model=bogus: {type(exc).__name__}: {exc}"
            failures.append(message)
            print(f"FAIL {message}", flush=True)
            results[f"{server}_invalid_model"] = {"pass": False, "error": message}

    py_stats = http_get(f"http://{args.host}:{args.python_port}/stats?last=bogus", args.http_timeout_s)
    cpp_stats = http_get(f"http://{args.host}:{args.cpp_port}/stats?last=bogus", args.http_timeout_s)
    for server, result in (("python", py_stats), ("cpp", cpp_stats)):
        try:
            validate_invalid_stats(server, result)
            print(f"PASS invalid-stats {server} status={result.status} body={result.body}", flush=True)
        except Exception as exc:  # noqa: BLE001
            message = f"{server} invalid /stats?last=bogus: {type(exc).__name__}: {exc}"
            failures.append(message)
            print(f"FAIL {message}", flush=True)
    if py_stats.status == 400 and cpp_stats.status == 400:
        py_canon = canonical_json(py_stats.payload)
        cpp_canon = canonical_json(cpp_stats.payload)
        if py_canon != cpp_canon:
            failures.append(
                f"/stats?last=bogus canonical body mismatch python={py_canon} cpp={cpp_canon}"
            )
    results["python_invalid_stats"] = dataclasses.asdict(py_stats)
    results["cpp_invalid_stats"] = dataclasses.asdict(cpp_stats)
    return results, failures


async def _run(args: argparse.Namespace) -> int:
    start_time = time.perf_counter()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = LOG_ROOT / f"server_compat_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"

    for path, label in (
        (args.python, "python venv"),
        (PYTHON_SERVER, "Python server"),
        (args.cpp_server, "C++ ws_server"),
        (args.density_main, "density_main"),
        (args.bundle, "audio bundle"),
        (args.artifact_dir, "artifact dir"),
        (args.steady_batch_dir, "steady batch dir"),
    ):
        if not Path(path).exists():
            raise CompatFailure(f"{label} not found: {path}")
    assert_port_free(args.host, args.python_port)
    assert_port_free(args.host, args.cpp_port)

    fixtures = load_fixtures(args.bundle, args.start, args.n, language=args.language)
    if len(fixtures) != args.n:
        raise CompatFailure(f"requested {args.n} fixture rows, got {len(fixtures)}")

    py_proc = ManagedProcess(
        "python",
        python_command(args),
        REPO_ROOT,
        python_env(args),
        run_dir / "python_server.log",
    )
    cpp_proc = ManagedProcess(
        "cpp",
        cpp_command(args),
        REPO_ROOT,
        cpp_env(args),
        run_dir / "cpp_ws_server.log",
    )
    processes = [py_proc, cpp_proc]
    summary: dict[str, Any] = {
        "run_dir": run_dir,
        "timestamp_utc": stamp,
        "fixtures": [
            {
                "utt": f.utt,
                "sample_index": f.sample_index,
                "sample_id": f.sample_id,
                "audio_samples": f.audio_samples,
            }
            for f in fixtures
        ],
        "commands": {
            "python": python_command(args),
            "cpp": cpp_command(args),
        },
        "logs": {
            "python": py_proc.log_path,
            "cpp": cpp_proc.log_path,
            "density": run_dir / "density_main.log",
        },
    }
    failures: list[str] = []

    def terminate_all() -> None:
        for proc in processes:
            proc.terminate(args.teardown_timeout_s)

    try:
        print(f"launch python server: {' '.join(py_proc.cmd)}", flush=True)
        py_proc.start()
        print(f"launch cpp ws_server: {' '.join(cpp_proc.cmd)}", flush=True)
        cpp_proc.start()
        py_health, cpp_health = await asyncio.gather(
            wait_health(
                name="python",
                url=f"http://{args.host}:{args.python_port}/health",
                timeout_s=args.server_start_timeout_s,
                process=py_proc,
            ),
            wait_health(
                name="cpp",
                url=f"http://{args.host}:{args.cpp_port}/health",
                timeout_s=args.server_start_timeout_s,
                process=cpp_proc,
            ),
        )
        summary["health"] = {"python": py_health, "cpp": cpp_health}
        print("both servers healthy", flush=True)

        correctness, correctness_failures = await run_correctness(args, fixtures)
        failures.extend(correctness_failures)
        summary["correctness"] = correctness

        invalid_results, invalid_failures = await run_invalid_query_checks(args)
        failures.extend(invalid_failures)
        summary["invalid_queries"] = invalid_results

        perf: dict[str, Any] = {"skipped": bool(args.skip_perf_gate)}
        if not args.skip_perf_gate:
            ws_ttfs = await measure_ws_ttfs(
                url=f"ws://{args.host}:{args.cpp_port}",
                fixtures=fixtures,
                args=args,
            )
            ws_p95 = percentile_p95(ws_ttfs)
            perf["ws_ttfs_ms"] = ws_ttfs
            perf["ws_p95_ms"] = ws_p95
            print(
                f"WS perf C++ N={args.perf_n} ttfs_ms="
                f"{[round(v, 3) for v in ws_ttfs]} ttfs_p95_ms={ws_p95:.3f}",
                flush=True,
            )

            terminate_all()
            density_p95, density_row = run_density_scheduler(args, run_dir / "density_main.log")
            ok, overhead, limit = perf_gate(ws_p95, density_p95)
            perf.update(
                {
                    "density_p95_ms": density_p95,
                    "ws_overhead_p95_ms": overhead,
                    "limit_ms": limit,
                    "pass": ok,
                    "density_row": density_row,
                }
            )
            verdict = "PASS" if ok else "FAIL"
            print(
                f"{verdict} perf-gate ws_p95_ms={ws_p95:.3f} density_p95_ms={density_p95:.3f} "
                f"overhead_ms={overhead:.3f} limit_ms={limit:.3f}",
                flush=True,
            )
            if not ok:
                failures.append(
                    f"WS overhead gate failed: ws_p95={ws_p95:.3f}ms density_p95={density_p95:.3f}ms "
                    f"overhead={overhead:.3f}ms limit={limit:.3f}ms"
                )
        summary["perf"] = perf
    finally:
        terminate_all()
        summary["elapsed_s"] = time.perf_counter() - start_time
        summary["failures"] = failures
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=_json_default, sort_keys=True)
            f.write("\n")
        print(f"wrote summary: {summary_path}", flush=True)

    if failures:
        print("COMPAT FAILURES:", flush=True)
        for failure in failures:
            print(f"  - {failure}", flush=True)
        return 1
    print(f"COMPAT PASS elapsed_s={summary['elapsed_s']:.1f}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--python-port", type=int, default=8080)
    parser.add_argument("--cpp-port", type=int, default=8081)
    parser.add_argument(
        "--language",
        default="",
        help="prompted (multilingual) model: ?language= value for every fixture connection",
    )
    parser.add_argument("--python", type=Path, default=PYTHON_EXE)
    parser.add_argument("--cpp-server", type=Path, default=CPP_SERVER)
    parser.add_argument("--density-main", type=Path, default=DENSITY_MAIN)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--steady-batch-dir", type=Path, default=STEADY_BATCH_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--right-context", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--perf-n", type=int, default=8)
    parser.add_argument("--density-rows", type=int, default=8)
    parser.add_argument("--chunk-bytes", type=int, default=CHUNK_BYTES)
    parser.add_argument("--chunk-ms", type=int, default=20)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--finalize-silence-ms", type=int, default=0)
    parser.add_argument("--admission-active-cap", type=int, default=64)
    parser.add_argument("--ws-lanes", type=int, default=None)
    parser.add_argument("--ws-finalize-runners", type=int, default=None)
    parser.add_argument("--batch-b-max", type=int, default=4)
    parser.add_argument("--batch-window-ms", type=int, default=10)
    parser.add_argument("--batch-lone-timeout-ms", type=int, default=0)
    parser.add_argument("--ws-steady-shadow", action="store_true")
    # C++ ws_server cold-loads the SharedRuntime AOTI packages (enc_first,
    # enc_steady, one scheduler loader set, and ~27 finalize buckets); measured ~330s
    # cold on sm_120. The Python server loads the same artifacts in ~10s. 600s
    # leaves margin so the slow side is not killed mid-load.
    parser.add_argument("--server-start-timeout-s", type=float, default=600.0)
    parser.add_argument("--final-timeout-s", type=float, default=90.0)
    parser.add_argument("--close-timeout-s", type=float, default=5.0)
    parser.add_argument("--http-timeout-s", type=float, default=5.0)
    parser.add_argument("--teardown-timeout-s", type=float, default=20.0)
    parser.add_argument("--density-timeout-s", type=float, default=900.0)
    parser.add_argument("--skip-perf-gate", action="store_true")
    args = parser.parse_args()
    if args.ws_lanes is not None and args.ws_lanes <= 0:
        parser.error("--ws-lanes must be positive")
    if args.ws_finalize_runners is not None and args.ws_finalize_runners <= 0:
        parser.error("--ws-finalize-runners must be positive")
    # Keep the venv symlink path. Resolving it executes the base interpreter
    # directly and loses the venv's site-packages.
    args.cpp_server = args.cpp_server.resolve()
    args.density_main = args.density_main.resolve()
    args.bundle = args.bundle.resolve()
    args.artifact_dir = args.artifact_dir.resolve()
    args.steady_batch_dir = args.steady_batch_dir.resolve()
    if args.chunk_bytes <= 0 or args.chunk_bytes % 2 != 0:
        parser.error("--chunk-bytes must be a positive even byte count")
    if args.n <= 0:
        parser.error("--n must be positive")
    if args.perf_n <= 0:
        parser.error("--perf-n must be positive")
    if args.density_rows <= 0:
        parser.error("--density-rows must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except CompatFailure as exc:
        print(f"COMPAT SETUP FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
