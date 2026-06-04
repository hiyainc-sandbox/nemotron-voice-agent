#!/usr/bin/env python3
"""Run the optional real-HAProxy health-check smoke for deploy/smoke_local.sh.

This helper proves the generated HAProxy config's HTTP health check and
``http-check expect rstring`` behavior by driving a local stub backend through:

  healthy -> loading -> healthy

It skips cleanly when haproxy is not installed or when 127.0.0.1:8080 is
already in use, because deploy/gen_haproxy.py hardcodes backend server lines to
port 8080.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Sequence


PASS = 0
FAIL = 1
SKIP = 77
BACKEND_NAME = "box_127-0-0-1"
BACKEND = "asr_pool"
HEALTHY = b'{"status":"healthy","model_loaded":true}\n'
LOADING = b'{"status":"loading","model_loaded":false}\n'


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def port_8080_available() -> tuple[bool, str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 8080))
    except OSError as exc:
        return False, str(exc)
    finally:
        sock.close()
    return True, ""


def wait_for_stdout_line(proc: subprocess.Popen[str], needle: str, timeout: float) -> str | None:
    assert proc.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    lines: list[str] = []
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            rest = proc.stdout.read() or ""
            if rest:
                lines.append(rest)
            return "".join(lines) or None
        remaining = max(0.05, deadline - time.monotonic())
        for key, _event_mask in selector.select(remaining):
            line = key.fileobj.readline()
            if not line:
                continue
            lines.append(line)
            if needle in line:
                return "".join(lines)
    return "".join(lines) or None


def read_process_tail(proc: subprocess.Popen[str]) -> str:
    if proc.poll() is None:
        return f"process still running pid={proc.pid}"
    pieces: list[str] = []
    for stream_name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
        if stream is None:
            continue
        try:
            data = stream.read()
        except Exception:
            data = ""
        if data:
            pieces.append(f"{stream_name}:\n{data}")
    return "\n".join(pieces)


def run_command(cmd: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def wait_for_socket(path: Path, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


def show_stat(socket_path: Path) -> str:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(2.0)
        sock.connect(str(socket_path))
        sock.sendall(b"show stat\n")
        sock.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")


def backend_status(socket_path: Path) -> tuple[str | None, str]:
    raw = show_stat(socket_path)
    lines = []
    for line in raw.splitlines():
        if line.startswith("# "):
            line = line[2:]
        elif line.startswith("#"):
            line = line[1:].lstrip()
        if line:
            lines.append(line)
    if not lines:
        return None, raw

    reader = csv.DictReader(lines)
    for row in reader:
        if row.get("pxname") == BACKEND and row.get("svname") == BACKEND_NAME:
            return row.get("status"), raw
    return None, raw


def wait_for_status(socket_path: Path, expected: str, timeout: float) -> tuple[bool, list[str], str]:
    deadline = time.monotonic() + timeout
    seen: list[str] = []
    last_raw = ""
    while time.monotonic() < deadline:
        try:
            status, raw = backend_status(socket_path)
            last_raw = raw
        except OSError as exc:
            status = f"socket-error:{exc}"
            raw = ""
            last_raw = raw
        seen.append(str(status))
        if status == expected:
            return True, seen, last_raw
        time.sleep(0.25)
    return False, seen, last_raw


def kill_pidfile(pidfile: Path) -> None:
    try:
        pid_text = pidfile.read_text(encoding="utf-8").strip()
        pid = int(pid_text)
    except (OSError, ValueError):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def terminate_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def fail(message: str) -> int:
    print(f"FAIL Generator health-check behavior (real haproxy + stub backend): {message}")
    return FAIL


def main(_argv: Sequence[str] | None = None) -> int:
    root = repo_root()
    haproxy = shutil.which("haproxy")
    if haproxy is None:
        print(
            "SKIP Generator health-check behavior (real haproxy + stub backend): "
            "WARN haproxy not found on PATH"
        )
        return SKIP

    available, reason = port_8080_available()
    if not available:
        print(
            "SKIP Generator health-check behavior (real haproxy + stub backend): "
            f"WARN 127.0.0.1:8080 is in use ({reason}); generator hardcodes backend port 8080"
        )
        return SKIP

    tmpdir = Path(tempfile.mkdtemp(prefix="nemotron-smoke-haproxy-"))
    state_file = tmpdir / "health.json"
    cfg = tmpdir / "haproxy.cfg"
    socket_path = tmpdir / "haproxy.sock"
    pidfile = tmpdir / "haproxy.pid"
    backend_proc: subprocess.Popen[str] | None = None

    try:
        state_file.write_bytes(HEALTHY)
        backend_proc = subprocess.Popen(
            [
                sys.executable,
                str(root / "deploy" / "_smoke_backend.py"),
                "--mode",
                "http",
                "--port",
                "8080",
                "--state-file",
                str(state_file),
            ],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        ready_output = wait_for_stdout_line(backend_proc, "listening", 5.0)
        if ready_output is None or "listening" not in ready_output:
            return fail(
                "expected stub backend to print 'listening' within 5s; "
                f"actual rc={backend_proc.poll()} output={ready_output!r} {read_process_tail(backend_proc)!r}"
            )

        generated = run_command(
            [
                sys.executable,
                "deploy/gen_haproxy.py",
                "--local-test",
                "--stats-socket",
                str(socket_path),
                "--boxes",
                "127.0.0.1",
                "-o",
                str(cfg),
            ],
            root,
        )
        if generated.returncode != 0:
            return fail(
                "expected gen_haproxy.py rc=0; "
                f"actual rc={generated.returncode} stdout={generated.stdout!r} stderr={generated.stderr!r}"
            )

        started = run_command([haproxy, "-f", str(cfg), "-D", "-p", str(pidfile)], root)
        if started.returncode != 0:
            return fail(
                "expected haproxy daemon start rc=0; "
                f"actual rc={started.returncode} stdout={started.stdout!r} stderr={started.stderr!r}"
            )
        if not wait_for_socket(socket_path, 5.0):
            return fail(f"expected Runtime API socket {socket_path} within 5s; actual missing")

        ok, seen, raw = wait_for_status(socket_path, "UP", 10.0)
        if not ok:
            return fail(
                "expected backend status UP within 10s while /health is healthy; "
                f"actual seen={seen} last_show_stat={raw!r}"
            )

        state_file.write_bytes(LOADING)
        ok, seen, raw = wait_for_status(socket_path, "DOWN", 12.0)
        if not ok:
            return fail(
                "expected backend status DOWN within 12s after /health returned loading; "
                f"actual seen={seen} last_show_stat={raw!r}"
            )

        state_file.write_bytes(HEALTHY)
        ok, seen, raw = wait_for_status(socket_path, "UP", 10.0)
        if not ok:
            return fail(
                "expected backend status UP within 10s after /health returned healthy again; "
                f"actual seen={seen} last_show_stat={raw!r}"
            )

        print(
            "PASS Generator health-check behavior (real haproxy + stub backend): "
            'observed healthy -> loading -> healthy as HAProxy UP -> DOWN -> UP via http-check expect rstring'
        )
        return PASS
    finally:
        kill_pidfile(pidfile)
        terminate_process(backend_proc)
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
