#!/usr/bin/env python3
"""Tiny stdlib-only backend used by deploy/smoke_local.sh.

Modes:
  echo: bind 127.0.0.1:PORT, accept TCP connections, and echo bytes back.
  http: bind 127.0.0.1:PORT and serve GET /health with the exact bytes read
        from --state-file on every request.

The process prints a line containing "listening" after the socket is bound so
the smoke harness can wait for readiness. SIGTERM/SIGINT request a clean exit.
"""

from __future__ import annotations

import argparse
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import signal
import sys
from typing import Sequence


HOST = "127.0.0.1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tiny TCP echo or HTTP /health backend for deploy/smoke_local.sh."
    )
    parser.add_argument("--mode", choices=("echo", "http"), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--state-file", help="HTTP mode only: file containing the /health JSON body")
    return parser


async def run_echo(port: int) -> int:
    stop = asyncio.Event()
    active_writers: set[asyncio.StreamWriter] = set()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        active_writers.add(writer)
        peer = writer.get_extra_info("peername")
        print(f"accepted {peer}", flush=True)
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        except Exception:
            pass
        finally:
            active_writers.discard(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    loop = asyncio.get_running_loop()

    def request_stop() -> None:
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, request_stop)
        except NotImplementedError:
            signal.signal(sig, lambda _signum, _frame: request_stop())

    server = await asyncio.start_server(handle, HOST, port)
    actual_port = server.sockets[0].getsockname()[1] if server.sockets else port
    print(f"listening {HOST}:{actual_port}", flush=True)
    async with server:
        await stop.wait()
        server.close()
        await server.wait_closed()

    for writer in list(active_writers):
        writer.close()
    return 0


class HealthServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], state_file: Path) -> None:
        self.state_file = state_file
        super().__init__(server_address, HealthHandler)


class HealthHandler(BaseHTTPRequestHandler):
    server: HealthServer

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path.split("?", 1)[0] != "/health":
            self.send_response(404)
            self.end_headers()
            return

        try:
            body = self.server.state_file.read_bytes()
        except OSError as exc:
            body = f'{{"status":"error","error":{str(exc)!r}}}'.encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def run_http(port: int, state_file: Path) -> int:
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    with HealthServer((HOST, port), state_file) as server:
        server.timeout = 0.25
        actual_port = server.server_address[1]
        print(f"listening {HOST}:{actual_port}", flush=True)
        while not stopping:
            server.handle_request()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    if args.mode == "http" and not args.state_file:
        parser.error("--state-file is required in --mode http")
    if args.mode == "echo" and args.state_file:
        parser.error("--state-file is only valid in --mode http")

    if args.mode == "echo":
        return asyncio.run(run_echo(args.port))
    return run_http(args.port, Path(args.state_file))


if __name__ == "__main__":
    raise SystemExit(main())
