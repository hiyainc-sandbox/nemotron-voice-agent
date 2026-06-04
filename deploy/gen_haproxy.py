#!/usr/bin/env python3
"""Generate the Phase 1 HAProxy config for Nemotron streaming ASR.

Purpose
-------
This stdlib-only tool renders an HAProxy config for the Phase 1 single-process
deployment described in deploy/RUNBOOK.md and deploy/DEPLOYMENT.md:
one ``python -m nemotron_speech.server`` process per L40S box on port 8080,
fronted by HAProxy in TCP mode with least-connections routing, health checks,
runtime-socket draining, and per-box ``maxconn`` caps.

Why leastconn is the right primitive here
-----------------------------------------
The rationale is lifted from deploy/haproxy.cfg.example: for this service,
one WebSocket connection equals one ASR stream and one unit of backend capacity.
WebSocket streams are sticky for their full lifetime once HAProxy picks a
backend, so active connection count is active stream count. Least-connections
routing therefore sends new streams to the box furthest below its stream budget.

Why maxconn defaults to 20
--------------------------
The 2026-05-27 L40S single-process measurement showed one full-GPU Python
server holding roughly 20 concurrent streams at acceptable latency for the
one-utterance-per-connection workload. That is why ``--maxconn`` defaults to 20
for one backend process per box. The result has not yet been validated under
sustained multi-turn traffic; use ``--maxconn-conservative`` to emit 12 when
you want the safer pre-validation value.

Queue policy
------------
``timeout queue 5s`` means HAProxy queues connection attempts that arrive after
a backend reaches its per-server ``maxconn``. If capacity opens within five
seconds, HAProxy forwards the connection; otherwise HAProxy fails it. This is
not a 1013-style server-side admission reject. HAProxy queues at the load
balancer, while server.py remains responsible for server-side admission
shedding via ``NEMOTRON_ADMISSION_MAX_BACKLOG``.

Operator examples
-----------------
Two-box development config without TLS::

    python3 deploy/gen_haproxy.py \\
      --boxes 10.0.1.10,10.0.1.11 \\
      -o /tmp/haproxy.cfg

Production config from a named fleet file with TLS and syntax check::

    cat >/tmp/asr-fleet.txt <<'EOF'
    box_a=10.0.1.10
    box_b=10.0.1.11
    EOF
    python3 deploy/gen_haproxy.py \\
      --boxes-file /tmp/asr-fleet.txt \\
      --maxconn 20 \\
      --tls-port 8443 \\
      --tls-pem /etc/haproxy/asr.pem \\
      -o /tmp/haproxy.cfg \\
      --check

Laptop-safe HAProxy smoke config::

    tmpdir=$(mktemp -d)
    python3 deploy/gen_haproxy.py \\
      --local-test \\
      --stats-socket "$tmpdir/haproxy.sock" \\
      --boxes 127.0.0.1,127.0.0.2 \\
      -o "$tmpdir/haproxy.cfg"

Graceful reload recipe
----------------------
After installing the generated config, reload rather than restart so active
WebSocket streams survive:

    systemctl reload haproxy

If running HAProxy directly, use the old-process handoff form:

    haproxy -sf $(pidof haproxy) -f /etc/haproxy/haproxy.cfg

Related artifacts
-----------------
* deploy/RUNBOOK.md - operator procedure.
* deploy/DEPLOYMENT.md - topology and sizing rationale.
* deploy/launch_single.sh - one-process-per-box server launcher.
* deploy/haproxy.cfg.example - original routing design artifact.
* deploy/drain.sh - runtime-socket drain helper planned in step 4.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import ipaddress
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Iterable, Sequence


GENERATOR = "deploy/gen_haproxy.py"
DEFAULT_FRONT_PORT = 8080
DEFAULT_TLS_PORT = 8443
DEFAULT_MAXCONN = 20
CONSERVATIVE_MAXCONN = 12
DEFAULT_STATS_SOCKET = "/run/haproxy/admin.sock"
SERVER_PORT = 8080
NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

# Production-mode global maxconn (the LB ceiling, not per-backend).
# 100000 is overscale for the realistic Phase-1 fleet (N boxes x ~20 maxconn)
# but matches HAProxy convention and leaves headroom for connection storms.
GLOBAL_MAXCONN_PROD = 100000
# --local-test global maxconn: keep small so the FD constraint stays portable
# (a laptop's default soft `nofile` may be 1024; 4096 ulimit-n supports
# maxconn up to ~2025 per the HAProxy 2*maxconn+42 formula).
GLOBAL_MAXCONN_LOCAL = 1000


def required_ulimit_n(global_maxconn: int) -> int:
    """HAProxy requires file descriptors >= 2 * maxconn + 42.

    Live-validation 2026-05-28 caught the original hardcoded ulimit-n 200000
    vs. maxconn 100000 mismatch:
      [ALERT] FD limit (200000) too low for maxconn=100000/maxsock=200042.
    We derive ulimit-n from maxconn with a small buffer so the two stay
    consistent forever.
    """
    return 2 * global_maxconn + 100


@dataclass(frozen=True)
class Backend:
    """Validated backend server entry."""

    name: str
    ip: str
    ip_obj: ipaddress._BaseAddress


class Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """argparse formatter that preserves examples and prints defaults."""


def positive_int(value: str) -> int:
    """Parse a positive integer for maxconn-like arguments."""
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def port_int(value: str) -> int:
    """Parse a TCP port."""
    parsed = positive_int(value)
    if parsed > 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    epilog = """\
Examples:
  # Two-box development config, plain TCP frontend on :8080.
  %(prog)s --boxes 10.0.1.10,10.0.1.11 -o /tmp/haproxy.cfg

  # Named production fleet with TLS frontend on :8443 and HAProxy syntax check.
  %(prog)s --boxes-file /tmp/asr-fleet.txt --maxconn 20 \\
    --tls-pem /etc/haproxy/asr.pem -o /tmp/haproxy.cfg --check

  # Conservative cap until sustained multi-turn load is validated.
  %(prog)s --boxes-file /tmp/asr-fleet.txt --maxconn-conservative -o /tmp/haproxy.cfg

  # Laptop-safe smoke config; pass an explicit socket under mktemp -d.
  tmpdir=$(mktemp -d)
  %(prog)s --local-test --stats-socket "$tmpdir/haproxy.sock" \\
    --boxes 127.0.0.1,127.0.0.2 -o "$tmpdir/haproxy.cfg"

Fleet file format:
  One backend per line, either "IP" or "name=IP". Blank lines and lines
  beginning with "#" are ignored. Names must match ^[a-zA-Z0-9_-]+$.

Cross-links:
  deploy/RUNBOOK.md, deploy/DEPLOYMENT.md, deploy/launch_single.sh,
  deploy/haproxy.cfg.example, deploy/drain.sh
"""
    parser = argparse.ArgumentParser(
        prog=GENERATOR,
        description=(
            "Generate the Phase 1 HAProxy config for Nemotron ASR WebSocket "
            "backends. The output is self-documenting and safe to validate "
            "locally with --local-test."
        ),
        epilog=epilog,
        formatter_class=Formatter,
    )
    fleet = parser.add_mutually_exclusive_group(required=True)
    fleet.add_argument(
        "--boxes",
        metavar="COMMA_LIST",
        help=(
            "Comma-separated backend IP list, for example "
            "10.0.1.10,10.0.1.11. Backend names are generated as "
            "box_<ip-with-dashes> and stay stable across input reordering."
        ),
    )
    fleet.add_argument(
        "--boxes-file",
        metavar="PATH",
        help=(
            "Path to a fleet file with one backend per line: either IP or "
            "name=IP. Example line: box_a=10.0.1.10."
        ),
    )
    parser.add_argument(
        "--maxconn",
        type=positive_int,
        default=DEFAULT_MAXCONN,
        metavar="INT",
        help=(
            "Per-backend HAProxy maxconn. Default 20 comes from the "
            "2026-05-27 L40S single-process one-utterance-per-connection "
            "measurement. Example: --maxconn 16."
        ),
    )
    parser.add_argument(
        "--maxconn-conservative",
        action="store_true",
        help=(
            "Use maxconn 12 instead of --maxconn. This is the safer value "
            "until sustained multi-turn load is validated. Example: "
            "--maxconn-conservative."
        ),
    )
    parser.add_argument(
        "--front-port",
        type=port_int,
        default=DEFAULT_FRONT_PORT,
        metavar="INT",
        help=(
            "Plain TCP frontend port used when --tls-pem is omitted. "
            "Example: --front-port 8080."
        ),
    )
    parser.add_argument(
        "--tls-port",
        type=port_int,
        default=DEFAULT_TLS_PORT,
        metavar="INT",
        help=(
            "TLS frontend port used when --tls-pem is provided. "
            "Example: --tls-port 8443."
        ),
    )
    parser.add_argument(
        "--tls-pem",
        metavar="PATH",
        help=(
            "Optional HAProxy PEM path for TLS termination. When set, the "
            "frontend bind uses --tls-port with 'ssl crt PATH'. Example: "
            "--tls-pem /etc/haproxy/asr.pem."
        ),
    )
    parser.add_argument(
        "--stats-socket",
        default=DEFAULT_STATS_SOCKET,
        metavar="PATH",
        help=(
            "HAProxy Runtime API socket path for drain.sh. In --local-test, "
            "an omitted value becomes a generated tmp path printed to stderr. "
            "Example: --stats-socket /run/haproxy/admin.sock."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help=(
            "Write rendered config to PATH. If omitted, write to stdout. "
            "Example: -o /tmp/haproxy.cfg."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Only valid with -o/--output. After writing, run "
            "'haproxy -c -f OUTPUT'. Example: --check."
        ),
    )
    parser.add_argument(
        "--local-test",
        action="store_true",
        help=(
            "Emit a laptop-safe config for non-root smoke tests: omit user/group "
            "haproxy, use stats socket mode 600, set ulimit-n 4096, and log to "
            "stdout with format raw daemon. Example: --local-test."
        ),
    )
    return parser


def option_was_given(argv: Sequence[str], option: str) -> bool:
    prefix = f"{option}="
    return any(arg == option or arg.startswith(prefix) for arg in argv)


def parse_boxes_list(value: str) -> list[tuple[str | None, str, str]]:
    entries: list[tuple[str | None, str, str]] = []
    for index, item in enumerate(value.split(","), start=1):
        token = item.strip()
        if not token:
            continue
        if "=" in token:
            raise SystemExit(
                f"{GENERATOR}: error: --boxes accepts IPs only; use --boxes-file for name=ip entries"
            )
        entries.append((None, token, f"--boxes item {index}"))
    return entries


def parse_boxes_file(path_text: str) -> list[tuple[str | None, str, str]]:
    path = Path(path_text)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SystemExit(f"{GENERATOR}: error: cannot read --boxes-file {path}: {exc}") from exc

    entries: list[tuple[str | None, str, str]] = []
    for lineno, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        source = f"{path}:{lineno}"
        if "=" in line:
            name, ip_text = (part.strip() for part in line.split("=", 1))
            if not name:
                raise SystemExit(f"{GENERATOR}: error: empty backend name at {source}")
            if not ip_text:
                raise SystemExit(f"{GENERATOR}: error: empty IP address at {source}")
            entries.append((name, ip_text, source))
        else:
            entries.append((None, line, source))
    return entries


def default_backend_name(ip_obj: ipaddress._BaseAddress) -> str:
    safe_ip = str(ip_obj).replace(".", "-").replace(":", "-")
    return f"box_{safe_ip}"


def validate_backends(raw_entries: Iterable[tuple[str | None, str, str]]) -> list[Backend]:
    backends: list[Backend] = []
    seen_ips: dict[ipaddress._BaseAddress, str] = {}
    seen_names: dict[str, str] = {}

    for explicit_name, ip_text, source in raw_entries:
        try:
            ip_obj = ipaddress.ip_address(ip_text)
        except ValueError as exc:
            raise SystemExit(f"{GENERATOR}: error: invalid IP address at {source}: {ip_text!r}") from exc

        canonical_ip = str(ip_obj)
        name = explicit_name or default_backend_name(ip_obj)
        if not NAME_RE.fullmatch(name):
            raise SystemExit(
                f"{GENERATOR}: error: invalid backend name at {source}: {name!r}; "
                "must match ^[a-zA-Z0-9_-]+$"
            )
        if ip_obj in seen_ips:
            raise SystemExit(
                f"{GENERATOR}: error: duplicate IP {canonical_ip} at {source}; "
                f"first seen at {seen_ips[ip_obj]}"
            )
        if name in seen_names:
            raise SystemExit(
                f"{GENERATOR}: error: duplicate backend name {name!r} at {source}; "
                f"first seen at {seen_names[name]}"
            )
        seen_ips[ip_obj] = source
        seen_names[name] = source
        backends.append(Backend(name=name, ip=canonical_ip, ip_obj=ip_obj))

    if not backends:
        raise SystemExit(f"{GENERATOR}: error: empty backend fleet")
    return backends


def haproxy_endpoint(backend: Backend) -> str:
    if backend.ip_obj.version == 6:
        return f"[{backend.ip}]:{SERVER_PORT}"
    return f"{backend.ip}:{SERVER_PORT}"


def shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def render_regenerate_hint(args: argparse.Namespace, maxconn: int, local_stats_socket_was_generated: bool) -> str:
    command = ["python3", GENERATOR]
    if args.boxes is not None:
        command.extend(["--boxes", args.boxes])
    else:
        command.extend(["--boxes-file", args.boxes_file])
    if args.maxconn_conservative:
        command.append("--maxconn-conservative")
    elif maxconn != DEFAULT_MAXCONN:
        command.extend(["--maxconn", str(maxconn)])
    if args.front_port != DEFAULT_FRONT_PORT:
        command.extend(["--front-port", str(args.front_port)])
    if args.tls_port != DEFAULT_TLS_PORT:
        command.extend(["--tls-port", str(args.tls_port)])
    if args.tls_pem:
        command.extend(["--tls-pem", args.tls_pem])
    if args.stats_socket != DEFAULT_STATS_SOCKET or args.local_test:
        command.extend(["--stats-socket", args.stats_socket])
    if args.local_test:
        command.append("--local-test")
    if args.output:
        command.extend(["-o", args.output])

    hint = shell_join(command)
    if local_stats_socket_was_generated:
        hint += "  # stats socket path was auto-generated for this local-test run"
    return hint


def render_config(
    *,
    backends: Sequence[Backend],
    maxconn: int,
    front_port: int,
    tls_port: int,
    tls_pem: str | None,
    stats_socket: str,
    local_test: bool,
    regenerate_hint: str,
) -> str:
    generated_at = _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="seconds")
    generated_at = generated_at.replace("+00:00", "Z")
    socket_mode = "600" if local_test else "660"
    log_line = "log stdout format raw daemon" if local_test else "log /dev/log local0"
    global_maxconn = GLOBAL_MAXCONN_LOCAL if local_test else GLOBAL_MAXCONN_PROD
    ulimit = str(required_ulimit_n(global_maxconn))
    bind_port = tls_port if tls_pem else front_port
    bind_suffix = f" ssl crt {tls_pem}" if tls_pem else ""
    fleet_inline = ", ".join(f"{backend.name}={backend.ip}" for backend in backends)

    lines: list[str] = [
        "# -----------------------------------------------------------------------------",
        f"# Generated by {GENERATOR}. Regenerate from repo root with:",
        f"#   {regenerate_hint}",
        f"# Generated at: {generated_at}",
        f"# Input fleet ({len(backends)} boxes): {fleet_inline}",
        "#",
        "# Purpose: route long-lived Nemotron ASR WebSocket streams to one",
        "# deploy/launch_single.sh server process per backend box on :8080.",
        "#",
        "# Related docs/artifacts:",
        "#   deploy/RUNBOOK.md      operator procedure",
        "#   deploy/DEPLOYMENT.md          topology, sizing, and tradeoffs",
        "#   deploy/launch_single.sh       backend process launcher",
        "#   deploy/haproxy.cfg.example    original leastconn design artifact",
        "#   deploy/drain.sh               runtime-socket drain helper (step 4)",
        "#",
        "# Stable backend names for drain.sh:",
    ]
    for backend in backends:
        lines.append(f"#   {backend.name} -> {backend.ip}")
    lines.extend(
        [
            "# -----------------------------------------------------------------------------",
            "",
            "global",
            f"    maxconn {global_maxconn}",
            f"    stats socket {stats_socket} mode {socket_mode} level admin",
        ]
    )
    if not local_test:
        lines.extend(
            [
                "    user haproxy",
                "    group haproxy",
            ]
        )
    lines.extend(
        [
            f"    {log_line}",
            f"    ulimit-n {ulimit}",
            "",
            "defaults",
            "    mode tcp",
            "    option tcplog",
            "    log global",
            "    timeout connect 5s",
            "    timeout client 1h",
            "    timeout server 1h",
            "    timeout queue 5s",
            "    # timeout client/server 1h: ASR WebSocket streams are long-lived.",
            "    # timeout queue 5s: HAProxy queues past maxconn briefly; server.py",
            "    # still owns 1013-style admission shedding.",
            "",
            "backend asr_pool",
            "    balance leastconn",
            "    # leastconn works because one WebSocket connection is one active ASR stream.",
            "    option httpchk",
            "    http-check send meth GET uri /health",
            "    # /health always returns HTTP 200; require the JSON status field to be healthy.",
            '    http-check expect rstring "\\"status\\"[ ]*:[ ]*\\"healthy\\""',
            f"    # maxconn {maxconn}: cap each single-process L40S backend at the measured",
            "    # stream budget; use --maxconn-conservative for 12 until sustained",
            "    # multi-turn traffic is validated.",
        ]
    )
    for backend in backends:
        lines.append(
            f"    server {backend.name} {haproxy_endpoint(backend)} "
            f"check inter 2s fall 3 rise 2 maxconn {maxconn}"
        )
    lines.extend(
        [
            "",
            "frontend asr_ws",
            f"    bind *:{bind_port}{bind_suffix}",
            "    default_backend asr_pool",
            "",
        ]
    )
    return "\n".join(lines)


def run_haproxy_check(output_path: str) -> int:
    haproxy = shutil.which("haproxy")
    if haproxy is None:
        raise SystemExit(f"{GENERATOR}: error: --check requested but haproxy was not found on PATH")
    completed = subprocess.run([haproxy, "-c", "-f", output_path], check=False)
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    stats_socket_was_given = option_was_given(argv, "--stats-socket")
    maxconn_was_given = option_was_given(argv, "--maxconn")
    args = parser.parse_args(argv)

    if args.check and not args.output:
        parser.error("--check is only valid with -o/--output")
    if args.maxconn_conservative and maxconn_was_given:
        parser.error("--maxconn and --maxconn-conservative cannot be used together")

    local_stats_socket_was_generated = False
    if args.local_test and not stats_socket_was_given:
        tmpdir = tempfile.mkdtemp(prefix="nemotron-haproxy-")
        args.stats_socket = os.path.join(tmpdir, "admin.sock")
        local_stats_socket_was_generated = True
        print(
            f"{GENERATOR}: --local-test using generated stats socket {args.stats_socket}",
            file=sys.stderr,
        )

    maxconn = CONSERVATIVE_MAXCONN if args.maxconn_conservative else args.maxconn
    raw_entries = parse_boxes_list(args.boxes) if args.boxes is not None else parse_boxes_file(args.boxes_file)
    backends = validate_backends(raw_entries)
    regenerate_hint = render_regenerate_hint(args, maxconn, local_stats_socket_was_generated)
    config = render_config(
        backends=backends,
        maxconn=maxconn,
        front_port=args.front_port,
        tls_port=args.tls_port,
        tls_pem=args.tls_pem,
        stats_socket=args.stats_socket,
        local_test=args.local_test,
        regenerate_hint=regenerate_hint,
    )

    if args.output:
        Path(args.output).write_text(config, encoding="utf-8")
    else:
        sys.stdout.write(config)

    if args.check:
        return run_haproxy_check(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
