#!/usr/bin/env bash
# Operator drain helper for the Phase 1 Nemotron ASR HAProxy backend.
#
# Purpose:
#   Wrap the HAProxy Runtime API socket for the Phase 1 Nemotron streaming-ASR
#   deployment. Operators use this on the load-balancer host to stop new
#   WebSocket streams from being assigned to one backend server, wait until its
#   active stream count reaches zero, restart deploy/nemotron-asr.service on
#   that box, and then mark the box ready again. Backend server names are the
#   stable names emitted by deploy/gen_haproxy.py, for example box_10-0-1-10.
#
# Usage:
#   deploy/drain.sh drain <box>
#     Issues: set server asr_pool/<box> state drain
#     Example: deploy/drain.sh drain box_10-0-1-10
#
#   deploy/drain.sh ready <box>
#     Issues: set server asr_pool/<box> state ready
#     Example: deploy/drain.sh ready box_10-0-1-10
#
#   deploy/drain.sh status
#     Runs the Runtime API command "show stat" and prints one tab-separated
#     line per asr_pool server: svname, status, scur, smax.
#     Example: deploy/drain.sh status
#
#   deploy/drain.sh wait-empty <box> [timeout=300]
#     Polls "show stat" every 2 seconds until asr_pool/<box> has scur == 0, or
#     until timeout seconds elapse.
#     Example: deploy/drain.sh wait-empty box_10-0-1-10 300
#
#   deploy/drain.sh -h
#   deploy/drain.sh --help
#     Prints the usage block.
#
# Rolling deploy procedure:
#   deploy/drain.sh drain box_X && \
#   deploy/drain.sh wait-empty box_X 300 && \
#   ssh box_X 'sudo systemctl restart nemotron-asr' && \
#   until curl -fsS "http://box_X:8080/health" | grep -q '"status"[[:space:]]*:[[:space:]]*"healthy"'; do sleep 2; done && \
#   deploy/drain.sh ready box_X
#
# Environment:
#   HAPROXY_SOCK
#     Default: /run/haproxy/admin.sock
#     Meaning: HAProxy Runtime API socket path. Production usage expects a UNIX
#       socket created by the generated HAProxy "stats socket" directive. For
#       fixture parsing and the local smoke test only, HAPROXY_SOCK=/dev/stdin
#       makes status and wait-empty read one show-stat CSV from stdin instead
#       of contacting socat.
#
# Exit codes and operator action:
#   0  Success, or wait-empty observed scur == 0. Continue the deploy step.
#   1  wait-empty timed out. The script prints the remaining scur count to
#      stdout and a timeout message to stderr; keep waiting, investigate stuck
#      streams, or extend the drain window before restarting the server.
#   2  Backend asr_pool or the requested server name was not found in show stat,
#      or HAProxy rejected a state-change command because the server/backend did
#      not exist. Check deploy/gen_haproxy.py output and the box name.
#   3  Runtime socket could not be reached, permission was denied, or a required
#      local prerequisite such as socat/python3 is unavailable. On the LB host,
#      check that HAProxy is running and that HAPROXY_SOCK points to the socket.
#   4  Bad arguments. Re-run with -h and fix the command line.
#
# 1013 backpressure:
#   HAProxy is generated in mode tcp for WebSocket safety. In TCP mode it cannot
#   inspect a WebSocket close frame and cannot react to the server's 1013 close
#   code. The real overload/backpressure mechanism is server-side admission via
#   NEMOTRON_ADMISSION_MAX_BACKLOG in deploy/nemotron-asr.service's environment;
#   this script is only the operator-initiated drain path for rolling deploys.
#
# Permissions and prerequisites:
#   socat is required on the load-balancer host; deploy/RUNBOOK.md
#   installs it. The operator must either be in the haproxy group for a socket
#   created with mode 660, run this script under sudo, or configure HAProxy's
#   global "stats socket" with an explicit user/group matching the operator.
#
# Runtime API details:
#   The command is exactly "show stat". Do not use "show stat;csv": the ;csv
#   suffix is stats-page URI syntax, not Runtime API socket syntax. CSV is the
#   default format on the socket. The first header row begins "# pxname,...";
#   the parser strips the leading "# " and reads columns by header name.
#
# Related documentation:
#   Service unit: deploy/nemotron-asr.service
#   HAProxy generator and backend-name contract: deploy/gen_haproxy.py
#   Runbook/how: deploy/RUNBOOK.md
#   Rationale/why: deploy/DEPLOYMENT.md

set -uo pipefail

DEFAULT_HAPROXY_SOCK="/run/haproxy/admin.sock"
BACKEND="asr_pool"
POLL_SECONDS=2

usage() {
  cat <<'USAGE'
Usage:
  deploy/drain.sh drain <box>
      Set HAProxy server asr_pool/<box> to drain.
      Example: deploy/drain.sh drain box_10-0-1-10

  deploy/drain.sh ready <box>
      Set HAProxy server asr_pool/<box> to ready.
      Example: deploy/drain.sh ready box_10-0-1-10

  deploy/drain.sh status
      Run "show stat" and print: svname<TAB>status<TAB>scur<TAB>smax.
      Example: deploy/drain.sh status

  deploy/drain.sh wait-empty <box> [timeout=300]
      Poll "show stat" every 2s until asr_pool/<box> has scur == 0.
      Example: deploy/drain.sh wait-empty box_10-0-1-10 300

Environment:
  HAPROXY_SOCK
      HAProxy Runtime API UNIX socket path.
      Default: /run/haproxy/admin.sock
      Fixture mode: HAPROXY_SOCK=/dev/stdin reads one show-stat CSV from stdin.

Rolling deploy:
  deploy/drain.sh drain box_X && \
  deploy/drain.sh wait-empty box_X 300 && \
  ssh box_X 'sudo systemctl restart nemotron-asr' && \
  until curl -fsS "http://box_X:8080/health" | grep -q '"status"[[:space:]]*:[[:space:]]*"healthy"'; do sleep 2; done && \
  deploy/drain.sh ready box_X

Exit codes:
  0  success / empty
  1  wait-empty timeout; stdout contains the remaining scur count
  2  backend or server not found
  3  socket unreachable / permission denied / missing local prerequisite
  4  bad arguments

Notes:
  Uses the Runtime API command "show stat" exactly; do not append ";csv".
  HAProxy mode tcp cannot react to WebSocket close code 1013. Backpressure is
  server-side NEMOTRON_ADMISSION_MAX_BACKLOG; this script is operator drain.
  The operator needs haproxy-group access to the mode-660 socket, sudo, or an
  explicit stats socket user/group matching the operator. socat is required.

Related:
  deploy/nemotron-asr.service
  deploy/gen_haproxy.py
  deploy/RUNBOOK.md
  deploy/DEPLOYMENT.md
USAGE
}

bad_args() {
  printf '[drain] bad arguments\n\n' >&2
  usage >&2
  exit 4
}

validate_box() {
  local box=$1

  if [[ ! "$box" =~ ^[A-Za-z0-9_-]+$ ]]; then
    printf '[drain] invalid box name %q; expected ^[A-Za-z0-9_-]+$\n' "$box" >&2
    exit 4
  fi
}

validate_timeout() {
  local timeout=$1

  if [[ ! "$timeout" =~ ^[0-9]+$ ]]; then
    printf '[drain] timeout must be a non-negative integer number of seconds: %q\n' "$timeout" >&2
    exit 4
  fi
}

socket_path() {
  printf '%s\n' "${HAPROXY_SOCK:-$DEFAULT_HAPROXY_SOCK}"
}

runtime_api() {
  local command_text=$1
  local sock
  local output

  sock=$(socket_path)

  if [[ "$sock" == "/dev/stdin" ]]; then
    if [[ "$command_text" == "show stat" ]]; then
      cat
      return 0
    fi
    printf '[drain] HAPROXY_SOCK=/dev/stdin only supports show-stat parsing, not state changes\n' >&2
    return 3
  fi

  if ! command -v socat >/dev/null 2>&1; then
    printf '[drain] socat is required to contact HAProxy Runtime API socket %s\n' "$sock" >&2
    return 3
  fi

  output=$(printf '%s\n' "$command_text" | socat -T 5 stdio "UNIX-CONNECT:${sock}" 2>&1)
  if [[ $? -ne 0 ]]; then
    printf '[drain] cannot use HAProxy Runtime API socket %s: %s\n' "$sock" "$output" >&2
    return 3
  fi

  if [[ "$output" == *"Permission denied"* ]] || [[ "$output" == *"Operation not permitted"* ]]; then
    printf '[drain] HAProxy Runtime API socket %s denied the command: %s\n' "$sock" "$output" >&2
    return 3
  fi

  printf '%s\n' "$output"
}

parse_show_stat() {
  if ! command -v python3 >/dev/null 2>&1; then
    printf '[drain] python3 is required to parse HAProxy show stat CSV\n' >&2
    return 3
  fi

  python3 /dev/fd/3 "$@" 3<<'PY'
import csv
import sys

BACKEND = "asr_pool"
SERVER_EXCLUDE = {"", "BACKEND", "FRONTEND"}


def fail(message, code):
    print(f"[drain] {message}", file=sys.stderr)
    raise SystemExit(code)


def normalize_lines(raw):
    lines = []
    for line in raw.splitlines():
        if line.startswith("# "):
            line = line[2:]
        elif line.startswith("#"):
            line = line[1:].lstrip()
        if line:
            lines.append(line)
    return lines


def main():
    if len(sys.argv) < 2:
        fail("internal parser error: missing parser mode", 4)

    mode = sys.argv[1]
    if mode not in {"status", "scur"}:
        fail(f"internal parser error: unknown parser mode {mode!r}", 4)
    if mode == "scur" and len(sys.argv) != 3:
        fail("internal parser error: scur mode requires a server name", 4)

    raw = sys.stdin.read()
    lines = normalize_lines(raw)
    if not lines:
        fail("no HAProxy show stat CSV received", 2)

    reader = csv.DictReader(lines)
    fieldnames = reader.fieldnames or []
    required = ["pxname", "svname", "scur", "smax", "status"]
    missing = [name for name in required if name not in fieldnames]
    if missing:
        fail("show stat CSV missing required column(s): " + ", ".join(missing), 2)

    backend_rows = []
    server_rows = []
    for row in reader:
        if row.get("pxname") != BACKEND:
            continue
        backend_rows.append(row)
        if row.get("svname", "") not in SERVER_EXCLUDE:
            server_rows.append(row)

    if not backend_rows:
        fail(f"backend {BACKEND} not found in show stat", 2)

    if mode == "status":
        if not server_rows:
            fail(f"backend {BACKEND} has no server rows in show stat", 2)
        for row in server_rows:
            print(
                "\t".join(
                    [
                        row.get("svname", ""),
                        row.get("status", ""),
                        row.get("scur", ""),
                        row.get("smax", ""),
                    ]
                )
            )
        return

    box = sys.argv[2]
    for row in server_rows:
        if row.get("svname") != box:
            continue
        scur = row.get("scur", "")
        try:
            value = int(scur)
        except ValueError:
            fail(f"server {BACKEND}/{box} has non-integer scur={scur!r}", 2)
        print(value)
        return

    fail(f"server {BACKEND}/{box} not found in show stat", 2)


if __name__ == "__main__":
    main()
PY
}

show_stat_csv() {
  local csv

  csv=$(runtime_api "show stat")
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    return "$rc"
  fi

  printf '%s\n' "$csv"
}

cmd_status() {
  local csv
  local rc

  csv=$(show_stat_csv)
  rc=$?
  if [[ $rc -ne 0 ]]; then
    exit "$rc"
  fi

  printf '%s\n' "$csv" | parse_show_stat status
  exit "$?"
}

current_scur() {
  local box=$1
  local csv
  local rc

  csv=$(show_stat_csv)
  rc=$?
  if [[ $rc -ne 0 ]]; then
    return "$rc"
  fi

  printf '%s\n' "$csv" | parse_show_stat scur "$box"
}

cmd_wait_empty() {
  local box=$1
  local timeout=$2
  local deadline=$((SECONDS + timeout))
  local sessions
  local rc

  while true; do
    sessions=$(current_scur "$box")
    rc=$?
    if [[ $rc -ne 0 ]]; then
      exit "$rc"
    fi

    if [[ "$sessions" -eq 0 ]]; then
      printf '[drain] %s/%s is empty\n' "$BACKEND" "$box"
      exit 0
    fi

    if (( SECONDS >= deadline )); then
      printf '%s\n' "$sessions"
      printf '[drain] timeout waiting for %s/%s: %s sessions remaining\n' "$BACKEND" "$box" "$sessions" >&2
      exit 1
    fi

    sleep "$POLL_SECONDS"
  done
}

response_means_missing_server() {
  local response=$1

  [[ "$response" == *"No such server"* ]] ||
    [[ "$response" == *"No such backend"* ]] ||
    [[ "$response" == *"No such proxy"* ]] ||
    [[ "$response" == *"not found"* ]] ||
    [[ "$response" == *"does not exist"* ]] ||
    [[ "$response" == *"doesn't exist"* ]]
}

cmd_set_state() {
  local state=$1
  local box=$2
  local response
  local rc

  response=$(runtime_api "set server ${BACKEND}/${box} state ${state}")
  rc=$?
  if [[ $rc -ne 0 ]]; then
    exit "$rc"
  fi

  if response_means_missing_server "$response"; then
    if [[ -n "$response" ]]; then
      printf '%s\n' "$response" >&2
    fi
    printf '[drain] server %s/%s not found\n' "$BACKEND" "$box" >&2
    exit 2
  fi

  if [[ -n "$response" ]]; then
    printf '%s\n' "$response"
  fi
  printf '[drain] %s/%s state %s requested\n' "$BACKEND" "$box" "$state"
}

main() {
  if [[ $# -eq 0 ]]; then
    bad_args
  fi

  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    drain)
      [[ $# -eq 2 ]] || bad_args
      validate_box "$2"
      cmd_set_state "drain" "$2"
      ;;
    ready)
      [[ $# -eq 2 ]] || bad_args
      validate_box "$2"
      cmd_set_state "ready" "$2"
      ;;
    status)
      [[ $# -eq 1 ]] || bad_args
      cmd_status
      ;;
    wait-empty)
      [[ $# -eq 2 || $# -eq 3 ]] || bad_args
      validate_box "$2"
      local timeout="${3:-300}"
      validate_timeout "$timeout"
      cmd_wait_empty "$2" "$timeout"
      ;;
    *)
      bad_args
      ;;
  esac
}

main "$@"
