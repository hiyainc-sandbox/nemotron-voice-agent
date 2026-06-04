#!/usr/bin/env bash
# Laptop-safe local smoke test for the Phase 1 Nemotron ASR deploy artifacts.
#
# Run:
#   deploy/smoke_local.sh
#
# This script takes no arguments and runs five independent checks:
#   1. ec2-bench/local_lb.py leastconn spread plus shed-on-full behavior using
#      two deploy/_smoke_backend.py echo backends and five held TCP clients.
#   2. deploy/gen_haproxy.py FD-vs-maxconn invariant: parses the rendered cfg
#      for both production and --local-test modes and asserts
#      ulimit-n >= 2 * global maxconn + 42 (the HAProxy 2.x runtime constraint
#      that haproxy -c does NOT check; caught live on g6e.4xlarge 2026-05-28).
#   3. deploy/gen_haproxy.py syntax under "haproxy -c" for a two-box config.
#   4. Optional real-HAProxy health-check behavior using deploy/_smoke_backend.py
#      HTTP mode on 127.0.0.1:8080 and deploy/_smoke_haproxy_check.py.
#   5. deploy/drain.sh fixture parsing against deploy/_drain_fixtures/*.csv via
#      HAPROXY_SOCK=/dev/stdin.
#
# SKIP interpretation:
#   - Check 2 SKIPs with WARN when haproxy is not on PATH; the generated config
#     has not been validated by HAProxy's parser on this laptop.
#   - Check 3 SKIPs with WARN when haproxy is not on PATH, or when
#     127.0.0.1:8080 is already in use. The generator hardcodes backend server
#     lines to port 8080, so the stub must bind that port for this proof.
#   - SKIPs are not failures. The script exits 0 when every non-skipped check
#     PASSes, and exits nonzero if any check FAILs.
#
# Not covered:
#   - No real GPU, model load, CUDA, or production nemotron_speech.server.
#   - No actual WebSocket protocol semantics beyond TCP connection routing.
#   - No cloud security group, TLS, rolling deploy, or sustained load behavior.
#     Those belong in the cloud smoke in deploy/RUNBOOK.md step 5.
#
# Related artifacts:
#   deploy/gen_haproxy.py
#   ec2-bench/local_lb.py
#   deploy/drain.sh
#   deploy/_drain_fixtures/
#   deploy/RUNBOOK.md
#   deploy/DEPLOYMENT.md

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
FAILED=0
PASSED=0
SKIPPED=0

pass_check() {
  PASSED=$((PASSED + 1))
  printf 'PASS %s: %s\n' "$1" "$2"
}

skip_check() {
  SKIPPED=$((SKIPPED + 1))
  printf 'SKIP %s: %s\n' "$1" "$2"
}

fail_check() {
  FAILED=$((FAILED + 1))
  printf 'FAIL %s: %s\n' "$1" "$2"
}

alloc_port() {
  "$PYTHON_BIN" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

terminate_pid() {
  local pid=${1:-}

  if [[ -z "$pid" ]]; then
    return 0
  fi
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
    wait "$pid" >/dev/null 2>&1 || true
  fi
}

wait_for_pattern() {
  local file=$1
  local pattern=$2
  local timeout=$3
  local deadline=$((SECONDS + timeout))

  while (( SECONDS <= deadline )); do
    if [[ -f "$file" ]] && grep -q "$pattern" "$file"; then
      return 0
    fi
    sleep 0.05
  done
  return 1
}

file_excerpt() {
  local label=$1
  local file=$2

  printf '%s=' "$label"
  if [[ -f "$file" ]]; then
    sed -n '1,40p' "$file" | "$PYTHON_BIN" -c 'import sys; print(repr(sys.stdin.read()))'
  else
    printf '<missing>\n'
  fi
}

check_local_lb() {
  local check='local_lb.py leastconn spread + shed-on-full'
  local tmpdir
  local b1_pid=
  local b2_pid=
  local lb_pid=
  local p1
  local p2
  local front
  local client_output
  local rc

  tmpdir=$(mktemp -d)
  p1=$(alloc_port)
  p2=$(alloc_port)
  front=$(alloc_port)

  (
    cd "$ROOT" &&
      "$PYTHON_BIN" deploy/_smoke_backend.py --mode echo --port "$p1" >"$tmpdir/backend1.log" 2>"$tmpdir/backend1.err"
  ) &
  b1_pid=$!
  (
    cd "$ROOT" &&
      "$PYTHON_BIN" deploy/_smoke_backend.py --mode echo --port "$p2" >"$tmpdir/backend2.log" 2>"$tmpdir/backend2.err"
  ) &
  b2_pid=$!

  if ! wait_for_pattern "$tmpdir/backend1.log" "listening" 5 ||
    ! wait_for_pattern "$tmpdir/backend2.log" "listening" 5; then
    fail_check "$check" "expected both echo backends to print listening within 5s; actual $(file_excerpt backend1.log "$tmpdir/backend1.log") $(file_excerpt backend1.err "$tmpdir/backend1.err") $(file_excerpt backend2.log "$tmpdir/backend2.log") $(file_excerpt backend2.err "$tmpdir/backend2.err")"
    terminate_pid "$lb_pid"
    terminate_pid "$b1_pid"
    terminate_pid "$b2_pid"
    rm -rf "$tmpdir"
    return
  fi

  (
    cd "$ROOT" &&
      "$PYTHON_BIN" ec2-bench/local_lb.py --front "$front" --backends "$p1,$p2" --maxconn 2 >"$tmpdir/lb.log" 2>"$tmpdir/lb.err"
  ) &
  lb_pid=$!

  if ! wait_for_pattern "$tmpdir/lb.log" "local_lb up" 5; then
    fail_check "$check" "expected local_lb.py to print ready within 5s; actual $(file_excerpt lb.log "$tmpdir/lb.log") $(file_excerpt lb.err "$tmpdir/lb.err")"
    terminate_pid "$lb_pid"
    terminate_pid "$b1_pid"
    terminate_pid "$b2_pid"
    rm -rf "$tmpdir"
    return
  fi

  client_output=$(
    "$PYTHON_BIN" - "$front" "$tmpdir/backend1.log" "$tmpdir/backend2.log" <<'PY'
import json
import socket
import sys
import time
from pathlib import Path

front = int(sys.argv[1])
log1 = Path(sys.argv[2])
log2 = Path(sys.argv[3])


def accepted_count(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.startswith("accepted "))
    except OSError:
        return -1


sockets = []
results = []
for idx in range(5):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    payload = f"smoke-client-{idx}\n".encode("ascii")
    try:
        sock.connect(("127.0.0.1", front))
        sock.sendall(payload)
        try:
            data = sock.recv(len(payload))
        except socket.timeout:
            data = b""
        if data == payload:
            results.append({"client": idx, "result": "echo"})
            sockets.append(sock)
        else:
            results.append({"client": idx, "result": "closed_or_no_echo", "data": data.decode("utf-8", errors="replace")})
            sock.close()
    except OSError as exc:
        results.append({"client": idx, "result": "connect_or_send_error", "error": str(exc)})
        sock.close()
    time.sleep(0.05)

deadline = time.monotonic() + 3.0
count1 = accepted_count(log1)
count2 = accepted_count(log2)
while time.monotonic() < deadline:
    count1 = accepted_count(log1)
    count2 = accepted_count(log2)
    if count1 == 2 and count2 == 2:
        break
    time.sleep(0.05)

echoed = sum(1 for item in results if item["result"] == "echo")
non_echo = len(results) - echoed
actual = {
    "echoed_clients": echoed,
    "shed_or_closed_clients": non_echo,
    "backend1_accepts": count1,
    "backend2_accepts": count2,
    "client_results": results,
}
expected = {
    "echoed_clients": 4,
    "shed_or_closed_clients": 1,
    "backend1_accepts": 2,
    "backend2_accepts": 2,
}
print("expected=" + json.dumps(expected, sort_keys=True))
print("actual=" + json.dumps(actual, sort_keys=True))

for sock in sockets:
    sock.close()

raise SystemExit(0 if all(actual[key] == value for key, value in expected.items()) else 1)
PY
  )
  rc=$?

  terminate_pid "$lb_pid"
  terminate_pid "$b1_pid"
  terminate_pid "$b2_pid"

  if [[ $rc -eq 0 ]]; then
    pass_check "$check" "2 accepted on each backend plus 1 overflow closed. local_lb.py SHEDS overflow while HAProxy QUEUES per timeout queue - these are NOT equivalent; this test asserts local_lb's documented behavior, not HAProxy's."
  else
    fail_check "$check" "leastconn/shed assertion mismatch. ${client_output//$'\n'/; } lb_log=$(sed -n '1,20p' "$tmpdir/lb.log" | "$PYTHON_BIN" -c 'import sys; print(repr(sys.stdin.read()))') lb_err=$(sed -n '1,20p' "$tmpdir/lb.err" | "$PYTHON_BIN" -c 'import sys; print(repr(sys.stdin.read()))') backend1_log=$(sed -n '1,20p' "$tmpdir/backend1.log" | "$PYTHON_BIN" -c 'import sys; print(repr(sys.stdin.read()))') backend2_log=$(sed -n '1,20p' "$tmpdir/backend2.log" | "$PYTHON_BIN" -c 'import sys; print(repr(sys.stdin.read()))')"
  fi

  rm -rf "$tmpdir"
}

check_generator_fd_invariant() {
  # HAProxy requires ulimit-n >= 2 * global maxconn + 42 (per HAProxy 2.x runtime).
  # haproxy -c -f validates SYNTAX, not this RUNTIME FD constraint, so the
  # generator can silently emit configs that pass -c but fail at startup with
  # "[ALERT] FD limit (X) too low for maxconn=Y/maxsock=Z" — caught live
  # 2026-05-28 on g6e.4xlarge. This check parses gen_haproxy.py output and
  # asserts the invariant for both production and --local-test modes.
  local check='Generator FD-vs-maxconn invariant'
  local tmpdir
  local rc

  tmpdir=$(mktemp -d)
  trap "rm -rf '$tmpdir'" RETURN

  local prod_rc local_rc
  (cd "$ROOT" && "$PYTHON_BIN" deploy/gen_haproxy.py --boxes 10.0.1.10,10.0.1.11 -o "$tmpdir/prod.cfg" >/dev/null 2>&1)
  prod_rc=$?
  (cd "$ROOT" && "$PYTHON_BIN" deploy/gen_haproxy.py --local-test --stats-socket "$tmpdir/local.sock" --boxes 127.0.0.1 -o "$tmpdir/local.cfg" >/dev/null 2>&1)
  local_rc=$?
  if [[ $prod_rc -ne 0 || $local_rc -ne 0 ]]; then
    fail_check "$check" "gen_haproxy.py failed: prod_rc=$prod_rc local_rc=$local_rc"
    return
  fi

  local result
  result=$("$PYTHON_BIN" - "$tmpdir/prod.cfg" "$tmpdir/local.cfg" <<'PY'
import re, sys
for path in sys.argv[1:]:
    txt = open(path).read()
    mc = int(re.search(r'^    maxconn (\d+)', txt, re.M).group(1))
    ul = int(re.search(r'^    ulimit-n (\d+)', txt, re.M).group(1))
    need = 2 * mc + 42
    ok = ul >= need
    label = path.rsplit('/', 1)[-1]
    print(f'{label}: maxconn={mc} ulimit-n={ul} need>={need} OK={ok}')
    if not ok:
        sys.exit(1)
PY
  )
  rc=$?
  if [[ $rc -eq 0 ]]; then
    pass_check "$check" "$(echo "$result" | tr '\n' '; ' | sed 's/; $//')"
  else
    fail_check "$check" "$result"
  fi
}

check_generator_syntax() {
  local check='Generator syntax (haproxy -c)'
  local tmpdir
  local cfg
  local output
  local rc

  if ! command -v haproxy >/dev/null 2>&1; then
    skip_check "$check" "WARN haproxy not found on PATH; install haproxy to validate generated config syntax with haproxy -c"
    return
  fi

  tmpdir=$(mktemp -d)
  cfg="$tmpdir/test.cfg"

  output=$(
    cd "$ROOT" &&
      "$PYTHON_BIN" deploy/gen_haproxy.py --boxes 10.0.1.10,10.0.1.11 --maxconn 20 -o "$cfg" 2>&1
  )
  rc=$?
  if [[ $rc -ne 0 ]]; then
    fail_check "$check" "expected gen_haproxy.py rc=0; actual rc=$rc output=$(printf '%s' "$output" | "$PYTHON_BIN" -c 'import sys; print(repr(sys.stdin.read()))')"
    rm -rf "$tmpdir"
    return
  fi

  output=$(haproxy -c -f "$cfg" 2>&1)
  rc=$?
  if [[ $rc -eq 0 ]]; then
    pass_check "$check" "haproxy -c accepted generated two-box config"
  else
    fail_check "$check" "expected haproxy -c rc=0; actual rc=$rc output=$(printf '%s' "$output" | "$PYTHON_BIN" -c 'import sys; print(repr(sys.stdin.read()))') cfg=$(sed -n '1,160p' "$cfg" | "$PYTHON_BIN" -c 'import sys; print(repr(sys.stdin.read()))')"
  fi

  rm -rf "$tmpdir"
}

check_haproxy_health() {
  local output
  local rc

  output=$(cd "$ROOT" && "$PYTHON_BIN" deploy/_smoke_haproxy_check.py 2>&1)
  rc=$?
  printf '%s\n' "$output"
  case "$rc" in
    0)
      PASSED=$((PASSED + 1))
      ;;
    77)
      SKIPPED=$((SKIPPED + 1))
      ;;
    *)
      FAILED=$((FAILED + 1))
      ;;
  esac
}

run_drain_cmd() {
  local fixture=$1
  shift

  (
    cd "$ROOT" &&
      HAPROXY_SOCK=/dev/stdin deploy/drain.sh "$@" <"$fixture"
  )
}

check_drain_fixtures() {
  local check='drain.sh fixture parsing'
  local fixture_dir="$ROOT/deploy/_drain_fixtures"
  local drained="$fixture_dir/server-drained-empty.csv"
  local traffic="$fixture_dir/server-up-with-traffic.csv"
  local missing="$fixture_dir/server-missing.csv"
  local failures=()
  local stdout_file
  local stderr_file
  local rc
  local stdout
  local stderr

  for fixture in "$drained" "$traffic" "$missing"; do
    if [[ ! -f "$fixture" ]]; then
      failures+=("missing fixture $fixture")
    fi
  done
  if [[ ${#failures[@]} -gt 0 ]]; then
    fail_check "$check" "expected all drain fixtures to exist; actual ${failures[*]}"
    return
  fi

  stdout_file=$(mktemp)
  stderr_file=$(mktemp)

  run_drain_cmd "$drained" wait-empty box_test_a 0 >"$stdout_file" 2>"$stderr_file"
  rc=$?
  stdout=$(cat "$stdout_file")
  stderr=$(cat "$stderr_file")
  if [[ $rc -ne 0 ]]; then
    failures+=("drained-empty wait-empty expected rc=0; actual rc=$rc stdout=$(printf '%s' "$stdout" | "$PYTHON_BIN" -c 'import sys; print(repr(sys.stdin.read()))') stderr=$(printf '%s' "$stderr" | "$PYTHON_BIN" -c 'import sys; print(repr(sys.stdin.read()))')")
  fi

  run_drain_cmd "$traffic" wait-empty box_test_a 0 >"$stdout_file" 2>"$stderr_file"
  rc=$?
  stdout=$(cat "$stdout_file")
  stderr=$(cat "$stderr_file")
  if [[ $rc -ne 1 || "$stdout" != "3" ]]; then
    failures+=("up-with-traffic wait-empty expected rc=1 stdout=3; actual rc=$rc stdout=$(printf '%s' "$stdout" | "$PYTHON_BIN" -c 'import sys; print(repr(sys.stdin.read()))') stderr=$(printf '%s' "$stderr" | "$PYTHON_BIN" -c 'import sys; print(repr(sys.stdin.read()))')")
  fi

  run_drain_cmd "$missing" wait-empty box_test_a 0 >"$stdout_file" 2>"$stderr_file"
  rc=$?
  stdout=$(cat "$stdout_file")
  stderr=$(cat "$stderr_file")
  if [[ $rc -ne 2 ]]; then
    failures+=("missing wait-empty expected rc=2; actual rc=$rc stdout=$(printf '%s' "$stdout" | "$PYTHON_BIN" -c 'import sys; print(repr(sys.stdin.read()))') stderr=$(printf '%s' "$stderr" | "$PYTHON_BIN" -c 'import sys; print(repr(sys.stdin.read()))')")
  fi

  run_drain_cmd "$drained" status >"$stdout_file" 2>"$stderr_file"
  rc=$?
  stdout=$(cat "$stdout_file")
  stderr=$(cat "$stderr_file")
  if [[ $rc -ne 0 || "$stdout" != *"box_test_a"* || "$stdout" != *"DRAIN"* ]]; then
    failures+=("drained-empty status expected rc=0 and output containing box_test_a and DRAIN; actual rc=$rc stdout=$(printf '%s' "$stdout" | "$PYTHON_BIN" -c 'import sys; print(repr(sys.stdin.read()))') stderr=$(printf '%s' "$stderr" | "$PYTHON_BIN" -c 'import sys; print(repr(sys.stdin.read()))')")
  fi

  rm -f "$stdout_file" "$stderr_file"

  if [[ ${#failures[@]} -eq 0 ]]; then
    pass_check "$check" "wait-empty and status matched expected fixture rc/output for drained-empty, up-with-traffic, and missing-server cases"
  else
    fail_check "$check" "$(printf '%s | ' "${failures[@]}")"
  fi
}

main() {
  if [[ $# -ne 0 ]]; then
    printf 'Usage: deploy/smoke_local.sh\n' >&2
    exit 2
  fi

  cd "$ROOT" || exit 1

  check_local_lb
  check_generator_fd_invariant
  check_generator_syntax
  check_haproxy_health
  check_drain_fixtures

  printf 'SUMMARY pass=%d skip=%d fail=%d\n' "$PASSED" "$SKIPPED" "$FAILED"
  if [[ $FAILED -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
