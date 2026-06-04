#!/usr/bin/env bash
# CI/build guard: every torch::jit::load in cpp/ must route through the ONE
# serialized wrapper load_jit_serialized() (cpp/lib/runtime_io/jit_load.cpp) to
# avoid the torch global-registry concurrent-load livelock. Fails (exit 1) if a
# raw load reappears anywhere else, OR if the wrapper file stops containing
# exactly one raw load (so a SECOND raw load added to the wrapper file is also
# caught). Whitespace-tolerant around `::` so `torch :: jit :: load` is caught.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ALLOWLIST="cpp/lib/runtime_io/jit_load.cpp"
EXPECTED_WRAPPER_HITS=1
# Whitespace-tolerant: torch :: jit :: load  and the bare  jit :: load(  form.
PATTERN='torch[[:space:]]*::[[:space:]]*jit[[:space:]]*::[[:space:]]*load|(^|[^[:alnum:]_:])jit[[:space:]]*::[[:space:]]*load[[:space:]]*\('
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

if command -v rg >/dev/null 2>&1; then
  set +e
  rg -n --no-heading -e "$PATTERN" cpp > "$TMP"
  SEARCH_STATUS=$?
  set -e
else
  set +e
  grep -RInE "$PATTERN" cpp > "$TMP"
  SEARCH_STATUS=$?
  set -e
fi

# grep/rg exit 1 == "no matches" (fine); >1 == real error.
if [[ "$SEARCH_STATUS" -gt 1 ]]; then
  exit "$SEARCH_STATUS"
fi

FAIL=0
WRAPPER_HITS=0
while IFS= read -r match; do
  [[ -z "$match" ]] && continue
  file="${match%%:*}"
  if [[ "$file" == "$ALLOWLIST" ]]; then
    WRAPPER_HITS=$((WRAPPER_HITS + 1))
  else
    if [[ "$FAIL" -eq 0 ]]; then
      echo "raw JIT load call(s) found outside $ALLOWLIST (must route through load_jit_serialized):"
    fi
    echo "$match"
    FAIL=1
  fi
done < "$TMP"

if [[ "$FAIL" -ne 0 ]]; then
  exit 1
fi

if [[ "$WRAPPER_HITS" -ne "$EXPECTED_WRAPPER_HITS" ]]; then
  echo "guard expected exactly $EXPECTED_WRAPPER_HITS raw load in $ALLOWLIST but found $WRAPPER_HITS;"
  echo "the wrapper must contain exactly the one serialized torch::jit::load."
  exit 1
fi

echo "OK: exactly $EXPECTED_WRAPPER_HITS serialized raw JIT load (in $ALLOWLIST); no raw loads elsewhere."
