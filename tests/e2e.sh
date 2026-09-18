#!/usr/bin/env bash
# End-to-end REST check: starts pcd_server on a free port, posts the triage
# fixture twice and asserts a schema-cache miss followed by a hit.
# Requires PCD_TEST_GGUF (or models/Qwen3.5-0.8B-Q8_0.gguf), jq and curl.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
server="${PCD_SERVER_BIN:-$root/build/pcd_server}"
model="${PCD_TEST_GGUF:-$root/models/Qwen3.5-0.8B-Q8_0.gguf}"
fixture="$root/tests/data/triage-request.json"
work="$(mktemp -d)"

[[ -x "$server" ]] || { echo "server binary not found: $server (build pcd_server first)" >&2; exit 1; }
[[ -f "$model" ]] || { echo "model not found: $model" >&2; exit 1; }
[[ -f "$fixture" ]] || { echo "fixture not found: $fixture" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }

port="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
"$server" --model "$model" --port "$port" >"$work/server.log" 2>&1 &
pid=$!
cleanup() {
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    rm -rf "$work"
}
trap cleanup EXIT

for _ in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:$port/health" >"$work/health.json" 2>/dev/null; then
        break
    fi
    sleep 0.2
done
test "$(jq -r '.ready' "$work/health.json")" = "true" || { echo "server never became ready"; cat "$work/server.log"; exit 1; }

post() {
    curl -sf -X POST "http://127.0.0.1:$port/v1/pcd/decode" -H 'Content-Type: application/json' --data-binary "@$fixture"
}
post >"$work/first.json"
post >"$work/second.json"

test "$(jq -r '.metrics.schemaCacheStatus' "$work/first.json")" = "miss"
test "$(jq -r '.metrics.schemaCacheStatus' "$work/second.json")" = "hit"
jq -e '.values.priority and (.values.fraudulent | type == "boolean")' "$work/second.json" >/dev/null
jq -e '.values.tier | startswith("TIER_")' "$work/second.json" >/dev/null
test "$(jq -c '.values' "$work/first.json")" = "$(jq -c '.values' "$work/second.json")"

# Validation and model errors keep the stable envelope.
code="$(curl -s -o "$work/err.json" -w '%{http_code}' -X POST "http://127.0.0.1:$port/v1/pcd/decode" -H 'Content-Type: application/json' -d '{"context":"x","fields":[]}')"
test "$code" = "400"
test "$(jq -r '.code' "$work/err.json")" = "invalid_request"
code="$(curl -s -o "$work/err.json" -w '%{http_code}' -X POST "http://127.0.0.1:$port/v1/models/select" -H 'Content-Type: application/json' -d '{"id":"nope.gguf"}')"
test "$code" = "404"

echo "e2e ok: cold $(jq -r '.metrics.elapsedMs' "$work/first.json") ms, cached $(jq -r '.metrics.elapsedMs' "$work/second.json") ms, values $(jq -c '.values' "$work/second.json")"
