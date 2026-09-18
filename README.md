# pcd_server — native Parallel Constrained Decoding server

A standalone C++20 REST server that embeds a pinned [llama.cpp](https://github.com/ggml-org/llama.cpp) and performs **Parallel Constrained Decoding (PCD)**: given a text and a list of bounded fields, it fills every field with exactly one allowed value and returns the full probability map per field. The JSON is assembled programmatically — the model never generates JSON syntax, only scores the allowed candidate tokens.

Scope of this version: **booleans and bounded string enums only**. No free strings, numbers, arrays, arbitrary JSON Schema, authentication or multi-model residency.

## Prerequisites

**macOS (Apple Silicon, Metal)**

- Xcode Command Line Tools with an accepted license (`sudo xcodebuild -license accept`)
- CMake ≥ 3.25 (`brew install cmake`)
- `curl` and `shasum` (bundled), `jq` for the end-to-end script (`brew install jq`)
- A Metal-capable Apple Silicon Mac. llama.cpp offloads every layer to the GPU; the numbers below come from an M-series machine.

**Linux**

- GCC ≥ 12 or Clang ≥ 15, CMake ≥ 3.25, `git`, `curl`, `jq`
- Runs on CPU by default (llama.cpp picks the platform backend; pass the usual `GGML_CUDA=ON` etc. through CMake if you want a GPU backend). Expect cold decodes in the hundreds of milliseconds on CPU for the 0.8B model.

## Getting the default model

```bash
./scripts/download-default-model.sh
```

Downloads `Qwen3.5-0.8B-Q8_0.gguf` from `ggml-org/Qwen3.5-0.8B-GGUF` into `models/` and verifies its SHA-256 (`37ae482d…7814f`). GGUF files are git-ignored.

## Build, test, run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/pcd_server --model models/Qwen3.5-0.8B-Q8_0.gguf
```

Dependencies are fetched by CMake at pinned tags: llama.cpp `v0.4.1`, cpp-httplib `v0.56.0`, nlohmann/json `v3.12.0`, Catch2 `v3.15.2`. The first configure/build compiles llama.cpp and takes a few minutes.

### Server flags

| Flag | Default | Meaning |
|---|---|---|
| `--bind ADDRESS` | `127.0.0.1` | Listen address |
| `--port PORT` | `8080` | Listen port |
| `--models-dir PATH` | `models` | Directory scanned for selectable `.gguf` files |
| `--model PATH` | – | Model to load at startup (overrides `PCD_GGUF`) |
| `--cache-entries N` | `32` | Max cached schema checkpoints |
| `--cache-bytes BYTES` | `536870912` | Max total checkpoint bytes (512 MiB) |

Startup model precedence: `--model`, then `PCD_GGUF`, then `models/Qwen3.5-0.8B-Q8_0.gguf`. If none exists the server starts without an active model and decode returns `503` until one is selected. `SIGINT`/`SIGTERM` stop the listener and release the model.

## REST API (`v1`)

All responses are `application/json`. Errors use a stable envelope `{"code": "...", "message": "..."}`.

### `GET /health`

```bash
curl -s http://127.0.0.1:8080/health
# {"apiVersion":"v1","backend":"MTL0","description":"qwen35 0.8B Q8_0","model":"Qwen3.5-0.8B-Q8_0.gguf","ready":true,"status":"ok"}
```

### `GET /v1/models`

Rescans `--models-dir` on every call, so a freshly copied `.gguf` becomes selectable without a restart.

```bash
curl -s http://127.0.0.1:8080/v1/models
# {"active":"Qwen3.5-0.8B-Q8_0.gguf","activeModel":{...},"models":[{"active":true,"bytes":833592096,"id":"Qwen3.5-0.8B-Q8_0.gguf"}]}
```

### `POST /v1/models/select`

Loads and validates the replacement completely before swapping it in. If loading fails the current model stays active. Only identifiers from `/v1/models` are accepted; client input is never used as a path.

```bash
curl -s -X POST http://127.0.0.1:8080/v1/models/select \
  -H 'Content-Type: application/json' \
  -d '{"id":"Qwen3.5-0.8B-Q8_0.gguf"}'
```

### `POST /v1/pcd/decode`

Fields are an ordered array. Choices in one field must all be strings (≥ 2 distinct, non-empty) or exactly `[false, true]`.

```bash
curl -s -X POST http://127.0.0.1:8080/v1/pcd/decode \
  -H 'Content-Type: application/json' \
  --data-binary @tests/data/triage-request.json
```

Response:

```json
{
  "model": "Qwen3.5-0.8B-Q8_0.gguf",
  "values": { "priority": "HIGH", "fraudulent": true, "tier": "TIER_3_HIGH" },
  "fields": [
    {
      "name": "priority",
      "value": "HIGH",
      "probability": 0.76,
      "probabilities": { "LOW": 0.03, "MEDIUM": 0.21, "HIGH": 0.76 },
      "levels": 1
    }
  ],
  "metrics": {
    "elapsedMs": 39.8,
    "forwardPasses": 2,
    "schemaCacheStatus": "hit",
    "checkpointBytes": 22332068,
    "phasesMs": { "tokenize": 0.2, "restoreOrPrefill": 5.8, "dynamicContext": 13.5, "broadcast": 0.04, "suffix": 20.8, "tree": 0.0 }
  }
}
```

- `values` — the assembled result, one entry per field in request order.
- `fields[].probabilities` — softmax over the *allowed* candidates only; sums to 1.
- `fields[].levels` — how many collision-tree levels were needed (1 when every choice starts with a distinct token).
- `metrics.schemaCacheStatus` — `miss` (schema prefix computed and checkpointed), `hit` (checkpoint restored), `fallback` (a cached checkpoint failed to restore; the prefix was recomputed cold and the bad checkpoint dropped).
- `metrics.forwardPasses` — number of `llama_decode` calls.

### Status codes

| Code | When |
|---|---|
| `400` | Malformed JSON, invalid schema, duplicate names, request exceeds limits or context window |
| `404` | Unknown model identifier, unknown route |
| `422` | The chat template or candidate tokenization cannot be compiled safely |
| `500` | Native inference failed |
| `503` | No model is loaded |

Limits: 1–63 fields, 2–256 choices per field, 64 KiB context, 4096 total candidate tokens, 1 MiB request body. The whole request (schema prefix + context + every field suffix and candidate path) must fit the 8192-token context window.

## How a decode works

1. Validate the request; build a canonical schema key from model fingerprint, chat template, prompt format version and ordered schema.
2. **Schema prefix** (cacheable): render the model's chat template with a system prompt that lists every field, description and allowed value, up to a marker where the user text goes. On a miss, decode it into sequence 0 and save a **full sequence checkpoint**; on a hit, clear memory and restore that checkpoint.
3. **Dynamic context**: decode the user text, the user-turn close, the assistant header (plus Qwen3.5's empty `<think></think>` block) and `{\n`.
4. **Broadcast**: copy sequence 0 to one sequence per field.
5. **Suffix**: decode every field's `  "name": "` (or `  "name":` for booleans) in one batch, requesting logits only on each final token.
6. **Score**: softmax over the allowed next tokens only. Choices that share the winning token stay live and are resolved level by level in batched decodes (max 24), e.g. `BILLING` vs `BILLING_DISPUTE`.
7. Assemble the JSON from the winners.

### Why full checkpoints instead of partial rollback

Qwen3.5 is a hybrid model with recurrent (Gated DeltaNet) layers alongside attention. Recurrent state cannot be partially rewound: `llama_memory_seq_rm` on a sub-range of positions returns `false` for such memory, and continuing after a failed trim would decode against a stale state. The server therefore never trims. It clears memory and restores a **complete** host-side checkpoint (`llama_state_seq_*_ext` with `LLAMA_STATE_SEQ_FLAGS_NONE`, covering the KV cache and the recurrent state) taken right after the schema prefix. Save and restore byte counts are verified exactly, the restored position is checked, and any mismatch falls back to a cold prefill. Two consecutive restore failures disable caching for that schema until the model is reloaded. Checkpoints are immutable after publication and the cache is bounded by entries and bytes (LRU). One 3-field schema checkpoint on Qwen3.5-0.8B is ≈ 22 MB and restores in ≈ 6 ms versus ≈ 36 ms for a cold prefill.

## Native integration tests

Unit tests run without a model. Tests tagged `[native]` are skipped unless `PCD_TEST_GGUF` points at a GGUF:

```bash
PCD_TEST_GGUF=models/Qwen3.5-0.8B-Q8_0.gguf build/pcd_tests "[native]"
PCD_TEST_GGUF=models/Qwen3.5-0.8B-Q8_0.gguf ./tests/e2e.sh   # REST round-trip: miss then hit
```

They cover checkpoint save/restore, cold-vs-cached equivalence, alternating contexts and schemas, forced restore failure → fallback, 20 consecutive cache hits, shared-prefix collision resolution, failed model switches, and the HTTP happy path.

## Benchmark

```bash
./build/pcd_bench --model models/Qwen3.5-0.8B-Q8_0.gguf --request tests/data/triage-request.json --runs 5
```

Performs one warm-up, then for each run evicts the schema entry, measures a cold decode and a cached decode, and prints medians per phase, checkpoint bytes, cache statuses and the selected values. No timing thresholds are enforced in CTest.

## Concurrency

HTTP handling and model discovery run concurrently. All inference on a model context is serialized by one mutex (llama sequence memory is mutable). Model replacement loads the new engine outside that mutex and only briefly takes a write lock to swap; requests already running finish on the old engine, which is released when the last of them completes.
