# Native PCD Server Design

**Date:** 2026-09-18

## Goal

Build a standalone C++20 REST server that embeds a pinned llama.cpp source dependency and exposes Parallel Constrained Decoding (PCD) to LangChain4j and other clients without requiring Java FFM.

## Scope

The first version supports structured extraction where every field is either a boolean or a bounded string enum. It loads one active GGUF model at a time, discovers additional models placed in `models/`, and switches models through REST. It does not implement unconstrained strings, numbers, arrays, arbitrary JSON Schema, multi-model concurrent residency, authentication, or distributed scheduling.

The default model is `Qwen3.5-0.8B-Q8_0.gguf` from `ggml-org/Qwen3.5-0.8B-GGUF`. The model is downloaded by a script and is not committed.

## Architecture

The project lives in `/Users/stephan/projects/pcdServer` and uses CMake with pinned source dependencies:

- llama.cpp for model loading, tokenization, Metal/CPU execution, sequence memory, and logits
- cpp-httplib for HTTP transport
- nlohmann/json for request and response JSON
- Catch2 for unit and integration tests

The server contains three layers:

1. `PcdEngine`: owns the active llama model and context and implements PCD.
2. `ModelCatalog` and `SchemaCache`: discover GGUF models and maintain compiled schemas plus immutable prefix checkpoints.
3. `HttpServer`: validates transport concerns, calls the engine, and maps domain failures to HTTP responses.

Inference logic does not depend on HTTP types. The REST layer does not call llama.cpp directly.

The initial source structure is:

```text
pcdServer/
├── CMakeLists.txt
├── cmake/
├── include/pcd/
├── src/
│   ├── engine/
│   ├── http/
│   └── main.cpp
├── tests/
├── scripts/download-default-model.sh
├── models/.gitkeep
└── README.md
```

## Model lifecycle

At startup the server chooses the model in this order:

1. `--model PATH`
2. `PCD_GGUF`
3. `models/Qwen3.5-0.8B-Q8_0.gguf`

`GET /v1/models` rescans the configured models directory so a newly copied `.gguf` becomes selectable without restarting. `POST /v1/models/select` loads and validates a replacement engine before acquiring the short critical section that swaps it into service. If loading fails, the existing active engine remains available. A successful switch invalidates all compiled schemas and checkpoints belonging to the previous model.

The first implementation keeps one active model/context. Decode operations against that context are serialized with a mutex because llama sequence state is mutable.

## REST API

### Health

`GET /health` returns server readiness, active model, backend, and version information.

### Models

`GET /v1/models` returns discovered GGUF identifiers, sizes, architectures, and which model is active.

`POST /v1/models/select` accepts:

```json
{ "id": "Qwen3.5-0.8B-Q8_0.gguf" }
```

### Decode

`POST /v1/pcd/decode` accepts fields as an array so evaluation order is explicit:

```json
{
  "context": "Customer requested an unusual transfer...",
  "fields": [
    {
      "name": "priority",
      "description": "Required handling priority",
      "choices": ["LOW", "MEDIUM", "HIGH"]
    },
    {
      "name": "fraudulent",
      "description": "Whether fraud is suspected",
      "choices": [false, true]
    }
  ]
}
```

All choices in a field must have the same JSON scalar type. Boolean fields must contain exactly `false` and `true`. String fields must contain at least two distinct, non-empty values.

The response contains programmatically assembled values, full bounded-choice probability maps, and diagnostic metadata:

```json
{
  "model": "Qwen3.5-0.8B-Q8_0.gguf",
  "values": {
    "priority": "HIGH",
    "fraudulent": true
  },
  "fields": [
    {
      "name": "priority",
      "value": "HIGH",
      "probability": 0.85,
      "probabilities": {
        "LOW": 0.02,
        "MEDIUM": 0.13,
        "HIGH": 0.85
      },
      "levels": 1
    }
  ],
  "metrics": {
    "elapsedMs": 108.4,
    "forwardPasses": 2,
    "schemaCacheStatus": "hit",
    "phasesMs": {
      "tokenize": 0.4,
      "restoreOrPrefill": 42.0,
      "broadcast": 0.2,
      "suffix": 64.9,
      "tree": 0.9
    }
  }
}
```

## PCD data flow

1. Validate the request and canonicalize the ordered schema.
2. Compute a schema key from model identity, embedded chat template, prompt version, and canonical schema.
3. Compile per-field metadata once: JSON suffix tokens, common string prefixes, choice token trees, and legal closing tokens.
4. Render the model chat template with a unique marker as user content. Split the rendered text around that marker into a cacheable schema prefix and the suffix that closes the user turn and opens the assistant turn.
5. On a cache miss, clear model memory, decode only the schema prefix into sequence 0, synchronize, and save its complete sequence state.
6. On a cache hit, clear model memory and restore the complete sequence-0 checkpoint.
7. Tokenize and decode the dynamic context plus chat-template suffix into sequence 0.
8. Copy the completed sequence-0 prefix to one sequence per field using llama.cpp sequence-copy operations.
9. Decode every field's JSON suffix in one batch, requesting logits only at each suffix's final token.
10. Apply a softmax over only the allowed next tokens. Choices sharing the winning token remain live.
11. Resolve shared candidate prefixes level by level, using one batched decode per level across all fields that remain ambiguous.
12. Assemble JSON from the winners without asking the model to generate JSON syntax.

## Schema-prefix checkpoint safety

Qwen3.5 uses hybrid/recurrent state. Partial `llama_memory_seq_rm` rollback is not assumed to work. The server uses full host-side sequence checkpoints through:

- `llama_state_seq_get_size_ext`
- `llama_state_seq_get_data_ext`
- `llama_state_seq_set_data_ext`

Checkpoints use `LLAMA_STATE_SEQ_FLAGS_NONE`, covering attention KV and recurrent state. The server verifies that save and restore byte counts exactly match the expected checkpoint size. Checkpoint buffers are immutable after publication.

If a restore fails, the request discards that checkpoint, clears model memory, recomputes the schema prefix cold, and continues. The response reports `schemaCacheStatus: "fallback"`. Repeated failures disable caching for that schema until the model is reloaded. No inference proceeds after a failed partial trim or incomplete state restore.

The cache is bounded by both entry count and total checkpoint bytes. Least-recently-used entries are evicted. Cache keys include the active model fingerprint, so checkpoints can never cross model instances.

## Concurrency

HTTP request parsing and model discovery may run concurrently. Each active inference context has one mutex covering memory clear/restore, decode, sequence broadcast, collision resolution, and result extraction. Model loading happens outside that mutex; only engine replacement requires exclusive ownership. Requests already executing finish on the old shared engine instance, while later requests use the replacement.

This design favors correctness and predictable latency over concurrent mutation of one llama context. Multiple contexts or a scheduler may be added only after the single-context implementation is correct and benchmarked.

## Validation and limits

Configurable limits protect native allocations and batch construction:

- Maximum context bytes
- Maximum number of fields
- Maximum choices per field
- Maximum total candidate tokens
- Maximum batch tokens
- Maximum checkpoint entries and bytes

Duplicate field names, mixed-type choices, empty names or choices, unsupported value types, and schemas that exceed limits are rejected before inference.

## Error handling

- `400 Bad Request`: malformed JSON, invalid schema, duplicate names, or exceeded request limits
- `404 Not Found`: unknown model identifier
- `409 Conflict`: model lifecycle conflict that cannot safely wait
- `422 Unprocessable Entity`: chat-template split or candidate tokenization cannot be performed safely
- `500 Internal Server Error`: native inference fails after cold fallback
- `503 Service Unavailable`: no model is loaded

Error responses use a stable envelope with `code`, `message`, and optional `details`. Native return codes and model identifiers are logged server-side, while clients receive actionable messages without filesystem disclosure.

## Testing

Unit tests without a model cover:

- Request and schema validation
- Canonical schema keys
- Common-prefix extraction
- Token-tree traversal with synthetic token IDs and logits
- Probability normalization
- LRU entry-count and byte-size eviction
- REST serialization and HTTP error mapping

Native integration tests run when `PCD_TEST_GGUF` names a model and cover:

- Cold and cached runs select identical values
- Probability maps match within a documented floating-point tolerance
- Repeated cached requests do not accumulate context
- Alternating schemas do not leak sequence state
- Alternating dynamic contexts restore the same schema prefix correctly
- Forced restore failure succeeds through cold fallback
- Failed model switching preserves the active model
- Multi-token shared-prefix choices resolve correctly
- Qwen3.5-0.8B performs repeated cache hits without invalid-position decode errors

HTTP integration tests start the server on an ephemeral port and exercise health, model listing, selection, valid decode, validation failures, and graceful shutdown.

A separate benchmark executable reports cold, checkpoint-save, checkpoint-restore, dynamic-context, broadcast, suffix, and tree timings. It compares cold and cached medians after warm-up. Ordinary CI does not contain fragile timing assertions.

## Build and distribution

CMake pins every fetched dependency to an immutable tag or commit. The default build enables the platform backend selected by llama.cpp; Metal is enabled on Apple Silicon. The README documents reproducible macOS and Linux builds, model download, startup flags, REST examples, integration tests, and benchmarks.

The produced executable remains dynamically independent of Java. LangChain4j integration is a separate HTTP client/provider built against the stable `/v1/pcd/decode` contract.

## Acceptance criteria

1. The server builds from a clean checkout with its pinned dependencies.
2. The default download script obtains Qwen3.5-0.8B-Q8_0 and verifies its published SHA-256.
3. The REST server lists and selects models while preserving the current model on failed replacement.
4. PCD returns valid bounded values and normalized probabilities.
5. Qwen3.5 cached results match cold results and survive alternating schemas and contexts.
6. Any checkpoint failure performs a cold fallback rather than continuing with stale state.
7. A benchmark demonstrates cache status and phase timings without relying on Java or FFM.
