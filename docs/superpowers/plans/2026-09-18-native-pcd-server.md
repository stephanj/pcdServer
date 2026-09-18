# Native PCD Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone C++20 REST server that embeds pinned llama.cpp v0.4.1, performs Parallel Constrained Decoding, and safely reuses full Qwen3.5 sequence checkpoints.

**Architecture:** A transport-independent `PcdEngine` owns one llama model/context and serializes inference. It compiles bounded boolean/enum schemas, caches immutable full sequence checkpoints, restores them before decoding dynamic context, broadcasts the completed prefix to field sequences, and scores only allowed candidate tokens. `ModelManager` transactionally swaps engines, while a thin cpp-httplib layer exposes health, model, and decode endpoints.

**Tech Stack:** C++20, CMake/CTest, llama.cpp v0.4.1, cpp-httplib v0.56.0, nlohmann/json v3.12.0, Catch2 v3.15.2, Metal on Apple Silicon.

---

## File map

- `CMakeLists.txt` — pinned dependencies, core/server/test/benchmark targets, platform options.
- `.gitignore` — build outputs and GGUF files.
- `include/pcd/types.hpp`, `src/types.cpp` — transport-independent requests, responses, validation, and JSON conversion.
- `include/pcd/schema.hpp`, `src/engine/schema.cpp` — canonical schema keys, prompts, common prefixes, and compiled token metadata.
- `include/pcd/scorer.hpp`, `src/engine/scorer.cpp` — pure bounded-token scoring and collision-tree state.
- `include/pcd/llama_runtime.hpp`, `src/engine/llama_runtime.cpp` — RAII wrapper around llama.cpp's public C API.
- `include/pcd/schema_cache.hpp`, `src/engine/schema_cache.cpp` — byte- and entry-bounded LRU compiled-schema/checkpoint cache.
- `include/pcd/pcd_engine.hpp`, `src/engine/pcd_engine.cpp` — cold prefill, checkpoint restore/fallback, sequence broadcast, suffix decode, and result assembly.
- `include/pcd/model_catalog.hpp`, `src/model_catalog.cpp` — safe GGUF discovery.
- `include/pcd/model_manager.hpp`, `src/model_manager.cpp` — transactional active-model replacement.
- `include/pcd/http_server.hpp`, `src/http/http_server.cpp`, `src/main.cpp` — REST transport and CLI startup.
- `src/benchmark.cpp` — reproducible cold-versus-cached benchmark.
- `tests/*.cpp` — unit, native integration, and HTTP tests.
- `scripts/download-default-model.sh` — checksum-verified Qwen3.5-0.8B-Q8_0 download.
- `README.md` — build, run, API, tests, and benchmark documentation.

## Task 1: Reproducible CMake project and default-model download

**Files:**
- Create: `CMakeLists.txt`
- Create: `.gitignore`
- Create: `include/pcd/version.hpp`
- Create: `tests/version_test.cpp`
- Create: `scripts/download-default-model.sh`
- Create: `models/.gitkeep`

- [ ] **Step 1: Write the failing build smoke test**

Create `tests/version_test.cpp` before `version.hpp` exists:

```cpp
#include <catch2/catch_test_macros.hpp>
#include "pcd/version.hpp"

TEST_CASE("server has a stable API version") {
    REQUIRE(pcd::api_version == "v1");
}
```

- [ ] **Step 2: Add pinned CMake dependencies and verify the test fails**

Create `CMakeLists.txt` with these exact pins and targets:

```cmake
cmake_minimum_required(VERSION 3.25)
project(pcd_server VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_IS_DEV OFF CACHE BOOL "" FORCE)

FetchContent_Declare(llama_cpp GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git GIT_TAG v0.4.1 GIT_SHALLOW TRUE)
FetchContent_Declare(httplib GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git GIT_TAG v0.56.0 GIT_SHALLOW TRUE)
FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.12.0 GIT_SHALLOW TRUE)
FetchContent_Declare(Catch2 GIT_REPOSITORY https://github.com/catchorg/Catch2.git GIT_TAG v3.15.2 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(llama_cpp httplib json Catch2)

add_library(pcd_core STATIC src/types.cpp)
target_include_directories(pcd_core PUBLIC include)
target_link_libraries(pcd_core PUBLIC llama nlohmann_json::nlohmann_json)

enable_testing()
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
add_executable(pcd_tests tests/version_test.cpp)
target_link_libraries(pcd_tests PRIVATE pcd_core Catch2::Catch2WithMain)
catch_discover_tests(pcd_tests)
```

Create an empty `src/types.cpp`, then run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target pcd_tests -j
```

Expected: compilation fails because `pcd/version.hpp` does not exist.

- [ ] **Step 3: Add the minimal version header and project ignores**

Create `include/pcd/version.hpp`:

```cpp
#pragma once
#include <string_view>
namespace pcd { inline constexpr std::string_view api_version = "v1"; }
```

Create `.gitignore`:

```gitignore
/build/
/models/*.gguf
!/models/.gitkeep
.DS_Store
compile_commands.json
```

Create `models/.gitkeep`.

- [ ] **Step 4: Add the checksum-verifying download script**

Create `scripts/download-default-model.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
dst="$root/models/Qwen3.5-0.8B-Q8_0.gguf"
url="https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf"
expected="37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f"
mkdir -p "$root/models"
curl -L --fail --continue-at - -o "$dst" "$url"
actual="$(shasum -a 256 "$dst" | awk '{print $1}')"
[[ "$actual" == "$expected" ]] || { echo "checksum mismatch: $actual" >&2; exit 1; }
echo "$dst"
```

Make it executable.

- [ ] **Step 5: Build and run the smoke test**

Run:

```bash
cmake --build build --target pcd_tests -j
ctest --test-dir build --output-on-failure
```

Expected: one passing test.

- [ ] **Step 6: Commit the scaffold**

```bash
git add CMakeLists.txt .gitignore include/pcd/version.hpp src/types.cpp tests/version_test.cpp scripts/download-default-model.sh models/.gitkeep
git commit -m "build: scaffold native PCD server"
```

## Task 2: Domain types, JSON parsing, and validation

**Files:**
- Create: `include/pcd/types.hpp`
- Modify: `src/types.cpp`
- Create: `tests/types_test.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing validation tests**

Create `tests/types_test.cpp` covering one valid enum/boolean request plus duplicate names, mixed types, empty choices, and non-boolean boolean sets:

```cpp
#include <catch2/catch_test_macros.hpp>
#include "pcd/types.hpp"

using nlohmann::json;

TEST_CASE("valid decode requests preserve field order") {
    auto request = pcd::parse_decode_request(json::parse(R"({
      "context":"ticket",
      "fields":[
        {"name":"priority","description":"handling","choices":["LOW","HIGH"]},
        {"name":"fraud","description":"fraudulent","choices":[false,true]}
      ]
    })"));
    REQUIRE(request.fields.size() == 2);
    REQUIRE(request.fields[0].name == "priority");
    REQUIRE(request.fields[1].kind == pcd::FieldKind::Boolean);
}

TEST_CASE("duplicate field names are rejected") {
    json body = {{"context", "x"}, {"fields", {
        {{"name", "same"}, {"description", "a"}, {"choices", {"A", "B"}}},
        {{"name", "same"}, {"description", "b"}, {"choices", {"C", "D"}}}
    }}};
    REQUIRE_THROWS_WITH(pcd::parse_decode_request(body), "duplicate field name: same");
}

TEST_CASE("mixed choice types are rejected") {
    json body = {{"context", "x"}, {"fields", {{{"name", "f"}, {"description", "d"}, {"choices", {"A", true}}}}}};
    REQUIRE_THROWS(pcd::parse_decode_request(body));
}
```

- [ ] **Step 2: Register and run the tests to verify RED**

Add `tests/types_test.cpp` to `pcd_tests`, run:

```bash
cmake --build build --target pcd_tests -j
```

Expected: compilation fails because the domain API is missing.

- [ ] **Step 3: Implement minimal domain types and parser**

Define in `include/pcd/types.hpp`:

```cpp
#pragma once
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace pcd {
using Choice = std::variant<bool, std::string>;
enum class FieldKind { Boolean, StringEnum };
struct FieldSpec { std::string name; std::string description; FieldKind kind; std::vector<Choice> choices; };
struct DecodeRequest { std::string context; std::vector<FieldSpec> fields; };
struct FieldResult { std::string name; Choice value; double probability; std::vector<double> probabilities; int levels; };
struct PhaseTimings { double tokenize_ms{}, restore_or_prefill_ms{}, broadcast_ms{}, suffix_ms{}, tree_ms{}; };
struct DecodeMetrics { double elapsed_ms{}; int forward_passes{}; std::string schema_cache_status; PhaseTimings phases; };
struct DecodeResponse { std::string model; std::vector<FieldResult> fields; DecodeMetrics metrics; };
class ValidationError : public std::invalid_argument { using std::invalid_argument::invalid_argument; };
DecodeRequest parse_decode_request(const nlohmann::json & body);
nlohmann::json to_json_value(const Choice & choice);
nlohmann::json to_json_response(const DecodeResponse & response);
}
```

Implement strict type and uniqueness checks in `src/types.cpp`. Enforce non-empty context, 1–63 fields, 2–256 unique choices per field, and exactly `{false,true}` for boolean fields. Preserve field and choice order.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
cmake --build build --target pcd_tests -j
ctest --test-dir build --output-on-failure
```

Expected: all domain tests pass.

- [ ] **Step 5: Commit domain validation**

```bash
git add include/pcd/types.hpp src/types.cpp tests/types_test.cpp CMakeLists.txt
git commit -m "feat: validate bounded PCD requests"
```

## Task 3: Canonical schemas and compiled token metadata

**Files:**
- Create: `include/pcd/schema.hpp`
- Create: `src/engine/schema.cpp`
- Create: `tests/schema_test.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing schema utility tests**

Create tests for common prefixes, deterministic cache keys, template marker splitting, and boolean suffixes:

```cpp
#include <catch2/catch_test_macros.hpp>
#include "pcd/schema.hpp"

TEST_CASE("common prefix is computed over every choice") {
    REQUIRE(pcd::common_prefix({"TIER_1_LOW", "TIER_3_HIGH"}) == "TIER_");
}

TEST_CASE("schema keys change when order changes") {
    auto a = pcd::parse_decode_request(nlohmann::json::parse(R"({"context":"x","fields":[{"name":"a","description":"d","choices":["X","Y"]},{"name":"b","description":"d","choices":[false,true]}]})"));
    auto b = a;
    std::swap(b.fields[0], b.fields[1]);
    REQUIRE(pcd::canonical_schema(a.fields) != pcd::canonical_schema(b.fields));
}

TEST_CASE("rendered templates must preserve the user marker") {
    REQUIRE_THROWS_WITH(pcd::split_template("no marker", "MARK"), "chat template did not preserve user marker");
}
```

- [ ] **Step 2: Run to verify RED**

Add the new source/test to CMake and build. Expected: missing `pcd/schema.hpp`.

- [ ] **Step 3: Implement the schema interfaces**

Define:

```cpp
namespace pcd {
struct TemplateParts { std::string prefix; std::string suffix; };
struct CompiledField { FieldSpec spec; std::vector<int32_t> suffix_tokens; std::string common_prefix; std::vector<std::vector<int32_t>> choice_tokens; };
struct CompiledSchema { std::string key; std::vector<int32_t> prefix_tokens; std::string user_suffix; std::vector<int32_t> closing_tokens; std::vector<CompiledField> fields; };
std::string common_prefix(const std::vector<std::string> & choices);
std::string canonical_schema(const std::vector<FieldSpec> & fields);
TemplateParts split_template(std::string_view rendered, std::string_view marker);
std::string schema_system_prompt(const std::vector<FieldSpec> & fields);
}
```

Use canonical ordered JSON for keys and include prompt format version `pcd-prompt-v1`. Render system instructions that list every field, description, and allowed choice. Keep tokenization out of this pure utility; `PcdEngine` fills `CompiledSchema` using `LlamaRuntime`.

- [ ] **Step 4: Run tests and commit**

```bash
cmake --build build --target pcd_tests -j
ctest --test-dir build --output-on-failure
git add include/pcd/schema.hpp src/engine/schema.cpp tests/schema_test.cpp CMakeLists.txt
git commit -m "feat: canonicalize and compile PCD schemas"
```

## Task 4: Pure candidate scorer and token-tree collision state

**Files:**
- Create: `include/pcd/scorer.hpp`
- Create: `src/engine/scorer.cpp`
- Create: `tests/scorer_test.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing scorer tests**

Use synthetic logits so tests need no model:

```cpp
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "pcd/scorer.hpp"

TEST_CASE("unique first tokens resolve in one level") {
    pcd::CandidateState state({{10}, {20}, {30}}, {"A", "B", "C"});
    auto result = state.advance([](int token) { return token == 20 ? 4.0 : 1.0; });
    REQUIRE(result.resolved);
    REQUIRE(state.winner() == 1);
    REQUIRE(state.probabilities()[0] + state.probabilities()[1] + state.probabilities()[2] == Catch::Approx(1.0));
}

TEST_CASE("shared first tokens remain live for another level") {
    pcd::CandidateState state({{10, 11}, {10, 12}, {20}}, {"AX", "AY", "B"});
    REQUIRE_FALSE(state.advance([](int token) { return token == 10 ? 5.0 : 1.0; }).resolved);
    REQUIRE(state.last_token() == 10);
    REQUIRE(state.advance([](int token) { return token == 12 ? 3.0 : 0.0; }).resolved);
    REQUIRE(state.winner() == 1);
}
```

- [ ] **Step 2: Verify RED**

Register files and build. Expected: missing scorer API.

- [ ] **Step 3: Implement the minimal scorer**

`CandidateState` stores live flags, matched positions, cumulative choice probabilities, level count, last token, and winner. `advance` groups live choices by next token, performs stable softmax using max-logit subtraction, retains the winning-token group, splits eliminated group mass evenly, and resolves when one choice remains. Cap engine traversal at 24 levels.

- [ ] **Step 4: Verify GREEN and commit**

```bash
cmake --build build --target pcd_tests -j
ctest --test-dir build --output-on-failure
git add include/pcd/scorer.hpp src/engine/scorer.cpp tests/scorer_test.cpp CMakeLists.txt
git commit -m "feat: score bounded candidate token trees"
```

## Task 5: llama.cpp RAII runtime and safe sequence checkpoints

**Files:**
- Create: `include/pcd/llama_runtime.hpp`
- Create: `src/engine/llama_runtime.cpp`
- Create: `tests/llama_runtime_test.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the gated native test first**

Create `tests/llama_runtime_test.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include "pcd/llama_runtime.hpp"
#include <cstdlib>

TEST_CASE("a complete sequence checkpoint restores its position", "[native]") {
    const char * path = std::getenv("PCD_TEST_GGUF");
    if (!path) SKIP("PCD_TEST_GGUF is not set");
    pcd::LlamaRuntime runtime({.model_path = path, .context_size = 2048, .max_sequences = 8, .batch_size = 2048});
    auto tokens = runtime.tokenize("checkpoint test", true, true);
    runtime.decode_single_sequence(tokens, 0, 0, false);
    auto saved = runtime.save_sequence(0);
    REQUIRE_FALSE(saved.bytes.empty());
    runtime.clear_memory();
    REQUIRE(runtime.restore_sequence(0, saved));
    REQUIRE(runtime.sequence_max_position(0) == static_cast<int>(tokens.size()) - 1);
}
```

- [ ] **Step 2: Verify RED**

Add the test/source to CMake and build. Expected: missing runtime API.

- [ ] **Step 3: Implement model/context RAII**

Define move-disabled `LlamaRuntime`, `SequenceCheckpoint`, and `BatchToken`:

```cpp
struct RuntimeOptions { std::filesystem::path model_path; uint32_t context_size{8192}; uint32_t max_sequences{64}; uint32_t batch_size{2048}; };
struct SequenceCheckpoint { std::vector<std::byte> bytes; std::size_t expected_size{}; int max_position{-1}; };
struct BatchToken { llama_token token; llama_pos position; std::vector<llama_seq_id> sequences; bool logits; };
```

The constructor must call `llama_backend_init`, load the model with `n_gpu_layers = -1`, create a context with `n_ctx=8192`, `n_batch=n_ubatch=2048`, `n_seq_max=64`, Flash Attention enabled, `kv_unified=true`, `offload_kqv=true`, and `op_offload=true`. Use RAII destructors for batch, context, and model.

Implement tokenization, token-to-piece, metadata/chat-template access, batch decoding, `llama_synchronize`, logits access, memory clear/copy/remove, and sequence-position queries. `remove_sequence` returns the native boolean and no caller may ignore it.

- [ ] **Step 4: Implement full host-side checkpoint validation**

`save_sequence` must use `LLAMA_STATE_SEQ_FLAGS_NONE`:

```cpp
const auto size = llama_state_seq_get_size_ext(ctx_, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
SequenceCheckpoint out{{}, size, sequence_max_position(seq)};
out.bytes.resize(size);
const auto written = llama_state_seq_get_data_ext(ctx_, reinterpret_cast<uint8_t *>(out.bytes.data()), size, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
if (written != size) throw NativeError("sequence checkpoint size mismatch");
```

`restore_sequence` rejects empty or size-mismatched buffers before calling native code and returns true only when `llama_state_seq_set_data_ext` consumes exactly `expected_size`. It never uses `PARTIAL_ONLY` or `ON_DEVICE`.

- [ ] **Step 5: Verify non-native and Qwen native tests**

Run:

```bash
cmake --build build --target pcd_tests -j
ctest --test-dir build --output-on-failure
PCD_TEST_GGUF=models/Qwen3.5-0.8B-Q8_0.gguf build/pcd_tests "[native]"
```

Expected: checkpoint restores Qwen3.5 sequence position without partial-removal errors.

- [ ] **Step 6: Commit the runtime**

```bash
git add include/pcd/llama_runtime.hpp src/engine/llama_runtime.cpp tests/llama_runtime_test.cpp CMakeLists.txt
git commit -m "feat: wrap llama runtime and full sequence checkpoints"
```

## Task 6: Byte-bounded schema checkpoint LRU

**Files:**
- Create: `include/pcd/schema_cache.hpp`
- Create: `src/engine/schema_cache.cpp`
- Create: `tests/schema_cache_test.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing LRU tests**

Test recency, byte eviction, replacement accounting, explicit erase, and per-key disablement:

```cpp
TEST_CASE("cache evicts least recently used entries by bytes") {
    pcd::SchemaCache cache({.max_entries = 3, .max_bytes = 10});
    cache.put("a", fixture_entry(4));
    cache.put("b", fixture_entry(4));
    REQUIRE(cache.get("a") != nullptr);
    cache.put("c", fixture_entry(4));
    REQUIRE(cache.get("b") == nullptr);
    REQUIRE(cache.get("a") != nullptr);
    REQUIRE(cache.get("c") != nullptr);
}
```

- [ ] **Step 2: Verify RED**

Register files and build. Expected: missing cache API.

- [ ] **Step 3: Implement immutable cache entries**

Define `SchemaCacheEntry` containing `shared_ptr<const CompiledSchema>`, `shared_ptr<const SequenceCheckpoint>`, checkpoint byte size, restore failure count, and caching-disabled flag. Use `std::list<std::string>` plus an unordered-map index. `get` updates recency; `put`, `erase`, and `record_restore_failure` maintain exact total bytes. Disable a key after two restore failures until cache/model reset.

- [ ] **Step 4: Verify and commit**

```bash
cmake --build build --target pcd_tests -j
ctest --test-dir build --output-on-failure
git add include/pcd/schema_cache.hpp src/engine/schema_cache.cpp tests/schema_cache_test.cpp CMakeLists.txt
git commit -m "feat: bound schema checkpoints with LRU eviction"
```

## Task 7: PCD engine with cold miss, full restore, and fallback

**Files:**
- Create: `include/pcd/pcd_engine.hpp`
- Create: `src/engine/pcd_engine.cpp`
- Create: `tests/pcd_engine_test.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the native cold/cache equivalence test**

Create a Qwen-gated test that decodes a two-field schema twice:

```cpp
TEST_CASE("Qwen cached decode matches cold decode", "[native]") {
    const char * path = std::getenv("PCD_TEST_GGUF");
    if (!path) SKIP("PCD_TEST_GGUF is not set");
    pcd::PcdEngine engine(path);
    auto request = pcd::parse_decode_request(nlohmann::json::parse(R"({
      "context":"Payment from a new country exceeded the normal amount.",
      "fields":[
        {"name":"risk","description":"Risk level","choices":["LOW","MEDIUM","HIGH"]},
        {"name":"review","description":"Needs manual review","choices":[false,true]}
      ]
    })"));
    auto cold = engine.decode(request);
    auto cached = engine.decode(request);
    REQUIRE(cold.metrics.schema_cache_status == "miss");
    REQUIRE(cached.metrics.schema_cache_status == "hit");
    REQUIRE(cached.fields.size() == cold.fields.size());
    for (std::size_t i = 0; i < cold.fields.size(); ++i) {
        REQUIRE(cached.fields[i].value == cold.fields[i].value);
        REQUIRE(cached.fields[i].probability == Catch::Approx(cold.fields[i].probability).margin(1e-5));
    }
}
```

- [ ] **Step 2: Write a failing fallback test**

Expose constructor injection of a `SchemaCache` instance. Insert a checkpoint whose `bytes.size()` differs from `expected_size`; require `decode` to return `schema_cache_status == "fallback"` and a valid result instead of calling native restore with malformed data.

- [ ] **Step 3: Verify RED**

Register the engine files and build. Expected: missing engine API.

- [ ] **Step 4: Implement schema compilation**

Use marker `\x01PCD-USER-CONTENT\x01`. Render the embedded model chat template with the schema system prompt and marker user content, require exactly one marker occurrence, tokenize the prefix with special parsing enabled, and tokenize each field suffix/candidate remainder independently. For strings, remove the longest common character prefix before tokenizing candidate remainders; booleans use literal `true`/`false` tokens.

- [ ] **Step 5: Implement restore-or-cold-prefill**

Inside one inference mutex:

```cpp
runtime_.clear_memory();
if (entry && entry->checkpoint && !entry->caching_disabled && runtime_.restore_sequence(0, *entry->checkpoint)) {
    cache_status = "hit";
} else {
    if (entry && entry->checkpoint) {
        cache_.record_restore_failure(key);
        cache_status = "fallback";
    } else {
        cache_status = "miss";
    }
    runtime_.clear_memory();
    runtime_.decode_single_sequence(compiled.prefix_tokens, 0, 0, false);
    runtime_.synchronize();
    if (!cache_.disabled(key)) {
        auto checkpoint = runtime_.save_sequence(0);
        cache_.put(key, make_entry(compiled, std::move(checkpoint)));
    }
}
```

Never attempt a partial sequence removal for prefix reuse.

- [ ] **Step 6: Implement dynamic context, broadcast, suffix batch, and collision levels**

Tokenize `request.context + compiled.user_suffix + "{\n"`, decode it at positions beginning at `prefix_tokens.size()`, synchronize, then copy `[0, completed_prefix_length)` from sequence 0 to sequences 1..N. Batch every field suffix without padding, request logits only on its final token, and create one `CandidateState` per field. Continue ambiguous fields in batched levels up to 24, then force-resolve the maximum remaining probability. Assemble booleans as booleans and enums as strings.

- [ ] **Step 7: Verify native correctness and alternating-state safety**

Add integration sections for:

- Same schema with alternating contexts: miss, hit, hit.
- Alternating schemas: miss A, miss B, hit A.
- Probability vectors sum to `1.0 ± 1e-5`.
- Invalid checkpoint triggers fallback.
- No `llama_decode` invalid-position errors over 20 Qwen3.5 cache hits.

Run:

```bash
cmake --build build --target pcd_tests -j
ctest --test-dir build --output-on-failure
PCD_TEST_GGUF=models/Qwen3.5-0.8B-Q8_0.gguf build/pcd_tests "[native]"
```

- [ ] **Step 8: Commit the engine**

```bash
git add include/pcd/pcd_engine.hpp src/engine/pcd_engine.cpp tests/pcd_engine_test.cpp CMakeLists.txt
git commit -m "feat: decode with safe Qwen schema checkpoints"
```

## Task 8: Model catalog and transactional model manager

**Files:**
- Create: `include/pcd/model_catalog.hpp`
- Create: `src/model_catalog.cpp`
- Create: `include/pcd/model_manager.hpp`
- Create: `src/model_manager.cpp`
- Create: `tests/model_manager_test.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write catalog and failed-switch tests**

Use a temporary directory containing `.gguf`, non-GGUF, symlink, and subdirectory fixtures. Require only regular `.gguf` files beneath the configured directory. Add a native-gated manager test that selects a missing/corrupt model and confirms the existing `shared_ptr<PcdEngine>` remains active.

- [ ] **Step 2: Verify RED**

Register files and build. Expected: missing catalog/manager APIs.

- [ ] **Step 3: Implement safe discovery**

`ModelCatalog::list()` uses `std::filesystem::directory_iterator`, rejects paths outside the configured directory after weak canonicalization, accepts case-insensitive `.gguf`, and returns identifier, path, and byte size. `find(id)` matches identifiers only; REST input never becomes an arbitrary filesystem path.

- [ ] **Step 4: Implement transactional switching**

`ModelManager` stores `std::shared_ptr<PcdEngine>` behind `std::shared_mutex`. `decode` copies the shared pointer under a read lock and performs inference after releasing the manager lock. `select` resolves the catalog entry and constructs the replacement engine before taking a write lock and swapping. Existing requests keep the old engine alive.

- [ ] **Step 5: Verify and commit**

```bash
cmake --build build --target pcd_tests -j
ctest --test-dir build --output-on-failure
git add include/pcd/model_catalog.hpp src/model_catalog.cpp include/pcd/model_manager.hpp src/model_manager.cpp tests/model_manager_test.cpp CMakeLists.txt
git commit -m "feat: discover and transactionally switch models"
```

## Task 9: REST server and command-line startup

**Files:**
- Create: `include/pcd/http_server.hpp`
- Create: `src/http/http_server.cpp`
- Create: `src/main.cpp`
- Create: `tests/http_server_test.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing HTTP tests**

Start `HttpServer` on port `0` with a manager fixture and exercise:

- `GET /health` returns API version and readiness.
- Malformed JSON returns `400` with `{code,message}`.
- Unknown model returns `404`.
- No active model returns `503`.
- Valid decode returns ordered fields and metrics.

Use cpp-httplib's client in the same process and stop the server through RAII.

- [ ] **Step 2: Verify RED**

Register sources/tests and build. Expected: missing HTTP API.

- [ ] **Step 3: Implement stable route/error mapping**

Register:

```cpp
server.Get("/health", ...);
server.Get("/v1/models", ...);
server.Post("/v1/models/select", ...);
server.Post("/v1/pcd/decode", ...);
```

All responses use `application/json`. Catch `ValidationError` as `400`, `UnknownModel` as `404`, schema compilation errors as `422`, `NoModelLoaded` as `503`, and remaining exceptions as `500`. Never include absolute filesystem paths in client errors.

- [ ] **Step 4: Implement startup configuration**

`main.cpp` parses:

```text
--bind ADDRESS           default 127.0.0.1
--port PORT              default 8080
--models-dir PATH        default models
--model PATH             overrides PCD_GGUF and default model
--cache-entries N        default 32
--cache-bytes BYTES      default 536870912
```

Model precedence is `--model`, `PCD_GGUF`, then `models/Qwen3.5-0.8B-Q8_0.gguf`. Install SIGINT/SIGTERM handling that calls `HttpServer::stop()` and allows RAII cleanup.

- [ ] **Step 5: Verify HTTP tests and manual health check**

```bash
cmake --build build --target pcd_server pcd_tests -j
ctest --test-dir build --output-on-failure
./build/pcd_server --model models/Qwen3.5-0.8B-Q8_0.gguf &
pid=$!
curl --fail http://127.0.0.1:8080/health
kill "$pid"
wait "$pid" || true
```

Expected: health returns HTTP 200 and identifies Qwen3.5.

- [ ] **Step 6: Commit REST transport**

```bash
git add include/pcd/http_server.hpp src/http/http_server.cpp src/main.cpp tests/http_server_test.cpp CMakeLists.txt
git commit -m "feat: expose PCD REST API"
```

## Task 10: Benchmark, end-to-end tests, and documentation

**Files:**
- Create: `src/benchmark.cpp`
- Create: `tests/data/triage-request.json`
- Create: `tests/e2e.sh`
- Create: `README.md`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add an end-to-end test that initially fails**

Create `tests/e2e.sh` that starts the built server on a free port, waits for `/health`, posts `tests/data/triage-request.json` twice, and asserts with `jq`:

```bash
test "$(jq -r '.metrics.schemaCacheStatus' first.json)" = "miss"
test "$(jq -r '.metrics.schemaCacheStatus' second.json)" = "hit"
jq -e '.values.priority and (.values.fraudulent | type == "boolean")' second.json >/dev/null
```

Run before the request fixture and benchmark/docs are present. Expected: failure due to missing fixture.

- [ ] **Step 2: Add the representative fixture and benchmark executable**

The fixture contains a fraud context, one three-value risk enum, one boolean, and a shared-prefix enum such as `TIER_1_LOW`/`TIER_3_HIGH`.

`pcd_bench` accepts `--model`, `--request`, and `--runs`. It performs one cache-building run, then reports JSON containing cold median, cached median, checkpoint bytes, cache status, per-phase medians, and selected values. Do not enforce timing thresholds inside CTest.

- [ ] **Step 3: Document build and operation**

`README.md` must include:

- macOS prerequisites: CMake, compiler, curl, Metal-capable Apple Silicon.
- Linux prerequisites and CPU expectations.
- `./scripts/download-default-model.sh`.
- Configure/build/test commands.
- Server flags and environment variables.
- Complete curl examples for all four endpoints.
- Response schema and cache status meanings.
- `PCD_TEST_GGUF` native integration tests.
- Benchmark invocation.
- Explicit scope: booleans and bounded string enums only.
- Why full checkpoints are used for Qwen3.5 instead of partial rollback.

- [ ] **Step 4: Run full verification**

Run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
PCD_TEST_GGUF=models/Qwen3.5-0.8B-Q8_0.gguf build/pcd_tests "[native]"
PCD_TEST_GGUF=models/Qwen3.5-0.8B-Q8_0.gguf ./tests/e2e.sh
./build/pcd_bench --model models/Qwen3.5-0.8B-Q8_0.gguf --request tests/data/triage-request.json --runs 5
```

Expected: all tests pass; the second identical REST request reports `hit`; the benchmark prints cold and cached phase medians without invalid-position errors.

- [ ] **Step 5: Inspect and sanitize the finished tree**

Run:

```bash
git status --short
git diff --check
find . -name '*.gguf' -o -name build | sort
```

Expected: no GGUF or build output is tracked, and `git diff --check` prints nothing.

- [ ] **Step 6: Commit documentation and end-to-end verification**

```bash
git add src/benchmark.cpp tests/data/triage-request.json tests/e2e.sh README.md CMakeLists.txt
git commit -m "docs: add native PCD server workflow and benchmark"
```
EOF