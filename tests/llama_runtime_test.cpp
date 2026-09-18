#include <catch2/catch_test_macros.hpp>
#include "pcd/llama_runtime.hpp"

#include <cstdlib>

namespace {
const bool quiet_native_logs = (pcd::LlamaRuntime::quiet_logging(), true);
}

TEST_CASE("a complete sequence checkpoint restores its position", "[native]") {
    const char * path = std::getenv("PCD_TEST_GGUF");
    if (!path) SKIP("PCD_TEST_GGUF is not set");
    pcd::LlamaRuntime runtime({.model_path = path, .context_size = 2048, .max_sequences = 8, .batch_size = 2048});
    auto tokens = runtime.tokenize("checkpoint test", true, true);
    REQUIRE_FALSE(tokens.empty());
    runtime.decode_single_sequence(tokens, 0, 0, false);
    runtime.synchronize();
    auto saved = runtime.save_sequence(0);
    REQUIRE_FALSE(saved.bytes.empty());
    REQUIRE(saved.bytes.size() == saved.expected_size);
    REQUIRE(saved.max_position == static_cast<int>(tokens.size()) - 1);
    runtime.clear_memory();
    REQUIRE(runtime.sequence_max_position(0) == -1);
    REQUIRE(runtime.restore_sequence(0, saved));
    REQUIRE(runtime.sequence_max_position(0) == static_cast<int>(tokens.size()) - 1);
}

TEST_CASE("malformed checkpoints are rejected before native restore", "[native]") {
    const char * path = std::getenv("PCD_TEST_GGUF");
    if (!path) SKIP("PCD_TEST_GGUF is not set");
    pcd::LlamaRuntime runtime({.model_path = path, .context_size = 2048, .max_sequences = 8, .batch_size = 2048});
    pcd::SequenceCheckpoint empty;
    REQUIRE_FALSE(runtime.restore_sequence(0, empty));
    pcd::SequenceCheckpoint mismatched{std::vector<std::byte>(16), 32, 3};
    REQUIRE_FALSE(runtime.restore_sequence(0, mismatched));
}

TEST_CASE("runtime exposes vocabulary, template and metadata", "[native]") {
    const char * path = std::getenv("PCD_TEST_GGUF");
    if (!path) SKIP("PCD_TEST_GGUF is not set");
    pcd::LlamaRuntime runtime({.model_path = path, .context_size = 2048, .max_sequences = 8, .batch_size = 2048});
    REQUIRE(runtime.vocab_size() > 0);
    REQUIRE_FALSE(runtime.chat_template().empty());
    REQUIRE_FALSE(runtime.model_description().empty());
    auto tokens = runtime.tokenize("hello", false, false);
    REQUIRE(runtime.token_to_piece(tokens.front()).find("hello") != std::string::npos);
    auto rendered = runtime.render_chat({{"system", "S"}, {"user", "U"}});
    REQUIRE(rendered.find("S") != std::string::npos);
    REQUIRE(rendered.find("U") != std::string::npos);
}

TEST_CASE("batch decode returns logits only where requested", "[native]") {
    const char * path = std::getenv("PCD_TEST_GGUF");
    if (!path) SKIP("PCD_TEST_GGUF is not set");
    pcd::LlamaRuntime runtime({.model_path = path, .context_size = 2048, .max_sequences = 8, .batch_size = 2048});
    auto tokens = runtime.tokenize("shared prefix", true, true);
    runtime.decode_single_sequence(tokens, 0, 0, false);
    runtime.synchronize();
    const auto next = static_cast<int>(tokens.size());
    runtime.copy_sequence(0, 1, 0, next);
    runtime.copy_sequence(0, 2, 0, next);
    std::vector<pcd::BatchToken> batch = {
        {tokens.front(), next, {1}, true},
        {tokens.front(), next, {2}, true},
    };
    runtime.decode_batch(batch);
    runtime.synchronize();
    const float * a = runtime.logits(0);
    const float * b = runtime.logits(1);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(a[0] == b[0]);
    REQUIRE(runtime.sequence_max_position(1) == next);
    REQUIRE(runtime.remove_sequence(1, -1, -1));
    REQUIRE(runtime.sequence_max_position(1) == -1);
}

TEST_CASE("Qwen3.5 refuses partial sequence removal, so prefix reuse needs full checkpoints", "[native]") {
    const char * path = std::getenv("PCD_TEST_GGUF");
    if (!path) SKIP("PCD_TEST_GGUF is not set");
    pcd::LlamaRuntime runtime({.model_path = path, .context_size = 2048, .max_sequences = 8, .batch_size = 2048});
    auto tokens = runtime.tokenize("one two three four five six seven", true, true);
    runtime.decode_single_sequence(tokens, 0, 0, false);
    runtime.synchronize();
    const auto last = static_cast<int>(tokens.size()) - 1;
    if (runtime.metadata("general.architecture").rfind("qwen35", 0) != 0) {
        SKIP("not a Qwen3.5 hybrid model");
    }
    REQUIRE_FALSE(runtime.remove_sequence(0, 2, -1));
    REQUIRE(runtime.sequence_max_position(0) == last);
    REQUIRE(runtime.remove_sequence(0, -1, -1));
    REQUIRE(runtime.sequence_max_position(0) == -1);
}
