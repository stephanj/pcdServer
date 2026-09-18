#pragma once
#include <llama.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pcd {

// Failures reported by llama.cpp's C API (HTTP 500 unless a fallback applies).
class NativeError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct RuntimeOptions {
    std::filesystem::path model_path;
    uint32_t context_size{8192};
    uint32_t max_sequences{64};
    uint32_t batch_size{2048};
    int32_t threads{0};  // 0 = derive from hardware concurrency
};

// Complete host-side copy of one sequence's state (attention KV plus any
// recurrent state). Immutable once published to a cache.
struct SequenceCheckpoint {
    std::vector<std::byte> bytes;
    std::size_t expected_size{};
    int max_position{-1};
};

struct BatchToken {
    llama_token token;
    llama_pos position;
    std::vector<llama_seq_id> sequences;
    bool logits;
};

struct ChatMessage {
    std::string role;
    std::string content;
};

// RAII owner of one llama model, context and reusable batch. Not thread-safe;
// callers serialize access.
class LlamaRuntime {
public:
    explicit LlamaRuntime(RuntimeOptions options);
    ~LlamaRuntime();
    LlamaRuntime(const LlamaRuntime &) = delete;
    LlamaRuntime & operator=(const LlamaRuntime &) = delete;
    LlamaRuntime(LlamaRuntime &&) = delete;
    LlamaRuntime & operator=(LlamaRuntime &&) = delete;

    // Vocabulary and metadata.
    std::vector<llama_token> tokenize(std::string_view text, bool add_special, bool parse_special) const;
    std::string token_to_piece(llama_token token) const;
    int32_t vocab_size() const;
    std::string chat_template() const;
    std::string render_chat(const std::vector<ChatMessage> & messages) const;
    std::string metadata(const char * key) const;
    std::string model_description() const;
    std::string model_fingerprint() const;
    std::string backend_name() const;
    uint32_t context_size() const;
    uint32_t batch_size() const;
    uint32_t max_sequences() const;
    const RuntimeOptions & options() const { return options_; }

    // Decoding. Both throw NativeError when llama_decode reports failure.
    void decode_single_sequence(const std::vector<llama_token> & tokens, llama_seq_id seq, llama_pos start, bool logits_on_last);
    void decode_batch(const std::vector<BatchToken> & tokens);
    void synchronize();
    // Logits for batch index `index` of the most recent decode_batch call, or
    // the last output of the most recent decode_single_sequence when -1.
    const float * logits(int32_t index) const;

    // Sequence memory.
    void clear_memory();
    void copy_sequence(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1);
    [[nodiscard]] bool remove_sequence(llama_seq_id seq, llama_pos p0, llama_pos p1);
    int sequence_max_position(llama_seq_id seq) const;

    // Full checkpoints (LLAMA_STATE_SEQ_FLAGS_NONE only).
    SequenceCheckpoint save_sequence(llama_seq_id seq) const;
    [[nodiscard]] bool restore_sequence(llama_seq_id seq, const SequenceCheckpoint & checkpoint);

    // Suppresses llama.cpp INFO/DEBUG output process-wide.
    static void quiet_logging();

private:
    void decode_or_throw(const llama_batch & batch);

    RuntimeOptions options_;
    llama_model * model_{nullptr};
    llama_context * ctx_{nullptr};
    const llama_vocab * vocab_{nullptr};
    llama_batch batch_{};
    bool batch_allocated_{false};
};

}  // namespace pcd
