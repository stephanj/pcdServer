#include "pcd/llama_runtime.hpp"

#include <ggml-backend.h>

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <thread>

namespace pcd {

namespace {

void ensure_backend() {
    static std::once_flag once;
    // llama_backend_free is intentionally never called: several runtimes may
    // coexist briefly while a replacement model is loaded.
    std::call_once(once, [] { llama_backend_init(); });
}

void quiet_log_callback(ggml_log_level level, const char * text, void * /*user_data*/) {
    static thread_local ggml_log_level last = GGML_LOG_LEVEL_NONE;
    if (level != GGML_LOG_LEVEL_CONT) {
        last = level;
    }
    if (last >= GGML_LOG_LEVEL_WARN) {
        std::fputs(text, stderr);
    }
}

int32_t default_threads() {
    const auto hw = static_cast<int32_t>(std::thread::hardware_concurrency());
    return std::clamp(hw, 1, 16);
}

}  // namespace

void LlamaRuntime::quiet_logging() {
    llama_log_set(quiet_log_callback, nullptr);
}

LlamaRuntime::LlamaRuntime(RuntimeOptions options) : options_(std::move(options)) {
    ensure_backend();
    if (options_.threads <= 0) {
        options_.threads = default_threads();
    }

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;
    model_ = llama_model_load_from_file(options_.model_path.string().c_str(), model_params);
    if (!model_) {
        throw NativeError("failed to load model");
    }
    vocab_ = llama_model_get_vocab(model_);

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = options_.context_size;
    ctx_params.n_batch = options_.batch_size;
    ctx_params.n_ubatch = options_.batch_size;
    ctx_params.n_seq_max = options_.max_sequences;
    ctx_params.n_threads = options_.threads;
    ctx_params.n_threads_batch = options_.threads;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    ctx_params.kv_unified = true;
    ctx_params.offload_kqv = true;
    ctx_params.op_offload = true;
    ctx_params.no_perf = true;
    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        llama_model_free(model_);
        model_ = nullptr;
        throw NativeError("failed to create llama context");
    }

    batch_ = llama_batch_init(static_cast<int32_t>(options_.batch_size), 0, static_cast<int32_t>(options_.max_sequences));
    batch_allocated_ = true;
}

LlamaRuntime::~LlamaRuntime() {
    if (batch_allocated_) {
        llama_batch_free(batch_);
    }
    if (ctx_) {
        llama_free(ctx_);
    }
    if (model_) {
        llama_model_free(model_);
    }
}

std::vector<llama_token> LlamaRuntime::tokenize(std::string_view text, bool add_special, bool parse_special) const {
    std::vector<llama_token> tokens(text.size() + 16);
    auto n = llama_tokenize(vocab_, text.data(), static_cast<int32_t>(text.size()), tokens.data(), static_cast<int32_t>(tokens.size()), add_special, parse_special);
    if (n < 0) {
        tokens.resize(static_cast<std::size_t>(-n));
        n = llama_tokenize(vocab_, text.data(), static_cast<int32_t>(text.size()), tokens.data(), static_cast<int32_t>(tokens.size()), add_special, parse_special);
        if (n < 0) {
            throw NativeError("tokenization failed");
        }
    }
    tokens.resize(static_cast<std::size_t>(n));
    return tokens;
}

std::string LlamaRuntime::token_to_piece(llama_token token) const {
    std::string piece(64, '\0');
    auto n = llama_token_to_piece(vocab_, token, piece.data(), static_cast<int32_t>(piece.size()), 0, true);
    if (n < 0) {
        piece.resize(static_cast<std::size_t>(-n));
        n = llama_token_to_piece(vocab_, token, piece.data(), static_cast<int32_t>(piece.size()), 0, true);
        if (n < 0) {
            throw NativeError("token_to_piece failed");
        }
    }
    piece.resize(static_cast<std::size_t>(n));
    return piece;
}

int32_t LlamaRuntime::vocab_size() const {
    return llama_vocab_n_tokens(vocab_);
}

std::string LlamaRuntime::chat_template() const {
    const char * tmpl = llama_model_chat_template(model_, nullptr);
    return tmpl ? std::string(tmpl) : std::string();
}

std::string LlamaRuntime::render_chat(const std::vector<ChatMessage> & messages) const {
    const auto tmpl = chat_template();
    std::vector<llama_chat_message> native;
    native.reserve(messages.size());
    std::size_t total = 0;
    for (const auto & message : messages) {
        native.push_back({message.role.c_str(), message.content.c_str()});
        total += message.role.size() + message.content.size();
    }
    std::string out(2 * total + 256, '\0');
    auto n = llama_chat_apply_template(tmpl.empty() ? nullptr : tmpl.c_str(), native.data(), native.size(), true, out.data(), static_cast<int32_t>(out.size()));
    if (n < 0) {
        throw NativeError("chat template is not supported by llama_chat_apply_template");
    }
    if (static_cast<std::size_t>(n) > out.size()) {
        out.resize(static_cast<std::size_t>(n));
        n = llama_chat_apply_template(tmpl.empty() ? nullptr : tmpl.c_str(), native.data(), native.size(), true, out.data(), static_cast<int32_t>(out.size()));
        if (n < 0) {
            throw NativeError("chat template is not supported by llama_chat_apply_template");
        }
    }
    out.resize(static_cast<std::size_t>(n));
    return out;
}

std::string LlamaRuntime::metadata(const char * key) const {
    std::string value(256, '\0');
    auto n = llama_model_meta_val_str(model_, key, value.data(), value.size());
    if (n < 0) {
        return {};
    }
    if (static_cast<std::size_t>(n) >= value.size()) {
        value.resize(static_cast<std::size_t>(n) + 1);
        n = llama_model_meta_val_str(model_, key, value.data(), value.size());
        if (n < 0) {
            return {};
        }
    }
    value.resize(static_cast<std::size_t>(n));
    return value;
}

std::string LlamaRuntime::model_description() const {
    std::string desc(256, '\0');
    auto n = llama_model_desc(model_, desc.data(), desc.size());
    if (n < 0) {
        return {};
    }
    desc.resize(std::min(static_cast<std::size_t>(n), desc.size() - 1));
    return desc;
}

std::string LlamaRuntime::model_fingerprint() const {
    return model_description() + "|" + std::to_string(llama_model_size(model_)) + "|" + metadata("general.name") + "|" + options_.model_path.filename().string();
}

std::string LlamaRuntime::backend_name() const {
    std::string name = "CPU";
    for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            return ggml_backend_dev_name(dev);
        }
    }
    return name;
}

uint32_t LlamaRuntime::context_size() const { return llama_n_ctx(ctx_); }
uint32_t LlamaRuntime::batch_size() const { return llama_n_batch(ctx_); }
uint32_t LlamaRuntime::max_sequences() const { return llama_n_seq_max(ctx_); }

void LlamaRuntime::decode_or_throw(const llama_batch & batch) {
    const auto rc = llama_decode(ctx_, batch);
    if (rc != 0) {
        throw NativeError("llama_decode failed with code " + std::to_string(rc));
    }
}

void LlamaRuntime::decode_single_sequence(const std::vector<llama_token> & tokens, llama_seq_id seq, llama_pos start, bool logits_on_last) {
    const auto chunk = static_cast<std::size_t>(options_.batch_size);
    for (std::size_t offset = 0; offset < tokens.size(); offset += chunk) {
        const auto count = std::min(chunk, tokens.size() - offset);
        batch_.n_tokens = static_cast<int32_t>(count);
        for (std::size_t i = 0; i < count; ++i) {
            batch_.token[i] = tokens[offset + i];
            batch_.pos[i] = start + static_cast<llama_pos>(offset + i);
            batch_.n_seq_id[i] = 1;
            batch_.seq_id[i][0] = seq;
            batch_.logits[i] = 0;
        }
        if (logits_on_last && offset + count == tokens.size()) {
            batch_.logits[count - 1] = 1;
        }
        decode_or_throw(batch_);
    }
}

void LlamaRuntime::decode_batch(const std::vector<BatchToken> & tokens) {
    if (tokens.empty()) {
        return;
    }
    if (tokens.size() > static_cast<std::size_t>(options_.batch_size)) {
        throw NativeError("batch exceeds configured batch size");
    }
    batch_.n_tokens = static_cast<int32_t>(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const auto & entry = tokens[i];
        if (entry.sequences.empty() || entry.sequences.size() > static_cast<std::size_t>(options_.max_sequences)) {
            throw NativeError("batch token has an invalid sequence list");
        }
        batch_.token[i] = entry.token;
        batch_.pos[i] = entry.position;
        batch_.n_seq_id[i] = static_cast<int32_t>(entry.sequences.size());
        for (std::size_t s = 0; s < entry.sequences.size(); ++s) {
            batch_.seq_id[i][s] = entry.sequences[s];
        }
        batch_.logits[i] = entry.logits ? 1 : 0;
    }
    decode_or_throw(batch_);
}

void LlamaRuntime::synchronize() {
    llama_synchronize(ctx_);
}

const float * LlamaRuntime::logits(int32_t index) const {
    const float * out = llama_get_logits_ith(ctx_, index);
    if (!out) {
        throw NativeError("no logits available at batch index " + std::to_string(index));
    }
    return out;
}

void LlamaRuntime::clear_memory() {
    llama_memory_clear(llama_get_memory(ctx_), true);
}

void LlamaRuntime::copy_sequence(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1) {
    llama_memory_seq_cp(llama_get_memory(ctx_), src, dst, p0, p1);
}

bool LlamaRuntime::remove_sequence(llama_seq_id seq, llama_pos p0, llama_pos p1) {
    return llama_memory_seq_rm(llama_get_memory(ctx_), seq, p0, p1);
}

int LlamaRuntime::sequence_max_position(llama_seq_id seq) const {
    return llama_memory_seq_pos_max(llama_get_memory(ctx_), seq);
}

SequenceCheckpoint LlamaRuntime::save_sequence(llama_seq_id seq) const {
    const auto size = llama_state_seq_get_size_ext(ctx_, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (size == 0) {
        throw NativeError("sequence checkpoint size is zero");
    }
    SequenceCheckpoint out{{}, size, sequence_max_position(seq)};
    out.bytes.resize(size);
    const auto written = llama_state_seq_get_data_ext(ctx_, reinterpret_cast<uint8_t *>(out.bytes.data()), size, seq, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (written != size) {
        throw NativeError("sequence checkpoint size mismatch");
    }
    return out;
}

bool LlamaRuntime::restore_sequence(llama_seq_id seq, const SequenceCheckpoint & checkpoint) {
    if (checkpoint.bytes.empty() || checkpoint.expected_size == 0 || checkpoint.bytes.size() != checkpoint.expected_size) {
        return false;
    }
    const auto consumed = llama_state_seq_set_data_ext(ctx_, reinterpret_cast<const uint8_t *>(checkpoint.bytes.data()), checkpoint.bytes.size(), seq, LLAMA_STATE_SEQ_FLAGS_NONE);
    return consumed == checkpoint.expected_size;
}

}  // namespace pcd
