#include "pcd/pcd_engine.hpp"

#include "pcd/scorer.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <set>
#include <unordered_map>

namespace pcd {

namespace {

constexpr const char * user_marker = "\x01PCD-USER-CONTENT\x01";
constexpr const char * empty_think_block = "<think>\n\n</think>\n\n";
constexpr llama_seq_id prefix_seq = 0;

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

RuntimeOptions with_path(RuntimeOptions runtime, const std::filesystem::path & path) {
    runtime.model_path = path;
    return runtime;
}

std::string hex_hash(const std::string & text) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016zx", std::hash<std::string>{}(text));
    return buf;
}

}  // namespace

PcdEngine::PcdEngine(std::filesystem::path model_path, EngineOptions options)
    : options_(std::move(options)),
      runtime_(with_path(options_.runtime, model_path)),
      cache_(options_.cache),
      model_id_(model_path.filename().string()),
      description_(runtime_.model_description()),
      backend_(runtime_.backend_name()),
      architecture_(runtime_.metadata("general.architecture")) {
    options_.runtime.model_path = std::move(model_path);
    const auto tmpl = runtime_.chat_template();
    template_fingerprint_ = hex_hash(tmpl);
    // Qwen3-style templates close an empty think block before the answer when
    // thinking is disabled; the built-in ChatML renderer omits it.
    if (tmpl.find(empty_think_block) != std::string::npos) {
        user_suffix_extra_ = empty_think_block;
    }
}

std::string PcdEngine::schema_key(const std::vector<FieldSpec> & fields) const {
    return runtime_.model_fingerprint() + "|" + template_fingerprint_ + "|" + canonical_schema(fields);
}

std::shared_ptr<const CompiledSchema> PcdEngine::compile(const std::vector<FieldSpec> & fields) const {
    auto compiled = std::make_shared<CompiledSchema>();
    compiled->key = schema_key(fields);

    std::string rendered;
    try {
        rendered = runtime_.render_chat({{"system", schema_system_prompt(fields)}, {"user", user_marker}});
    } catch (const NativeError & error) {
        throw SchemaError(error.what());
    }
    auto parts = split_template(rendered, user_marker);
    compiled->prefix_tokens = runtime_.tokenize(parts.prefix, true, true);
    compiled->user_suffix = parts.suffix + user_suffix_extra_;
    compiled->closing_tokens = runtime_.tokenize("\"", false, false);
    if (compiled->prefix_tokens.empty() || compiled->closing_tokens.empty()) {
        throw SchemaError("prompt prefix tokenized to nothing");
    }

    std::size_t candidate_tokens = 0;
    for (const auto & spec : fields) {
        CompiledField field;
        field.spec = spec;
        if (spec.kind == FieldKind::StringEnum) {
            std::vector<std::string> texts;
            for (const auto & choice : spec.choices) {
                texts.push_back(std::get<std::string>(choice));
            }
            field.common_prefix = common_prefix(texts);
        }
        field.suffix_tokens = runtime_.tokenize(field_suffix_text(spec, field.common_prefix), false, false);
        if (field.suffix_tokens.empty()) {
            throw SchemaError("field " + spec.name + " suffix tokenized to nothing");
        }
        std::set<std::vector<int32_t>> distinct;
        for (const auto & text : candidate_texts(spec, field.common_prefix)) {
            auto tokens = runtime_.tokenize(text, false, false);
            if (tokens.empty()) {
                throw SchemaError("field " + spec.name + " has a choice that tokenizes to nothing");
            }
            if (!distinct.insert(tokens).second) {
                throw SchemaError("field " + spec.name + " has choices that are indistinguishable after tokenization");
            }
            candidate_tokens += tokens.size();
            field.choice_tokens.push_back(std::move(tokens));
        }
        compiled->fields.push_back(std::move(field));
    }
    if (candidate_tokens > options_.max_candidate_tokens) {
        throw SchemaError("schema exceeds " + std::to_string(options_.max_candidate_tokens) + " candidate tokens");
    }
    return compiled;
}

void PcdEngine::ensure_fits(const CompiledSchema & compiled, std::size_t context_tokens) const {
    if (compiled.fields.size() + 1 > runtime_.max_sequences()) {
        throw ValidationError("request has more fields than the configured sequence limit (" + std::to_string(runtime_.max_sequences() - 1) + ")");
    }
    std::size_t per_field = 0;
    for (const auto & field : compiled.fields) {
        std::size_t deepest = 0;
        for (const auto & tokens : field.choice_tokens) {
            deepest = std::max(deepest, tokens.size());
        }
        per_field += field.suffix_tokens.size() + std::min<std::size_t>(deepest, max_tree_levels);
    }
    const auto suffix_tokens = runtime_.tokenize(compiled.user_suffix + "{\n", false, true).size();
    const auto needed = compiled.prefix_tokens.size() + context_tokens + suffix_tokens + per_field;
    if (needed > runtime_.context_size()) {
        throw ValidationError("request needs " + std::to_string(needed) + " tokens but the context window holds " + std::to_string(runtime_.context_size()));
    }
}

DecodeResponse PcdEngine::decode(const DecodeRequest & request) {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    const auto started = Clock::now();
    DecodeResponse response;
    response.model = model_id_;
    auto & metrics = response.metrics;
    int forward_passes = 0;

    // Phase: tokenize (schema lookup/compile plus dynamic context).
    auto phase = Clock::now();
    const auto key = schema_key(request.fields);
    auto entry = cache_.get(key);
    std::shared_ptr<const CompiledSchema> compiled = entry ? entry->schema : nullptr;
    if (!compiled) {
        compiled = compile(request.fields);
    }
    auto context_tokens = runtime_.tokenize(request.context, false, false);
    auto dynamic_tokens = context_tokens;
    for (auto token : runtime_.tokenize(compiled->user_suffix + "{\n", false, true)) {
        dynamic_tokens.push_back(token);
    }
    ensure_fits(*compiled, context_tokens.size());
    metrics.phases.tokenize_ms = ms_since(phase);

    // Phase: restore the schema-prefix checkpoint or prefill it cold.
    phase = Clock::now();
    const auto prefix_len = static_cast<llama_pos>(compiled->prefix_tokens.size());
    runtime_.clear_memory();
    const bool restored = entry && entry->checkpoint && !cache_.disabled(key)
        && runtime_.restore_sequence(prefix_seq, *entry->checkpoint)
        && runtime_.sequence_max_position(prefix_seq) == prefix_len - 1;
    if (restored) {
        metrics.schema_cache_status = "hit";
        metrics.checkpoint_bytes = entry->checkpoint->bytes.size();
    } else {
        if (entry && entry->checkpoint) {
            cache_.record_restore_failure(key);
            metrics.schema_cache_status = "fallback";
        } else {
            metrics.schema_cache_status = "miss";
        }
        runtime_.clear_memory();
        runtime_.decode_single_sequence(compiled->prefix_tokens, prefix_seq, 0, false);
        forward_passes += static_cast<int>((compiled->prefix_tokens.size() + runtime_.batch_size() - 1) / runtime_.batch_size());
        runtime_.synchronize();
        if (!cache_.disabled(key)) {
            auto checkpoint = std::make_shared<const SequenceCheckpoint>(runtime_.save_sequence(prefix_seq));
            metrics.checkpoint_bytes = checkpoint->bytes.size();
            cache_.put(key, SchemaCacheEntry{compiled, std::move(checkpoint)});
        }
    }
    metrics.phases.restore_or_prefill_ms = ms_since(phase);

    // Phase: dynamic context (user text, user close, assistant open, JSON object start).
    phase = Clock::now();
    runtime_.decode_single_sequence(dynamic_tokens, prefix_seq, prefix_len, false);
    forward_passes += static_cast<int>((dynamic_tokens.size() + runtime_.batch_size() - 1) / runtime_.batch_size());
    runtime_.synchronize();
    const auto completed = prefix_len + static_cast<llama_pos>(dynamic_tokens.size());
    if (runtime_.sequence_max_position(prefix_seq) != completed - 1) {
        throw NativeError("prefix sequence position mismatch after dynamic context");
    }
    metrics.phases.dynamic_context_ms = ms_since(phase);

    // Phase: broadcast the completed prefix to one sequence per field.
    phase = Clock::now();
    const auto field_count = compiled->fields.size();
    for (std::size_t f = 0; f < field_count; ++f) {
        runtime_.copy_sequence(prefix_seq, static_cast<llama_seq_id>(f + 1), -1, -1);
    }
    metrics.phases.broadcast_ms = ms_since(phase);

    // Phase: decode every field suffix, logits only on each final token, then
    // score the first candidate level.
    phase = Clock::now();
    std::vector<std::unique_ptr<CandidateState>> states;
    std::vector<llama_pos> next_position(field_count);
    states.reserve(field_count);
    for (std::size_t f = 0; f < field_count; ++f) {
        const auto & field = compiled->fields[f];
        std::vector<std::string> labels;
        for (const auto & choice : field.spec.choices) {
            labels.push_back(choice_label(choice));
        }
        states.push_back(std::make_unique<CandidateState>(field.choice_tokens, std::move(labels)));
        next_position[f] = completed + static_cast<llama_pos>(field.suffix_tokens.size());
    }

    // Runs `batch`, then hands each requested field its logits. Batches are
    // flushed at the configured batch size so a field's logits are always read
    // from the decode that produced them.
    std::vector<BatchToken> batch;
    std::vector<std::pair<std::size_t, std::size_t>> pending;  // (field, batch index)
    const auto flush = [&](const std::function<void(std::size_t, const float *)> & on_logits) {
        if (batch.empty()) {
            return;
        }
        runtime_.decode_batch(batch);
        ++forward_passes;
        runtime_.synchronize();
        for (const auto & [field, index] : pending) {
            on_logits(field, runtime_.logits(static_cast<int32_t>(index)));
        }
        batch.clear();
        pending.clear();
    };
    const auto score = [&](std::size_t f, const float * logits) {
        states[f]->advance([logits](int32_t token) { return static_cast<double>(logits[token]); });
    };
    const auto batch_limit = static_cast<std::size_t>(runtime_.batch_size());

    for (std::size_t f = 0; f < field_count; ++f) {
        const auto & field = compiled->fields[f];
        const auto seq = static_cast<llama_seq_id>(f + 1);
        for (std::size_t i = 0; i < field.suffix_tokens.size(); ++i) {
            if (batch.size() == batch_limit) {
                flush(score);
            }
            const bool last = i + 1 == field.suffix_tokens.size();
            batch.push_back({field.suffix_tokens[i], completed + static_cast<llama_pos>(i), {seq}, last});
            if (last) {
                pending.emplace_back(f, batch.size() - 1);
            }
        }
    }
    flush(score);
    metrics.phases.suffix_ms = ms_since(phase);

    // Phase: resolve shared candidate prefixes level by level.
    phase = Clock::now();
    for (int level = 0; level < max_tree_levels; ++level) {
        bool any = false;
        for (std::size_t f = 0; f < field_count; ++f) {
            if (states[f]->resolved()) {
                continue;
            }
            any = true;
            if (batch.size() == batch_limit) {
                flush(score);
            }
            batch.push_back({states[f]->last_token(), next_position[f], {static_cast<llama_seq_id>(f + 1)}, true});
            pending.emplace_back(f, batch.size() - 1);
            ++next_position[f];
        }
        if (!any) {
            break;
        }
        flush(score);
    }
    for (auto & state : states) {
        state->force_resolve();
    }
    metrics.phases.tree_ms = ms_since(phase);

    for (std::size_t f = 0; f < field_count; ++f) {
        const auto & field = compiled->fields[f];
        const auto & state = *states[f];
        const auto winner = static_cast<std::size_t>(state.winner());
        response.fields.push_back({
            field.spec.name,
            field.spec.choices[winner],
            state.probabilities()[winner],
            state.probabilities(),
            state.labels(),
            std::max(state.levels(), 1),
        });
    }
    metrics.forward_passes = forward_passes;
    metrics.elapsed_ms = ms_since(started);
    return response;
}

}  // namespace pcd
