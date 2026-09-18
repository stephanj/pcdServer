#pragma once
#include "pcd/llama_runtime.hpp"
#include "pcd/schema.hpp"
#include "pcd/schema_cache.hpp"
#include "pcd/types.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

namespace pcd {

struct EngineOptions {
    RuntimeOptions runtime;                  // model_path is filled from the constructor argument
    SchemaCacheOptions cache;
    std::size_t max_candidate_tokens{4096};  // total candidate tokens across all fields
};

// Owns one model/context and performs Parallel Constrained Decoding. All
// inference on the context is serialized by one mutex.
class PcdEngine {
public:
    explicit PcdEngine(std::filesystem::path model_path, EngineOptions options = {});

    DecodeResponse decode(const DecodeRequest & request);

    std::string model_id() const { return model_id_; }
    std::string model_description() const { return description_; }
    std::string backend() const { return backend_; }
    std::string architecture() const { return architecture_; }
    const EngineOptions & options() const { return options_; }

    // Cache key for a schema on this model instance.
    std::string schema_key(const std::vector<FieldSpec> & fields) const;
    // Direct cache access for tests and diagnostics. Only safe when no decode
    // is running.
    SchemaCache & cache() { return cache_; }

private:
    struct CompiledSchemaInternal;
    std::shared_ptr<const CompiledSchema> compile(const std::vector<FieldSpec> & fields) const;
    void ensure_fits(const CompiledSchema & compiled, std::size_t context_tokens) const;

    EngineOptions options_;
    LlamaRuntime runtime_;
    SchemaCache cache_;
    std::mutex inference_mutex_;
    std::string model_id_;
    std::string description_;
    std::string backend_;
    std::string architecture_;
    std::string template_fingerprint_;
    std::string user_suffix_extra_;  // e.g. an empty think block appended after the assistant header
};

}  // namespace pcd
