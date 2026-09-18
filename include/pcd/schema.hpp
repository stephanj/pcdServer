#pragma once
#include "pcd/types.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pcd {

// Bumped whenever the rendered prompt or suffix layout changes so stale
// checkpoints are never reused across prompt formats.
inline constexpr std::string_view prompt_format_version = "pcd-prompt-v1";

// Raised when a schema cannot be turned into a safe token layout (HTTP 422).
class SchemaError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct TemplateParts {
    std::string prefix;  // rendered text before the user marker: cacheable
    std::string suffix;  // rendered text after the marker: closes user turn, opens assistant turn
};

struct CompiledField {
    FieldSpec spec;
    std::vector<int32_t> suffix_tokens;                // `  "name": "` (+ common prefix) or `  "name": `
    std::string common_prefix;                         // longest shared character prefix of string choices
    std::vector<std::vector<int32_t>> choice_tokens;   // per choice: remainder tokens (+ closing quote for strings)
};

struct CompiledSchema {
    std::string key;
    std::vector<int32_t> prefix_tokens;   // system prompt + user turn opening, up to the marker
    std::string user_suffix;              // text after the marker (user close + assistant open)
    std::vector<int32_t> closing_tokens;  // tokens of the closing quote for string values
    std::vector<CompiledField> fields;
};

// Longest common character prefix of all choices ("" when empty).
std::string common_prefix(const std::vector<std::string> & choices);

// Deterministic, order-sensitive canonical form used as part of the cache key.
std::string canonical_schema(const std::vector<FieldSpec> & fields);

// Splits rendered chat-template text around a marker that must occur exactly once.
TemplateParts split_template(std::string_view rendered, std::string_view marker);

// System instructions that enumerate every field, description and legal value.
std::string schema_system_prompt(const std::vector<FieldSpec> & fields);

// Text placed after `{\n` that opens the JSON member for one field.
std::string field_suffix_text(const FieldSpec & field, std::string_view common_prefix);

// Per-choice text that must follow the field suffix; strings are JSON escaped
// and end with the closing quote so exhausted candidates disambiguate cleanly.
std::vector<std::string> candidate_texts(const FieldSpec & field, std::string_view common_prefix);

// JSON string body (escaped, without surrounding quotes).
std::string json_escape(std::string_view text);

}  // namespace pcd
