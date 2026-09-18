#pragma once
#include <nlohmann/json.hpp>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace pcd {

// Bounded request limits. Requests exceeding these are rejected before any
// native allocation happens.
inline constexpr std::size_t max_fields = 63;
inline constexpr std::size_t min_choices = 2;
inline constexpr std::size_t max_choices = 256;
inline constexpr std::size_t max_context_bytes = 64 * 1024;
inline constexpr std::size_t max_name_bytes = 128;
inline constexpr std::size_t max_choice_bytes = 256;
inline constexpr std::size_t max_description_bytes = 1024;

using Choice = std::variant<bool, std::string>;
enum class FieldKind { Boolean, StringEnum };

struct FieldSpec {
    std::string name;
    std::string description;
    FieldKind kind;
    std::vector<Choice> choices;
};

struct DecodeRequest {
    std::string context;
    std::vector<FieldSpec> fields;
};

struct FieldResult {
    std::string name;
    Choice value;
    double probability;
    std::vector<double> probabilities;   // parallel to labels, same order as the request choices
    std::vector<std::string> labels;     // JSON key for each choice ("true"/"false" for booleans)
    int levels;
};

struct PhaseTimings {
    double tokenize_ms{}, restore_or_prefill_ms{}, dynamic_context_ms{}, broadcast_ms{}, suffix_ms{}, tree_ms{};
};

struct DecodeMetrics {
    double elapsed_ms{};
    int forward_passes{};
    std::string schema_cache_status;  // "miss", "hit" or "fallback"
    std::size_t checkpoint_bytes{};
    PhaseTimings phases;
};

struct DecodeResponse {
    std::string model;
    std::vector<FieldResult> fields;
    DecodeMetrics metrics;
};

// Client-caused request problems (HTTP 400).
class ValidationError : public std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

DecodeRequest parse_decode_request(const nlohmann::json & body);
std::string choice_label(const Choice & choice);
nlohmann::json to_json_value(const Choice & choice);
nlohmann::json to_json_response(const DecodeResponse & response);

}  // namespace pcd
