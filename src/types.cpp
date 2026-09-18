#include "pcd/types.hpp"

#include <unordered_set>

namespace pcd {

using nlohmann::json;

namespace {

const json & require_member(const json & object, const char * key, json::value_t type, const char * what) {
    auto it = object.find(key);
    if (it == object.end()) {
        throw ValidationError(std::string("missing ") + what);
    }
    if (it->type() != type) {
        throw ValidationError(std::string(what) + " has the wrong type");
    }
    return *it;
}

std::string require_string(const json & object, const char * key, const char * what, std::size_t max_bytes, bool allow_empty) {
    auto value = require_member(object, key, json::value_t::string, what).get<std::string>();
    if (!allow_empty && value.empty()) {
        throw ValidationError(std::string(what) + " must not be empty");
    }
    if (value.size() > max_bytes) {
        throw ValidationError(std::string(what) + " exceeds " + std::to_string(max_bytes) + " bytes");
    }
    return value;
}

FieldSpec parse_field(const json & field) {
    if (!field.is_object()) {
        throw ValidationError("field must be an object");
    }
    FieldSpec spec;
    spec.name = require_string(field, "name", "field name", max_name_bytes, false);
    spec.description = require_string(field, "description", "field description", max_description_bytes, true);

    const auto & choices = require_member(field, "choices", json::value_t::array, "field choices");
    if (choices.size() < min_choices) {
        throw ValidationError("field " + spec.name + " needs at least " + std::to_string(min_choices) + " choices");
    }
    if (choices.size() > max_choices) {
        throw ValidationError("field " + spec.name + " exceeds " + std::to_string(max_choices) + " choices");
    }

    const bool boolean = choices.front().is_boolean();
    if (!boolean && !choices.front().is_string()) {
        throw ValidationError("field " + spec.name + " choices must be booleans or strings");
    }
    spec.kind = boolean ? FieldKind::Boolean : FieldKind::StringEnum;

    std::unordered_set<std::string> seen;
    for (const auto & choice : choices) {
        if (boolean) {
            if (!choice.is_boolean()) {
                throw ValidationError("field " + spec.name + " mixes choice types");
            }
            spec.choices.emplace_back(choice.get<bool>());
        } else {
            if (!choice.is_string()) {
                throw ValidationError("field " + spec.name + " mixes choice types");
            }
            auto text = choice.get<std::string>();
            if (text.empty()) {
                throw ValidationError("field " + spec.name + " has an empty choice");
            }
            if (text.size() > max_choice_bytes) {
                throw ValidationError("field " + spec.name + " has a choice longer than " + std::to_string(max_choice_bytes) + " bytes");
            }
            spec.choices.emplace_back(std::move(text));
        }
        if (!seen.insert(choice_label(spec.choices.back())).second) {
            throw ValidationError("field " + spec.name + " has duplicate choice " + choice_label(spec.choices.back()));
        }
    }
    if (boolean && spec.choices.size() != 2) {
        throw ValidationError("field " + spec.name + " must offer exactly false and true");
    }
    return spec;
}

}  // namespace

DecodeRequest parse_decode_request(const json & body) {
    if (!body.is_object()) {
        throw ValidationError("request body must be a JSON object");
    }
    DecodeRequest request;
    request.context = require_string(body, "context", "context", max_context_bytes, false);

    const auto & fields = require_member(body, "fields", json::value_t::array, "fields");
    if (fields.empty()) {
        throw ValidationError("fields must not be empty");
    }
    if (fields.size() > max_fields) {
        throw ValidationError("fields exceeds " + std::to_string(max_fields) + " entries");
    }

    std::unordered_set<std::string> names;
    for (const auto & field : fields) {
        auto spec = parse_field(field);
        if (!names.insert(spec.name).second) {
            throw ValidationError("duplicate field name: " + spec.name);
        }
        request.fields.push_back(std::move(spec));
    }
    return request;
}

std::string choice_label(const Choice & choice) {
    if (const auto * flag = std::get_if<bool>(&choice)) {
        return *flag ? "true" : "false";
    }
    return std::get<std::string>(choice);
}

json to_json_value(const Choice & choice) {
    if (const auto * flag = std::get_if<bool>(&choice)) {
        return json(*flag);
    }
    return json(std::get<std::string>(choice));
}

json to_json_response(const DecodeResponse & response) {
    json values = json::object();
    json fields = json::array();
    for (const auto & field : response.fields) {
        json probabilities = json::object();
        for (std::size_t i = 0; i < field.labels.size() && i < field.probabilities.size(); ++i) {
            probabilities[field.labels[i]] = field.probabilities[i];
        }
        values[field.name] = to_json_value(field.value);
        fields.push_back({
            {"name", field.name},
            {"value", to_json_value(field.value)},
            {"probability", field.probability},
            {"probabilities", std::move(probabilities)},
            {"levels", field.levels},
        });
    }
    const auto & m = response.metrics;
    return {
        {"model", response.model},
        {"values", std::move(values)},
        {"fields", std::move(fields)},
        {"metrics", {
            {"elapsedMs", m.elapsed_ms},
            {"forwardPasses", m.forward_passes},
            {"schemaCacheStatus", m.schema_cache_status},
            {"checkpointBytes", m.checkpoint_bytes},
            {"phasesMs", {
                {"tokenize", m.phases.tokenize_ms},
                {"restoreOrPrefill", m.phases.restore_or_prefill_ms},
                {"dynamicContext", m.phases.dynamic_context_ms},
                {"broadcast", m.phases.broadcast_ms},
                {"suffix", m.phases.suffix_ms},
                {"tree", m.phases.tree_ms},
            }},
        }},
    };
}

}  // namespace pcd
