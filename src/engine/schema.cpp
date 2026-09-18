#include "pcd/schema.hpp"

#include <algorithm>

namespace pcd {

using nlohmann::json;

std::string common_prefix(const std::vector<std::string> & choices) {
    if (choices.empty()) {
        return {};
    }
    std::string prefix = choices.front();
    for (const auto & choice : choices) {
        auto mismatch = std::mismatch(prefix.begin(), prefix.end(), choice.begin(), choice.end());
        prefix.erase(mismatch.first, prefix.end());
        if (prefix.empty()) {
            break;
        }
    }
    return prefix;
}

std::string canonical_schema(const std::vector<FieldSpec> & fields) {
    json canonical = json::array();
    for (const auto & field : fields) {
        json choices = json::array();
        for (const auto & choice : field.choices) {
            choices.push_back(to_json_value(choice));
        }
        // Ordered array of arrays keeps key order independent of any object sorting.
        canonical.push_back(json::array({
            field.name,
            field.description,
            field.kind == FieldKind::Boolean ? "boolean" : "enum",
            std::move(choices),
        }));
    }
    return std::string(prompt_format_version) + ":" + canonical.dump();
}

TemplateParts split_template(std::string_view rendered, std::string_view marker) {
    const auto first = rendered.find(marker);
    if (marker.empty() || first == std::string_view::npos) {
        throw SchemaError("chat template did not preserve user marker");
    }
    if (rendered.find(marker, first + marker.size()) != std::string_view::npos) {
        throw SchemaError("chat template duplicated user marker");
    }
    return {
        std::string(rendered.substr(0, first)),
        std::string(rendered.substr(first + marker.size())),
    };
}

std::string json_escape(std::string_view text) {
    auto dumped = json(std::string(text)).dump();
    return dumped.substr(1, dumped.size() - 2);
}

std::string schema_system_prompt(const std::vector<FieldSpec> & fields) {
    std::string prompt =
        "You are a precise structured-data extraction engine. Read the user's text and "
        "decide the value of every field below. Each field must be set to exactly one of "
        "its allowed values. Respond only with a JSON object containing every field.\n\n"
        "Fields:\n";
    for (const auto & field : fields) {
        prompt += "- \"" + json_escape(field.name) + "\"";
        if (!field.description.empty()) {
            prompt += ": " + field.description;
        }
        prompt += ". Allowed values: ";
        for (std::size_t i = 0; i < field.choices.size(); ++i) {
            if (i > 0) {
                prompt += ", ";
            }
            prompt += to_json_value(field.choices[i]).dump();
        }
        prompt += "\n";
    }
    prompt += "\nOutput format:\n{\n";
    for (std::size_t i = 0; i < fields.size(); ++i) {
        prompt += "  \"" + json_escape(fields[i].name) + "\": <value>";
        prompt += (i + 1 < fields.size()) ? ",\n" : "\n";
    }
    prompt += "}";
    return prompt;
}

std::string field_suffix_text(const FieldSpec & field, std::string_view common_prefix) {
    std::string text = "  \"" + json_escape(field.name) + "\": ";
    if (field.kind == FieldKind::StringEnum) {
        text += "\"";
        text += json_escape(common_prefix);
    }
    return text;
}

std::vector<std::string> candidate_texts(const FieldSpec & field, std::string_view common_prefix) {
    std::vector<std::string> out;
    out.reserve(field.choices.size());
    for (const auto & choice : field.choices) {
        if (field.kind == FieldKind::Boolean) {
            out.push_back(choice_label(choice));
            continue;
        }
        const auto & text = std::get<std::string>(choice);
        out.push_back(json_escape(std::string_view(text).substr(common_prefix.size())) + "\"");
    }
    return out;
}

}  // namespace pcd
