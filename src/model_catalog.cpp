#include "pcd/model_catalog.hpp"

#include <algorithm>
#include <cctype>
#include <system_error>

namespace pcd {

namespace fs = std::filesystem;

namespace {

bool has_gguf_extension(const fs::path & path) {
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return ext == ".gguf";
}

bool is_within(const fs::path & root, const fs::path & candidate) {
    auto rel = candidate.lexically_relative(root);
    return !rel.empty() && rel.native().rfind("..", 0) != 0;
}

}  // namespace

ModelCatalog::ModelCatalog(fs::path directory) : directory_(std::move(directory)) {}

std::vector<ModelInfo> ModelCatalog::list() const {
    std::vector<ModelInfo> models;
    std::error_code ec;
    if (!fs::is_directory(directory_, ec)) {
        return models;
    }
    const auto root = fs::weakly_canonical(directory_, ec);
    if (ec) {
        return models;
    }
    for (const auto & entry : fs::directory_iterator(directory_, ec)) {
        if (ec) {
            break;
        }
        if (entry.is_symlink(ec) || !entry.is_regular_file(ec) || !has_gguf_extension(entry.path())) {
            continue;
        }
        const auto canonical = fs::weakly_canonical(entry.path(), ec);
        if (ec || !is_within(root, canonical)) {
            continue;
        }
        const auto bytes = entry.file_size(ec);
        if (ec) {
            continue;
        }
        models.push_back({entry.path().filename().string(), entry.path(), bytes});
    }
    std::sort(models.begin(), models.end(), [](const ModelInfo & a, const ModelInfo & b) { return a.id < b.id; });
    return models;
}

std::optional<ModelInfo> ModelCatalog::find(const std::string & id) const {
    if (id.empty() || id.find('/') != std::string::npos || id.find('\\') != std::string::npos) {
        return std::nullopt;
    }
    for (auto & model : list()) {
        if (model.id == id) {
            return model;
        }
    }
    return std::nullopt;
}

}  // namespace pcd
