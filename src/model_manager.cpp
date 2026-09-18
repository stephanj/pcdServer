#include "pcd/model_manager.hpp"

namespace pcd {

ModelManager::ModelManager(ModelCatalog catalog, EngineOptions options)
    : catalog_(std::move(catalog)), options_(std::move(options)) {}

void ModelManager::load_path(const std::filesystem::path & path) {
    std::lock_guard<std::mutex> load(load_mutex_);
    replace(std::make_shared<PcdEngine>(path, options_));
}

void ModelManager::select(const std::string & id) {
    auto info = catalog_.find(id);
    if (!info) {
        throw UnknownModel("unknown model: " + id);
    }
    std::lock_guard<std::mutex> load(load_mutex_);
    // Construction happens outside the swap lock so readers keep decoding on
    // the current engine; a throwing constructor leaves engine_ untouched.
    replace(std::make_shared<PcdEngine>(info->path, options_));
}

void ModelManager::replace(std::shared_ptr<PcdEngine> engine) {
    std::shared_ptr<PcdEngine> previous;
    {
        std::unique_lock<std::shared_mutex> lock(swap_mutex_);
        previous = std::move(engine_);
        engine_ = std::move(engine);
    }
    // `previous` is destroyed here, outside the lock, once no decode holds it.
}

DecodeResponse ModelManager::decode(const DecodeRequest & request) {
    auto engine = active();
    if (!engine) {
        throw NoModelLoaded("no model is loaded");
    }
    return engine->decode(request);
}

std::shared_ptr<PcdEngine> ModelManager::active() const {
    std::shared_lock<std::shared_mutex> lock(swap_mutex_);
    return engine_;
}

std::optional<std::string> ModelManager::active_id() const {
    auto engine = active();
    if (!engine) {
        return std::nullopt;
    }
    return engine->model_id();
}

}  // namespace pcd
