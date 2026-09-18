#pragma once
#include "pcd/model_catalog.hpp"
#include "pcd/pcd_engine.hpp"

#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>

namespace pcd {

class UnknownModel : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

class NoModelLoaded : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Holds the active engine and replaces it transactionally: a replacement is
// fully loaded before the short swap, and a failed load leaves the current
// engine untouched. In-flight decodes keep their engine alive.
class ModelManager {
public:
    ModelManager(ModelCatalog catalog, EngineOptions options);

    // Loads an explicit path (startup override); throws NativeError on failure.
    void load_path(const std::filesystem::path & path);
    // Loads a catalog identifier; throws UnknownModel or NativeError.
    void select(const std::string & id);

    DecodeResponse decode(const DecodeRequest & request);

    std::shared_ptr<PcdEngine> active() const;
    std::optional<std::string> active_id() const;
    const ModelCatalog & catalog() const { return catalog_; }
    const EngineOptions & options() const { return options_; }

private:
    void replace(std::shared_ptr<PcdEngine> engine);

    ModelCatalog catalog_;
    EngineOptions options_;
    mutable std::shared_mutex swap_mutex_;
    std::mutex load_mutex_;  // serializes replacement loads
    std::shared_ptr<PcdEngine> engine_;
};

}  // namespace pcd
