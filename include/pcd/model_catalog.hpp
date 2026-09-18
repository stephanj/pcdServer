#pragma once
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace pcd {

struct ModelInfo {
    std::string id;               // file name, the only identifier clients may use
    std::filesystem::path path;
    std::uintmax_t bytes{};
};

// Discovers regular `.gguf` files directly inside one directory. Client input
// is matched against identifiers only and never interpreted as a path.
class ModelCatalog {
public:
    explicit ModelCatalog(std::filesystem::path directory);

    // Rescans the directory on every call, sorted by identifier.
    std::vector<ModelInfo> list() const;
    std::optional<ModelInfo> find(const std::string & id) const;
    const std::filesystem::path & directory() const { return directory_; }

private:
    std::filesystem::path directory_;
};

}  // namespace pcd
