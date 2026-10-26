#pragma once

#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>

namespace pcd::test {

// Portable replacement for mkdtemp: creates a fresh, uniquely named directory
// under the system temp directory and returns its path.
inline std::filesystem::path make_temp_dir(const std::string & prefix) {
    namespace fs = std::filesystem;
    std::random_device seed;
    std::mt19937_64 rng(seed());
    for (int attempt = 0; attempt < 100; ++attempt) {
        const auto path = fs::temp_directory_path() / (prefix + std::to_string(rng()));
        if (fs::create_directory(path)) {
            return path;
        }
    }
    throw std::runtime_error("could not create a temporary directory");
}

}  // namespace pcd::test
