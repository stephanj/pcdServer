#include <catch2/catch_test_macros.hpp>
#include "pcd/model_catalog.hpp"
#include "pcd/model_manager.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

struct TempDir {
    fs::path path;
    TempDir() {
        std::string tmpl = (fs::temp_directory_path() / "pcd-catalog-XXXXXX").string();
        REQUIRE(mkdtemp(tmpl.data()) != nullptr);
        path = tmpl;
    }
    ~TempDir() { fs::remove_all(path); }
    void file(const std::string & name, std::size_t bytes) const {
        std::ofstream out(path / name, std::ios::binary);
        out << std::string(bytes, 'x');
    }
};

pcd::DecodeRequest sample_request() {
    return pcd::parse_decode_request(nlohmann::json::parse(R"({"context":"A normal purchase.","fields":[{"name":"risk","description":"Risk","choices":["LOW","HIGH"]}]})"));
}

}  // namespace

TEST_CASE("catalog lists only regular gguf files in the directory") {
    TempDir dir;
    dir.file("alpha.gguf", 10);
    dir.file("BETA.GGUF", 20);
    dir.file("notes.txt", 5);
    dir.file("archive.gguf.bak", 5);
    fs::create_directories(dir.path / "nested.gguf");
    fs::create_directories(dir.path / "sub");
    dir.file("sub/inner.gguf", 5);
    fs::create_symlink(dir.path / "alpha.gguf", dir.path / "link.gguf");

    pcd::ModelCatalog catalog(dir.path);
    auto models = catalog.list();
    REQUIRE(models.size() == 2);
    REQUIRE(models[0].id == "BETA.GGUF");
    REQUIRE(models[0].bytes == 20);
    REQUIRE(models[1].id == "alpha.gguf");
    REQUIRE(models[1].bytes == 10);
    REQUIRE(models[1].path == dir.path / "alpha.gguf");
}

TEST_CASE("catalog find matches identifiers only") {
    TempDir dir;
    dir.file("alpha.gguf", 10);
    pcd::ModelCatalog catalog(dir.path);
    REQUIRE(catalog.find("alpha.gguf").has_value());
    REQUIRE_FALSE(catalog.find("missing.gguf").has_value());
    REQUIRE_FALSE(catalog.find("../alpha.gguf").has_value());
    REQUIRE_FALSE(catalog.find((dir.path / "alpha.gguf").string()).has_value());
    REQUIRE_FALSE(catalog.find("").has_value());
}

TEST_CASE("catalog tolerates a missing directory") {
    pcd::ModelCatalog catalog(fs::temp_directory_path() / "pcd-does-not-exist");
    REQUIRE(catalog.list().empty());
    REQUIRE_FALSE(catalog.find("x.gguf").has_value());
}

TEST_CASE("manager without a model rejects decode and reports no active model") {
    TempDir dir;
    pcd::ModelManager manager(pcd::ModelCatalog(dir.path), {});
    REQUIRE_FALSE(manager.active_id().has_value());
    REQUIRE(manager.active() == nullptr);
    REQUIRE_THROWS_AS(manager.decode(sample_request()), pcd::NoModelLoaded);
    REQUIRE_THROWS_AS(manager.select("missing.gguf"), pcd::UnknownModel);
}

TEST_CASE("manager keeps the active engine when a replacement fails to load", "[native]") {
    const char * path = std::getenv("PCD_TEST_GGUF");
    if (!path) SKIP("PCD_TEST_GGUF is not set");
    TempDir dir;
    dir.file("corrupt.gguf", 4096);
    pcd::ModelManager manager(pcd::ModelCatalog(dir.path), {});
    manager.load_path(path);
    auto before = manager.active();
    REQUIRE(before != nullptr);
    REQUIRE(manager.active_id() == fs::path(path).filename().string());

    REQUIRE_THROWS_AS(manager.select("missing.gguf"), pcd::UnknownModel);
    REQUIRE_THROWS_AS(manager.select("corrupt.gguf"), pcd::NativeError);
    REQUIRE(manager.active() == before);
    auto response = manager.decode(sample_request());
    REQUIRE(response.model == fs::path(path).filename().string());
}

TEST_CASE("manager switches to a catalog model transactionally", "[native]") {
    const char * path = std::getenv("PCD_TEST_GGUF");
    if (!path) SKIP("PCD_TEST_GGUF is not set");
    TempDir dir;
    fs::copy_file(path, dir.path / "copy.gguf");
    pcd::ModelManager manager(pcd::ModelCatalog(dir.path), {});
    manager.load_path(path);
    auto before = manager.active();
    auto held = manager.decode(sample_request());
    manager.select("copy.gguf");
    REQUIRE(manager.active() != before);
    REQUIRE(manager.active_id() == "copy.gguf");
    // The previous engine stays alive while a caller still holds it.
    REQUIRE(before->decode(sample_request()).model == held.model);
    REQUIRE(manager.decode(sample_request()).model == "copy.gguf");
}
