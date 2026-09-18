#include <catch2/catch_test_macros.hpp>
#include "pcd/http_server.hpp"
#include "pcd/ui_assets.hpp"
#include "pcd/version.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <unistd.h>

using nlohmann::json;
namespace fs = std::filesystem;

namespace {

struct Fixture {
    fs::path dir;
    pcd::ModelManager manager;
    pcd::HttpServer server;
    httplib::Client client;

    static fs::path make_dir() {
        std::string tmpl = (fs::temp_directory_path() / "pcd-http-XXXXXX").string();
        REQUIRE(mkdtemp(tmpl.data()) != nullptr);
        return tmpl;
    }

    Fixture()
        : dir(make_dir()),
          manager(pcd::ModelCatalog(dir), {}),
          server(manager, {.bind = "127.0.0.1", .port = 0}),
          client("127.0.0.1", (server.start(), server.port())) {}
    ~Fixture() { fs::remove_all(dir); }
};

const char * decode_body = R"({"context":"A regular customer paid their monthly subscription.","fields":[
  {"name":"risk","description":"Risk level","choices":["LOW","HIGH"]},
  {"name":"fraudulent","description":"Whether fraud is suspected","choices":[false,true]}]})";

}  // namespace

TEST_CASE("health reports version and readiness without a model") {
    Fixture fx;
    auto res = fx.client.Get("/health");
    REQUIRE(res);
    REQUIRE(res->status == 200);
    REQUIRE(res->get_header_value("Content-Type").find("application/json") == 0);
    auto body = json::parse(res->body);
    REQUIRE(body["apiVersion"] == pcd::api_version);
    REQUIRE(body["ready"] == false);
    REQUIRE(body["model"].is_null());
}

TEST_CASE("models endpoint lists the catalog") {
    Fixture fx;
    auto res = fx.client.Get("/v1/models");
    REQUIRE(res);
    REQUIRE(res->status == 200);
    auto body = json::parse(res->body);
    REQUIRE(body["models"].is_array());
    REQUIRE(body["models"].empty());
    REQUIRE(body["active"].is_null());
}

TEST_CASE("malformed JSON is a 400 with a stable error envelope") {
    Fixture fx;
    auto res = fx.client.Post("/v1/pcd/decode", "{not json", "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 400);
    auto body = json::parse(res->body);
    REQUIRE(body["code"] == "invalid_json");
    REQUIRE(body["message"].is_string());
}

TEST_CASE("invalid schemas are a 400") {
    Fixture fx;
    auto res = fx.client.Post("/v1/pcd/decode", R"({"context":"x","fields":[]})", "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 400);
    REQUIRE(json::parse(res->body)["code"] == "invalid_request");
}

TEST_CASE("unknown model is a 404") {
    Fixture fx;
    auto res = fx.client.Post("/v1/models/select", R"({"id":"missing.gguf"})", "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 404);
    REQUIRE(json::parse(res->body)["code"] == "unknown_model");
    auto invalid = fx.client.Post("/v1/models/select", R"({"id":12})", "application/json");
    REQUIRE(invalid->status == 400);
}

TEST_CASE("decode without an active model is a 503") {
    Fixture fx;
    auto res = fx.client.Post("/v1/pcd/decode", decode_body, "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 503);
    REQUIRE(json::parse(res->body)["code"] == "no_model");
}

TEST_CASE("unknown routes are a JSON 404") {
    Fixture fx;
    auto res = fx.client.Get("/nope");
    REQUIRE(res);
    REQUIRE(res->status == 404);
    REQUIRE(json::parse(res->body)["code"] == "not_found");
}

TEST_CASE("valid decode returns ordered fields and metrics", "[native]") {
    const char * path = std::getenv("PCD_TEST_GGUF");
    if (!path) SKIP("PCD_TEST_GGUF is not set");
    Fixture fx;
    fx.manager.load_path(path);

    auto health = json::parse(fx.client.Get("/health")->body);
    REQUIRE(health["ready"] == true);
    REQUIRE(health["model"] == fs::path(path).filename().string());
    REQUIRE(health["description"].get<std::string>().find("qwen") != std::string::npos);

    auto res = fx.client.Post("/v1/pcd/decode", decode_body, "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 200);
    auto body = json::parse(res->body);
    REQUIRE(body["fields"].size() == 2);
    REQUIRE(body["fields"][0]["name"] == "risk");
    REQUIRE(body["fields"][1]["name"] == "fraudulent");
    REQUIRE(body["values"]["fraudulent"].is_boolean());
    REQUIRE(body["metrics"]["schemaCacheStatus"] == "miss");
    REQUIRE(json::parse(fx.client.Post("/v1/pcd/decode", decode_body, "application/json")->body)["metrics"]["schemaCacheStatus"] == "hit");

    auto models = json::parse(fx.client.Get("/v1/models")->body);
    REQUIRE(models["active"] == fs::path(path).filename().string());
}

TEST_CASE("root serves the embedded playground page") {
    Fixture fx;
    auto res = fx.client.Get("/");
    REQUIRE(res);
    REQUIRE(res->status == 200);
    REQUIRE(res->get_header_value("Content-Type").find("text/html") == 0);
    REQUIRE(res->body.find("<!doctype html>") != std::string::npos);
    REQUIRE(res->body.find("/v1/pcd/decode") != std::string::npos);
    REQUIRE(res->body.size() == pcd::ui_index_html().size());
}

TEST_CASE("index.html is an alias for the playground") {
    Fixture fx;
    auto res = fx.client.Get("/index.html");
    REQUIRE(res);
    REQUIRE(res->status == 200);
    REQUIRE(res->get_header_value("Content-Type").find("text/html") == 0);
}
