#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include "pcd/types.hpp"

using nlohmann::json;

TEST_CASE("valid decode requests preserve field order") {
    auto request = pcd::parse_decode_request(json::parse(R"({
      "context":"ticket",
      "fields":[
        {"name":"priority","description":"handling","choices":["LOW","HIGH"]},
        {"name":"fraud","description":"fraudulent","choices":[false,true]}
      ]
    })"));
    REQUIRE(request.context == "ticket");
    REQUIRE(request.fields.size() == 2);
    REQUIRE(request.fields[0].name == "priority");
    REQUIRE(request.fields[0].kind == pcd::FieldKind::StringEnum);
    REQUIRE(request.fields[0].choices.size() == 2);
    REQUIRE(std::get<std::string>(request.fields[0].choices[1]) == "HIGH");
    REQUIRE(request.fields[1].kind == pcd::FieldKind::Boolean);
    REQUIRE(std::get<bool>(request.fields[1].choices[0]) == false);
    REQUIRE(std::get<bool>(request.fields[1].choices[1]) == true);
}

TEST_CASE("duplicate field names are rejected") {
    json body = {{"context", "x"}, {"fields", {
        {{"name", "same"}, {"description", "a"}, {"choices", {"A", "B"}}},
        {{"name", "same"}, {"description", "b"}, {"choices", {"C", "D"}}}
    }}};
    REQUIRE_THROWS_WITH(pcd::parse_decode_request(body), "duplicate field name: same");
}

TEST_CASE("mixed choice types are rejected") {
    json body = {{"context", "x"}, {"fields", {{{"name", "f"}, {"description", "d"}, {"choices", {"A", true}}}}}};
    REQUIRE_THROWS_AS(pcd::parse_decode_request(body), pcd::ValidationError);
}

TEST_CASE("structural problems are rejected") {
    SECTION("missing context") {
        REQUIRE_THROWS_AS(pcd::parse_decode_request(json::parse(R"({"fields":[]})")), pcd::ValidationError);
    }
    SECTION("empty context") {
        REQUIRE_THROWS_AS(pcd::parse_decode_request(json::parse(R"({"context":"","fields":[{"name":"f","description":"d","choices":["A","B"]}]})")), pcd::ValidationError);
    }
    SECTION("no fields") {
        REQUIRE_THROWS_AS(pcd::parse_decode_request(json::parse(R"({"context":"x","fields":[]})")), pcd::ValidationError);
    }
    SECTION("fields not an array") {
        REQUIRE_THROWS_AS(pcd::parse_decode_request(json::parse(R"({"context":"x","fields":{}})")), pcd::ValidationError);
    }
    SECTION("body not an object") {
        REQUIRE_THROWS_AS(pcd::parse_decode_request(json::parse(R"([])")), pcd::ValidationError);
    }
    SECTION("empty field name") {
        REQUIRE_THROWS_AS(pcd::parse_decode_request(json::parse(R"({"context":"x","fields":[{"name":"","description":"d","choices":["A","B"]}]})")), pcd::ValidationError);
    }
    SECTION("too many fields") {
        json fields = json::array();
        for (int i = 0; i < 64; ++i) {
            fields.push_back({{"name", "f" + std::to_string(i)}, {"description", "d"}, {"choices", {"A", "B"}}});
        }
        REQUIRE_THROWS_AS(pcd::parse_decode_request(json{{"context", "x"}, {"fields", fields}}), pcd::ValidationError);
    }
}

TEST_CASE("choice sets are validated") {
    SECTION("empty choices") {
        REQUIRE_THROWS_AS(pcd::parse_decode_request(json::parse(R"({"context":"x","fields":[{"name":"f","description":"d","choices":[]}]})")), pcd::ValidationError);
    }
    SECTION("single choice") {
        REQUIRE_THROWS_AS(pcd::parse_decode_request(json::parse(R"({"context":"x","fields":[{"name":"f","description":"d","choices":["ONLY"]}]})")), pcd::ValidationError);
    }
    SECTION("duplicate string choices") {
        REQUIRE_THROWS_AS(pcd::parse_decode_request(json::parse(R"({"context":"x","fields":[{"name":"f","description":"d","choices":["A","A"]}]})")), pcd::ValidationError);
    }
    SECTION("empty string choice") {
        REQUIRE_THROWS_AS(pcd::parse_decode_request(json::parse(R"({"context":"x","fields":[{"name":"f","description":"d","choices":["A",""]}]})")), pcd::ValidationError);
    }
    SECTION("numeric choices are unsupported") {
        REQUIRE_THROWS_AS(pcd::parse_decode_request(json::parse(R"({"context":"x","fields":[{"name":"f","description":"d","choices":[1,2]}]})")), pcd::ValidationError);
    }
    SECTION("boolean sets must be exactly false and true") {
        REQUIRE_THROWS_AS(pcd::parse_decode_request(json::parse(R"({"context":"x","fields":[{"name":"f","description":"d","choices":[true,true]}]})")), pcd::ValidationError);
        REQUIRE_THROWS_AS(pcd::parse_decode_request(json::parse(R"({"context":"x","fields":[{"name":"f","description":"d","choices":[true]}]})")), pcd::ValidationError);
        REQUIRE_NOTHROW(pcd::parse_decode_request(json::parse(R"({"context":"x","fields":[{"name":"f","description":"d","choices":[true,false]}]})")));
    }
    SECTION("too many choices") {
        json choices = json::array();
        for (int i = 0; i < 257; ++i) choices.push_back("C" + std::to_string(i));
        REQUIRE_THROWS_AS(pcd::parse_decode_request(json{{"context", "x"}, {"fields", {{{"name", "f"}, {"description", "d"}, {"choices", choices}}}}}), pcd::ValidationError);
    }
}

TEST_CASE("responses serialize to the documented JSON shape") {
    pcd::DecodeResponse response;
    response.model = "m.gguf";
    response.fields.push_back({"priority", std::string("HIGH"), 0.85, {0.02, 0.13, 0.85}, {"LOW", "MEDIUM", "HIGH"}, 1});
    response.fields.push_back({"fraudulent", true, 0.7, {0.3, 0.7}, {"false", "true"}, 1});
    response.metrics.elapsed_ms = 108.4;
    response.metrics.forward_passes = 2;
    response.metrics.schema_cache_status = "hit";
    response.metrics.phases = {0.4, 42.0, 3.5, 0.2, 64.9, 0.9};

    auto out = pcd::to_json_response(response);
    REQUIRE(out["model"] == "m.gguf");
    REQUIRE(out["values"]["priority"] == "HIGH");
    REQUIRE(out["values"]["fraudulent"] == true);
    REQUIRE(out["fields"].is_array());
    REQUIRE(out["fields"][0]["name"] == "priority");
    REQUIRE(out["fields"][0]["value"] == "HIGH");
    REQUIRE(out["fields"][0]["probability"].get<double>() == Catch::Approx(0.85));
    REQUIRE(out["fields"][0]["probabilities"]["MEDIUM"].get<double>() == Catch::Approx(0.13));
    REQUIRE(out["fields"][0]["levels"] == 1);
    REQUIRE(out["fields"][1]["value"] == true);
    REQUIRE(out["fields"][1]["probabilities"]["true"].get<double>() == Catch::Approx(0.7));
    REQUIRE(out["metrics"]["elapsedMs"].get<double>() == Catch::Approx(108.4));
    REQUIRE(out["metrics"]["forwardPasses"] == 2);
    REQUIRE(out["metrics"]["schemaCacheStatus"] == "hit");
    REQUIRE(out["metrics"]["phasesMs"]["restoreOrPrefill"].get<double>() == Catch::Approx(42.0));
    REQUIRE(out["metrics"]["phasesMs"]["dynamicContext"].get<double>() == Catch::Approx(3.5));
    REQUIRE(out["metrics"]["phasesMs"]["tree"].get<double>() == Catch::Approx(0.9));
}

TEST_CASE("choices convert to JSON scalars") {
    REQUIRE(pcd::to_json_value(pcd::Choice{true}) == json(true));
    REQUIRE(pcd::to_json_value(pcd::Choice{std::string("X")}) == json("X"));
}
