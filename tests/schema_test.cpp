#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "pcd/schema.hpp"

using Catch::Matchers::ContainsSubstring;

namespace {
pcd::DecodeRequest sample_request() {
    return pcd::parse_decode_request(nlohmann::json::parse(R"({"context":"x","fields":[{"name":"a","description":"first","choices":["X","Y"]},{"name":"b","description":"second","choices":[false,true]}]})"));
}
}

TEST_CASE("common prefix is computed over every choice") {
    REQUIRE(pcd::common_prefix({"TIER_1_LOW", "TIER_3_HIGH"}) == "TIER_");
    REQUIRE(pcd::common_prefix({"LOW", "MEDIUM", "HIGH"}).empty());
    REQUIRE(pcd::common_prefix({"LOW", "LOWER"}) == "LOW");
    REQUIRE(pcd::common_prefix({}).empty());
}

TEST_CASE("schema keys change when order changes") {
    auto a = sample_request();
    auto b = a;
    std::swap(b.fields[0], b.fields[1]);
    REQUIRE(pcd::canonical_schema(a.fields) != pcd::canonical_schema(b.fields));
    REQUIRE(pcd::canonical_schema(a.fields) == pcd::canonical_schema(sample_request().fields));
    REQUIRE_THAT(pcd::canonical_schema(a.fields), ContainsSubstring("pcd-prompt-v1"));
}

TEST_CASE("schema keys distinguish descriptions and choice order") {
    auto a = sample_request();
    auto b = a;
    b.fields[0].description = "changed";
    REQUIRE(pcd::canonical_schema(a.fields) != pcd::canonical_schema(b.fields));
    auto c = a;
    std::swap(c.fields[0].choices[0], c.fields[0].choices[1]);
    REQUIRE(pcd::canonical_schema(a.fields) != pcd::canonical_schema(c.fields));
}

TEST_CASE("rendered templates must preserve the user marker") {
    REQUIRE_THROWS_WITH(pcd::split_template("no marker", "MARK"), "chat template did not preserve user marker");
    REQUIRE_THROWS_AS(pcd::split_template("no marker", "MARK"), pcd::SchemaError);
    REQUIRE_THROWS_AS(pcd::split_template("MARK twice MARK", "MARK"), pcd::SchemaError);
    auto parts = pcd::split_template("<s>system\nuser: MARK<e>assistant\n", "MARK");
    REQUIRE(parts.prefix == "<s>system\nuser: ");
    REQUIRE(parts.suffix == "<e>assistant\n");
}

TEST_CASE("system prompt lists every field, description and choice") {
    auto prompt = pcd::schema_system_prompt(sample_request().fields);
    REQUIRE_THAT(prompt, ContainsSubstring("\"a\""));
    REQUIRE_THAT(prompt, ContainsSubstring("first"));
    REQUIRE_THAT(prompt, ContainsSubstring("\"X\""));
    REQUIRE_THAT(prompt, ContainsSubstring("\"Y\""));
    REQUIRE_THAT(prompt, ContainsSubstring("\"b\""));
    REQUIRE_THAT(prompt, ContainsSubstring("second"));
    REQUIRE_THAT(prompt, ContainsSubstring("false"));
    REQUIRE_THAT(prompt, ContainsSubstring("true"));
}

TEST_CASE("field suffix text opens the JSON member for each kind") {
    auto fields = sample_request().fields;
    REQUIRE(pcd::field_suffix_text(fields[0], "") == "  \"a\": \"");
    REQUIRE(pcd::field_suffix_text(fields[0], "TIER_") == "  \"a\": \"TIER_");
    REQUIRE(pcd::field_suffix_text(fields[1], "") == "  \"b\": ");
}

TEST_CASE("candidate remainders are JSON escaped and strings are closed") {
    auto fields = sample_request().fields;
    auto strings = pcd::candidate_texts(fields[0], "");
    REQUIRE(strings == std::vector<std::string>{"X\"", "Y\""});
    auto booleans = pcd::candidate_texts(fields[1], "");
    REQUIRE(booleans == std::vector<std::string>{"false", "true"});

    pcd::FieldSpec quoted{"q", "d", pcd::FieldKind::StringEnum, {std::string("a\"b"), std::string("a\\c")}};
    REQUIRE(pcd::candidate_texts(quoted, "a") == std::vector<std::string>{"\\\"b\"", "\\\\c\""});
}
