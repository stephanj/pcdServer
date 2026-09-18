#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "pcd/pcd_engine.hpp"

#include <cstdlib>
#include <numeric>

namespace {

const char * model_path() {
    return std::getenv("PCD_TEST_GGUF");
}

pcd::DecodeRequest fraud_request(const char * context = "Payment from a new country exceeded the normal amount.") {
    nlohmann::json body = {
        {"context", context},
        {"fields", {
            {{"name", "risk"}, {"description", "Risk level"}, {"choices", {"LOW", "MEDIUM", "HIGH"}}},
            {{"name", "review"}, {"description", "Needs manual review"}, {"choices", {false, true}}},
        }},
    };
    return pcd::parse_decode_request(body);
}

pcd::DecodeRequest tier_request() {
    return pcd::parse_decode_request(nlohmann::json::parse(R"({
      "context":"The customer disputes a small recurring charge.",
      "fields":[
        {"name":"tier","description":"Escalation tier","choices":["TIER_1_LOW","TIER_2_MEDIUM","TIER_3_HIGH"]},
        {"name":"category","description":"Ticket category","choices":["BILLING","BILLING_DISPUTE","FRAUD"]}
      ]
    })"));
}

double sum(const std::vector<double> & values) {
    return std::accumulate(values.begin(), values.end(), 0.0);
}

void require_valid(const pcd::DecodeResponse & response, const pcd::DecodeRequest & request) {
    REQUIRE(response.fields.size() == request.fields.size());
    for (std::size_t i = 0; i < request.fields.size(); ++i) {
        const auto & field = response.fields[i];
        const auto & spec = request.fields[i];
        REQUIRE(field.name == spec.name);
        REQUIRE(field.probabilities.size() == spec.choices.size());
        REQUIRE(sum(field.probabilities) == Catch::Approx(1.0).margin(1e-5));
        bool allowed = false;
        for (const auto & choice : spec.choices) {
            allowed = allowed || choice == field.value;
        }
        REQUIRE(allowed);
        REQUIRE(field.probability >= 0.0);
        REQUIRE(field.probability <= 1.0 + 1e-9);
        REQUIRE(field.levels >= 1);
    }
}

}  // namespace

TEST_CASE("Qwen cached decode matches cold decode", "[native]") {
    const char * path = model_path();
    if (!path) SKIP("PCD_TEST_GGUF is not set");
    pcd::PcdEngine engine(path);
    auto request = fraud_request();
    auto cold = engine.decode(request);
    auto cached = engine.decode(request);
    REQUIRE(cold.metrics.schema_cache_status == "miss");
    REQUIRE(cached.metrics.schema_cache_status == "hit");
    REQUIRE(cached.metrics.checkpoint_bytes > 0);
    REQUIRE(cached.metrics.checkpoint_bytes == cold.metrics.checkpoint_bytes);
    require_valid(cold, request);
    require_valid(cached, request);
    REQUIRE(cached.fields.size() == cold.fields.size());
    for (std::size_t i = 0; i < cold.fields.size(); ++i) {
        REQUIRE(cached.fields[i].value == cold.fields[i].value);
        REQUIRE(cached.fields[i].probability == Catch::Approx(cold.fields[i].probability).margin(1e-5));
        for (std::size_t j = 0; j < cold.fields[i].probabilities.size(); ++j) {
            REQUIRE(cached.fields[i].probabilities[j] == Catch::Approx(cold.fields[i].probabilities[j]).margin(1e-5));
        }
    }
    REQUIRE(cold.model == engine.model_id());
    REQUIRE(cold.metrics.forward_passes >= 2);
}

TEST_CASE("Qwen alternating contexts reuse one schema checkpoint", "[native]") {
    const char * path = model_path();
    if (!path) SKIP("PCD_TEST_GGUF is not set");
    pcd::PcdEngine engine(path);
    auto a = fraud_request("A regular customer paid their usual monthly subscription.");
    auto b = fraud_request("Fifteen rapid transfers to unknown accounts were made at 3am from a new device.");
    auto first = engine.decode(a);
    auto second = engine.decode(b);
    auto third = engine.decode(a);
    REQUIRE(first.metrics.schema_cache_status == "miss");
    REQUIRE(second.metrics.schema_cache_status == "hit");
    REQUIRE(third.metrics.schema_cache_status == "hit");
    require_valid(first, a);
    require_valid(second, b);
    require_valid(third, a);
    // The same context must reproduce the same answer after an intervening request.
    for (std::size_t i = 0; i < first.fields.size(); ++i) {
        REQUIRE(third.fields[i].value == first.fields[i].value);
        REQUIRE(third.fields[i].probability == Catch::Approx(first.fields[i].probability).margin(1e-5));
    }
    // Different contexts must not produce byte-identical probability vectors.
    bool differs = false;
    for (std::size_t i = 0; i < first.fields.size(); ++i) {
        differs = differs || std::abs(second.fields[i].probability - first.fields[i].probability) > 1e-9;
    }
    REQUIRE(differs);
}

TEST_CASE("Qwen alternating schemas keep separate checkpoints", "[native]") {
    const char * path = model_path();
    if (!path) SKIP("PCD_TEST_GGUF is not set");
    pcd::PcdEngine engine(path);
    auto a = fraud_request();
    auto b = tier_request();
    auto cold_a = engine.decode(a);
    auto cold_b = engine.decode(b);
    auto hit_a = engine.decode(a);
    auto hit_b = engine.decode(b);
    REQUIRE(cold_a.metrics.schema_cache_status == "miss");
    REQUIRE(cold_b.metrics.schema_cache_status == "miss");
    REQUIRE(hit_a.metrics.schema_cache_status == "hit");
    REQUIRE(hit_b.metrics.schema_cache_status == "hit");
    require_valid(cold_b, b);
    require_valid(hit_b, b);
    for (std::size_t i = 0; i < cold_a.fields.size(); ++i) {
        REQUIRE(hit_a.fields[i].value == cold_a.fields[i].value);
        REQUIRE(hit_a.fields[i].probability == Catch::Approx(cold_a.fields[i].probability).margin(1e-5));
    }
    for (std::size_t i = 0; i < cold_b.fields.size(); ++i) {
        REQUIRE(hit_b.fields[i].value == cold_b.fields[i].value);
        REQUIRE(hit_b.fields[i].probability == Catch::Approx(cold_b.fields[i].probability).margin(1e-5));
    }
    REQUIRE(engine.cache().size() == 2);
    const auto stats = engine.cache_stats();
    REQUIRE(stats.entries == engine.cache().size());
    REQUIRE(stats.bytes == engine.cache().total_bytes());
    REQUIRE(stats.bytes == cold_a.metrics.checkpoint_bytes + cold_b.metrics.checkpoint_bytes);
}

TEST_CASE("Qwen shared-prefix choices resolve through the collision tree", "[native]") {
    const char * path = model_path();
    if (!path) SKIP("PCD_TEST_GGUF is not set");
    pcd::PcdEngine engine(path);
    auto request = tier_request();
    auto response = engine.decode(request);
    require_valid(response, request);
    // BILLING vs BILLING_DISPUTE share every token of BILLING, so at least two levels are needed
    // unless FRAUD wins outright at level one.
    const auto & category = response.fields[1];
    if (std::get<std::string>(category.value) != "FRAUD") {
        REQUIRE(category.levels >= 2);
    }
}

TEST_CASE("Qwen invalid checkpoint falls back to cold prefill", "[native]") {
    const char * path = model_path();
    if (!path) SKIP("PCD_TEST_GGUF is not set");
    pcd::PcdEngine engine(path);
    auto request = fraud_request();
    auto cold = engine.decode(request);
    const auto key = engine.schema_key(request.fields);
    auto entry = engine.cache().get(key);
    REQUIRE(entry != nullptr);
    REQUIRE(entry->checkpoint != nullptr);

    auto corrupt = std::make_shared<pcd::SequenceCheckpoint>(*entry->checkpoint);
    corrupt->bytes.resize(corrupt->bytes.size() / 2);
    engine.cache().put(key, pcd::SchemaCacheEntry{entry->schema, corrupt});

    auto fallback = engine.decode(request);
    REQUIRE(fallback.metrics.schema_cache_status == "fallback");
    require_valid(fallback, request);
    for (std::size_t i = 0; i < cold.fields.size(); ++i) {
        REQUIRE(fallback.fields[i].value == cold.fields[i].value);
        REQUIRE(fallback.fields[i].probability == Catch::Approx(cold.fields[i].probability).margin(1e-5));
    }
    // The fallback re-published a valid checkpoint, so the next request hits again.
    REQUIRE(engine.decode(request).metrics.schema_cache_status == "hit");
}

TEST_CASE("Qwen repeated cache hits do not accumulate context", "[native]") {
    const char * path = model_path();
    if (!path) SKIP("PCD_TEST_GGUF is not set");
    pcd::PcdEngine engine(path);
    auto request = fraud_request();
    auto cold = engine.decode(request);
    auto baseline = engine.decode(request);
    REQUIRE(baseline.metrics.schema_cache_status == "hit");
    REQUIRE(baseline.metrics.forward_passes < cold.metrics.forward_passes);
    for (int i = 0; i < 20; ++i) {
        auto again = engine.decode(request);
        REQUIRE(again.metrics.schema_cache_status == "hit");
        REQUIRE(again.metrics.forward_passes == baseline.metrics.forward_passes);
        for (std::size_t f = 0; f < baseline.fields.size(); ++f) {
            REQUIRE(again.fields[f].value == baseline.fields[f].value);
            REQUIRE(again.fields[f].probability == Catch::Approx(baseline.fields[f].probability).margin(1e-5));
        }
    }
}

TEST_CASE("Qwen rejects requests that cannot fit the context window", "[native]") {
    const char * path = model_path();
    if (!path) SKIP("PCD_TEST_GGUF is not set");
    pcd::EngineOptions options;
    options.runtime.context_size = 512;
    options.runtime.batch_size = 512;
    options.runtime.max_sequences = 4;
    pcd::PcdEngine engine(path, options);
    std::string huge(20000, 'x');
    for (std::size_t i = 0; i < huge.size(); i += 7) huge[i] = ' ';
    REQUIRE_THROWS_AS(engine.decode(fraud_request(huge.c_str())), pcd::ValidationError);
    // Too many fields for the configured sequence count is also a client error.
    nlohmann::json fields = nlohmann::json::array();
    for (int i = 0; i < 5; ++i) {
        fields.push_back({{"name", "f" + std::to_string(i)}, {"description", "d"}, {"choices", {"A", "B"}}});
    }
    REQUIRE_THROWS_AS(engine.decode(pcd::parse_decode_request({{"context", "x"}, {"fields", fields}})), pcd::ValidationError);
    // A normal request still works on the small context.
    REQUIRE(engine.decode(fraud_request()).metrics.schema_cache_status == "miss");
}
