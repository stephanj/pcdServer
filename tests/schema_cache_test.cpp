#include <catch2/catch_test_macros.hpp>
#include "pcd/schema_cache.hpp"

namespace {
pcd::SchemaCacheEntry fixture_entry(std::size_t bytes) {
    auto schema = std::make_shared<pcd::CompiledSchema>();
    auto checkpoint = std::make_shared<pcd::SequenceCheckpoint>();
    checkpoint->bytes.resize(bytes);
    checkpoint->expected_size = bytes;
    return pcd::SchemaCacheEntry{std::move(schema), std::move(checkpoint)};
}
}

TEST_CASE("cache evicts least recently used entries by bytes") {
    pcd::SchemaCache cache({.max_entries = 3, .max_bytes = 10});
    cache.put("a", fixture_entry(4));
    cache.put("b", fixture_entry(4));
    REQUIRE(cache.get("a") != nullptr);
    cache.put("c", fixture_entry(4));
    REQUIRE(cache.get("b") == nullptr);
    REQUIRE(cache.get("a") != nullptr);
    REQUIRE(cache.get("c") != nullptr);
    REQUIRE(cache.total_bytes() == 8);
    REQUIRE(cache.size() == 2);
}

TEST_CASE("cache evicts by entry count") {
    pcd::SchemaCache cache({.max_entries = 2, .max_bytes = 1000});
    cache.put("a", fixture_entry(1));
    cache.put("b", fixture_entry(1));
    cache.put("c", fixture_entry(1));
    REQUIRE(cache.get("a") == nullptr);
    REQUIRE(cache.size() == 2);
}

TEST_CASE("replacing an entry accounts bytes exactly") {
    pcd::SchemaCache cache({.max_entries = 3, .max_bytes = 100});
    cache.put("a", fixture_entry(10));
    cache.put("a", fixture_entry(30));
    REQUIRE(cache.total_bytes() == 30);
    REQUIRE(cache.size() == 1);
    cache.erase("a");
    REQUIRE(cache.total_bytes() == 0);
    REQUIRE(cache.get("a") == nullptr);
}

TEST_CASE("entries larger than the byte budget are not stored") {
    pcd::SchemaCache cache({.max_entries = 3, .max_bytes = 10});
    cache.put("big", fixture_entry(11));
    REQUIRE(cache.get("big") == nullptr);
    REQUIRE(cache.total_bytes() == 0);
}

TEST_CASE("entries without a checkpoint weigh nothing") {
    pcd::SchemaCache cache({.max_entries = 3, .max_bytes = 10});
    cache.put("s", pcd::SchemaCacheEntry{std::make_shared<pcd::CompiledSchema>(), nullptr});
    REQUIRE(cache.get("s") != nullptr);
    REQUIRE(cache.get("s")->checkpoint == nullptr);
    REQUIRE(cache.total_bytes() == 0);
}

TEST_CASE("two restore failures disable caching for a key until reset") {
    pcd::SchemaCache cache({.max_entries = 3, .max_bytes = 100});
    cache.put("a", fixture_entry(10));
    REQUIRE_FALSE(cache.disabled("a"));
    cache.record_restore_failure("a");
    REQUIRE(cache.get("a") == nullptr);
    REQUIRE(cache.total_bytes() == 0);
    REQUIRE_FALSE(cache.disabled("a"));
    cache.put("a", fixture_entry(10));
    cache.record_restore_failure("a");
    REQUIRE(cache.disabled("a"));
    cache.put("a", fixture_entry(10));
    REQUIRE(cache.get("a") == nullptr);
    REQUIRE(cache.total_bytes() == 0);
    cache.clear();
    REQUIRE_FALSE(cache.disabled("a"));
    cache.put("a", fixture_entry(10));
    REQUIRE(cache.get("a") != nullptr);
}

TEST_CASE("entries are immutable shared snapshots") {
    pcd::SchemaCache cache({.max_entries = 3, .max_bytes = 100});
    cache.put("a", fixture_entry(10));
    auto first = cache.get("a");
    cache.erase("a");
    REQUIRE(first != nullptr);
    REQUIRE(first->checkpoint->bytes.size() == 10);
}
