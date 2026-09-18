#include <catch2/catch_test_macros.hpp>
#include "pcd/version.hpp"

TEST_CASE("server has a stable API version") {
    REQUIRE(pcd::api_version == "v1");
}
