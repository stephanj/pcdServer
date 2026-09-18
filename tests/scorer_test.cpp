#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "pcd/scorer.hpp"

#include <cmath>
#include <numeric>

namespace {
double sum(const std::vector<double> & values) {
    return std::accumulate(values.begin(), values.end(), 0.0);
}
}

TEST_CASE("unique first tokens resolve in one level") {
    pcd::CandidateState state({{10}, {20}, {30}}, {"A", "B", "C"});
    REQUIRE_FALSE(state.resolved());
    auto result = state.advance([](int token) { return token == 20 ? 4.0 : 1.0; });
    REQUIRE(result.resolved);
    REQUIRE(state.resolved());
    REQUIRE(state.winner() == 1);
    REQUIRE(state.levels() == 1);
    REQUIRE(sum(state.probabilities()) == Catch::Approx(1.0));
    REQUIRE(state.probabilities()[1] > state.probabilities()[0]);
    REQUIRE(state.probabilities()[0] == Catch::Approx(state.probabilities()[2]));
}

TEST_CASE("shared first tokens remain live for another level") {
    pcd::CandidateState state({{10, 11}, {10, 12}, {20}}, {"AX", "AY", "B"});
    REQUIRE_FALSE(state.advance([](int token) { return token == 10 ? 5.0 : 1.0; }).resolved);
    REQUIRE(state.last_token() == 10);
    REQUIRE(state.live_count() == 2);
    REQUIRE(state.advance([](int token) { return token == 12 ? 3.0 : 0.0; }).resolved);
    REQUIRE(state.winner() == 1);
    REQUIRE(state.levels() == 2);
    REQUIRE(sum(state.probabilities()) == Catch::Approx(1.0));
    // The eliminated choice keeps the mass of its first-level token.
    REQUIRE(state.probabilities()[2] == Catch::Approx(std::exp(1.0) / (std::exp(5.0) + std::exp(1.0))));
    REQUIRE(state.probabilities()[1] > state.probabilities()[0]);
}

TEST_CASE("eliminated groups share their mass evenly") {
    pcd::CandidateState state({{10, 1}, {10, 2}, {20}}, {"A1", "A2", "B"});
    REQUIRE(state.advance([](int token) { return token == 20 ? 1.0 : 0.0; }).resolved);
    REQUIRE(state.winner() == 2);
    const double p_b = std::exp(1.0) / (std::exp(1.0) + 1.0);
    REQUIRE(state.probabilities()[2] == Catch::Approx(p_b));
    REQUIRE(state.probabilities()[0] == Catch::Approx((1.0 - p_b) / 2.0));
    REQUIRE(state.probabilities()[1] == Catch::Approx((1.0 - p_b) / 2.0));
}

TEST_CASE("a closing token disambiguates a choice that is a prefix of another") {
    // "LOW" -> [10, 99] and "LOWER" -> [10, 11, 99] where 99 is the closing quote.
    pcd::CandidateState state({{10, 99}, {10, 11, 99}}, {"LOW", "LOWER"});
    REQUIRE_FALSE(state.advance([](int) { return 0.0; }).resolved);
    REQUIRE(state.advance([](int token) { return token == 99 ? 2.0 : 0.0; }).resolved);
    REQUIRE(state.winner() == 0);
    REQUIRE(sum(state.probabilities()) == Catch::Approx(1.0));
}

TEST_CASE("softmax is stable under large logits") {
    pcd::CandidateState state({{1}, {2}}, {"A", "B"});
    REQUIRE(state.advance([](int token) { return token == 1 ? 1e4 : 1e4 - 1.0; }).resolved);
    REQUIRE(state.winner() == 0);
    REQUIRE(sum(state.probabilities()) == Catch::Approx(1.0));
}

TEST_CASE("force resolve picks the live choice with the highest mass") {
    pcd::CandidateState state({{10, 11, 12}, {10, 11, 13}}, {"A", "B"});
    REQUIRE_FALSE(state.advance([](int) { return 0.0; }).resolved);
    state.force_resolve();
    REQUIRE(state.resolved());
    REQUIRE(sum(state.probabilities()) == Catch::Approx(1.0));
    REQUIRE((state.winner() == 0 || state.winner() == 1));
}

TEST_CASE("advancing a resolved state is rejected") {
    pcd::CandidateState state({{1}, {2}}, {"A", "B"});
    state.advance([](int) { return 0.0; });
    REQUIRE_THROWS(state.advance([](int) { return 0.0; }));
}
