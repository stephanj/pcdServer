#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace pcd {

// Upper bound on collision-tree levels an engine walks before force-resolving.
inline constexpr int max_tree_levels = 24;

struct AdvanceResult {
    bool resolved;
    int32_t winning_token;
};

// Tracks the bounded candidate set of one field while shared token prefixes
// are resolved level by level. Only allowed next tokens are ever scored.
class CandidateState {
public:
    using LogitFn = std::function<double(int32_t)>;

    CandidateState(std::vector<std::vector<int32_t>> candidates, std::vector<std::string> labels);

    // Scores the next token of every live candidate, keeps the winning token's
    // group live and assigns the eliminated groups their share of the mass.
    AdvanceResult advance(const LogitFn & logit_of);

    // Resolves an ambiguous field without further decoding by splitting the
    // remaining mass evenly and picking the first live candidate.
    void force_resolve();

    bool resolved() const { return winner_ >= 0; }
    int winner() const { return winner_; }
    int32_t last_token() const { return last_token_; }
    int levels() const { return levels_; }
    int live_count() const { return live_count_; }
    const std::vector<double> & probabilities() const { return probabilities_; }
    const std::vector<std::string> & labels() const { return labels_; }
    // Token positions consumed by the winning path so far (one per level).
    const std::vector<int32_t> & consumed_tokens() const { return consumed_; }

private:
    void finish(int index);

    std::vector<std::vector<int32_t>> candidates_;
    std::vector<std::string> labels_;
    std::vector<bool> live_;
    std::vector<double> probabilities_;
    std::vector<int32_t> consumed_;
    double live_mass_{1.0};
    int live_count_{0};
    int levels_{0};
    int32_t last_token_{-1};
    int winner_{-1};
};

}  // namespace pcd
