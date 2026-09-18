#include "pcd/scorer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace pcd {

CandidateState::CandidateState(std::vector<std::vector<int32_t>> candidates, std::vector<std::string> labels)
    : candidates_(std::move(candidates)),
      labels_(std::move(labels)),
      live_(candidates_.size(), true),
      probabilities_(candidates_.size(), 0.0),
      live_count_(static_cast<int>(candidates_.size())) {
    if (candidates_.empty() || candidates_.size() != labels_.size()) {
        throw std::invalid_argument("candidate token lists and labels must be non-empty and equal in size");
    }
    for (const auto & tokens : candidates_) {
        if (tokens.empty()) {
            throw std::invalid_argument("candidate token list must not be empty");
        }
    }
    if (candidates_.size() == 1) {
        finish(0);
    }
}

void CandidateState::finish(int index) {
    probabilities_[static_cast<std::size_t>(index)] = live_mass_;
    live_mass_ = 0.0;
    for (std::size_t i = 0; i < live_.size(); ++i) {
        live_[i] = false;
    }
    live_count_ = 0;
    winner_ = index;
}

AdvanceResult CandidateState::advance(const LogitFn & logit_of) {
    if (resolved()) {
        throw std::logic_error("candidate state already resolved");
    }
    const auto level = static_cast<std::size_t>(levels_);

    // Group live candidates by their next token; a candidate whose tokens are
    // exhausted while siblings remain live cannot be scored, so it wins outright.
    std::map<int32_t, std::vector<int>> groups;
    for (std::size_t i = 0; i < candidates_.size(); ++i) {
        if (!live_[i]) {
            continue;
        }
        if (level >= candidates_[i].size()) {
            for (std::size_t j = 0; j < candidates_.size(); ++j) {
                if (live_[j] && j != i) {
                    live_[j] = false;
                }
            }
            finish(static_cast<int>(i));
            return {true, last_token_};
        }
        groups[candidates_[i][level]].push_back(static_cast<int>(i));
    }

    // Stable softmax over the distinct allowed tokens.
    std::vector<std::pair<int32_t, double>> logits;
    logits.reserve(groups.size());
    double max_logit = -std::numeric_limits<double>::infinity();
    for (const auto & [token, _] : groups) {
        const double logit = logit_of(token);
        logits.emplace_back(token, logit);
        max_logit = std::max(max_logit, logit);
    }
    double denominator = 0.0;
    for (auto & [token, logit] : logits) {
        logit = std::exp(logit - max_logit);
        denominator += logit;
    }

    int32_t best_token = logits.front().first;
    double best_probability = -1.0;
    for (const auto & [token, weight] : logits) {
        const double probability = weight / denominator;
        if (probability > best_probability) {
            best_probability = probability;
            best_token = token;
        }
    }

    for (const auto & [token, weight] : logits) {
        const double probability = weight / denominator;
        const auto & members = groups[token];
        if (token == best_token) {
            continue;
        }
        const double share = live_mass_ * probability / static_cast<double>(members.size());
        for (int member : members) {
            probabilities_[static_cast<std::size_t>(member)] = share;
            live_[static_cast<std::size_t>(member)] = false;
        }
    }

    live_mass_ *= best_probability;
    live_count_ = static_cast<int>(groups[best_token].size());
    last_token_ = best_token;
    consumed_.push_back(best_token);
    ++levels_;

    if (live_count_ == 1) {
        finish(groups[best_token].front());
        return {true, best_token};
    }
    return {false, best_token};
}

void CandidateState::force_resolve() {
    if (resolved()) {
        return;
    }
    const double share = live_mass_ / static_cast<double>(live_count_);
    int first = -1;
    for (std::size_t i = 0; i < live_.size(); ++i) {
        if (live_[i]) {
            probabilities_[i] = share;
            if (first < 0) {
                first = static_cast<int>(i);
            }
        }
    }
    live_mass_ = share;
    finish(first);
}

}  // namespace pcd
