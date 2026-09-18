// Reproducible cold-versus-cached PCD benchmark. Prints one JSON document.
#include "pcd/pcd_engine.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Args {
    std::filesystem::path model{"models/Qwen3.5-0.8B-Q8_0.gguf"};
    std::filesystem::path request{"tests/data/triage-request.json"};
    int runs{5};
};

Args parse(int argc, char ** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto next = [&] {
            if (i + 1 >= argc) throw std::invalid_argument(flag + " requires a value");
            return std::string(argv[++i]);
        };
        if (flag == "--model") args.model = next();
        else if (flag == "--request") args.request = next();
        else if (flag == "--runs") args.runs = std::stoi(next());
        else if (flag == "--help" || flag == "-h") {
            std::cerr << "usage: pcd_bench [--model PATH] [--request PATH] [--runs N]\n";
            std::exit(0);
        } else throw std::invalid_argument("unknown argument: " + flag);
    }
    if (args.runs < 1) throw std::invalid_argument("--runs must be at least 1");
    return args;
}

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto n = values.size();
    return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

struct Samples {
    std::vector<double> elapsed, tokenize, restore_or_prefill, dynamic_context, broadcast, suffix, tree;
    void add(const pcd::DecodeMetrics & m) {
        elapsed.push_back(m.elapsed_ms);
        tokenize.push_back(m.phases.tokenize_ms);
        restore_or_prefill.push_back(m.phases.restore_or_prefill_ms);
        dynamic_context.push_back(m.phases.dynamic_context_ms);
        broadcast.push_back(m.phases.broadcast_ms);
        suffix.push_back(m.phases.suffix_ms);
        tree.push_back(m.phases.tree_ms);
    }
    nlohmann::json medians() const {
        return {
            {"elapsedMs", median(elapsed)},
            {"phasesMs", {
                {"tokenize", median(tokenize)},
                {"restoreOrPrefill", median(restore_or_prefill)},
                {"dynamicContext", median(dynamic_context)},
                {"broadcast", median(broadcast)},
                {"suffix", median(suffix)},
                {"tree", median(tree)},
            }},
        };
    }
};

}  // namespace

int main(int argc, char ** argv) {
    Args args;
    try {
        args = parse(argc, argv);
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << "\n";
        return 2;
    }
    try {
        pcd::LlamaRuntime::quiet_logging();
        std::ifstream in(args.request);
        if (!in) throw std::runtime_error("cannot read request " + args.request.string());
        auto request = pcd::parse_decode_request(nlohmann::json::parse(in));

        pcd::PcdEngine engine(args.model);
        const auto key = engine.schema_key(request.fields);

        // Warm-up: builds the schema entry once so cold runs measure prefill,
        // not first-touch allocation.
        engine.decode(request);

        Samples cold, cached;
        std::size_t checkpoint_bytes = 0;
        std::string last_cold_status, last_cached_status;
        pcd::DecodeResponse last;
        for (int run = 0; run < args.runs; ++run) {
            engine.cache().erase(key);
            auto c = engine.decode(request);
            cold.add(c.metrics);
            last_cold_status = c.metrics.schema_cache_status;
            auto h = engine.decode(request);
            cached.add(h.metrics);
            last_cached_status = h.metrics.schema_cache_status;
            checkpoint_bytes = h.metrics.checkpoint_bytes;
            last = std::move(h);
        }

        nlohmann::json values = nlohmann::json::object();
        nlohmann::json probabilities = nlohmann::json::object();
        for (const auto & field : last.fields) {
            values[field.name] = pcd::to_json_value(field.value);
            probabilities[field.name] = field.probability;
        }
        nlohmann::json out = {
            {"model", engine.model_id()},
            {"backend", engine.backend()},
            {"runs", args.runs},
            {"fields", request.fields.size()},
            {"checkpointBytes", checkpoint_bytes},
            {"coldStatus", last_cold_status},
            {"cachedStatus", last_cached_status},
            {"cold", cold.medians()},
            {"cached", cached.medians()},
            {"speedup", median(cold.elapsed) / std::max(median(cached.elapsed), 1e-9)},
            {"values", values},
            {"probabilities", probabilities},
        };
        std::cout << out.dump(2) << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
