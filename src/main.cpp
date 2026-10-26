#include "pcd/http_server.hpp"
#include "pcd/version.hpp"

#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <condition_variable>
#include <mutex>
#endif

namespace {

struct Config {
    std::string bind{"127.0.0.1"};
    int port{8090};
    std::filesystem::path models_dir{"models"};
    std::filesystem::path model;
    std::size_t cache_entries{32};
    std::size_t cache_bytes{536870912};
};

void usage(const char * argv0) {
    std::cerr << "usage: " << argv0 << " [options]\n"
              << "  --bind ADDRESS        default 127.0.0.1\n"
              << "  --port PORT           default 8090\n"
              << "  --models-dir PATH     default models\n"
              << "  --model PATH          overrides PCD_GGUF and the default model\n"
              << "  --cache-entries N     default 32\n"
              << "  --cache-bytes BYTES   default 536870912\n"
              << "  --help\n";
}

Config parse_args(int argc, char ** argv) {
    Config config;
    auto value = [&](int & i, const char * flag) -> std::string {
        if (i + 1 >= argc) {
            throw std::invalid_argument(std::string(flag) + " requires a value");
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--bind") {
            config.bind = value(i, "--bind");
        } else if (arg == "--port") {
            config.port = std::stoi(value(i, "--port"));
        } else if (arg == "--models-dir") {
            config.models_dir = value(i, "--models-dir");
        } else if (arg == "--model") {
            config.model = value(i, "--model");
        } else if (arg == "--cache-entries") {
            config.cache_entries = std::stoull(value(i, "--cache-entries"));
        } else if (arg == "--cache-bytes") {
            config.cache_bytes = std::stoull(value(i, "--cache-bytes"));
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (config.model.empty()) {
        if (const char * env = std::getenv("PCD_GGUF"); env && *env) {
            config.model = env;
        } else {
            config.model = config.models_dir / "Qwen3.5-0.8B-Q8_0.gguf";
        }
    }
    return config;
}

#ifdef _WIN32
// Console control events arrive on a thread the system creates; the handler
// only records the event and wakes the main thread, which runs the shutdown.
std::mutex shutdown_mutex;
std::condition_variable shutdown_cv;
DWORD shutdown_event = 0;
bool shutdown_requested = false;

BOOL WINAPI on_console_event(DWORD event) {
    {
        std::lock_guard lock(shutdown_mutex);
        shutdown_event = event;
        shutdown_requested = true;
    }
    shutdown_cv.notify_one();
    return TRUE;
}

void install_shutdown_handler() {
    SetConsoleCtrlHandler(on_console_event, TRUE);
}

int wait_for_shutdown_signal() {
    std::unique_lock lock(shutdown_mutex);
    shutdown_cv.wait(lock, [] { return shutdown_requested; });
    return static_cast<int>(shutdown_event);
}
#else
// SIGINT/SIGTERM are blocked on every thread and consumed here so shutdown
// runs on the main thread with ordinary (non-async-signal) code.
sigset_t shutdown_signals;

// Must run before any other thread starts so every thread inherits the mask.
void install_shutdown_handler() {
    sigemptyset(&shutdown_signals);
    sigaddset(&shutdown_signals, SIGINT);
    sigaddset(&shutdown_signals, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &shutdown_signals, nullptr);
}

int wait_for_shutdown_signal() {
    int signal = 0;
    sigwait(&shutdown_signals, &signal);
    return signal;
}
#endif

}  // namespace

int main(int argc, char ** argv) {
    Config config;
    try {
        config = parse_args(argc, argv);
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << "\n";
        usage(argv[0]);
        return 2;
    }

    install_shutdown_handler();

    pcd::LlamaRuntime::quiet_logging();
    pcd::EngineOptions options;
    options.cache.max_entries = config.cache_entries;
    options.cache.max_bytes = config.cache_bytes;
    pcd::ModelManager manager(pcd::ModelCatalog(config.models_dir), options);

    if (std::filesystem::is_regular_file(config.model)) {
        std::cerr << "loading " << config.model.string() << "\n";
        try {
            manager.load_path(config.model);
            auto engine = manager.active();
            std::cerr << "loaded " << engine->model_id() << " (" << engine->model_description() << ") on " << engine->backend() << "\n";
        } catch (const std::exception & error) {
            std::cerr << "error: " << error.what() << "\n";
            return 1;
        }
    } else {
        std::cerr << "warning: model " << config.model.string() << " not found; starting without an active model\n";
    }

    pcd::HttpServer server(manager, {.bind = config.bind, .port = config.port});
    try {
        server.start();
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
    std::cerr << "pcd_server " << pcd::api_version << " listening on http://" << config.bind << ":" << server.port() << "\n";

    const int signal = wait_for_shutdown_signal();
    std::cerr << "received signal " << signal << ", shutting down\n";
    server.stop();
    return 0;
}
