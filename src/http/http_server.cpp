#include "pcd/http_server.hpp"

#include "pcd/version.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <stdexcept>

namespace pcd {

using nlohmann::json;

namespace {

constexpr const char * json_type = "application/json";

void reply(httplib::Response & res, int status, const json & body) {
    res.status = status;
    res.set_content(body.dump(), json_type);
}

void reply_error(httplib::Response & res, int status, const char * code, const std::string & message) {
    reply(res, status, {{"code", code}, {"message", message}});
}

json parse_body(const httplib::Request & req) {
    return json::parse(req.body);
}

// Maps domain failures to stable HTTP statuses. Messages never contain
// filesystem paths: engine and catalog errors only mention identifiers.
template <typename Fn>
void guarded(httplib::Response & res, Fn && fn) {
    try {
        fn();
    } catch (const json::parse_error & error) {
        reply_error(res, 400, "invalid_json", error.what());
    } catch (const ValidationError & error) {
        reply_error(res, 400, "invalid_request", error.what());
    } catch (const UnknownModel & error) {
        reply_error(res, 404, "unknown_model", error.what());
    } catch (const SchemaError & error) {
        reply_error(res, 422, "schema_error", error.what());
    } catch (const NoModelLoaded & error) {
        reply_error(res, 503, "no_model", error.what());
    } catch (const NativeError & error) {
        reply_error(res, 500, "native_error", error.what());
    } catch (const std::exception & error) {
        reply_error(res, 500, "internal_error", error.what());
    }
}

json engine_summary(const std::shared_ptr<PcdEngine> & engine) {
    if (!engine) {
        return nullptr;
    }
    return {
        {"id", engine->model_id()},
        {"description", engine->model_description()},
        {"architecture", engine->architecture()},
        {"backend", engine->backend()},
        {"contextSize", engine->options().runtime.context_size},
        {"cacheEntries", engine->cache().size()},
        {"cacheBytes", engine->cache().total_bytes()},
    };
}

}  // namespace

HttpServer::HttpServer(ModelManager & manager, HttpServerOptions options)
    : manager_(manager), options_(std::move(options)), server_(std::make_unique<httplib::Server>()) {
    server_->set_payload_max_length(options_.max_body_bytes);
    register_routes();
}

HttpServer::~HttpServer() {
    stop();
}

void HttpServer::register_routes() {
    auto & server = *server_;

    server.Get("/health", [this](const httplib::Request &, httplib::Response & res) {
        guarded(res, [&] {
            auto engine = manager_.active();
            json body = {
                {"status", "ok"},
                {"apiVersion", std::string(api_version)},
                {"ready", engine != nullptr},
                {"model", engine ? json(engine->model_id()) : json(nullptr)},
                {"description", engine ? json(engine->model_description()) : json(nullptr)},
                {"backend", engine ? json(engine->backend()) : json(nullptr)},
            };
            reply(res, 200, body);
        });
    });

    server.Get("/v1/models", [this](const httplib::Request &, httplib::Response & res) {
        guarded(res, [&] {
            const auto active = manager_.active_id();
            json models = json::array();
            for (const auto & model : manager_.catalog().list()) {
                models.push_back({
                    {"id", model.id},
                    {"bytes", model.bytes},
                    {"active", active.has_value() && *active == model.id},
                });
            }
            reply(res, 200, {
                {"active", active ? json(*active) : json(nullptr)},
                {"activeModel", engine_summary(manager_.active())},
                {"models", std::move(models)},
            });
        });
    });

    server.Post("/v1/models/select", [this](const httplib::Request & req, httplib::Response & res) {
        guarded(res, [&] {
            auto body = parse_body(req);
            if (!body.is_object() || !body.contains("id") || !body["id"].is_string()) {
                throw ValidationError("id must be a string");
            }
            manager_.select(body["id"].get<std::string>());
            reply(res, 200, {{"active", *manager_.active_id()}, {"activeModel", engine_summary(manager_.active())}});
        });
    });

    server.Post("/v1/pcd/decode", [this](const httplib::Request & req, httplib::Response & res) {
        guarded(res, [&] {
            auto request = parse_decode_request(parse_body(req));
            reply(res, 200, to_json_response(manager_.decode(request)));
        });
    });

    server.set_error_handler([](const httplib::Request &, httplib::Response & res) {
        if (res.body.empty()) {
            const char * code = res.status == 404 ? "not_found" : res.status == 413 ? "payload_too_large" : "http_error";
            reply_error(res, res.status, code, httplib::status_message(res.status));
        }
    });

    server.set_exception_handler([](const httplib::Request &, httplib::Response & res, std::exception_ptr ep) {
        std::string message = "unhandled exception";
        try {
            if (ep) std::rethrow_exception(ep);
        } catch (const std::exception & error) {
            message = error.what();
        } catch (...) {
        }
        reply_error(res, 500, "internal_error", message);
    });
}

void HttpServer::start() {
    if (thread_.joinable()) {
        return;
    }
    if (options_.port == 0) {
        port_ = server_->bind_to_any_port(options_.bind);
        if (port_ <= 0) {
            throw std::runtime_error("failed to bind " + options_.bind);
        }
    } else {
        if (!server_->bind_to_port(options_.bind, options_.port)) {
            throw std::runtime_error("failed to bind " + options_.bind + ":" + std::to_string(options_.port));
        }
        port_ = options_.port;
    }
    thread_ = std::thread([this] { server_->listen_after_bind(); });
    server_->wait_until_ready();
}

void HttpServer::stop() {
    if (server_) {
        server_->stop();
    }
    if (thread_.joinable()) {
        thread_.join();
    }
}

bool HttpServer::running() const {
    return server_ && server_->is_running();
}

}  // namespace pcd
