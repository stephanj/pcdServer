#pragma once
#include "pcd/model_manager.hpp"

#include <memory>
#include <string>
#include <thread>

namespace httplib {
class Server;
}

namespace pcd {

struct HttpServerOptions {
    std::string bind{"127.0.0.1"};
    int port{8090};  // 0 selects a free port
    std::size_t max_body_bytes{1024 * 1024};
};

// Thin REST transport over ModelManager. The listener runs on its own thread;
// the destructor stops it.
class HttpServer {
public:
    HttpServer(ModelManager & manager, HttpServerOptions options);
    ~HttpServer();
    HttpServer(const HttpServer &) = delete;
    HttpServer & operator=(const HttpServer &) = delete;

    // Binds and starts serving; throws std::runtime_error if the bind fails.
    void start();
    void stop();
    int port() const { return port_; }
    bool running() const;

private:
    void register_routes();

    ModelManager & manager_;
    HttpServerOptions options_;
    std::unique_ptr<httplib::Server> server_;
    std::thread thread_;
    int port_{0};
};

}  // namespace pcd
