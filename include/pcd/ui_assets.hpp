#pragma once
#include <string_view>
#include <vector>

namespace pcd {

// Files under ui/, embedded at build time (see cmake/embed_resource.cmake).
std::string_view ui_index_html();
std::string_view ui_style_css();
std::string_view ui_app_js();
std::string_view ui_tetris_js();
std::string_view ui_presets_json();
std::string_view ui_openapi_json();
std::string_view ui_docs_html();

struct StaticAsset {
    const char * path;
    const char * mime;
    std::string_view (*content)();
};

// Every static route the HTTP server exposes besides the JSON API.
const std::vector<StaticAsset> & ui_assets();

}  // namespace pcd
