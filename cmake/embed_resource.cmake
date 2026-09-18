# Turns one file into a C++ source exposing it as a std::string_view.
# Usage: cmake -DINPUT=<file> -DOUTPUT=<cpp> -DSYMBOL=<function> -P embed_resource.cmake
file(READ "${INPUT}" content HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${content}")
file(WRITE "${OUTPUT}" "// Generated from ${INPUT}; do not edit.
#include <string_view>
namespace pcd {
namespace {
const unsigned char embedded_data[] = {${bytes}0x00};
}
std::string_view ${SYMBOL}() {
    return std::string_view(reinterpret_cast<const char *>(embedded_data), sizeof(embedded_data) - 1);
}
}
")
