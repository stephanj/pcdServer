cmake_minimum_required(VERSION 3.25)
project(pcd_server VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_IS_DEV OFF CACHE BOOL "" FORCE)

FetchContent_Declare(llama_cpp GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git GIT_TAG v0.4.1 GIT_SHALLOW TRUE)
FetchContent_Declare(httplib GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git GIT_TAG v0.56.0 GIT_SHALLOW TRUE)
FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.12.0 GIT_SHALLOW TRUE)
FetchContent_Declare(Catch2 GIT_REPOSITORY https://github.com/catchorg/Catch2.git GIT_TAG v3.15.2 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(llama_cpp httplib json Catch2)

add_library(pcd_core STATIC
    src/types.cpp
    src/engine/schema.cpp
    src/engine/scorer.cpp
    src/engine/llama_runtime.cpp
    src/engine/schema_cache.cpp
    src/engine/pcd_engine.cpp
)
target_include_directories(pcd_core PUBLIC include)
target_link_libraries(pcd_core PUBLIC llama nlohmann_json::nlohmann_json)

enable_testing()
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
add_executable(pcd_tests
    tests/version_test.cpp
    tests/types_test.cpp
    tests/schema_test.cpp
    tests/scorer_test.cpp
    tests/llama_runtime_test.cpp
    tests/schema_cache_test.cpp
    tests/pcd_engine_test.cpp
)
target_link_libraries(pcd_tests PRIVATE pcd_core Catch2::Catch2WithMain)
catch_discover_tests(pcd_tests)
