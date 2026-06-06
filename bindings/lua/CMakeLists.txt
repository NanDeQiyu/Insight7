# bindings/lua/CMakeLists.txt
#
# Lua/LuaJIT bindings via sol2 (header-only).
#
# Usage:
#   cmake .. -DINSIGHT_BUILD_LUA=ON -DLUA_DIR=/path/to/lua
#
# The built module is always _insight.so, loadable by require("_insight").

option(INSIGHT_BUILD_LUA "Build Lua bindings" ON)
if(NOT INSIGHT_BUILD_LUA)
    return()
endif()

# Resolve LUA_DIR from CMake variable, environment variable, or auto-detect
if(NOT LUA_DIR AND DEFINED ENV{LUA_DIR})
    set(LUA_DIR "$ENV{LUA_DIR}")
endif()

if(LUA_DIR)
    # Explicit Lua installation path
    find_library(LUA_LIB
        NAMES luajit-5.1 luajit lua5.4 lua5.3 lua5.2 lua5.1 lua
        HINTS "${LUA_DIR}/lib" "${LUA_DIR}/lib/x86_64-linux-gnu"
              "${LUA_DIR}/lib/x64" "${LUA_DIR}"
        NO_DEFAULT_PATH
    )
    find_path(LUA_INCLUDE_DIR
        NAMES lua.h
        HINTS "${LUA_DIR}/include" "${LUA_DIR}"
        PATH_SUFFIXES luajit-2.1 luajit-2.0
        NO_DEFAULT_PATH
    )
else()
    # Auto-detect
    find_library(LUA_LIB
        NAMES luajit-5.1 luajit lua5.4 lua5.3 lua5.2 lua5.1 lua
    )
    find_path(LUA_INCLUDE_DIR
        NAMES lua.h
        PATH_SUFFIXES luajit-2.1 luajit-2.0
    )
endif()

if(NOT LUA_LIB OR NOT LUA_INCLUDE_DIR)
    message(STATUS "Lua not found, skipping Lua bindings (set -DLUA_DIR=...)")
    return()
endif()

message(STATUS "Lua binding: ${LUA_LIB}")
message(STATUS "Lua headers: ${LUA_INCLUDE_DIR}")

add_library(insight_lua SHARED insight_lua.cpp)
set_target_properties(insight_lua PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_insight"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bindings/lua"
    # Windows: export all C++ symbols for sol2 template instantiation
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

target_include_directories(insight_lua PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${LUA_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/sol2/include
)

target_link_libraries(insight_lua PRIVATE
    insight_core
    ${LUA_LIB}
)

target_compile_features(insight_lua PRIVATE cxx_std_17)

# Ensure backend is built before POST_BUILD copy
add_dependencies(insight_lua insight_cpu_backend)

# Copy binding + CPU backend using cross-platform generator expressions
add_custom_command(TARGET insight_lua POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:insight_lua>
        "${CMAKE_SOURCE_DIR}/bindings/lua/$<TARGET_FILE_NAME:insight_lua>"
    # Copy CPU backend for dlopen
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:insight_cpu_backend>"
        "${CMAKE_SOURCE_DIR}/bindings/lua/$<TARGET_FILE_NAME:insight_cpu_backend>"
    COMMENT "Copying Lua binding + CPU backend"
)

message(STATUS "Lua bindings enabled")
