# bindings/lua/CMakeLists.txt
#
# Lua/LuaJIT bindings via sol2 (header-only).
#
# Usage:
#   # Auto-detect LuaJIT, then Lua 5.3, then Lua 5.1:
#   cmake .. -DINSIGHT_BUILD_LUA=ON
#
#   # Explicit LuaJIT:
#   cmake .. -DINSIGHT_BUILD_LUA=ON -DLUA_DIR=$HOME/public/luajit
#
#   # Explicit Lua 5.3:
#   cmake .. -DINSIGHT_BUILD_LUA=ON -DLUA_DIR=/usr
#
# The built module is always _insight.so, loadable by require("_insight").

option(INSIGHT_BUILD_LUA "Build Lua bindings" ON)
if(NOT INSIGHT_BUILD_LUA)
    return()
endif()

if(LUA_DIR)
    # User-specified Lua installation
    find_library(LUA_LIB
        NAMES luajit-5.1 luajit lua5.3 lua-5.3 lua5.1 lua-5.1 lua
        HINTS "${LUA_DIR}/lib" "${LUA_DIR}/lib/x86_64-linux-gnu"
              "${LUA_DIR}/lib/x64" "${LUA_DIR}/lib"
        NO_DEFAULT_PATH
    )
    find_path(LUA_INCLUDE_DIR
        NAMES lua.h
        HINTS "${LUA_DIR}/include/luajit-2.1"
              "${LUA_DIR}/include/lua5.3"
              "${LUA_DIR}/include/lua5.1"
              "${LUA_DIR}/include"
        NO_DEFAULT_PATH
    )
else()
    # Auto-detect: prefer LuaJIT, then Lua 5.3, then Lua 5.1
    find_library(LUA_LIB
        NAMES luajit-5.1 luajit
        HINTS "$ENV{HOME}/public/luajit/lib"
    )
    find_path(LUA_INCLUDE_DIR
        NAMES lua.h
        HINTS "$ENV{HOME}/public/luajit/include/luajit-2.1"
        PATHS /usr/include/luajit-2.1 /usr/local/include/luajit-2.1
    )
    if(NOT LUA_LIB OR NOT LUA_INCLUDE_DIR)
        find_library(LUA_LIB
            NAMES lua5.3 lua-5.3 lua
            PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
        )
        find_path(LUA_INCLUDE_DIR
            NAMES lua.h
            PATHS /usr/include/lua5.3 /usr/local/include/lua5.3
        )
    endif()
    # Windows: try common Lua installation paths
    if(WIN32 AND (NOT LUA_LIB OR NOT LUA_INCLUDE_DIR))
        find_library(LUA_LIB
            NAMES lua51 lua5.1 lua53 lua5.3 lua luajit
            PATHS
                "C:/Program Files (x86)/Lua/5.1/lib"
                "C:/Program Files/Lua/5.1/lib"
                "C:/Lua/5.1/lib"
                "C:/Program Files (x86)/Lua/5.3/lib"
                "C:/Program Files/Lua/5.3/lib"
                "$ENV{LUA_DIR}/lib"
        )
        find_path(LUA_INCLUDE_DIR
            NAMES lua.h
            PATHS
                "C:/Program Files (x86)/Lua/5.1/include"
                "C:/Program Files/Lua/5.1/include"
                "C:/Lua/5.1/include"
                "C:/Program Files (x86)/Lua/5.3/include"
                "C:/Program Files/Lua/5.3/include"
                "$ENV{LUA_DIR}/include"
        )
    endif()
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
