# bindings/lua/CMakeLists.txt
#
# Lua/LuaJIT bindings via sol2 (header-only).
#
# Usage:
#   cmake .. -DINSIGHT_BUILD_LUA=ON -DLUA_DIR=/path/to/lua
#
# The built module is always _insight.so, loadable by require("_insight").

include(FetchContent)

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
        NAMES luajit-5.1 luajit lua5.4 lua5.3 lua5.2 lua5.1 lua51 lua
        HINTS "${LUA_DIR}/lib" "${LUA_DIR}/lib/x86_64-linux-gnu"
              "${LUA_DIR}/lib/x64" "${LUA_DIR}"
        NO_DEFAULT_PATH
    )
    find_path(LUA_INCLUDE_DIR
        NAMES lua.h
        HINTS "${LUA_DIR}/include/luajit-2.1" "${LUA_DIR}/include/luajit-2.0"
              "${LUA_DIR}/include/lua5.4" "${LUA_DIR}/include/lua5.3"
              "${LUA_DIR}/include/lua5.2" "${LUA_DIR}/include/lua5.1"
              "${LUA_DIR}/include" "${LUA_DIR}"
        NO_DEFAULT_PATH
    )
else()
    # Auto-detect (CMake searches platform default paths automatically)
    find_library(LUA_LIB
        NAMES luajit-5.1 luajit lua5.4 lua5.3 lua5.2 lua5.1 lua51 lua
    )
    find_path(LUA_INCLUDE_DIR
        NAMES lua.h
        PATH_SUFFIXES luajit-2.1 luajit-2.0 lua5.4 lua5.3 lua5.2 lua5.1
    )
endif()

# Fallback: build Lua 5.1.5 from source if not found
if(NOT LUA_LIB OR NOT LUA_INCLUDE_DIR)
    message(STATUS "Lua not found via find_library, building from source...")

    FetchContent_Declare(
        lua_src
        URL https://www.lua.org/ftp/lua-5.1.5.tar.gz
        URL_HASH MD5=2e115fe26e435e33b0d5c022e4490567
    )
    FetchContent_MakeAvailable(lua_src)

    set(LUA_SOURCES
        ${lua_src_SOURCE_DIR}/src/lapi.c ${lua_src_SOURCE_DIR}/src/lcode.c
        ${lua_src_SOURCE_DIR}/src/ldebug.c ${lua_src_SOURCE_DIR}/src/ldo.c
        ${lua_src_SOURCE_DIR}/src/ldump.c ${lua_src_SOURCE_DIR}/src/lfunc.c
        ${lua_src_SOURCE_DIR}/src/lgc.c ${lua_src_SOURCE_DIR}/src/llex.c
        ${lua_src_SOURCE_DIR}/src/lmem.c ${lua_src_SOURCE_DIR}/src/lobject.c
        ${lua_src_SOURCE_DIR}/src/lopcodes.c ${lua_src_SOURCE_DIR}/src/lparser.c
        ${lua_src_SOURCE_DIR}/src/lstate.c ${lua_src_SOURCE_DIR}/src/lstring.c
        ${lua_src_SOURCE_DIR}/src/ltable.c ${lua_src_SOURCE_DIR}/src/ltm.c
        ${lua_src_SOURCE_DIR}/src/lundump.c ${lua_src_SOURCE_DIR}/src/lvm.c
        ${lua_src_SOURCE_DIR}/src/lzio.c ${lua_src_SOURCE_DIR}/src/lauxlib.c
        ${lua_src_SOURCE_DIR}/src/lbaselib.c ${lua_src_SOURCE_DIR}/src/ldblib.c
        ${lua_src_SOURCE_DIR}/src/liolib.c ${lua_src_SOURCE_DIR}/src/lmathlib.c
        ${lua_src_SOURCE_DIR}/src/loslib.c ${lua_src_SOURCE_DIR}/src/ltablib.c
        ${lua_src_SOURCE_DIR}/src/lstrlib.c ${lua_src_SOURCE_DIR}/src/loadlib.c
        ${lua_src_SOURCE_DIR}/src/linit.c
    )

    add_library(lua_src_lib SHARED ${LUA_SOURCES})
    target_include_directories(lua_src_lib PUBLIC ${lua_src_SOURCE_DIR}/src)
    set_target_properties(lua_src_lib PROPERTIES
        PREFIX ""
        OUTPUT_NAME "lua5.1"
    )
    if(MSVC)
        target_compile_definitions(lua_src_lib PRIVATE LUA_BUILD_AS_DLL)
    endif()
    if(UNIX)
        target_link_libraries(lua_src_lib PRIVATE m dl)
    endif()

    add_executable(lua_interpreter ${lua_src_SOURCE_DIR}/src/lua.c)
    target_link_libraries(lua_interpreter PRIVATE lua_src_lib)
    set_target_properties(lua_interpreter PROPERTIES OUTPUT_NAME "lua")

    set(LUA_LIB lua_src_lib)
    set(LUA_INCLUDE_DIR "${lua_src_SOURCE_DIR}/src")
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
