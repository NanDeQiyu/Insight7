# bindings/lua/CMakeLists.txt
#
# Lua/LuaJIT bindings via sol2 (header-only).

# Find LuaJIT
set(LUAJIT_DIR "$ENV{HOME}/public/luajit" CACHE PATH "LuaJIT installation directory")

find_library(LUAJIT_LIB
    NAMES luajit-5.1 luajit
    HINTS "${LUAJIT_DIR}/lib"
)

find_path(LUAJIT_INCLUDE_DIR
    NAMES lua.h
    HINTS "${LUAJIT_DIR}/include/luajit-2.1"
)

if(NOT LUAJIT_LIB OR NOT LUAJIT_INCLUDE_DIR)
    message(WARNING "LuaJIT not found at ${LUAJIT_DIR}, skipping Lua bindings")
    message(STATUS "  LUAJIT_LIB: ${LUAJIT_LIB}")
    message(STATUS "  LUAJIT_INCLUDE_DIR: ${LUAJIT_INCLUDE_DIR}")
    return()
endif()

message(STATUS "LuaJIT found: ${LUAJIT_LIB}")
message(STATUS "LuaJIT headers: ${LUAJIT_INCLUDE_DIR}")

# Build Lua module as shared library (loadable by require())
add_library(insight_lua SHARED insight_lua.cpp)
set_target_properties(insight_lua PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_insight"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bindings/lua"
)

target_include_directories(insight_lua PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${LUAJIT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/sol2/include
)

target_link_libraries(insight_lua PRIVATE
    insight_core
    ${LUAJIT_LIB}
)

# sol2 needs C++17
target_compile_features(insight_lua PRIVATE cxx_std_17)

message(STATUS "Lua bindings enabled (sol2 + LuaJIT)")
