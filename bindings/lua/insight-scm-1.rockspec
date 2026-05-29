package = "insight"
version = "scm-1"

source = {
    url = "https://github.com/PlumBlossomMaid/Insight7.git",
    branch = "main",
}

description = {
    summary = "Insight7 — lightweight scientific computing framework for Lua",
    detailed = [[
        Insight is a C++ tensor computing framework with GPU support.
        This package provides Lua/LuaJIT bindings via sol2 with
        Penlight-enhanced API wrappers.
    ]],
    homepage = "https://github.com/PlumBlossomMaid/Insight7",
    license = "MIT",
}

dependencies = {
    "lua >= 5.1",
    "penlight >= 1.5",
}

external_dependencies = {
    INSIGHT = {
        header = "insight/core/array.h",
        library = "insight_core",
    },
}

build = {
    type = "cmake",

    variables = {
        CMAKE_BUILD_TYPE = "Release",
        INSIGHT_BUILD_TESTS = "OFF",
        INSIGHT_BUILD_DEMOS = "OFF",
        INSIGHT_BUILD_BINDINGS = "ON",
        INSIGHT_BUILD_PYTHON_BINDING = "OFF",
        INSIGHT_BUILD_JULIA_BINDING = "OFF",
        INSIGHT_BUILD_LUA_BINDING = "ON",
        INSIGHT_WITH_CUDA = "OFF",
        LUA_INCLUDE_DIR = "$(LUA_INCDIR)",
        LUA_LIBDIR = "$(LUA_LIBDIR)",
        CMAKE_INSTALL_PREFIX = "$(PREFIX)",
        LUADIR = "$(LUADIR)",
        LIBDIR = "$(LIBDIR)",
    },

    install = {
        lua = {
            ["insight"] = "bindings/lua/insight/init.lua",
        },
        lib = {
            "insight.so",
        },
    },
}
