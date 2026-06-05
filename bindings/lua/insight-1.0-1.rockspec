package = "insight"
version = "1.0-1"

source = {
    url = "https://github.com/PlumBlossomMaid/Insight7.git",
    tag = "v1.0.0",
}

description = {
    summary = "Insight7 — lightweight scientific computing framework for Lua",
    detailed = [[
        Insight is a C++ tensor computing framework with GPU support.
        This package provides Lua/LuaJIT bindings via sol2 with
        Penlight-enhanced API wrappers. Supports signal processing,
        linear algebra, FFT, and more.
    ]],
    homepage = "https://github.com/PlumBlossomMaid/Insight7",
    license = "MIT",
}

dependencies = {
    "lua >= 5.1",
}

build = {
    type = "cmake",

    variables = {
        CMAKE_BUILD_TYPE = "Release",
        CMAKE_SKIP_INSTALL_RULES = "ON",
        INSIGHT_BUILD_TESTS = "OFF",
        INSIGHT_BUILD_DEMOS = "OFF",
        INSIGHT_BUILD_BINDINGS = "ON",
        INSIGHT_BUILD_PYTHON_BINDING = "OFF",
        INSIGHT_BUILD_JULIA_BINDING = "OFF",
        INSIGHT_BUILD_LUA_BINDING = "ON",
        INSIGHT_WITH_CUDA = "ON",
        INSIGHT_USE_FFTW3 = "ON",
        INSIGHT_USE_OPENBLAS = "ON",
        INSIGHT_USE_MATPLOT = "ON",
        LUA_INCLUDE_DIR = "$(LUA_INCDIR)",
        CMAKE_INSTALL_PREFIX = "$(PREFIX)",
        LUADIR = "$(LUADIR)",
        LIBDIR = "$(LIBDIR)",
    },

    install = {
        lua = {
            ["insight"] = "bindings/lua/insight/init.lua",
            ["insight.init"] = "bindings/lua/insight/init.lua",
        },
        lib = {
            "_insight",
        },
    },
}
