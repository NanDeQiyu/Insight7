---
name: luarocks-cmake-packaging
description: Create luarocks rockspec with cmake build, dynamic backend .so copying, cross-platform
source: auto-skill
extracted_at: '2026-06-05T14:22:08.096Z'
---

## Luarocks + CMake Packaging

**Problem**: luarocks `cmake` build type calls `cmake --install` which fails on FetchContent codec dependencies (ogg/flac/vorbis). Also, `install.lib` doesn't support glob patterns for dynamic backend discovery.

**Solution**: Use `command` build type with custom `build_command` + `install` section.

**Rockspec template** (`insight-1.0-1.rockspec`):

```lua
build = {
    type = "command",
    build_command = [[
        cmake -S . -B build.luarocks \
            -DCMAKE_BUILD_TYPE=Release \
            -DINSIGHT_BUILD_TESTS=OFF \
            -DINSIGHT_BUILD_DEMOS=OFF \
            -DINSIGHT_BUILD_BINDINGS=ON \
            -DINSIGHT_BUILD_PYTHON_BINDING=OFF \
            -DINSIGHT_BUILD_JULIA_BINDING=OFF \
            -DINSIGHT_BUILD_LUA_BINDING=ON \
            -DINSIGHT_WITH_CUDA=ON \
            -DINSIGHT_USE_FFTW3=ON \
            -DINSIGHT_USE_OPENBLAS=ON \
            -DINSIGHT_USE_MATPLOT=ON \
            -DLUA_INCLUDE_DIR="$(LUA_INCDIR)" \
            && cmake --build build.luarocks -j 24
    ]],
    install = {
        lua = {
            ["insight"] = "bindings/lua/insight/init.lua",
            ["insight.init"] = "bindings/lua/insight/init.lua",
        },
        lib = {
            "build.luarocks/bindings/lua/_insight",
        },
    },
}
```

**Key decisions**:
- `command` type avoids `cmake --install` (which fails on codec deps)
- `install.lib` only lists `_insight` (luarocks creates symlink)
- Backend .so files are copied by cmake POST_BUILD, NOT by luarocks
- `LUA_INCLUDE_DIR="$(LUA_INCDIR)"` passes luarocks' Lua headers to cmake
- Do NOT use `$(LUA_LIBDIR)` — not always set by luarocks

**Backend .so handling**:
- cmake POST_BUILD copies `libinsight_*_backend.so` to `bindings/lua/`
- Lua `init.lua` finds backends via `package.cpath` search + symlink resolution
- No hardcoded backend names in rockspec

**Parallel build**: `CMAKE_BUILD_PARALLEL_LEVEL=24` or `MAKEFLAGS="-j24"`

**Testing**:
```bash
luarocks --lua-version 5.3 make bindings/lua/insight-1.0-1.rockspec LUA_DIR=/usr --local
eval $(luarocks path --lua-version 5.3)
lua5.3 -e 'local ins = require("insight"); print(ins.zeros({3,3}))'
```

**Why:** `cmake` build type in luarocks calls `cmake --install` which triggers codec deps install rules. `command` type gives full control.
