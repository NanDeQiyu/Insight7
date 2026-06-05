---
name: install-language-bindings
description: Make Insight7 installable via pip/luarocks/Julia Pkg with backend .so auto-discovery
source: auto-skill
extracted_at: '2026-06-05T05:24:58.368Z'
---

# System-wide Install for Language Bindings

## Problem
After `cmake --build`, users must manually set `PYTHONPATH`, `LUA_CPATH/LUA_PATH`,
`LD_LIBRARY_PATH` etc. to use Insight from their code. This is error-prone and
doesn't simulate real user experience in CI.

## Solution: CMake POST_BUILD + Language Package Managers

### Architecture
```
cmake build → POST_BUILD copies .so to source dirs → pip/luarocks/Pkg installs from source
```

### 1. CMake POST_BUILD (bindings/{python,lua,julia}/CMakeLists.txt)

Each binding's CMakeLists.txt copies the native .so AND the CPU backend .so to
the source wrapper directory:

```cmake
add_custom_command(TARGET insight_python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:insight_python>
        "${CMAKE_SOURCE_DIR}/bindings/python/insight/$<TARGET_FILE_NAME:insight_python>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_BINARY_DIR}/backends/cpu/libinsight_cpu_backend.so"
        "${CMAKE_SOURCE_DIR}/bindings/python/insight/libinsight_cpu_backend.so"
    COMMENT "Copying Python binding + CPU backend"
)
```

Same pattern for Lua (`bindings/lua/`) and Julia (`bindings/julia/`).

**CRITICAL**: Add `add_dependencies(insight_xxx insight_cpu_backend)` to each binding's
CMakeLists.txt. Without this, the POST_BUILD copy runs before the backend .so is built,
causing `Error copying file` in CI. The binding target links against `insight_core` (static),
but that doesn't guarantee the backend .so is built first.

### 2. Backend .so Pre-loading (ALL backends, not just CPU)

The C++ `ins::init()` calls `dlopen("libinsight_cpu_backend.so")` which searches
`LD_LIBRARY_PATH`. At runtime, modifying `LD_LIBRARY_PATH` doesn't always work.
Solution: pre-load ALL backend .so files (CPU + GPU: cuda/npu/rocm/...) BEFORE
importing the native module. This matches C++ `discover_backends()` behavior.

**Python** (`bindings/python/insight/__init__.py`):
```python
import ctypes as _ct, os as _os, glob as _gl
_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
for _so in _gl.glob(_os.path.join(_pkg_dir, "libinsight_*_backend.so")):
    try:
        _ct.CDLL(_so, mode=_ct.RTLD_GLOBAL)
    except OSError:
        pass
from ._insight import *
```

**Lua** (`bindings/lua/insight/init.lua`):
Scan for ALL `libinsight_*_backend.so` in candidate directories:
```lua
local function _find_and_load_backends()
  local dirs = {}
  -- 1. Parent of this script's directory (dev/source layout)
  -- 2. package.cpath directories (luarocks lib layout)
  -- 3. LD_LIBRARY_PATH directories
  local ok_ffi, ffi = pcall(require, "ffi")
  if not ok_ffi then return false end
  pcall(ffi.cdef, [[
    int setenv(const char *name, const char *value, int overwrite);
    void *dlopen(const char *filename, int flag);
  ]])
  local seen = {}
  for _, dir in ipairs(dirs) do
    local pipe = io.popen('ls "' .. dir .. '"/libinsight_*_backend.so 2>/dev/null')
    if pipe then
      for line in pipe:lines() do
        if not seen[line] then
          seen[line] = true
          local f = io.open(line, "r")
          if f then
            f:close()
            -- setenv LD_LIBRARY_PATH + dlopen(RTLD_GLOBAL)
          end
        end
      end
      pipe:close()
    end
  end
end
```

**Julia** (`bindings/julia/Insight.jl`):
```julia
using Libdl
for _d in (_dir, _parent)
    if isdir(_d)
        for _f in readdir(_d; join=true)
            if occursin("libinsight_", _f) && endswith(_f, "_backend.so")
                try Libdl.dlopen(_f, Libdl.RTLD_GLOBAL) catch end
            end
        end
    end
end
```

**Why scan ALL backends**: Future hardware support (NPU, ROCm, etc.) — the
backend .so naming convention `libinsight_<name>_backend.so` enables auto-discovery.

### 3. Python: pyproject.toml

```toml
[tool.setuptools.packages.find]
where = ["bindings/python"]
include = ["insight*"]

[tool.setuptools.package-data]
"insight" = ["*.so", "*.dll", "*.dylib"]
```

Install: `pip install -e .`

### 4. Lua: rockspec

Create `bindings/lua/insight-1.0-1.rockspec` with **command** build type (NOT cmake type).

**Why not cmake type**: luarocks' cmake build type calls `cmake --install` which fails
because codec dependencies (ogg/flac/vorbis) don't have install rules. The `command`
type gives full control over build and install steps.

**Why enable all features**: Set `INSIGHT_WITH_CUDA=ON`, `INSIGHT_USE_OPENBLAS=ON`, etc.
cmake auto-disables unavailable features. One rockspec works everywhere.

**Don't use `$(LUA_LIBDIR)`** — not set on all systems (e.g., system Lua 5.3).

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
    install_command = [[
        mkdir -p "$(LIBDIR)" "$(LUADIR)/insight" && \
        cp build.luarocks/bindings/lua/_insight.so "$(LIBDIR)/" && \
        cp build.luarocks/backends/cpu/libinsight_cpu_backend.so "$(LIBDIR)/" && \
        cp bindings/lua/insight/init.lua "$(LUADIR)/insight/" && \
        cp bindings/lua/insight/*.lua "$(LUADIR)/insight/" 2>/dev/null; true
    ]],
}
```

Install commands:
```bash
# With sudo
sudo luarocks make bindings/lua/insight-1.0-1.rockspec LUA_DIR=/usr CMAKE_BUILD_DIR=build.luarocks

# Without sudo
luarocks make bindings/lua/insight-1.0-1.rockspec LUA_DIR=/usr CMAKE_BUILD_DIR=build.luarocks --local

# Non-standard OpenBLAS path
OPENBLAS_HOME=/path/to/OpenBLAS luarocks make ... --local

# Faster build (CMAKE_BUILD_PARALLEL_LEVEL is the cmake-native way)
CMAKE_BUILD_PARALLEL_LEVEL=24 luarocks make ... --local

# Use specific Lua version
luarocks --lua-version 5.3 make ... --local
```

After install, `require("insight")` works from any directory (no LUA_CPATH/LD_LIBRARY_PATH).

### 5. Julia: Project.toml + src/ structure

```
bindings/julia/
├── Project.toml    # name, uuid, version, [deps] Libdl
├── src/
│   └── Insight.jl  # copy of Insight.jl (cmake does this)
│   └── modules/    # copy of modules/ (cmake does this)
├── Insight.jl      # source of truth
└── libinsight_*.so # copied by cmake POST_BUILD
```

Julia `__init__()` checks both `@__DIR__` and parent dir for .so files.

### 6. CI Testing Pattern

```yaml
# Python — pip install then test from /tmp
- run: pip install -e .
- run: cd /tmp && python3 -m pytest $GITHUB_WORKSPACE/tests/cpu/python/

# Lua — source dir .so + LD_LIBRARY_PATH
- run: |
    export LUA_CPATH="$GITHUB_WORKSPACE/bindings/lua/?.so;;"
    export LD_LIBRARY_PATH=$GITHUB_WORKSPACE/bindings/lua:$LD_LIBRARY_PATH
    cd /tmp && busted ... $GITHUB_WORKSPACE/tests/cpu/lua/

# Julia — push LOAD_PATH, no LD_LIBRARY_PATH needed
- run: |
    cd /tmp && julia -e "push!(LOAD_PATH,\"$GITHUB_WORKSPACE/bindings/julia\"); ..."
```

## Key Gotchas

- **Lua `os.setenv` doesn't exist** in LuaJIT — use `ffi.C.setenv` instead
- **`dlopen` with absolute path ≠ filename search** — pre-loading with absolute
  path doesn't make `dlopen("libfoo.so")` find it. Must also set `LD_LIBRARY_PATH`.
- **Julia `Libdl.DL_LOAD_PATH`** doesn't affect C++ `dlopen` — must use
  `Libdl.dlopen` with `RTLD_GLOBAL` to pre-load.
- **Python `os.environ["LD_LIBRARY_PATH"]`** doesn't affect `dlopen` after
  process startup — must use `ctypes.CDLL` with `RTLD_GLOBAL`.
- **Julia needs `Project.toml` with UUID** for `Pkg.develop()` to work.
  Without UUID, `push!(LOAD_PATH, ...)` also works but `using` may fail
  if `Project.toml` exists without proper structure.
- **Julia `src/` directory** is required for standard package structure.
  `Insight.jl` at root won't be found by `using Insight` via LOAD_PATH.
- **Lua binding auto-init must catch exceptions** — `ins::init()` is called during
  `luaopen__insight` (module loading). If it throws, sol2 can't catch it →
  `std::terminate`. Always wrap in `try { ins::init(); } catch (...) {}`.
- **CMake `option()` vs `set(... FORCE)`** — `option()` respects stale cache values.
  Use `set(VAR ON CACHE BOOL "" FORCE)` when the value must be enforced regardless
  of previous cmake runs (e.g., `INSIGHT_USE_THRUST` when CUDA is ON).
