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

### 2. Backend .so Pre-loading

The C++ `ins::init()` calls `dlopen("libinsight_cpu_backend.so")` which searches
`LD_LIBRARY_PATH`. At runtime, modifying `LD_LIBRARY_PATH` doesn't always work.
Solution: pre-load the backend .so BEFORE importing the native module.

**Python** (`bindings/python/insight/__init__.py`):
```python
import ctypes as _ct, os as _os
_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
_backend_so = _os.path.join(_pkg_dir, "libinsight_cpu_backend.so")
if _os.path.isfile(_backend_so):
    _ct.CDLL(_backend_so, mode=_ct.RTLD_GLOBAL)
from ._insight import *
```

**Lua** (`bindings/lua/insight/init.lua`):
```lua
local ffi = require("ffi")
ffi.cdef[[int setenv(const char*, const char*, int); void *dlopen(const char*, int);]]
ffi.C.setenv("LD_LIBRARY_PATH", _parent_dir .. ":" .. _ld, 1)
ffi.C.dlopen(_backend, 258) -- RTLD_NOW | RTLD_GLOBAL
```

**Julia** (`bindings/julia/Insight.jl`):
```julia
using Libdl
Libdl.dlopen(_backend, Libdl.RTLD_GLOBAL)
```

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

Create `bindings/lua/insight-1.0-1.rockspec` with cmake build type.
Enable ALL features (CUDA, OpenBLAS, FFTW, Matplot) — cmake auto-disables
unavailable ones. Don't use `$(LUA_LIBDIR)` — not set on all systems.
Install: `luarocks make bindings/lua/insight-1.0-1.rockspec LUA_DIR=/usr CMAKE_BUILD_DIR=build --local`

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
