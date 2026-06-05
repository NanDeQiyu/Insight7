---
name: cross-platform-backend-discovery
description: Backend discovery for pip/luarocks installs — add_backend_search_path + LD_LIBRARY_PATH + ctypes pre-load
source: auto-skill
extracted_at: '2026-06-05T16:55:08.448Z'
---

## Cross-Platform Backend Discovery

**Problem**: Backend .so files may be in non-standard locations (luarocks, pip, editable installs). `dlopen("libinsight_cpu_backend.so")` with a simple name only searches `LD_LIBRARY_PATH` + system paths, NOT the package directory.

### Solution: Three-layer defense

1. **`add_backend_search_path(path)`** — C++ API, Python tells C++ the exact directory. `try_load_backend()` tries full path `path/libinsight_cpu_backend.so` as fallback. **This is the primary fix.**

2. **`LD_LIBRARY_PATH`** — Set via `os.environ` before importing native module. Works for standard installs but unreliable for `pip install -e .`.

3. **`ctypes.CDLL` pre-load** — Loads .so with `RTLD_GLOBAL` into process. Helps but doesn't guarantee `dlopen` finds it by simple name.

### Python `__init__.py` (complete pattern)

```python
import ctypes as _ct
import os as _os
import glob as _gl

_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))

# Layer 2: LD_LIBRARY_PATH
_ld = _os.environ.get("LD_LIBRARY_PATH", "")
if _pkg_dir not in _ld.split(":"):
    _os.environ["LD_LIBRARY_PATH"] = _pkg_dir + ":" + _ld if _ld else _pkg_dir

# chdir so discover_backends() scans "." correctly
_saved_cwd = _os.getcwd()
_os.chdir(_pkg_dir)

# Layer 3: Pre-load with ctypes
for _so in _gl.glob(_os.path.join(_pkg_dir, "libinsight_*_backend.so")):
    try: _ct.CDLL(_so, mode=_ct.RTLD_GLOBAL)
    except OSError: pass

from ._insight import *  # noqa
_os.chdir(_saved_cwd)

# Layer 1: add_backend_search_path (primary fix)
from ._insight import init as _native_init, is_initialized as _is_init
from ._insight import add_backend_search_path as _add_search_path
_add_search_path(_pkg_dir)

# Also search _insight.so directory (may differ in editable installs)
try:
    import importlib.util as _ilu
    _spec = _ilu.find_spec("insight._insight")
    if _spec and _spec.origin:
        _native_dir = _os.path.dirname(_os.path.abspath(_spec.origin))
        if _native_dir != _pkg_dir:
            _add_search_path(_native_dir)
except Exception:
    pass

if not _is_init():
    _native_init()
```

### C++ `add_backend_search_path` (in `src/init.cpp`)

```cpp
static std::vector<std::string> g_extra_search_paths;

void add_backend_search_path(const std::string &path) {
  g_extra_search_paths.push_back(path);
}

static bool try_load_backend(DeviceKind kind, const char *lib_name) {
  std::string lib_filename = std::string(backend_prefix()) + lib_name +
                             backend_extension();
  // Try default search (LD_LIBRARY_PATH, system paths)
  LibHandle lib = load_library(lib_filename.c_str());
  // Try extra search paths with FULL PATH (primary fix for pip installs)
  if (!lib) {
    for (const auto &path : g_extra_search_paths) {
      std::string full = path + "/" + lib_filename;
      lib = load_library(full.c_str());
      if (lib) break;
    }
  }
  if (!lib) return false;
  // ... InitPlugin registration ...
}
```

### CMake POST_BUILD (cross-platform, BUILD-time copy)

**⚠️ CRITICAL GOTCHA**: `file(GLOB)` runs at **configure time**, NOT build time. If backends aren't built yet (they never are at configure time), the glob is empty → no copy commands registered → `.so` files never copied → `pip install -e .` fails with "Failed to load CPU backend". This was the root cause of 5+ CI failures.

**Rule**: NEVER use `file(GLOB)` to find files produced by the build. Always use a `cmake -P` script that runs at BUILD time.

```cmake
# bindings/python/CMakeLists.txt
add_custom_command(TARGET insight_python POST_BUILD
    COMMAND ${CMAKE_COMMAND}
        -DSRC_DIR="${CMAKE_BINARY_DIR}/backends"
        -DDST_DIR="${CMAKE_SOURCE_DIR}/bindings/python/insight"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/copy_backends.cmake"
    COMMENT "Copying backend .so files to Python package"
)
```

```cmake
# bindings/python/copy_backends.cmake (runs at BUILD time)
file(GLOB _BACKEND_FILES
    "${SRC_DIR}/*/libinsight_*_backend.so"
    "${SRC_DIR}/*/libinsight_*_backend.dylib"
    "${SRC_DIR}/*/insight_*_backend.dll")
foreach(_f ${_BACKEND_FILES})
    get_filename_component(_name ${_f} NAME)
    file(COPY_FILE "${_f}" "${DST_DIR}/${_name}" ONLY_IF_DIFFERENT)
endforeach()
```

### Why NOT auto-init in PYBIND11_MODULE

Auto-init (`ins::init()` in `PYBIND11_MODULE`) runs BEFORE Python can call `add_backend_search_path()`. Deferred init lets Python set up the environment first.

**Why:** `pip install -e .` puts .so in source dir, `dlopen` can't find it via `LD_LIBRARY_PATH` alone. Full-path `dlopen` via `add_backend_search_path` is 100% reliable.
