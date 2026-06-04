---
name: smart-backend-init
description: ins::init() with optional auto-discover CPU+GPU backends, dynamic-only loading
source: auto-skill
extracted_at: '2026-06-04T10:52:54.680Z'
---

# Smart Backend Init

## Problem

Framework initialization needed to:
1. Auto-discover available backends (CPU, CUDA, ROCm, etc.)
2. Load CPU first (required), then first GPU found (alphabetical)
3. Support explicit backend specification for advanced users
4. Remove static backend linking (dynamic only)

## Solution

### C++ API

```cpp
// include/insight/init.h
#include <optional>
#include <string>
#include <vector>

namespace ins {
void init(std::optional<std::vector<std::string>> backends = std::nullopt);
void init(const std::vector<std::string> &backends);  // convenience overload
}
```

### Behavior matrix

| Call | Behavior |
|------|----------|
| `init()` | CPU required + auto-discover first GPU (alphabetical) |
| `init({})` | Load nothing |
| `init({"cpu"})` | CPU only |
| `init({"cpu","cuda"})` | Load both in order |
| `init({"cuda"})` | GPU only (no CPU check) |

### Backend discovery

```cpp
static std::vector<std::string> discover_backends() {
    std::vector<std::string> found;
    // Scan current directory for libinsight_*_backend.so (or .dll)
    // Match pattern: libinsight_<name>_backend.so
    // Sort alphabetically
    return found;
}
```

On Linux: `opendir(".")` + `readdir` matching `libinsight_*_backend.so`
On Windows: `FindFirstFileA("insight_*_backend.dll")`

### Dynamic loading only

Remove all static backend support:
- Remove `INSIGHT_DYNAMIC_CPU_BACKEND` / `INSIGHT_DYNAMIC_GPU_BACKEND` CMake options
- Remove `INSIGHT_CPU_BACKEND_EXPORTS` conditional compilation in init.cpp
- Remove `init_static_backend()` function
- Backend CMakeLists.txt always builds SHARED

```cmake
# backends/cpu/CMakeLists.txt — always shared
add_library(insight_cpu_backend SHARED ${SOURCES})
target_compile_definitions(insight_cpu_backend PRIVATE INSIGHT_CPU_BACKEND_EXPORTS)
target_link_libraries(insight_cpu_backend PRIVATE insight_core)
```

### Language binding auto-init

All bindings auto-init with smart discovery on import:

**Python** (pybind11):
```cpp
PYBIND11_MODULE(_insight, m) {
    if (!ins::is_initialized()) {
        ins::init();  // Smart discovery
    }
    m.def("init", [](py::args args) {
        if (args.size() == 0) ins::init(std::nullopt);
        else if (py::isinstance<py::list>(args[0])) {
            auto backends = args[0].cast<std::vector<std::string>>();
            if (backends.empty()) ins::init(std::vector<std::string>{});
            else ins::init(backends);
        }
    });
}
```

**Lua** (sol2):
```cpp
extern "C" int luaopen__insight(lua_State *L) {
    if (!ins::is_initialized()) {
        ins::init();  // Smart discovery
    }
    m["init"] = [](sol::table backends) {
        std::vector<std::string> be;
        for (size_t i = 1; i <= backends.size(); i++)
            be.push_back(backends.get<std::string>(i));
        ins::init(be);
    };
}
```

**Julia** (ccall):
```julia
function __init__()
    ccall((:insight_jl_init_cpu, LIB_INSIGHT), Cvoid, ())
end
```
Where `insight_jl_init_cpu` calls `ins::init()` (smart discovery).

### Demo pattern — GPU silent skip

All demos use smart init and silent GPU skip:

```cpp
// C++
ins::init();  // Smart: CPU + first GPU if available
// ... CPU work ...
if (has_device(DeviceKind::GPU)) {
    // GPU work — only shown when GPU available
}
```

```python
# Python
ins.init()  # Smart discovery
# ... CPU work ...
try:
    ins.load_backend("cuda")
    # GPU work
except Exception:
    pass  # Silent skip
```

```lua
-- Lua (auto-init on require, no explicit call needed)
-- ... CPU work ...
local ok = pcall(function() ins.load_backend("cuda") end)
if ok then
    -- GPU work
end
```

```julia
# Julia (auto-init via __init__)
# ... CPU work ...
if Insight.load_backend("cuda")
    # GPU work
end
```

**Why:** CI has no GPU → silent skip. User with GPU → full benchmark output.

### Convenience overload for backward compatibility

```cpp
// init.h
void init(std::optional<std::vector<std::string>> backends = std::nullopt);
void init(const std::vector<std::string> &backends);

// init.cpp
void init(const std::vector<std::string> &backends) {
    init(std::optional<std::vector<std::string>>(backends));
}
```

**Why needed:** `{"cpu"}` can't implicitly convert to `std::optional<std::vector<std::string>>`.
Without the overload, all existing call sites like `init({"cpu"})` would fail to compile.

### Error handling

- CPU backend not found → `INS_THROW("Failed to load CPU backend")`
- Specified GPU not found → `INS_THROW("Failed to load GPU backend: " + name)`
- Empty vector → no error (load nothing)
- Already initialized → no-op (safe to call multiple times)
