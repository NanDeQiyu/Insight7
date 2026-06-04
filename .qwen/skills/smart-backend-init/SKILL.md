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

### GPU auto-discover fallback

`discover_backends()` only scans the current directory (`.`). When running demos
from `build/bin/demos/`, the .so files are in `build/backends/cuda/` — not found.

**Fix**: After the scan, try `dlopen` for common GPU backend names directly.
`dlopen` uses `LD_LIBRARY_PATH`, so it finds .so files in any directory:

```cpp
// 2. Scan current directory for other backends
auto available = discover_backends();
for (const auto &name : available) {
    if (name == "cpu") continue;
    if (try_load_backend(DeviceKind::GPU,
                         ("insight_" + name + "_backend").c_str())) {
        break;
    }
}
// 3. If no GPU backend found via scan, try common names directly
//    (dlopen uses LD_LIBRARY_PATH, works when .so is in another directory)
if (!get_device_interface(DeviceKind::GPU)) {
    try_load_backend(DeviceKind::GPU, "insight_cuda_backend");
}
```

**Why:** Users run demos from various directories. `LD_LIBRARY_PATH` is the
standard way to locate shared libraries. The scan is a convenience for the
common case (running from the build directory), but the fallback ensures
GPU is found regardless of working directory.

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
    // Supports no-arg (auto-discover) and table of backend names
    m["init"] = [](sol::optional<sol::table> backends) {
        if (!backends.has_value()) {
            ins::init();  // auto-discover
        } else {
            std::vector<std::string> be;
            for (size_t i = 1; i <= backends->size(); i++)
                be.push_back(backends->get<std::string>(i));
            ins::init(be);
        }
    };
}
```
**Note**: `sol::optional<sol::table>` allows `ins.init()` with no args.
Without it, sol2 requires the table argument and crashes on `ins.init()`.

**Julia** (ccall):
```julia
function __init__()
    ccall((:insight_jl_init_cpu, LIB_INSIGHT), Cvoid, ())
end
```
Where `insight_jl_init_cpu` calls `ins::init()` (smart discovery).

### Demo pattern — GPU silent skip with has_device

All demos use `has_device` to check GPU availability BEFORE any GPU code.
The GPU section (including title/separator) is entirely inside the if block.

**CRITICAL**: Do NOT use `load_backend("cuda")` to check GPU — it returns true
even when no GPU is available (just loads the .so). Use `has_device("gpu")` which
checks if the GPU device interface is actually registered.

```cpp
// C++
ins::init();  // Smart: CPU + first GPU if available
// ... CPU work ...
if (ins::has_device(DeviceKind::GPU)) {
    separator("GPU Section Title");  // Title INSIDE if block
    // GPU work
}
// No else — completely silent when no GPU
```

```python
# Python
ins.init()  # Smart discovery
# ... CPU work ...
if ins.has_device("gpu"):
    separator("GPU Section Title")
    # GPU work
# No else — completely silent
```

```lua
-- Lua
ins.init()  -- Smart discovery
-- ... CPU work ...
if ins.has_device("gpu") then
    separator("GPU Section Title")
    -- GPU work
end
-- No else — completely silent
```

```julia
# Julia (auto-init via __init__)
# ... CPU work ...
if Insight.has_device(1)  # 1 = GPU
    separator("GPU Section Title")
    # GPU work
end
```

**Why has_device, not load_backend**: `load_backend("cuda")` just loads the .so file.
It returns true even without a physical GPU. `has_device("gpu")` checks if the GPU
device interface is registered AND functional. This is the only reliable check.

**Why title inside if block**: If the title is printed before the if check, CI output
shows "GPU Section Title" followed by nothing — confusing. Everything GPU-related
must be inside the if block.

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
