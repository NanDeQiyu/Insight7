---
name: lua-dual-abi-build
description: Build Lua binding as single target, switch Lua version via -DLUA_DIR cmake parameter
source: auto-skill
extracted_at: '2026-05-30T20:30:00.000Z'
---

# Lua Binding Build — Single Target, Switch via -DLUA_DIR

## Problem

LuaJIT uses Lua 5.1 C ABI, while Lua 5.3 has a different ABI. Loading a
.so compiled against one into the other causes **segfault** (not catchable
by `pcall`). The user's Lua version is unknown — we can't ship a single .so.

## Solution: One target, configurable via CMake

### CMakeLists.txt — Single `insight_lua` target

```cmake
option(INSIGHT_BUILD_LUA "Build Lua bindings" ON)

if(LUA_DIR)
    # User-specified Lua installation
    find_library(LUA_LIB
        NAMES luajit-5.1 luajit lua5.3 lua-5.3 lua5.1 lua-5.1 lua
        HINTS "${LUA_DIR}/lib" "${LUA_DIR}/lib/x86_64-linux-gnu"
        NO_DEFAULT_PATH)
    find_path(LUA_INCLUDE_DIR NAMES lua.h
        HINTS "${LUA_DIR}/include/luajit-2.1"
              "${LUA_DIR}/include/lua5.3"
              "${LUA_DIR}/include/lua5.1"
              "${LUA_DIR}/include"
        NO_DEFAULT_PATH)
else()
    # Auto-detect: prefer LuaJIT, then Lua 5.3, then Lua 5.1
    # ... standard find_library/find_path ...
endif()

add_library(insight_lua SHARED insight_lua.cpp)
set_target_properties(insight_lua PROPERTIES
    PREFIX "" OUTPUT_NAME "_insight"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bindings/lua")
target_include_directories(insight_lua PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${LUA_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/sol2/include)
target_link_libraries(insight_lua PRIVATE insight_core ${LUA_LIB})
target_compile_features(insight_lua PRIVATE cxx_std_17)
```

### Usage

```bash
# LuaJIT build
sed -i '/LUA_LIB\|LUA_INCLUDE\|LUA_DIR/d' build/CMakeCache.txt
cmake .. ... -DLUA_DIR=$HOME/public/luajit
make -j$(nproc) insight_lua

# Lua 5.3 build (switch)
sed -i '/LUA_LIB\|LUA_INCLUDE\|LUA_DIR/d' build/CMakeCache.txt
cmake .. ... -DLUA_DIR=/usr
make -j$(nproc) insight_lua
```

### Wrapper — Always `require("_insight")`

No version detection needed. The wrapper always loads `_insight.so`:

```lua
-- insight/init.lua
local ok, native = pcall(require, "_insight")
if not ok then
    error("Failed to load Insight native module: " .. tostring(native))
end
```

## Key Points

1. **Single `.so` output**: Always `_insight.so`. The Lua version is baked in at compile time.

2. **Cache clearing required**: CMake caches `LUA_LIB` and `LUA_INCLUDE_DIR`.
   When switching Lua versions, must clear these from CMakeCache.txt:
   ```bash
   sed -i '/LUA_LIB\|LUA_INCLUDE\|LUA_DIR/d' build/CMakeCache.txt
   ```

3. **ABI mismatch = segfault**: `pcall` cannot catch it. The process crashes
   before Lua error handling kicks in. This is why the wrapper must NOT
   try multiple .so files with pcall fallback.

4. **sol2 is ABI-agnostic**: sol2 is header-only and works with both
   Lua 5.1 and 5.3. The ABI difference is in the Lua C API linkage.

5. **Testing**: After building for a specific Lua version, test with that
   version's interpreter:
   ```bash
   # After LuaJIT build:
   ~/public/luajit/bin/luajit -e "require('insight'); print('OK')"
   
   # After Lua 5.3 build:
   lua5.3 -e "require('insight'); print('OK')"
   ```

6. **GPU testing**: Always set `LD_LIBRARY_PATH` to include both CPU and
   CUDA backend directories:
   ```bash
   LD_LIBRARY_PATH="build:build/backends/cpu:build/backends/cuda:$LD_LIBRARY_PATH"
   ```
   Then call `ins.load_backend("cuda")` before GPU operations.
