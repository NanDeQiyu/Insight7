---
name: fix-busted-cached-so
description: busted loads _insight.so from ~/.luarocks/lib first, not build/ — must copy .so after rebuild
source: auto-skill
extracted_at: '2026-06-07T01:46:20.559Z'
---

# Busted Loads Cached .so from ~/.luarocks/lib

## Problem

After rebuilding the Lua binding (`make insight_lua`), busted tests still use
the OLD `.so` file. Direct `luajit` tests work fine, but busted fails.

**Root cause**: busted's launcher script (`~/.luarocks/bin/busted`) sets
`package.cpath` to include `~/.luarocks/lib/lua/5.1/?.so` FIRST. Even with
`LUA_CPATH="build/bindings/lua/?.so;;"`, the luarocks path takes precedence
because the launcher explicitly sets `package.cpath`.

**Symptom**: Direct luajit test passes, but busted test fails with old behavior.

## Diagnosis

```bash
# Check which _insight.so files exist and their timestamps
ls -la bindings/lua/_insight.so build/bindings/lua/_insight.so ~/.luarocks/lib/lua/5.1/_insight.so

# Check if new code is in the .so
strings build/bindings/lua/_insight.so | grep "your_new_string"
strings ~/.luarocks/lib/lua/5.1/_insight.so | grep "your_new_string"
```

## Fix

After rebuilding, copy the new `.so` to the luarocks lib directory:

```bash
# Rebuild
make -j24 insight_lua

# Copy to luarocks lib (busted loads from here first)
cp build/bindings/lua/_insight.so ~/.luarocks/lib/lua/5.1/_insight.so
cp build/backends/cpu/libinsight_cpu_backend.so ~/.luarocks/lib/lua/5.1/libinsight_cpu_backend.so
```

## Alternative: Force LUA_CPATH

If you don't want to copy, you can try setting `LUA_CPATH` before the busted
launcher's `package.cpath` assignment. But this is fragile — copying is more
reliable.

## Key Files

- `~/.luarocks/bin/busted` — launcher script, sets `package.cpath`
- `~/.luarocks/lib/lua/5.1/_insight.so` — cached binding
- `build/bindings/lua/_insight.so` — freshly built binding
- `bindings/lua/_insight.so` — copy target (used by `LUA_CPATH="bindings/lua/?.so;;"`)

## Prevention

Add a post-build step to CMakeLists.txt that copies the .so to the luarocks
lib directory automatically.
