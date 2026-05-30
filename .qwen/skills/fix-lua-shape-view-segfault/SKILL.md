---
name: fix-lua-shape-view-segfault
description: "Lua shape property segfaults on non-contiguous view arrays (permute/transpose) — use ndim/numel instead"
source: auto-skill
extracted_at: '2026-05-30T15:34:48.924Z'
---

# Fix Lua shape Property Segfault on View Arrays

## The Problem

In the Lua (sol2) binding, accessing the `shape` property on a non-contiguous
view array causes a segfault. This happens with `permute()`, `transpose()`,
`swapaxes()`, `moveaxis()`, and other operations that return views with
non-standard strides.

```lua
local a = ins.from_table({{1,2,3},{4,5,6}})
local r = ins.permute(a, {1, 0})
print(r.numel)   -- ✅ works: 6
print(r.ndim)    -- ✅ works: 2
print(r.shape)   -- ❌ SEGFAULT
```

The `numel` and `ndim` properties work fine because they access the `Shape`
object directly. The `shape` property crashes because it creates a Lua table
by iterating over `shape().dim(i)`, and the sol2 property accessor may be
dereferencing a dangling reference on view arrays.

## Root Cause

The `shape` property accessor in `insight_lua.cpp`:
```cpp
array_type["shape"] = sol::property([](const Array &a) {
    sol::table t;
    for (int i = 0; i < a.shape().ndim(); i++) {
        t[i + 1] = a.shape().dim(i);
    }
    return t;
});
```

`a.shape()` returns a `Shape` by value. Calling it twice (once for `ndim()`
loop bound, once for `dim(i)`) creates two temporaries. On view arrays, the
second `a.shape()` call may produce a corrupted temporary due to sol2's
reference management.

## The Fix

**In the binding code:** Add `sol::this_state` to the property accessor so
sol2 has access to the correct `lua_State` for table creation:

```cpp
// ❌ Segfaults on view arrays
array_type["shape"] = sol::property([](const Array &a) {
    sol::table t;
    for (int i = 0; i < a.shape().ndim(); i++) {
        t[i + 1] = a.shape().dim(i);
    }
    return t;
});

// ✅ Works on all arrays including views
array_type["shape"] = sol::property([](const Array &a, sol::this_state L) {
    sol::table t = sol::table::create(L.lua_state(), a.shape().ndim());
    for (int i = 0; i < a.shape().ndim(); i++) {
        t[i + 1] = a.shape().dim(i);
    }
    return t;
});
```

The root cause is that `sol::property` lambdas that create `sol::table`
objects need the `lua_State` pointer. Without `sol::this_state`, sol2 may
use a stale or wrong state pointer, causing a segfault when the property
is accessed on certain array types (views with non-standard strides).

**In tests (workaround):** If you can't fix the binding, use `ndim` and
`numel` instead of `shape` for view arrays:
```lua
-- ✅ Safe on all arrays
assert.equals(6, r.numel)
assert.equals(2, r.ndim)
```

## Affected Operations

Any operation that returns a non-contiguous view:
- `permute(x, axes)` — reorders dimensions
- `transpose(x)` — swaps last two dims
- `swapaxes(x, axis1, axis2)` — swaps two dims
- `moveaxis(x, src, dst)` — moves a dim
- `fliplr(x)` / `flipud(x)` — reversed view
- `diag(x)` / `diagonal(x)` — diagonal view
- `tril(x)` / `triu(x)` — triangular view

## How to Detect

The symptom is a segfault (exit code 139) when running Lua tests with busted.
busted will stop reporting tests after the crashing `it` block, so the total
test count will be lower than expected.

To isolate: run the suspect operation in `luajit -e` directly:
```bash
luajit -e "
local ins = require('_insight')
local a = ins.from_table({{1,2,3},{4,5,6}})
local r = ins.permute(a, {1, 0})
print(r.shape)  -- crashes here?
"
```

## Related Skills

- `fix-strided-view-data-access` — C++ side: `data<T>()[i]` ignores strides
- `fix-test-dangling-pointer` — different cause, same symptom (segfault/garbage)
