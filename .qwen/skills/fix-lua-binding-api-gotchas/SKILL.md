---
name: fix-lua-binding-api-gotchas
description: Common Lua binding API issues — comparison operators, item/get, __tostring, scalar inputs
source: auto-skill
extracted_at: '2026-06-02T06:00:00.000Z'
---

# Lua Binding API Gotchas

## 1. Comparison operators return native Lua booleans, not Arrays

In Lua, `==`, `<`, `<=`, `>`, `>=` on Array objects return native Lua booleans, NOT Insight Array objects.

```lua
-- ❌ WRONG — c is a Lua boolean, not an Array
local c = (a == b)
assert.are.equal(3, c.numel)  -- ERROR: attempt to index field 'numel' (a nil value)

-- ✅ CORRECT — use Insight native functions
local c = ins.equal(a, b)
assert.are.equal(3, c.numel)
```

Available comparison functions:
- `ins.equal(a, b)` instead of `a == b`
- `ins.not_equal(a, b)` instead of `a ~= b`
- `ins.less(a, b)` instead of `a < b`
- `ins.less_equal(a, b)` instead of `a <= b`
- `ins.greater(a, b)` instead of `a > b`
- `ins.greater_equal(a, b)` instead of `a >= b`

**Why:** Sol2 registers `__eq`/`__lt`/`__le` metamethods that return bool for Lua compatibility. The Insight comparison functions return Array.

## 2. `item()` vs `get()` — scalar extraction

- `arr:item()` — returns double for 0-dimensional scalar arrays ONLY. Takes NO arguments. Returns nil for multi-element arrays.
- `arr:get(idx)` — returns double for element at 0-based index. Works on any array.

```lua
-- ❌ WRONG — item(idx) doesn't work, item() only for scalars
local sum_val = ins.sum(arr)
print(sum_val:item(0))  -- nil! item doesn't take an index

-- ❌ WRONG — item() on multi-element array returns nil
local data = ins.from_table({1.0, 2.0, 3.0})
print(data:item())  -- nil! not a scalar

-- ✅ CORRECT — use get() for element access
local sum_val = ins.sum(arr)
print(sum_val:get(0))  -- works

local data = ins.from_table({1.0, 2.0, 3.0})
print(data:get(0))  -- 1.0
print(data:get(1))  -- 2.0
```

**Common mistake in tests**: Writing `c:item(0)` when you mean `c:get(0)`. The `item()` function ignores its argument and returns nil for non-scalar arrays. Always use `get(idx)` for indexed access.

## 3. `ins.item(arr, idx)` does NOT exist as module function

`item` and `get` are methods on the Array usertype, not module-level functions.

```lua
-- ❌ WRONG
local v = ins.item(arr, 0)

-- ✅ CORRECT
local v = arr:item(0)  -- or arr:get(0)
```

## 4. `__tostring` metamethod — uses C++ `ins::to_string()`

The `__tostring` metamethod calls `insight_array_tostring()` (C API) which delegates to `ins::to_string()`. Output matches C++ format:

```lua
print(arr)
-- Array(shape=[2, 2], dtype=float64, place=cpu:0,
--       [[1., 2.],
--        [3., 4.]])
```

If you see `<insight.Array shape=...>` (old format), the binding is using the fallback path. Ensure `insight_array_tostring` is linked correctly.

## 4b. sol2 default parameters DON'T WORK — use lambda wrapper

When binding a C++ function with default parameters via `sig["func"] = &ns::func`, sol2 does NOT pass the defaults. This causes segfaults (not exceptions).

```cpp
// ❌ WRONG — default bool append=true not passed, segfaults
sig["write_bin"] = &signal::write_bin;

// ✅ CORRECT — use lambda + sol::optional
sig["write_bin"] = [](const std::string &file, const Array &data,
                       sol::optional<bool> append) {
  signal::write_bin(file, data, append.value_or(true));
};
```

## 5. Signal functions expect Arrays, not scalars

Lua signal functions (mel2hz, hz2mel, etc.) are registered via sol2 and expect `const Array&`, not Lua numbers.

```lua
-- ❌ WRONG — passes scalar, crashes
local hz = ins.signal.mel2hz(1000.0)

-- ✅ CORRECT — wrap scalar in Array
local hz_arr = ins.signal.mel2hz(ins.from_table({1000.0}))
local hz = hz_arr:get(0)
```

Window functions that take integer M (hann, hamming, etc.) work with plain integers since sol2 auto-converts int → int64_t.

## 6. Busted test file discovery

Busted defaults to `*_spec.lua` pattern. For `test_*.lua` files:

```bash
busted --pattern="test_.+%.lua" .
```

## 7. Tile/reshape need Shape objects

```lua
-- ❌ WRONG — raw table causes segfault
ins.tile(a, {2})

-- ✅ CORRECT
ins.tile(a, ins.Shape({2}))
```

Similarly, `reshape` is a method:
```lua
-- ❌ WRONG
ins.reshape(a, {2, 3})

-- ✅ CORRECT
a:reshape({2, 3})
```
