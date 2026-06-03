---
name: lua-dual-calling-convention
description: Add positional and named-table calling convention to Lua bindings using _wrap helper
source: auto-skill
extracted_at: '2026-06-01T14:22:00.581Z'
---

# Lua Dual Calling Convention (argparse)

Insight7 Lua bindings support both positional and named-table argument styles:

```lua
-- All equivalent:
ins.sum(x, 1)           -- positional
ins.sum{x, axis=1}      -- mixed positional + named
ins.sum{x=x, axis=1}    -- fully named
```

## Implementation

### Step 1: Add `_wrap` helper to each submodule

Add this local function right after `local M = {}` in every Lua wrapper file:

```lua
local function _wrap(names, fn)
  return function(...)
    if select("#", ...) == 1 and type(select(1, ...)) == "table" then
      local t = select(1, ...)
      local has_names = false
      for k, _ in pairs(t) do
        if type(k) ~= "number" then has_names = true; break end
      end
      if has_names then
        local pos = {}
        for i, name in ipairs(names) do
          pos[i] = t[name]
          if pos[i] == nil then pos[i] = t[i] end
        end
        return fn(table.unpack(pos, 1, #names))
      end
    end
    return fn(...)
  end
end
```

**How it works**:
1. If called with a single table argument that has named keys → extract positional args from named keys (falling back to numeric indices)
2. Otherwise → pass through as positional args

### Step 2: Convert each function

```lua
-- Before (positional only):
function M.sum(x, axis, keepdims)
  return native.sum(x, axis, keepdims or false)
end

-- After (dual mode):
M.sum = _wrap({"x", "axis", "keepdims"}, function(x, axis, keepdims)
  return native.sum(x, axis, keepdims or false)
end)
```

Keep ALL existing LDoc docstrings (`---` comments) above the `_wrap` call.

### Step 3: Add `args()` and `wrap()` to init.lua

For the top-level `init.lua`, add public versions:

```lua
function M.args(names, ...)
  -- Same logic as _wrap but returns parsed table instead of calling fn
end

function M.wrap(arg_names, fn)
  -- Same as _wrap but available as ins.wrap() for user code
end
```

## Key Design Decisions

1. **Local `_wrap` per file**: Avoids circular dependency (submodules can't require init.lua)
2. **Named-key detection**: Checks if any key is non-numeric (`type(k) ~= "number"`)
3. **Fallback to numeric**: `t[i]` as fallback allows `{x, y, axis=1}` mixed style
4. **LDoc preserved**: Docstrings go above the `M.func = _wrap(...)` assignment

## Files Updated

All 25 Lua wrapper modules:
- Top-level: elementwise, unary, reduction, manipulation, linalg, fft, complex, random, cast, indexing
- Signal: windows, waveforms, bsplines, filter_design, convolution, filtering, spectral, wavelets, acoustics, demod, peak_finding, radar, estimation, io

## Pitfalls

1. **Table arrays vs named tables**: A table `{1, 2, 3}` has only numeric keys → treated as positional. A table `{x=1, y=2}` has named keys → treated as named.
2. **Boolean false**: `func{x=false}` correctly sets x=false (not nil).
3. **Optional params**: Use `or default` inside the wrapped function, not in the names list.
