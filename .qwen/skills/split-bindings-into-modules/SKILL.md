---
name: split-bindings-into-modules
description: Split monolithic language bindings into per-module wrapper files aligned with C++ source structure
source: auto-skill
extracted_at: '2026-06-01T00:24:01.968Z'
---

# Split Monolithic Bindings into Per-Module Wrappers

## Problem

Language bindings (Python/Lua/Julia) start as monolithic files where all
functions are bound in one place. The requirement is to split them into
per-module wrapper files that:
1. Align with C++ source structure (elementwise, unary, reduction, etc.)
2. Add docstrings (Google-style for Python, LDoc for Lua, Julia docstrings)
3. Keep the native binding as-is (no C++ changes needed)

## Architecture

```
bindings/python/insight/
├── __init__.py          # Imports from wrapper modules
├── creation.py          # Wrapper for ins::creation functions
├── elementwise.py       # Wrapper for ins::elementwise functions
├── unary.py             # Wrapper for ins::unary functions
├── reduction.py         # Wrapper for ins::reduction functions
├── manipulation.py      # Wrapper for ins::manipulation functions
├── linalg.py            # Wrapper for ins::linalg functions
├── fft.py               # Wrapper for ins::fft functions
├── complex.py           # Wrapper for ins::complex functions
├── random.py            # Wrapper for ins::random functions
├── cast.py              # Wrapper for ins::cast functions
├── indexing.py          # Wrapper for ins::indexing functions
└── signal/              # Already split (14 files)
    ├── __init__.py
    ├── windows.py
    └── ...

bindings/lua/insight/
├── init.lua             # Loads all submodules
├── signal.lua           # Loader for insight.signal.init
├── elementwise.lua      # Wrapper module
├── unary.lua
├── reduction.lua
├── manipulation.lua
├── linalg.lua
├── fft.lua
├── complex.lua
├── random.lua
├── cast.lua
├── indexing.lua
└── signal/
    ├── init.lua         # Merges all signal sub-modules
    ├── windows.lua
    └── ...

bindings/julia/modules/
├── elementwise.jl       # Wrapper with docstrings
├── unary.jl
├── reduction.jl
├── manipulation.jl
├── linalg.jl
├── fft.jl
├── complex.jl
├── random.jl
├── cast.jl
├── indexing.jl
└── signal/
    ├── windows.jl
    └── ...
```

## Python Wrapper Pattern

Each wrapper file imports from the native module and adds docstrings:

```python
"""Elementwise binary operations for Insight7.

Provides add, sub, mul, div, pow, mod, maximum, minimum,
comparison, and logical operations.
"""

from insight._insight import (
    add as _add,
    sub as _sub,
    # ...
)

__all__ = ["add", "sub", "mul", "div"]


def add(a, b):
    """Element-wise addition of two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        Element-wise sum of a and b.
    """
    return _add(a, b)
```

### __init__.py update

Replace direct `from ._insight import *` with imports from wrapper modules:

```python
# Core types still from native
from ._insight import DType, Place, Shape, Slice, Array, init, ...

# Wrapped modules
from .creation import *     # noqa: F401,F403
from .elementwise import *  # noqa: F401,F403
from .unary import *        # noqa: F401,F403
# ... etc

# Signal submodule
from . import signal as _signal_module
signal = _signal_module
```

The wildcard imports ensure backward compatibility — `ins.add(a, b)` still works.

## Lua Wrapper Pattern

```lua
--- Elementwise binary operations.
-- @module insight.elementwise
local native = require("_insight")
local M = {}

--- Element-wise addition.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Element-wise sum.
function M.add(a, b)
    return native.add(a, b)
end

return M
```

### init.lua update

Add submodule loading at the end:

```lua
-- Signal submodule
local ok_signal, signal_mod = pcall(require, "insight.signal")
if ok_signal then M.signal = signal_mod end

-- Non-signal sub-modules
local submodules = {"elementwise", "unary", "reduction", ...}
for _, mod_name in ipairs(submodules) do
    local ok_mod, mod = pcall(require, "insight." .. mod_name)
    if ok_mod then
        for k, v in pairs(mod) do
            if M[k] == nil then M[k] = v end
        end
    end
end
```

Use `pcall` so missing modules don't crash the loader.

## Julia Wrapper Pattern

Julia uses `include()` to split files. Sub-files access parent module scope:

```julia
# modules/elementwise.jl
# Elementwise binary operations for Insight7.

"""
    add(a::InsightArray, b::InsightArray) -> InsightArray

Element-wise addition of two arrays.

# Arguments
- `a`: First input array.
- `b`: Second input array.

# Returns
Element-wise sum of `a` and `b`.
"""
function add(a::InsightArray, b::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_add, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}), a.ptr, b.ptr)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end
```

## Key Decisions

1. **Native binding stays monolithic** — no C++ changes needed
2. **Wrapper files are thin** — just import + docstring + delegate
3. **Wildcard imports in __init__.py** — backward compatible
4. **pcall in Lua** — graceful degradation if modules missing
5. **Signal submodule already done** — don't redo, just align non-signal modules

## Verification

```bash
# Build
cmake --build build -j$(nproc)

# Python
PYTHONPATH=bindings/python LD_LIBRARY_PATH=build/backends/cpu \
    python3 -c "import insight as ins; a = ins.zeros([2,3]); print(ins.add(a, a))"

# Lua
LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" \
    LD_LIBRARY_PATH=build/backends/cpu \
    luajit -e "local ins = require('insight'); print(ins.add(ins.zeros({2,3}), ins.zeros({2,3})))"

# Julia
LD_LIBRARY_PATH=build/backends/cpu \
    julia -e 'push!(LOAD_PATH, "bindings/julia"); using Insight; println(Insight.add(Insight.zeros([2,3]), Insight.zeros([2,3])))'
```

## Wrapping Core Types (Array, DType, Place, Shape, Slice)

In addition to function modules, core types need wrapper modules with docstrings.

### Python: types.py

Create `bindings/python/insight/types.py` to wrap DType, Place, Shape, Slice:

```python
"""Core types: DType, Place, Shape, Slice."""
from ._insight import DType, Place, Shape, Slice, CPUPlace, GPUPlace

DType.__doc__ = """Data type enum. Available: bool, uint8..uint64, float16, bfloat16, float32, float64, complex64, complex128."""
Place.__doc__ = """Device placement descriptor. Use CPUPlace() or GPUPlace(id)."""
CPUPlace.__doc__ = """CPU device placement."""
GPUPlace.__doc__ = """GPU device placement. Args: device_id (int, default 0)."""
Shape.__doc__ = """Array shape. Args: dims (list of int)."""
Slice.__doc__ = """Slice descriptor. Args: start, stop, step."""
__all__ = ["DType", "Place", "Shape", "Slice", "CPUPlace", "GPUPlace"]
```

### Python: array.py

Create `bindings/python/insight/array.py` to wrap Array class methods:

```python
"""Array class wrapper with documented methods."""
from ._insight import Array as _NativeArray

# Monkey-patch docstrings onto native class methods
_NativeArray.reshape.__doc__ = """Return a view with a new shape.\n\nArgs:\n    new_shape: Target shape.\nReturns:\n    Array: View with new shape (shares data)."""
_NativeArray.transpose.__doc__ = """Return a view with axes transposed."""
_NativeArray.to.__doc__ = """Transfer to device.\n\nArgs:\n    place: CPUPlace or GPUPlace.\nReturns:\n    Array on target device."""
_NativeArray.numpy.__doc__ = """Convert to NumPy array (CPU only)."""
# ... etc for squeeze, unsqueeze, contiguous, copy

Array = _NativeArray
__all__ = ["Array"]
```

### Python: __init__.py update for types

Replace direct native imports with wrapper module imports:

```python
# BEFORE (direct from native):
from ._insight import DType, Place, Shape, Slice, Array, CPUPlace, GPUPlace

# AFTER (from wrapper modules):
from .types import DType, Place, Shape, Slice, CPUPlace, GPUPlace
from .array import Array
```

Keep `from ._insight import *` for backward compatibility (catches anything not yet wrapped).

### Lua: Documentation tables

Add to `insight/init.lua` after all function definitions:

```lua
--- Array type: properties (shape, dtype, place, numel, ndim), methods
-- (contiguous, reshape, transpose, squeeze, to, copy, get, item),
-- metamethods (+, -, *, /, ==, <).
M._array_docs = "See Array usertype"

--- Place types: CPUPlace(), GPUPlace(id).
--- DType shortcuts: float32, float64, int32, ... , complex128.
```

### Julia: modules/types.jl

Create `bindings/julia/modules/types.jl` with Julia docstrings for InsightArray, dtype constants, etc.

## Cross-Language Test Alignment

When creating aligned tests across Python/Lua/Julia, use this pattern:

1. Define a shared test specification (35 test cases across 7 categories)
2. Each language implements the EXACT same tests in the EXACT same order
3. Use language-native assertion mechanisms:
   - Python: `assert`, `pytest`
   - Lua: busted `assert.are.equal`
   - Julia: custom `check(name, cond)` function
4. Verify counts match: `grep -c 'def test_' *.py`, `grep -c 'it(' *.lua`, `grep -c 'check(' *.jl`

## Demo File Alignment

When aligning demos across 4 languages:
1. Move existing C++ demos to `demos/cpp/`
2. Port to Python/Lua/Julia in `demos/{python,lua,julia}/`
3. Verify file counts: `ls demos/{cpp,python,lua,julia}/ | wc -l` must be equal
4. Each demo should produce similar output (print same values)
5. GPU demos wrapped in try/catch for non-CUDA platforms

## Pre-commit

All new `.py`, `.lua`, `.jl` files must pass pre-commit:
```bash
PATH="$HOME/.local/bin:$PATH" pre-commit run --all-files
```
