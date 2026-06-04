---
name: python-wrapper-auto-conversion
description: Add _to_shape auto-conversion in Python wrappers so users can pass lists instead of Shape objects
source: auto-skill
extracted_at: '2026-06-02T12:00:00.000Z'
---

# Python Wrapper Auto-Conversion

Insight7 Python wrappers should accept native Python types (lists, tuples) in addition to Insight types (Shape, DType, Place). This makes the API user-friendly: `ins.rand([3,4])` instead of `ins.rand(ins.Shape([3,4]))`.

## Pattern

Add a `_to_shape` helper to each wrapper module that accepts Shape arguments:

```python
from typing import Union
from insight._insight import Shape

def _to_shape(s: Union[list, tuple, Shape]) -> Shape:
    """Convert a list/tuple to Shape if needed."""
    if isinstance(s, Shape):
        return s
    return Shape(s)
```

Then use it in all functions that pass shape to native calls:

```python
def rand(shape, dtype=None, place=None):
    return _native_rand(_to_shape(shape), dtype, place)
```

## Files to update

Apply this pattern to:
- `bindings/python/insight/random.py` — rand, randn, randint, normal, uniform, exponential, gamma, beta, binomial, poisson
- `bindings/python/insight/indexing.py` — indices
- `bindings/python/insight/creation.py` — zeros, ones, full, eye (if they accept shape)

## Important: Watch for parameter name collisions

The `gamma` function has a parameter named `shape` (the distribution shape parameter, a float) AND an `out_shape` parameter (the array shape). Apply `_to_shape` to `out_shape`, NOT `shape`:

```python
# ❌ WRONG — applies _to_shape to the distribution parameter
return _native_gamma(_to_shape(shape), rate, out_shape, dtype, place)

# ✅ CORRECT — applies _to_shape to the output array shape
return _native_gamma(shape, rate, _to_shape(out_shape), dtype, place)
```

## Lua equivalent

Lua bindings handle this via sol2's automatic table-to-Shape conversion. The native `Shape` constructor is registered with sol2, so `native.rand({3, 4})` works automatically. No wrapper changes needed.

## Julia equivalent

Julia bindings should accept `Vector{Int64}` or `Tuple` and convert to the ccall-compatible format. Use `collect(Int64, shape)` if needed.

## How to apply

When adding new wrapper functions that take shape arguments:
1. Add `_to_shape` helper if not already present
2. Wrap all shape parameters with `_to_shape()`
3. Verify with: `ins.rand([3,4])` should work without explicit Shape()
4. Also verify: `ins.rand(ins.Shape([3,4]))` still works (idempotent)
