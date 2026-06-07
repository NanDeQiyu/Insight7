---
name: fix-julia-nullptr-check
description: Julia C API bindings must check for nullptr return — silent failure otherwise
source: auto-skill
extracted_at: '2026-06-07T01:46:20.559Z'
---

# Julia C API nullptr Check

## Problem

The C API wrappers in `insight_julia_capi.cpp` catch exceptions and return
`nullptr` on error. But the Julia bindings in `Insight.jl` don't check for
`C_NULL` — they just wrap the nullptr in an `InsightArray` and return it.

This causes silent failures: the function appears to succeed but returns a
broken array that segfaults or produces garbage on use.

**Symptom**: No error message, just wrong results or segfaults later.

## Pattern

```cpp
// C API — catches exception, returns nullptr
Array *insight_jl_reshape(const Array *x, const int64_t *dims, int32_t ndim) {
  try {
    Shape shape(std::vector<int64_t>(dims, dims + ndim));
    return new Array(x->reshape(shape));
  } catch (...) {
    return nullptr;
  }
}
```

```julia
# ❌ WRONG — no nullptr check
function reshape(x::InsightArray, dims::Vector{Int64})::InsightArray
    ptr = ccall((:insight_jl_reshape, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Int64}, Int32),
                x, dims, Int32(length(dims)))
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

# ✅ CORRECT — check for nullptr
function reshape(x::InsightArray, dims::Vector{Int64})::InsightArray
    ptr = ccall((:insight_jl_reshape, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Int64}, Int32),
                x, dims, Int32(length(dims)))
    if ptr == C_NULL
        error("reshape failed: invalid dimensions")
    end
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end
```

## Testing

```julia
# Error case should throw, not silently fail
try
    Insight.reshape(a, Int64[-1, -1])
    check("reshape_err_multi_neg1", false)  # should not reach here
catch
    check("reshape_err_multi_neg1", true)
end
```

## Files

- `bindings/julia/insight_julia_capi.cpp` — C API wrappers (return nullptr on error)
- `bindings/julia/src/Insight.jl` — Julia bindings (must check C_NULL)
- `bindings/julia/Insight.jl` — old copy (keep in sync)
