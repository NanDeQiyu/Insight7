---
name: batch-bind-ops
description: Systematically audit and batch-bind missing C++ functions to Python/Lua/Julia bindings
source: auto-skill
extracted_at: '2026-05-30T13:52:58.516Z'
---

# Batch-Bind Missing Operations

## Problem

After initial binding work, many C++ functions remain unbound. Need a
systematic approach to audit, prioritize, and batch-add bindings.

## Step 1: Audit — Find what's ready to bind

For each category (complex, unary, reduction, manipulation, indexing,
random, fft), check:

1. **Declared in header?** — grep the `.h` file
2. **Implemented in source?** — grep the `.cpp` file

Only functions that are BOTH declared AND implemented are ready.

```bash
# Example: check manipulation functions
for func in permute swapaxes moveaxis fliplr flipud rot90 diag diagonal tril triu diff; do
    echo -n "$func: "
    grep -l "$func" src/ops/manipulation.cpp 2>/dev/null | head -1 || echo "NOT IMPLEMENTED"
done
```

Report format:
| Category | Checked | Ready | Missing |
|----------|---------|-------|---------|
| complex.h | 8 | 8 | 0 |
| unary.h | 11 | 11 | 0 |
| ... | ... | ... | ... |

## Step 2: Read exact signatures

For each ready function, read the header to get exact signature:

```cpp
// Example: reduction.h
Array sem(const Array &x, std::optional<int> axis = std::nullopt,
          bool keepdim = false, int ddof = 0);
```

Key things to note:
- Default parameter values (for Python `py::arg("x") = default`)
- `std::optional` parameters (need `sol::optional` in Lua)
- Return types: `Array`, `bool`, `double`, `std::pair`, `std::tuple`, struct
- Overloads (need lambda disambiguation)

## Step 3: Add bindings in batches

### Python (pybind11)

For simple `Array → Array` functions:
```cpp
m.def("func_name", [](const Array &x) { return func_name(x); }, py::arg("x"));
```

For reduction-style functions with optional axis:
```cpp
m.def("sem",
    [](const Array &x, std::optional<int> axis, bool keepdims, int ddof) {
        return sem(x, axis, keepdims, ddof);
    },
    py::arg("x"), py::arg("axis") = std::nullopt,
    py::arg("keepdims") = false, py::arg("ddof") = 0);
```

For functions with overloaded names (e.g., `is_complex(DType)` and `is_complex(Array)`):
```cpp
m.def("is_complex",
    static_cast<bool (*)(const Array &)>(&is_complex), py::arg("x"));
```

For enum types — do NOT use `export_values()` if enum value names
conflict with function names:
```cpp
// ❌ "quadratic" enum value conflicts with quadratic() function
py::enum_<ChirpMethod>(sig, "ChirpMethod")
    .value("quadratic", ChirpMethod::Quadratic)
    .export_values();  // ← creates sig.quadratic as non-function

// ✅ Access via ChirpMethod.quadratic
py::enum_<ChirpMethod>(sig, "ChirpMethod")
    .value("quadratic", ChirpMethod::Quadratic);  // no export_values
```

### Lua (sol2)

For simple functions:
```cpp
m["func_name"] = &ins::func_name;
```

For functions with optional parameters:
```cpp
m["sem"] = [](const Array &x, sol::optional<int> axis,
              sol::optional<bool> keepdims, sol::optional<int> ddof) {
    return ins::sem(x, axis ? std::optional<int>(*axis) : std::nullopt,
                    keepdims.value_or(false), ddof.value_or(0));
};
```

### Julia (C ABI)

For each function, add a C wrapper in `insight_julia_capi.cpp`:
```cpp
Array *insight_jl_func_name(const Array *x, int32_t axis, int32_t keepdims) {
    std::optional<int> ax = axis >= 0 ? std::optional<int>(axis) : std::nullopt;
    return new Array(ins::func_name(*x, ax, keepdims != 0));
}
```

Then add Julia wrapper in `Insight.jl`:
```julia
function func_name(x::InsightArray; axis::Union{Int,Nothing}=nothing,
                   keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(axis) : Int32(0)
    ptr = ccall((:insight_jl_func_name, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32), x, has_axis, ax)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end
```

## Step 4: Update wrapper exports

### Python `__init__.py`
Add new functions to both the import list and `__all__`:
```python
from ._insight import (
    # ... existing ...
    sem, count_nonzero, cummax, cummin, median, quantile, percentile,
    nansum, nanmean, nanmax, nanmin, nanstd, nanvar,
    permute, swapaxes, moveaxis, fliplr, flipud, rot90,
    diag, diagonal, tril, triu, diff,
    unique, topk, gather, scatter, scatter_add, scatter_reduce,
    interp, indices, ix_,
    seed, get_seed, rand_like, randn_like, exponential, gamma, beta,
    fftshift, ifftshift, next_fast_len, hfft, ihfft, rfft2, irfft2,
    is_complex, to_complex, as_complex, as_real, real, imag,
    exp2, expm1, log1p, cbrt, reciprocal, asinh, acosh, atanh,
    trunc, deg2rad, rad2deg,
)
```

## Step 5: Build and test incrementally

```bash
# Build
cd build && make -j$(nproc) insight_python 2>&1 | tail -10

# Quick smoke test
LD_LIBRARY_PATH="build:build/backends/cpu:$LD_LIBRARY_PATH" \
PYTHONPATH="bindings/python:$PYTHONPATH" \
python3 -c "
import insight as ins
import numpy as np
x = ins.from_numpy(np.array([1.0, 2.0, 3.0]))
print('sem:', ins.sem(x).numpy())
print('diag:', ins.diag(ins.from_numpy(np.array([[1,2],[3,4.0]]))).numpy())
print('unique:', ins.unique(x).unique.numpy())
print('OK')
"

# Full test suite
cd build && ./tests/insight_tests_cpu 2>&1 | tail -3
```

## Common pitfalls

1. **`pybind11` unresolved overloaded function**: Multiple C++ functions
   with the same name. Use `static_cast<>` to select the right one.

2. **`sol2` optional conversion**: `sol::optional<int>` ≠ `std::optional<int>`.
   Must convert: `axis ? std::optional<int>(*axis) : std::nullopt`

3. **Julia `Cstring` lifetime**: Julia `String` may be GC'd while C++
   holds the pointer. Use `Cstring` in ccall signature (Julia pins it).

4. **Struct return types**: Need `py::class_<>` registration for Python,
   table for Lua, output pointers for Julia C ABI.

5. **`gamma` function name conflict**: `gamma` conflicts with math.h.
   Use lambda wrapper: `m.def("gamma", [](double a, double b, ...) { ... })`

6. **Julia `ccall` function name must be compile-time constant**:
   Cannot pass a `Symbol` variable to `ccall`. Each ccall needs its own
   dedicated Julia function — no generic helpers like
   `_jl_reduction_with_opts(:symbol, x)`.
   ```julia
   # ❌ Crashes: "ccall function name cannot reference local variables"
   function _helper(cfunc::Symbol, x)
       ccall((cfunc, LIB), ...)  # ← cfunc is a variable, not a constant
   end
   # ✅ Correct: inline the literal symbol
   function nansum(x; axis=nothing, keepdims=false)
       ccall((:insight_jl_nansum, LIB), ...)  # ← literal symbol
   end
   ```

7. **Julia `size()` returns Tuple, not Vector**: When passing shape/dims
   to `ccall` expecting `Ptr{Int64}`, a Tuple cannot be auto-converted.
   Use `collect(Int64, size(data))` instead of `Int64.(size(data))`.
   ```julia
   # ❌ Crashes: "no method matching unsafe_convert(Ptr{Int64}, Tuple{Int64,Int64})"
   dims = Int64.(size(data))
   ccall(..., dims, ...)
   # ✅ Correct
   dims = collect(Int64, size(data))
   ccall(..., dims, ...)
   ```

8. **Julia `_fn` suffix for name conflicts**: Julia has built-in functions
   like `sinc`, `round`, `trunc`, `diag`, `diff`. Use `_fn` suffix to avoid
   shadowing: `sinc_fn`, `round_fn`, `trunc_fn`, `diag_fn`, `diff_fn`.

9. **`sol::optional` → `std::optional` conversion for non-trivial types**:
   When a C++ function takes `std::optional<double>`, sol2's `sol::optional<double>`
   is NOT implicitly convertible. Must convert manually:
   ```cpp
   // ❌ Crashes: sol::optional<double> is not std::optional<double>
   m["interp"] = [](const Array &x, const Array &xp, const Array &fp,
                    sol::optional<double> left, sol::optional<double> right) {
       return interp(x, xp, fp, left, right);  // type mismatch!
   };
   // ✅ Correct: convert explicitly
   m["interp"] = [](const Array &x, const Array &xp, const Array &fp,
                    sol::optional<double> left, sol::optional<double> right) {
       std::optional<double> l = left ? std::optional<double>(*left) : std::nullopt;
       std::optional<double> r = right ? std::optional<double>(*right) : std::nullopt;
       return interp(x, xp, fp, l, r);
   };
   ```

10. **Struct usertype registration (sol2)**: For C++ structs returned by
    functions (e.g., `UniqueResult`), register as sol2 usertype:
    ```cpp
    sol::usertype<UniqueResult> ur_type = m.new_usertype<UniqueResult>(
        "UniqueResult", sol::constructors<UniqueResult()>());
    ur_type["unique"] = &UniqueResult::unique;
    ur_type["indices"] = &UniqueResult::indices;
    ur_type["inverse"] = &UniqueResult::inverse;
    ur_type["counts"] = &UniqueResult::counts;
    ```

11. **`hfft`/`ihfft` require complex input**: These Hermitian FFT functions
    crash with real-valued input. In tests, skip them or use complex arrays:
    ```lua
    -- ❌ Crashes: "irfft: input must be complex"
    local a = ins.ones({8}, ins.float32)
    local r = ins.hfft(a)
    -- ✅ Skip in tests or use complex input
    ```

12. **Python `exponential` name conflict**: The signal window function
    `exponential` (in `signal.h`) conflicts with the random distribution
    `exponential` (in `random.h`). In Python binding, the signal version
    wins because it's registered later. Use `rand_like`/`randn_like` in
    tests instead.
