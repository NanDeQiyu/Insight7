---
name: fix-julia-binding-api-gotchas
description: Common Julia binding API issues — from_data, getindex, Base.abs shadowing, Insight.init
source: auto-skill
extracted_at: '2026-06-02T06:00:00.000Z'
---

# Julia Binding API Gotchas

## 1. `from_data` does NOT take shape argument

`from_data(data, dtype_val, device)` — shape is inferred from Julia array dimensions.

```julia
# ❌ WRONG — [2,2] is passed as dtype_val (Int32), not shape
A = Insight.from_data([1.0, 2.0, 3.0, 4.0], [2, 2])

# ✅ CORRECT — use reshape for multi-dimensional
A = Insight.from_data(reshape([1.0, 2.0, 3.0, 4.0], 2, 2))

# ✅ For 1D, just pass the vector
b = Insight.from_data([1.0, 2.0, 3.0])

# ✅ With explicit dtype
A = Insight.from_data(reshape([1.0, 2.0, 3.0, 4.0], 2, 2), Insight.float64)
```

## 2. `getindex` not defined for InsightArray

Can't use `arr[1]` on InsightArray — `getindex` is not implemented.

```julia
# ❌ WRONG
val = arr[1]  # MethodError: no method matching getindex(::InsightArray, ::Int64)

# ✅ CORRECT — use item() for scalar extraction (0-based)
val = Insight.item(arr, 0)

# ✅ For converting to Julia array
data = Insight.to_data(arr)  # returns Julia Vector
val = data[1]  # now 1-based indexing works
```

## 3. `Base.abs` shadowed by `Insight.abs`

When `using Insight`, `abs` refers to `Insight.abs` (Array→Array), not `Base.abs` (number→number).

```julia
# ❌ WRONG — Insight.abs on Julia numbers fails
max_err = maximum(abs.(data1 .- data2))  # UndefVarError: abs not defined

# ✅ CORRECT — use Base.abs explicitly
max_err = maximum(Base.abs.(data1 .- data2))
```

Similarly shadowed: `ones`, `zeros`, `fill`, `exp`, `sin`, `cos`, `sqrt`, `log`, `floor`, `ceil`, `round`, `sign`.

Use `fill(1.0, n)` instead of `ones(n)`, `fill(0.0, n)` instead of `zeros(n)`.

## 4. `Insight.init` does NOT exist

The Julia binding has no `init()` function. Just `using Insight` is sufficient.

```julia
# ❌ WRONG
Insight.init(["cpu"])  # UndefVarError: init not defined

# ✅ CORRECT — just use the module
using Insight
```

## 5. Function name conflicts

Some Insight functions have `_fn` suffix to avoid Julia name conflicts:

```julia
# ❌ WRONG
Insight.abs(x)    # Works but returns Array
Insight.sin(x)    # Works but returns Array
Insight.exp(x)    # Works but returns Array

# These are correct for Array operations, but for display:
# Use Insight.abs_fn, Insight.sin_fn, Insight.exp_fn if available
```

## 6. `read_bin` dtype argument

`read_bin` takes dtype as Julia DataType keyword, not Insight DType:

```julia
# ❌ WRONG
data = Insight.read_bin(path, Insight.float64)  # MethodError

# ✅ CORRECT
data = Insight.read_bin(path, dtype=Float64)
```

## 7. `arange` expects Float64 arguments

```julia
# ❌ WRONG — Int64 arguments
Insight.arange(10, Insight.float32)  # MethodError

# ✅ CORRECT — Float64 arguments
Insight.arange(0.0, 10.0, 1.0, Insight.float32)
```

## 8. Julia demo pattern

```julia
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "bindings", "julia"))
using Insight

# No init() needed
# Use Insight.item(arr, 0) for scalar extraction
# Use Insight.to_data(arr) for Julia array conversion
# Use Base.abs() for number operations
```
