---
name: fix-julia-binding-api-gotchas
description: Common Julia binding API issues — from_data column-major, getindex, Base.abs shadowing, load_backend, complex dtypes, axis reversal
source: auto-skill
extracted_at: '2026-06-04T03:57:53.684Z'
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

## 8. `from_data` auto-detects dtype from Julia element type

As of 2026-06-03, `from_data` automatically maps Julia element types to Insight DTypes:

```julia
# Auto-detected: Float64 → float64, Float32 → float32, Int32 → int32, etc.
A = Insight.from_data(reshape([1.0, 2.0, 3.0, 4.0], 2, 2))  # Float64 → F64

# Explicit override still works
A = Insight.from_data(reshape([1.0, 2.0, 3.0, 4.0], 2, 2), Insight.float32)
```

Complex types: `Complex{Float32}` → C32, `Complex{Float64}` → C64.

## 9. `to_data` handles complex dtypes

`to_data` returns `Vector{Complex{Float32}}` for C32 and `Vector{Complex{Float64}}` for C64.

```julia
# FFT output is complex
X = Insight.rfft(signal)  # C64
data = Insight.to_data(X)  # Vector{Complex{Float64}}

# Extract real part before comparing
recon = Insight.real_part(Insight.irfft(X, n))
```

## 10. `conj_fn` / `angle_fn` naming (avoid Base conflicts)

Julia's `Base.conj` and `Base.angle` are shadowed. Insight exports `conj_fn` and `angle_fn`:

```julia
# ❌ WRONG
z_conj = Insight.conj(z)  # may conflict with Base.conj

# ✅ CORRECT
z_conj = Insight.conj_fn(z)
z_angle = Insight.angle_fn(z)
```

## 11. `load_backend` returns Bool, does NOT throw

```julia
# ❌ WRONG — always returns true
function gpu_available()
    try
        Insight.load_backend("cuda")
        return true
    catch
        return false
    end
end

# ✅ CORRECT — check return value
function gpu_available()
    try
        return Insight.load_backend("cuda")
    catch
        return false
    end
end
```

## 12. `fftfreq` and `fftshift` are available

```julia
freqs = Insight.fftfreq(n, d)      # frequency bins
shifted = Insight.fftshift(fftout)  # shift zero-freq to center
```

## 13. Column-major layout: `from_data` reverses dims

Julia is column-major, Insight is row-major. `from_data` automatically reverses
dims when creating Insight Arrays from Julia data:

```julia
# Julia (2, 3) column-major → Insight (3, 2) row-major
a = reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
arr = Insight.from_data(a)
Insight.shape(arr)  # [3, 2] — reversed from Julia (2, 3)
Insight.to_data(arr) == a  # true — roundtrip preserves data
```

`shape()` returns the reversed dims (Julia convention). `to_data()` reshapes
with the reversed dims, so the roundtrip is correct.

### Axis reversal: `_julia_axis(arr, k) = ndim - k`

All functions that take axis parameters use `_julia_axis` to convert Julia's
1-based axis to Insight's 0-based axis (reversed due to dim reversal):

```julia
# Julia axis 1 = Insight axis (ndim-1)
# Julia axis 2 = Insight axis (ndim-2)
# axis <= 0 means "all axes" — pass through unchanged

Insight.sum(arr, axis=1)  # sums along Julia's first dimension
Insight.sum(arr, axis=2)  # sums along Julia's second dimension
Insight.fft(x, axis=1)    # FFT along Julia's first dimension
```

### When adding new axis-taking functions

Always use `_julia_axis(x, axis)` in the Julia binding:

```julia
function my_func(x::InsightArray; axis::Int=-1)::InsightArray
    ptr = ccall((:insight_jl_my_func, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(_julia_axis(x, axis)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end
```

### 2D data with `permutedims` for Insight operations

For operations that need correct 2D semantics (like CFAR), use `permutedims`
before `from_data` and after `to_data`:

```julia
# Julia (N_PULSES, N) → permutedims → (N, N_PULSES) → from_data → Insight (N_PULSES, N)
energy_ins = Insight.from_data(Base.permutedims(energy))
det = Insight.signal.ca_cfar(energy_ins, ...)
det_jl = Base.permutedims(Base.reshape(Insight.to_data(det), N, N_PULSES))
```

### 1D noise arrays avoid layout issues

For radar/signal demos, generate 1D noise arrays and use linear indexing:

```julia
Insight.seed(42)
noise_flat = Insight.to_data(Insight.randn(Int64[N_PULSES * N], Insight.float64) * sigma)
# Access: noise_flat[(p-1)*N+1 : p*N] for pulse p
```

## 14. Julia demo pattern

```julia
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "bindings", "julia"))
using Insight

# No init() needed
# Use Insight.item(arr, 0) for scalar extraction
# Use Insight.to_data(arr) for Julia array conversion
# Use Base.abs() for number operations
```
