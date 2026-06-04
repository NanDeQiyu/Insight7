---
name: julia-column-major-layout
description: Handle Julia column-major ↔ Insight row-major layout: from_data dim reversal, axis conversion, to_data reshape
source: auto-skill
extracted_at: '2026-06-04T10:52:54.680Z'
---

# Julia Column-Major Layout Handling

## Problem

Julia uses column-major (Fortran) order. Insight uses row-major (C) order.
When `from_data(julia_2d_array)` is called, the data is copied as-is but the
shape interpretation differs:

```
Julia pc[128, 1000] column-major:
  memory: [col0_row0, col0_row1, ..., col0_row127, col1_row0, ...]

Insight (128, 1000) row-major reads as:
  row0 = [col0_row0, col0_row1, ..., col0_row127]  ← WRONG! This is column 0
```

Without dim reversal, 2D FFT/CFAR/conv operations process wrong dimensions.

## Solution: Reverse dims at boundary

### C++ layer (`insight_julia_capi.cpp`)

**`from_data`** — reverse dims when creating Insight Array:
```cpp
Array *insight_jl_from_data(const void *data, const int64_t *dims,
                            int32_t ndim, int32_t dtype, int32_t device_type) {
    std::vector<int64_t> rdims(dims, dims + ndim);
    std::reverse(rdims.begin(), rdims.end());  // Julia (128,1000) → Insight (1000,128)
    Shape shape(rdims);
    // ... memcpy data as-is (memory layout matches) ...
}
```

**`shape_reversed`** — return reversed dims for Julia:
```cpp
void insight_jl_shape_reversed(const Array *arr, int64_t *out, int32_t max_ndim) {
    int nd = arr->shape().ndim();
    for (int i = 0; i < nd && i < max_ndim; ++i)
        out[i] = arr->shape().dims()[nd - 1 - i];
}
```

**`zeros/ones/full/randn/reshape`** — NO reversal (these are Insight-native):
```cpp
Array *insight_jl_zeros(const int64_t *dims, int32_t ndim, ...) {
    Shape shape(std::vector<int64_t>(dims, dims + ndim));  // NO reversal
    return new Array(zeros(shape, dt, place));
}
```

**Key distinction**: `from_data` reverses (Julia→Insight boundary).
`zeros/ones/full/randn/reshape` do NOT reverse (Insight-internal operations).

### Julia binding layer (`Insight.jl`)

**`shape()`** — use reversed shape:
```julia
function shape(arr::InsightArray)::Vector{Int64}
    n = ndim(arr)
    dims = Vector{Int64}(undef, n)
    ccall((:insight_jl_shape_reversed, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Ptr{Int64}, Int32), arr, dims, Int32(n))
    return dims
end
```

**`to_array`** — reshape with Julia convention (uses `shape()` which is reversed):
```julia
function to_array(arr::InsightArray)
    # ... ccall to copy data ...
    return Base.reshape(dst, Tuple(shape(arr)))  # shape() returns Julia dims
end
```

### Axis conversion

Julia axes are 1-based. Insight axes are 0-based. With dim reversal:
- Julia axis 1 = Insight axis (ndim - 1)
- Julia axis 2 = Insight axis (ndim - 2)
- axis ≤ 0 means "all axes" — pass through unchanged

```julia
# Helper in Insight.jl
_julia_axis(arr::InsightArray, axis::Int) = axis <= 0 ? axis : ndim(arr) - axis
```

Apply to ALL functions with axis parameters:
```julia
function sum(x::InsightArray; axis::Union{Int,Nothing}=nothing, ...)
    ax = axis !== nothing ? Int32(_julia_axis(x, axis)) : Int32(0)
    # ... ccall with ax ...
end

function fft(x::InsightArray; n=nothing, axis=nothing)
    if axis !== nothing
        ax = Int32(_julia_axis(x, axis))
        # ... ccall with ax ...
    end
end
```

**Functions needing axis conversion**: sum, mean, min_fn, max_fn, cummax, cummin,
cumsum, cumprod, fft, ifft, rfft, irfft, fftshift, ifftshift, convolve,
fftconvolve, unwrap, topk, and all signal functions with axis parameter.

### Test updates

Tests using `axis=0` (old 0-based) need updating to `axis=1` (Julia 1-based):
```julia
# ❌ OLD — axis=0 passes 0 to Insight (which is correct for row-major)
s = Insight.sum(b, axis=0)

# ✅ NEW — axis=1 becomes Insight axis (ndim-1) = correct for reversed dims
s = Insight.sum(b, axis=1)
```

### Data roundtrip verification

```julia
a = reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
arr = Insight.from_data(a)
@test Insight.shape(arr) == [2, 3]  # Julia convention
b = Insight.to_data(arr)
@test a == b  # Exact roundtrip
```

**Why it works**: `from_data` reverses (2,3)→(3,2) for Insight. Data is memcpy'd
as-is. `to_data` copies back and reshapes with reversed shape (3,2)→(2,3) for Julia.
The memory layout is identical throughout.

### Radar demo pattern

For 2D signal processing (radar demo), use 1D noise to avoid layout issues:
```julia
Insight.seed(42)
noise_r_flat = Insight.to_data(Insight.randn(Int64[N_PULSES * N], Insight.float64) * sigma)
noise_i_flat = Insight.to_data(Insight.randn(Int64[N_PULSES * N], Insight.float64) * sigma)
# Access: noise_r_flat[(p-1)*N+1 : p*N] for pulse p
```

For CFAR on 2D energy array — pass directly (reversed dims handle it):
```julia
energy_ins = Insight.from_data(energy)  # Julia (128,1000) → Insight (1000,128)
threshold, detections = Insight.signal.ca_cfar(energy_ins, Int32[2,2], Int32[4,4], 1e-5)
det_jl = reshape(Insight.to_data(detections) .!= 0, N_PULSES, N)
```

**Why:** The CFAR operates on the reversed Insight array, but the guard/reference
cells are symmetric, so the results are correct. `to_data` + reshape restores
Julia convention.
