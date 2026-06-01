---
name: add-julia-signal-binding
description: Pattern for adding Julia signal function bindings via C ABI wrappers
source: auto-skill
extracted_at: '2026-05-31T03:00:00.000Z'
---

# Adding Julia Signal Function Bindings

## Architecture

Julia bindings use C ABI wrappers (ccall) — each function needs:
1. A C wrapper function in `bindings/julia/insight_julia_capi.cpp`
2. A Julia wrapper function in `bindings/julia/Insight.jl`
3. An export entry in the `export` list
4. A re-export in the `module signal` section

## Step 1: C ABI Wrapper (insight_julia_capi.cpp)

```cpp
// Simple array → array function
Array *insight_jl_fm_demod(const Array *x, int32_t axis) {
  return new Array(signal::fm_demod(*x, axis));
}

// Scalar return (use double directly, NOT Array*)
double insight_jl_cfar_alpha(double pfa, int32_t N) {
  return signal::cfar_alpha(pfa, N);
}

// Array with vector parameters
Array *insight_jl_firwin2(int64_t numtaps, const double *freq, int32_t freq_len,
                          const double *gain, int32_t gain_len,
                          const char *window) {
  std::vector<double> fv(freq, freq + freq_len);
  std::vector<double> gv(gain, gain + gain_len);
  return new Array(signal::firwin2(numtaps, fv, gv, 0, window));
}

// Function returning index array (fill pre-allocated buffer)
int64_t insight_jl_argrelmax(const Array *data, int32_t axis, int32_t order,
                             int64_t *out_indices) {
  auto result = signal::argrelmax(*data, axis, order, "clip");
  if (result.empty()) return 0;
  int64_t n = result[0].numel();
  const int64_t *src = result[0].data<int64_t>();
  for (int64_t i = 0; i < n; ++i) out_indices[i] = src[i];
  return n;
}
```

**Key rules:**
- Array functions return `Array *` (heap-allocated, Julia will free via `_free`)
- Scalar functions return `double`/`int64_t` directly (NOT wrapped in Array)
- Vector params: pass `(const double *data, int32_t len)` pairs
- String params: use `const char *` (Julia Cstring auto-converts)
- Bool params: use `int32_t` (0/1)
- `std::function` params: NOT bindable from Julia (skip cwt)

## Step 2: Julia Wrapper (Insight.jl)

```julia
function fm_demod(x::InsightArray, axis::Int = -1)::InsightArray
    ptr = ccall((:insight_jl_fm_demod, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(axis))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function cfar_alpha(pfa::Float64, N::Int)::Float64
    return ccall((:insight_jl_cfar_alpha, LIB_INSIGHT), Float64,
                 (Float64, Int32), pfa, Int32(N))
end

function argrelmax(data::InsightArray, axis::Int = 0, order::Int = 1)::Vector{Int64}
    out = Vector{Int64}(undef, numel(data))
    n = ccall((:insight_jl_argrelmax, LIB_INSIGHT), Int64,
              (Ptr{Cvoid}, Int32, Int32, Ptr{Int64}),
              data, Int32(axis), Int32(order), out)
    return out[1:n]
end
```

**Key rules:**
- Always `finalizer(_free, arr)` for returned InsightArrays
- `Int32()` for C int32_t parameters
- `Int64()` for C int64_t parameters
- Pre-allocate output buffers for index-returning functions

## Step 3: Export List

Add to the `export` block at the top of `Insight.jl`:
```julia
export fm_demod, argrelmax, argrelmin, cfar_alpha, ...
```

## Step 4: Signal Submodule Re-export

Add to the `module signal` section at the bottom:
```julia
const fm_demod = Insight.fm_demod
const argrelmax = Insight.argrelmax
```

## Binding Struct-Returning Functions (SpectralResult, SpectrogramResult, etc.)

Functions like `csd`, `coherence`, `spectrogram`, `stft` return C++ structs. Julia can't receive these directly via ccall. Use **decomposed output pointers**:

```cpp
// C API wrapper — decompose struct into output pointers
void insight_jl_csd(const Array *x, const Array *y, double fs,
                    const char *window, int64_t nperseg, int64_t noverlap,
                    int64_t nfft, Array **out_f, Array **out_Pxx) {
  auto result = signal::csd(*x, *y, fs, window, nperseg, noverlap, nfft);
  *out_f = new Array(result.f);
  *out_Pxx = new Array(result.Pxx);
}
```

```julia
# Julia wrapper — use Ref{} for output pointers, return NamedTuple
function csd(x::InsightArray, y::InsightArray; fs::Float64 = 1.0,
             window::String = "hann", nperseg::Int = 256, ...)
    f_ref = Ref{Ptr{Cvoid}}(C_NULL)
    Pxx_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_csd, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Ptr{Cvoid}, Float64, Cstring, Int64, Int64, Int64,
           Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
          x, y, fs, window, Int64(nperseg), Int64(noverlap), Int64(nfft),
          f_ref, Pxx_ref)
    f_arr = InsightArray(f_ref[]); finalizer(_free, f_arr)
    Pxx_arr = InsightArray(Pxx_ref[]); finalizer(_free, Pxx_arr)
    return (f=f_arr, Pxx=Pxx_arr)
end
```

Same pattern for `pair<Array, Array>` returns (e.g., `firfilter_zi_state`):
```cpp
void insight_jl_firfilter_zi_state(const Array *b, const Array *x,
                                   const Array *zi, int32_t axis,
                                   Array **out_y, Array **out_zf) {
  auto result = signal::firfilter_zi_state(*b, *x, *zi, axis);
  *out_y = new Array(result.first);
  *out_zf = new Array(result.second);
}
```

For `pair<double, double>` returns (e.g., `vectorstrength`):
```cpp
void insight_jl_vectorstrength(const Array *events, double period,
                               double *out_strength, double *out_phase) {
  auto result = signal::vectorstrength(*events, period);
  *out_strength = result.first;
  *out_phase = result.second;
}
```

## Common Pitfalls

1. **`_dtype_code` must be defined**: Add a helper that maps Julia DataType → C enum values
2. **`to_f64` doesn't exist**: Use `Float64` return type for scalar functions instead of wrapping in Array
3. **`ccall` needs compile-time constants**: Function names must be string literals, not variables
4. **`size()` returns Tuple**: Use `collect(Int64, size(data))` for ccall pointer args
5. **C API duplicates**: Check for existing wrappers before adding new ones (e.g., `insight_jl_cosine_win` already existed)
6. **Function signature must match C++**: If C++ function has `const std::string &window` parameter, the C API wrapper must include `const char *window` and Julia must pass `Cstring`
7. **`Ref{Ptr{Cvoid}}` for output pointers**: Always initialize with `C_NULL`, dereference with `ref[]`

## Binding C++ Classes (KalmanFilter Pattern)

For stateful C++ classes that Julia can't bind directly via ccall, create opaque pointer wrappers:

### C ABI Layer (insight_julia_capi.cpp)

```cpp
// Constructor — return opaque void*
void *insight_jl_kalman_filter_new(int32_t dim_x, int32_t dim_z, int32_t dim_u,
                                   int32_t points, int32_t dtype) {
  DType dt = static_cast<DType>(dtype);
  auto *kf = new signal::KalmanFilter(dim_x, dim_z, dim_u, points, dt);
  return static_cast<void *>(kf);
}

// Destructor
void insight_jl_kalman_filter_free(void *kf) {
  delete static_cast<signal::KalmanFilter *>(kf);
}

// Methods
void insight_jl_kalman_filter_predict(void *kf) {
  static_cast<signal::KalmanFilter *>(kf)->predict();
}
void insight_jl_kalman_filter_update(void *kf, const Array *z) {
  static_cast<signal::KalmanFilter *>(kf)->update(*z);
}

// Getters — return new heap-allocated Array (shared ref)
Array *insight_jl_kf_get_x(void *kf) {
  return new Array(static_cast<signal::KalmanFilter *>(kf)->x);
}

// Setters — accept const Array*
void insight_jl_kf_set_x(void *kf, const Array *v) {
  static_cast<signal::KalmanFilter *>(kf)->x = *v;
}

// Read-only integer properties
int32_t insight_jl_kf_get_dim_x(void *kf) {
  return static_cast<signal::KalmanFilter *>(kf)->dim_x;
}
```

### Julia Wrapper (modules/signal/estimation.jl)

```julia
mutable struct KalmanFilter
    ptr::Ptr{Cvoid}
    function KalmanFilter(ptr::Ptr{Cvoid})
        kf = new(ptr)
        finalizer(kf) do k
            if k.ptr != C_NULL
                ccall((:insight_jl_kalman_filter_free, LIB_INSIGHT),
                      Cvoid, (Ptr{Cvoid},), k.ptr)
                k.ptr = C_NULL
            end
        end
        return kf
    end
end

# Keyword constructor
function KalmanFilter(dim_x::Int, dim_z::Int; dim_u::Int=0, points::Int=1,
                      dtype=DType(float64))::KalmanFilter
    dt_val = dtype isa DType ? Int32(dtype.val) : Int32(dtype)
    ptr = ccall((:insight_jl_kalman_filter_new, LIB_INSIGHT), Ptr{Cvoid},
                (Int32, Int32, Int32, Int32, Int32),
                Int32(dim_x), Int32(dim_z), Int32(dim_u), Int32(points), dt_val)
    return KalmanFilter(ptr)
end

# Property accessors via getproperty/setproperty! overloads
function Base.getproperty(kf::KalmanFilter, sym::Symbol)
    if sym === :x
        ptr = ccall((:insight_jl_kf_get_x, LIB_INSIGHT), Ptr{Cvoid},
                    (Ptr{Cvoid},), kf.ptr)
        a = InsightArray(ptr); finalizer(_free, a); return a
    elseif sym === :dim_x
        return Int(ccall((:insight_jl_kf_get_dim_x, LIB_INSIGHT), Int32,
                         (Ptr{Cvoid},), kf.ptr))
    elseif sym === :ptr
        return getfield(kf, :ptr)
    else
        error("KalmanFilter has no property :$sym")
    end
end

function Base.setproperty!(kf::KalmanFilter, sym::Symbol, v)
    arr = v isa InsightArray ? v : error("expected InsightArray")
    if sym === :x
        ccall((:insight_jl_kf_set_x, LIB_INSIGHT), Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}), kf.ptr, arr.ptr)
    else
        error("cannot set :$sym on KalmanFilter")
    end
    return v
end
```

### Key Rules for Class Binding

- **Constructor**: Returns `void*` via `new T(...)`, Julia wraps in mutable struct with finalizer
- **Destructor**: Takes `void*`, calls `delete static_cast<T*>(ptr)`
- **Getters**: Return `new Array(member)` (heap copy, Julia frees via `_free`)
- **Setters**: Accept `const Array*`, assign `kf->member = *v`
- **Methods**: Cast `void*` to `T*`, call method
- **Read-only ints**: Return `int32_t` directly
- **Re-export**: Add `const KalmanFilter = Insight.KalmanFilter` to `module signal`
