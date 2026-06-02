# bindings/julia/Insight.jl
#
# Julia bindings for the Insight7 scientific computing framework.
# Uses ccall() to call into libinsight_julia.so (C ABI wrappers).
# API style: PaddlePaddle (Insight.float32, Insight.CPUPlace(), etc.)
#
# Usage:
#   push!(LOAD_PATH, "/path/to/bindings/julia")
#   using Insight
#
#   a = Insight.zeros([2, 3], Insight.float32)
#   b = Insight.ones([2, 3], Insight.float32)
#   c = a + b

module Insight

export Array, zeros, ones, full, arange, linspace, eye,
       add, sub, mul, div, pow, matmul, dot, det, inv, solve, svd,
       fft, ifft, rand, randn, cast,
       # Convenience
       item, to_data, negative, abs, equal, greater, less,
       max, min, argmax, argmin, prod, outer, norm, trace, DType,
       # Complex
       is_complex, has_complex_shape, to_complex, as_complex, as_real,
       real_part, imag_part,
       # Additional unary
       exp2_fn, expm1_fn, log1p_fn, cbrt_fn, reciprocal_fn,
       asinh_fn, acosh_fn, atanh_fn, trunc_fn, deg2rad_fn, rad2deg_fn,
       # Additional reduction
       cummax, cummin, sem, count_nonzero, median, quantile, percentile,
       nansum, nanmean, nanmax, nanmin, nanstd, nanvar,
       # Additional manipulation
       permute, swapaxes, moveaxis, fliplr, flipud, rot90,
       diag_fn, diagonal, tril, triu, diff_fn,
       # Additional indexing
       unique_ins, topk, gather, scatter, scatter_add, scatter_reduce,
       interp, indices_fn, ix_fn,
       # Additional random
       seed, get_seed, rand_like, randn_like,
       exponential, gamma_dist, beta_dist, binomial_dist, poisson_dist,
       # Additional FFT
       fftshift, ifftshift, next_fast_len, hfft, ihfft,
       rfft2, irfft2, rfftn, irfftn,
       # Additional linalg
       lstsq, cond_fn, matrix_rank,
       # Signal
       convolve, unwrap, sinc,
       hann, hamming, blackman, kaiser, gaussian, boxcar, triang,
       bartlett, flattop, nuttall, blackmanharris, tukey, chebwin, taylor,
       get_window, sawtooth, square_wf, chirp, unit_impulse,
       gauss_spline, cubic, quadratic,
       kaiser_beta, kaiser_atten, firwin, cmplx_sort,
       fftconvolve, correlate, correlation_lags,
       hilbert, detrend, lfilter, filtfilt, decimate, resample, freq_shift,
       welch_jl, periodogram_jl,
       morlet, ricker,
       mel2hz, hz2mel, mel_frequencies, hz2bark, bark2hz,
       fm_demod, argrelmax, argrelmin, cfar_alpha,
       pulse_compression, pulse_doppler, mvdr,
       read_bin, write_bin, pack_bin, unpack_bin, read_sigmf, write_sigmf,
       cosine_win, general_hamming, parzen_win, bohman_win, barthann_win,
       exponential_win, general_gaussian_win,
       firwin2, convolve2d, correlate2d,
       hilbert2, wiener, firfilter, lfilter_zi, resample_poly,
       morlet2,
       csd, coherence, spectrogram, stft, vectorstrength,
       choose_conv_method, firfilter_zi_state,
       # Device info
       device_name, cuda_version, driver_version, compute_capability,
       device_memory, gpu_count,
       # Signal submodule
       signal

# ============================================================================
# Configuration
# ============================================================================

const LIB_INSIGHT = let
    _local = joinpath(@__DIR__, "libinsight_julia.so")
    if isfile(_local)
        _local
    else
        "libinsight_julia.so"
    end
end

# Auto-initialize CPU backend on module load
function __init__()
    ccall((:insight_jl_init_cpu, LIB_INSIGHT), Cvoid, ())
end

# DType enum mapping (matches InsightDType in c_api/dtype.h)
function _dtype_code(dt::DataType)::Int32
    dt === Bool && return Int32(1)      # INSIGHT_DTYPE_BOOL
    dt === UInt8 && return Int32(2)     # INSIGHT_DTYPE_U8
    dt === Int8 && return Int32(3)      # INSIGHT_DTYPE_I8
    dt === Int16 && return Int32(4)     # INSIGHT_DTYPE_I16
    dt === Int32 && return Int32(5)     # INSIGHT_DTYPE_I32
    dt === Int64 && return Int32(6)     # INSIGHT_DTYPE_I64
    dt === Float32 && return Int32(9)   # INSIGHT_DTYPE_F32
    dt === Float64 && return Int32(10)  # INSIGHT_DTYPE_F64
    dt === UInt16 && return Int32(15)   # INSIGHT_DTYPE_U16
    dt === UInt32 && return Int32(16)   # INSIGHT_DTYPE_U32
    dt === UInt64 && return Int32(17)   # INSIGHT_DTYPE_U64
    error("Unsupported dtype: $dt")
end

# ============================================================================
# Device information
# ============================================================================

function device_name(device_id::Int=0)::String
    buf = Vector{UInt8}(undef, 256)
    ccall((:insight_jl_device_name, LIB_INSIGHT), Cvoid,
          (Int32, Ptr{UInt8}, Csize_t), Int32(device_id), buf, 256)
    return String(buf[1:findfirst(==(0x00), buf)-1])
end

function cuda_version()::Int
    Int(ccall((:insight_jl_cuda_version, LIB_INSIGHT), Int32, ()))
end

function driver_version()::Int
    Int(ccall((:insight_jl_driver_version, LIB_INSIGHT), Int32, ()))
end

function compute_capability(device_id::Int=0)::Int
    Int(ccall((:insight_jl_compute_capability, LIB_INSIGHT), Int32,
              (Int32,), Int32(device_id)))
end

function device_memory(device_id::Int=0)
    total = Ref{Csize_t}(0)
    free = Ref{Csize_t}(0)
    ccall((:insight_jl_device_memory, LIB_INSIGHT), Cvoid,
          (Int32, Ptr{Csize_t}, Ptr{Csize_t}), Int32(device_id), total, free)
    return (total=total[], free=free[])
end

function gpu_count()::Int
    Int(ccall((:insight_jl_gpu_count, LIB_INSIGHT), Int32, ()))
end

# ============================================================================
# DType enum (mirrors InsightDType in C)
# ============================================================================

module DTypeValues
const UNKNOWN  = Int32(0)
const BOOL     = Int32(1)
const U8       = Int32(2)
const I8       = Int32(3)
const I16      = Int32(4)
const I32      = Int32(5)
const I64      = Int32(6)
const F16      = Int32(7)
const BF16     = Int32(8)
const F32      = Int32(9)
const F64      = Int32(10)
const C32      = Int32(11)
const C64      = Int32(12)
const F8_E4M3  = Int32(13)
const F8_E5M2  = Int32(14)
const U16      = Int32(15)
const U32      = Int32(16)
const U64      = Int32(17)
end

# Module-level dtype shortcuts
const bool     = DTypeValues.BOOL
const uint8    = DTypeValues.U8
const int8     = DTypeValues.I8
const int16    = DTypeValues.I16
const int32    = DTypeValues.I32
const int64    = DTypeValues.I64

# DType wrapper struct for API compatibility
struct DType
    val::Int32
end
const uint16   = DTypeValues.U16
const uint32   = DTypeValues.U32
const uint64   = DTypeValues.U64
const float16  = DTypeValues.F16
const bfloat16 = DTypeValues.BF16
const float32  = DTypeValues.F32
const float64  = DTypeValues.F64
const complex64  = DTypeValues.C32
const complex128 = DTypeValues.C64

# Place shortcuts
const CPU = Int32(0)
const GPU = Int32(1)

# ============================================================================
# Array wrapper
# ============================================================================

mutable struct InsightArray
    ptr::Ptr{Cvoid}  # Pointer to C++ Array object
end

function _free(arr::InsightArray)
    if arr.ptr != C_NULL
        ccall((:insight_jl_array_free, LIB_INSIGHT), Cvoid, (Ptr{Cvoid},),
              arr.ptr)
        arr.ptr = C_NULL
    end
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, arr::InsightArray) = arr.ptr

# ============================================================================
# Metadata
# ============================================================================

function ndim(arr::InsightArray)::Int
    Int(ccall((:insight_jl_ndim, LIB_INSIGHT), Int32, (Ptr{Cvoid},), arr))
end

function numel(arr::InsightArray)::Int
    Int(ccall((:insight_jl_numel, LIB_INSIGHT), Int64, (Ptr{Cvoid},), arr))
end

function dtype(arr::InsightArray)::Int32
    ccall((:insight_jl_dtype, LIB_INSIGHT), Int32, (Ptr{Cvoid},), arr)
end

function shape(arr::InsightArray)::Vector{Int64}
    n = ndim(arr)
    dims = Vector{Int64}(undef, n)
    ccall((:insight_jl_shape, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Ptr{Int64}), arr, dims)
    return dims
end

function device_type(arr::InsightArray)::Int32
    ccall((:insight_jl_device_type, LIB_INSIGHT), Int32, (Ptr{Cvoid},), arr)
end

# ============================================================================
# Array creation
# ============================================================================

"""
    zeros(dims::Vector{Int64}, dtype_val::Int32=float32, device::Int32=CPU) -> InsightArray

Create an array filled with zeros.

# Arguments
- `dims`: Shape of the array, e.g. `[2, 3]`.
- `dtype_val`: Data type (default `float32`).
- `device`: Device placement (default `CPU`).

# Returns
- `InsightArray`: Array of zeros with the given shape and dtype.

# Example
```julia
a = Insight.zeros([2, 3], Insight.float32)
Insight.numel(a)  # 6
```
"""
function zeros(dims::Vector{Int64}, dtype_val::Int32=float32,
               device::Int32=CPU)::InsightArray
    ptr = ccall((:insight_jl_zeros, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Int64}, Int32, Int32, Int32),
                dims, Int32(length(dims)), dtype_val, device)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

"""
    ones(dims::Vector{Int64}, dtype_val::Int32=float32, device::Int32=CPU) -> InsightArray

Create an array filled with ones.

# Arguments
- `dims`: Shape of the array, e.g. `[2, 3]`.
- `dtype_val`: Data type (default `float32`).
- `device`: Device placement (default `CPU`).

# Returns
- `InsightArray`: Array of ones with the given shape and dtype.
"""
function ones(dims::Vector{Int64}, dtype_val::Int32=float32,
              device::Int32=CPU)::InsightArray
    ptr = ccall((:insight_jl_ones, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Int64}, Int32, Int32, Int32),
                dims, Int32(length(dims)), dtype_val, device)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function full(dims::Vector{Int64}, fill_value::Float64,
              dtype_val::Int32=float32, device::Int32=CPU)::InsightArray
    ptr = ccall((:insight_jl_full, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Int64}, Int32, Float64, Int32, Int32),
                dims, Int32(length(dims)), fill_value, dtype_val, device)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function eye(n::Int64, m::Int64=Int64(-1), dtype_val::Int32=float32,
             device::Int32=CPU)::InsightArray
    ptr = ccall((:insight_jl_eye, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int64, Int32, Int32), n, m, dtype_val, device)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function arange(start::Float64, stop::Float64, step::Float64=1.0,
                dtype_val::Int32=int64, device::Int32=CPU)::InsightArray
    ptr = ccall((:insight_jl_arange, LIB_INSIGHT), Ptr{Cvoid},
                (Float64, Float64, Float64, Int32, Int32),
                start, stop, step, dtype_val, device)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function linspace(start::Float64, stop::Float64, num::Int64,
                  dtype_val::Int32=float32, device::Int32=CPU)::InsightArray
    ptr = ccall((:insight_jl_linspace, LIB_INSIGHT), Ptr{Cvoid},
                (Float64, Float64, Int64, Int32, Int32),
                start, stop, num, dtype_val, device)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function from_data(data::AbstractArray{T}, dtype_val::Int32=float32,
                   device::Int32=CPU) where T
    # Convert data to match the requested dtype before passing to C
    _julia_type(dt::Int32) = dt == DTypeValues.F32 ? Float32 :
                             dt == DTypeValues.F64 ? Float64 :
                             dt == DTypeValues.I32 ? Int32 :
                             dt == DTypeValues.I64 ? Int64 :
                             dt == DTypeValues.I8  ? Int8  :
                             dt == DTypeValues.U8  ? UInt8 :
                             dt == DTypeValues.I16 ? Int16 :
                             dt == DTypeValues.U16 ? UInt16 :
                             dt == DTypeValues.U32 ? UInt32 :
                             dt == DTypeValues.U64 ? UInt64 :
                             dt == DTypeValues.BOOL ? Bool : Float32
    target_type = _julia_type(dtype_val)
    converted = T === target_type ? data : convert(Array{target_type}, data)
    dims = collect(Int64, size(converted))
    ptr = ccall((:insight_jl_from_data, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Int64}, Int32, Int32, Int32),
                pointer(converted), dims, Int32(length(dims)), dtype_val, device)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function to_array(arr::InsightArray)::Array
    n = numel(arr)
    dt = dtype(arr)
    if dt == DTypeValues.F32
        dst = Vector{Float32}(undef, n)
    elseif dt == DTypeValues.F64
        dst = Vector{Float64}(undef, n)
    elseif dt == DTypeValues.I32
        dst = Vector{Int32}(undef, n)
    elseif dt == DTypeValues.I64
        dst = Vector{Int64}(undef, n)
    else
        dst = Vector{Float64}(undef, n)
    end
    ccall((:insight_jl_to_data, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Ptr{Cvoid}), arr, dst)
    return Base.reshape(dst, Tuple(shape(arr)))
end

# ============================================================================
# Arithmetic operators
# ============================================================================

macro jl_binary(name, cfunc)
    quote
        function $(esc(name))(a::InsightArray, b::InsightArray)::InsightArray
            ptr = ccall(($cfunc, LIB_INSIGHT), Ptr{Cvoid},
                        (Ptr{Cvoid}, Ptr{Cvoid}), a, b)
            arr = InsightArray(ptr)
            finalizer(_free, arr)
            return arr
        end
    end
end

macro jl_unary(name, cfunc)
    quote
        function $(esc(name))(x::InsightArray)::InsightArray
            ptr = ccall(($cfunc, LIB_INSIGHT), Ptr{Cvoid},
                        (Ptr{Cvoid},), x)
            arr = InsightArray(ptr)
            finalizer(_free, arr)
            return arr
        end
    end
end

@jl_binary add  :insight_jl_add
@jl_binary sub  :insight_jl_sub
@jl_binary mul  :insight_jl_mul
@jl_binary div  :insight_jl_div
@jl_binary pow  :insight_jl_pow

# Operator overloading
Base.:+(a::InsightArray, b::InsightArray) = add(a, b)
Base.:-(a::InsightArray, b::InsightArray) = sub(a, b)
Base.:*(a::InsightArray, b::InsightArray) = mul(a, b)
Base.:/(a::InsightArray, b::InsightArray) = div(a, b)
Base.:-(x::InsightArray) = begin
    ptr = ccall((:insight_jl_negative, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

# ============================================================================
# Unary math
# ============================================================================

@jl_unary abs_fn  :insight_jl_abs
@jl_unary sqrt_fn :insight_jl_sqrt
@jl_unary exp_fn  :insight_jl_exp
@jl_unary log_fn  :insight_jl_log
@jl_unary sin_fn  :insight_jl_sin
@jl_unary cos_fn  :insight_jl_cos
@jl_unary tan_fn  :insight_jl_tan
@jl_unary floor_fn :insight_jl_floor
@jl_unary ceil_fn  :insight_jl_ceil
@jl_unary round_fn :insight_jl_round

# ============================================================================
# Additional Unary (Phase D)
# ============================================================================

@jl_unary exp2_fn      :insight_jl_exp2
@jl_unary expm1_fn     :insight_jl_expm1
@jl_unary log1p_fn     :insight_jl_log1p
@jl_unary cbrt_fn      :insight_jl_cbrt
@jl_unary reciprocal_fn :insight_jl_reciprocal
@jl_unary asinh_fn     :insight_jl_asinh
@jl_unary acosh_fn     :insight_jl_acosh
@jl_unary atanh_fn     :insight_jl_atanh
@jl_unary trunc_fn     :insight_jl_trunc_unary
@jl_unary deg2rad_fn   :insight_jl_deg2rad
@jl_unary rad2deg_fn   :insight_jl_rad2deg

# ============================================================================
# Complex
# ============================================================================

function is_complex(x::InsightArray)::Bool
    ccall((:insight_jl_is_complex, LIB_INSIGHT), Int32, (Ptr{Cvoid},), x) != 0
end

function has_complex_shape(x::InsightArray)::Bool
    ccall((:insight_jl_has_complex_shape, LIB_INSIGHT), Int32,
          (Ptr{Cvoid},), x) != 0
end

function to_complex(real::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_to_complex, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), real)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function to_complex(real::InsightArray, imag_arr::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_to_complex2, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}), real, imag_arr)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function as_complex(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_as_complex, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function as_real(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_as_real, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function real_part(z::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_real_part, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), z)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function imag_part(z::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_imag_part, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), z)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# ============================================================================
# Additional Reduction (Phase D)
# ============================================================================

function cummax(x::InsightArray, axis::Int)::InsightArray
    ptr = ccall((:insight_jl_cummax, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(axis))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function cummin(x::InsightArray, axis::Int)::InsightArray
    ptr = ccall((:insight_jl_cummin, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(axis))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function sem(x::InsightArray; axis::Union{Int,Nothing}=nothing,
             keepdims::Bool=false, ddof::Int=0)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(axis) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_sem, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32, Int32),
                x, has_axis, ax, kd, Int32(ddof))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function count_nonzero(x::InsightArray; axis::Union{Int,Nothing}=nothing,
                       keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(axis) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_count_nonzero, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function median(x::InsightArray; axis::Union{Int,Nothing}=nothing,
                keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(axis) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_median, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function quantile(x::InsightArray, q::Float64;
                  axis::Union{Int,Nothing}=nothing,
                  keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(axis) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_quantile, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Float64, Int32, Int32, Int32),
                x, q, has_axis, ax, kd)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function percentile(x::InsightArray, q::Float64;
                    axis::Union{Int,Nothing}=nothing,
                    keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(axis) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_percentile, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Float64, Int32, Int32, Int32),
                x, q, has_axis, ax, kd)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function nansum(x::InsightArray; axis::Union{Int,Nothing}=nothing,
                keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(axis) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_nansum, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function nanmean(x::InsightArray; axis::Union{Int,Nothing}=nothing,
                 keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(axis) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_nanmean, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function nanmax(x::InsightArray; axis::Union{Int,Nothing}=nothing,
                keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(axis) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_nanmax, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function nanmin(x::InsightArray; axis::Union{Int,Nothing}=nothing,
                keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(axis) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_nanmin, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function nanstd(x::InsightArray; axis::Union{Int,Nothing}=nothing,
                keepdims::Bool=false, ddof::Int=0)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(axis) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_nanstd, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32, Int32),
                x, has_axis, ax, kd, Int32(ddof))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function nanvar(x::InsightArray; axis::Union{Int,Nothing}=nothing,
                keepdims::Bool=false, ddof::Int=0)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(axis) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_nanvar, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32, Int32),
                x, has_axis, ax, kd, Int32(ddof))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# ============================================================================
# Reduction
# ============================================================================

"""
    sum(x::InsightArray; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false) -> InsightArray

Sum of array elements over a given axis.

# Arguments
- `x`: Input array.
- `axis`: Axis along which to sum (default: sum over all elements).
- `keepdims`: If `true`, retains reduced axes with size 1.

# Returns
- `InsightArray`: Sum result.
"""
function sum(x::InsightArray; axis::Union{Int,Nothing}=nothing,
             keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(axis) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_sum, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

"""
    mean(x::InsightArray; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false) -> InsightArray

Mean of array elements over a given axis.

# Arguments
- `x`: Input array.
- `axis`: Axis along which to compute mean (default: all elements).
- `keepdims`: If `true`, retains reduced axes with size 1.

# Returns
- `InsightArray`: Mean result.
"""
function mean(x::InsightArray; axis::Union{Int,Nothing}=nothing,
              keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(axis) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_mean, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

# ============================================================================
# Linear Algebra
# ============================================================================

"""
    matmul(a::InsightArray, b::InsightArray) -> InsightArray

Matrix multiplication of two 2-D arrays.

# Arguments
- `a`: Left matrix (M x K).
- `b`: Right matrix (K x N).

# Returns
- `InsightArray`: Result matrix (M x N).
"""
function matmul(a::InsightArray, b::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_matmul, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}), a, b)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function det(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_det, LIB_INSIGHT), Ptr{Cvoid}, (Ptr{Cvoid},), x)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function inv(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_inv, LIB_INSIGHT), Ptr{Cvoid}, (Ptr{Cvoid},), x)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function solve(a::InsightArray, b::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_solve, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}), a, b)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function svd(x::InsightArray)
    u_ref = Ref{Ptr{Cvoid}}(C_NULL)
    s_ref = Ref{Ptr{Cvoid}}(C_NULL)
    vt_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_svd, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
          x, u_ref, s_ref, vt_ref)
    u = InsightArray(u_ref[]); finalizer(_free, u)
    s = InsightArray(s_ref[]); finalizer(_free, s)
    vt = InsightArray(vt_ref[]); finalizer(_free, vt)
    return (u, s, vt)
end

function cholesky(x::InsightArray; lower::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_cholesky, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, lower ? Int32(1) : Int32(0))
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

# ============================================================================
# FFT
# ============================================================================

"""
    fft(x::InsightArray; n::Union{Int,Nothing}=nothing) -> InsightArray

1-D discrete Fourier transform.

# Arguments
- `x`: Input array (real or complex).
- `n`: Length of the transform (default: same as input length).

# Returns
- `InsightArray`: Complex-valued DFT result.
"""
function fft(x::InsightArray; n::Union{Int,Nothing}=nothing)::InsightArray
    has_n = n !== nothing ? Int32(1) : Int32(0)
    nv = n !== nothing ? Int64(n) : Int64(-1)
    ptr = ccall((:insight_jl_fft, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int64), x, has_n, nv)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function ifft(x::InsightArray; n::Union{Int,Nothing}=nothing)::InsightArray
    has_n = n !== nothing ? Int32(1) : Int32(0)
    nv = n !== nothing ? Int64(n) : Int64(-1)
    ptr = ccall((:insight_jl_ifft, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int64), x, has_n, nv)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

# ============================================================================
# Random
# ============================================================================

function rand(dims::Vector{Int64}, dtype_val::Int32=float32,
              device::Int32=CPU)::InsightArray
    ptr = ccall((:insight_jl_rand, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Int64}, Int32, Int32, Int32),
                dims, Int32(length(dims)), dtype_val, device)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function randn(dims::Vector{Int64}, dtype_val::Int32=float32,
               device::Int32=CPU)::InsightArray
    ptr = ccall((:insight_jl_randn, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Int64}, Int32, Int32, Int32),
                dims, Int32(length(dims)), dtype_val, device)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

# ============================================================================
# Cast
# ============================================================================

function cast(x::InsightArray, dtype_val::Int32)::InsightArray
    ptr = ccall((:insight_jl_cast, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, dtype_val)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

# ============================================================================
# Manipulation
# ============================================================================

function reshape(x::InsightArray, dims::Vector{Int64})::InsightArray
    ptr = ccall((:insight_jl_reshape, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Int64}, Int32),
                x, dims, Int32(length(dims)))
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function transpose(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_transpose, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function Base.copy(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_copy, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

# ============================================================================
# Additional Manipulation (Phase D)
# ============================================================================

function permute(x::InsightArray, axes::Vector{Int32})::InsightArray
    ptr = ccall((:insight_jl_permute, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Int32}, Int32),
                x, axes, Int32(length(axes)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function swapaxes(x::InsightArray, axis1::Int, axis2::Int)::InsightArray
    ptr = ccall((:insight_jl_swapaxes, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32), x, Int32(axis1), Int32(axis2))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function moveaxis(x::InsightArray, source::Int, destination::Int)::InsightArray
    ptr = ccall((:insight_jl_moveaxis, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32), x, Int32(source),
                Int32(destination))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function fliplr(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_fliplr, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function flipud(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_flipud, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function rot90(x::InsightArray; k::Int=1,
               axes::Vector{Int32}=Int32[0, 1])::InsightArray
    ptr = ccall((:insight_jl_rot90, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Ptr{Int32}, Int32),
                x, Int32(k), axes, Int32(length(axes)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function diag_fn(x::InsightArray; k::Int=0)::InsightArray
    ptr = ccall((:insight_jl_diag, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(k))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function diagonal(x::InsightArray; offset::Int=0, axis1::Int=0,
                  axis2::Int=1)::InsightArray
    ptr = ccall((:insight_jl_diagonal, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32),
                x, Int32(offset), Int32(axis1), Int32(axis2))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function tril(x::InsightArray; k::Int=0)::InsightArray
    ptr = ccall((:insight_jl_tril, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(k))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function triu(x::InsightArray; k::Int=0)::InsightArray
    ptr = ccall((:insight_jl_triu, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(k))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function diff_fn(x::InsightArray; n::Int=1, axis::Int=-1)::InsightArray
    ptr = ccall((:insight_jl_diff, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32), x, Int32(n), Int32(axis))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# ============================================================================
# Additional Indexing (Phase D)
# ============================================================================

function unique_ins(x::InsightArray; return_indices::Bool=false,
                    return_inverse::Bool=false,
                    return_counts::Bool=false)
    u_ref = Ref{Ptr{Cvoid}}(C_NULL)
    idx_ref = Ref{Ptr{Cvoid}}(C_NULL)
    inv_ref = Ref{Ptr{Cvoid}}(C_NULL)
    cnt_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_unique, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Int32, Int32, Int32,
           Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}},
           Ptr{Ptr{Cvoid}}),
          x, return_indices ? Int32(1) : Int32(0),
          return_inverse ? Int32(1) : Int32(0),
          return_counts ? Int32(1) : Int32(0),
          u_ref, idx_ref, inv_ref, cnt_ref)
    u = InsightArray(u_ref[]); finalizer(_free, u)
    idx = InsightArray(idx_ref[]); finalizer(_free, idx)
    inv = InsightArray(inv_ref[]); finalizer(_free, inv)
    cnt = InsightArray(cnt_ref[]); finalizer(_free, cnt)
    return (unique=u, indices=idx, inverse=inv, counts=cnt)
end

function topk(x::InsightArray, k::Int; axis::Int=-1, largest::Bool=true,
              sorted::Bool=true)
    v_ref = Ref{Ptr{Cvoid}}(C_NULL)
    i_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_topk, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Int64, Int32, Int32, Int32,
           Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
          x, Int64(k), Int32(axis), largest ? Int32(1) : Int32(0),
          sorted ? Int32(1) : Int32(0), v_ref, i_ref)
    vals = InsightArray(v_ref[]); finalizer(_free, vals)
    idx = InsightArray(i_ref[]); finalizer(_free, idx)
    return (vals, idx)
end

function gather(x::InsightArray, dim::Int, index::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_gather, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Ptr{Cvoid}), x, Int32(dim), index)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function scatter(x::InsightArray, dim::Int, index::InsightArray,
                 src::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_scatter, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Ptr{Cvoid}, Ptr{Cvoid}),
                x, Int32(dim), index, src)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function scatter_add(x::InsightArray, dim::Int, index::InsightArray,
                     src::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_scatter_add, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Ptr{Cvoid}, Ptr{Cvoid}),
                x, Int32(dim), index, src)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function scatter_reduce(x::InsightArray, dim::Int, index::InsightArray,
                        src::InsightArray;
                        reduce::String="replace")::InsightArray
    ptr = ccall((:insight_jl_scatter_reduce, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Ptr{Cvoid}, Ptr{Cvoid}, Cstring),
                x, Int32(dim), index, src, reduce)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function interp(x::InsightArray, xp::InsightArray, fp::InsightArray;
                left::Union{Float64,Nothing}=nothing,
                right::Union{Float64,Nothing}=nothing)::InsightArray
    has_l = left !== nothing ? Int32(1) : Int32(0)
    lv = left !== nothing ? left : 0.0
    has_r = right !== nothing ? Int32(1) : Int32(0)
    rv = right !== nothing ? right : 0.0
    ptr = ccall((:insight_jl_interp, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
                 Int32, Float64, Int32, Float64),
                x, xp, fp, has_l, lv, has_r, rv)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function indices_fn(dims::Vector{Int64}; sparse::Bool=false)::InsightArray
    ptr = ccall((:insight_jl_indices, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Int64}, Int32, Int32),
                dims, Int32(length(dims)), sparse ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function ix_fn(arrays::Vector{InsightArray})
    n = length(arrays)
    ptrs = [a.ptr for a in arrays]
    out_ptrs = Vector{Ptr{Cvoid}}(undef, n)
    out_count = Ref{Int32}(0)
    ccall((:insight_jl_ix_, LIB_INSIGHT), Cvoid,
          (Ptr{Ptr{Cvoid}}, Int32, Ptr{Ptr{Cvoid}}, Ptr{Int32}),
          ptrs, Int32(n), out_ptrs, out_count)
    result = InsightArray[]
    for i in 1:out_count[]
        arr = InsightArray(out_ptrs[i])
        finalizer(_free, arr)
        push!(result, arr)
    end
    return result
end

# ============================================================================
# Additional Random (Phase D)
# ============================================================================

function seed(s::UInt64)
    ccall((:insight_jl_seed, LIB_INSIGHT), Cvoid, (UInt64,), s)
end
function seed(s::Integer)
    ccall((:insight_jl_seed, LIB_INSIGHT), Cvoid, (UInt64,), UInt64(s))
end

function get_seed()::UInt64
    ccall((:insight_jl_get_seed, LIB_INSIGHT), UInt64, ())
end

function rand_like(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_rand_like, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function randn_like(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_randn_like, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function exponential(scale::Float64, dims::Vector{Int64};
                     dtype_val::Int32=float32,
                     device::Int32=CPU)::InsightArray
    ptr = ccall((:insight_jl_exponential, LIB_INSIGHT), Ptr{Cvoid},
                (Float64, Ptr{Int64}, Int32, Int32, Int32),
                scale, dims, Int32(length(dims)), dtype_val, device)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function gamma_dist(shape_param::Float64, rate::Float64,
                    dims::Vector{Int64}; dtype_val::Int32=float32,
                    device::Int32=CPU)::InsightArray
    ptr = ccall((:insight_jl_gamma, LIB_INSIGHT), Ptr{Cvoid},
                (Float64, Float64, Ptr{Int64}, Int32, Int32, Int32),
                shape_param, rate, dims, Int32(length(dims)),
                dtype_val, device)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function beta_dist(a::Float64, b::Float64, dims::Vector{Int64};
                   dtype_val::Int32=float32,
                   device::Int32=CPU)::InsightArray
    ptr = ccall((:insight_jl_beta_dist, LIB_INSIGHT), Ptr{Cvoid},
                (Float64, Float64, Ptr{Int64}, Int32, Int32, Int32),
                a, b, dims, Int32(length(dims)), dtype_val, device)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function binomial_dist(n::Int64, p::Float64, dims::Vector{Int64};
                       dtype_val::Int32=int64,
                       device::Int32=CPU)::InsightArray
    ptr = ccall((:insight_jl_binomial, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Ptr{Int64}, Int32, Int32, Int32),
                n, p, dims, Int32(length(dims)), dtype_val, device)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function poisson_dist(lam::Float64, dims::Vector{Int64};
                      dtype_val::Int32=int64,
                      device::Int32=CPU)::InsightArray
    ptr = ccall((:insight_jl_poisson, LIB_INSIGHT), Ptr{Cvoid},
                (Float64, Ptr{Int64}, Int32, Int32, Int32),
                lam, dims, Int32(length(dims)), dtype_val, device)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# ============================================================================
# Additional FFT (Phase D)
# ============================================================================

function fftshift(x::InsightArray; axis::Int=-1)::InsightArray
    ptr = ccall((:insight_jl_fftshift, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(axis))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function ifftshift(x::InsightArray; axis::Int=-1)::InsightArray
    ptr = ccall((:insight_jl_ifftshift, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(axis))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function next_fast_len(target::Int)::Int
    Int(ccall((:insight_jl_next_fast_len, LIB_INSIGHT), Int32,
              (Int32,), Int32(target)))
end

function hfft(x::InsightArray; n::Union{Int,Nothing}=nothing)::InsightArray
    has_n = n !== nothing ? Int32(1) : Int32(0)
    nv = n !== nothing ? Int64(n) : Int64(-1)
    ptr = ccall((:insight_jl_hfft, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int64), x, has_n, nv)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function ihfft(x::InsightArray; n::Union{Int,Nothing}=nothing)::InsightArray
    has_n = n !== nothing ? Int32(1) : Int32(0)
    nv = n !== nothing ? Int64(n) : Int64(-1)
    ptr = ccall((:insight_jl_ihfft, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int64), x, has_n, nv)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function rfft2(x::InsightArray; s::Vector{Int64}=Int64[],
               axes::Vector{Int32}=Int32[-2, -1])::InsightArray
    ptr = ccall((:insight_jl_rfft2, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Int64}, Int32, Ptr{Int32}, Int32),
                x, s, Int32(length(s)), axes, Int32(length(axes)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function irfft2(x::InsightArray; s::Vector{Int64}=Int64[],
                axes::Vector{Int32}=Int32[-2, -1])::InsightArray
    ptr = ccall((:insight_jl_irfft2, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Int64}, Int32, Ptr{Int32}, Int32),
                x, s, Int32(length(s)), axes, Int32(length(axes)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function rfftn(x::InsightArray; s::Vector{Int64}=Int64[],
               axes::Vector{Int32}=Int32[])::InsightArray
    ptr = ccall((:insight_jl_rfftn, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Int64}, Int32, Ptr{Int32}, Int32),
                x, s, Int32(length(s)), axes, Int32(length(axes)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function irfftn(x::InsightArray; s::Vector{Int64}=Int64[],
                axes::Vector{Int32}=Int32[])::InsightArray
    ptr = ccall((:insight_jl_irfftn, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Int64}, Int32, Ptr{Int32}, Int32),
                x, s, Int32(length(s)), axes, Int32(length(axes)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# ============================================================================
# Additional Linalg (Phase D)
# ============================================================================

function lstsq(a::InsightArray, b::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_lstsq, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}), a, b)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function cond_fn(x::InsightArray; p::Float64=2.0)::InsightArray
    ptr = ccall((:insight_jl_cond, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Float64), x, p)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function matrix_rank(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_matrix_rank, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# ============================================================================
# Convenience functions (aliases and missing bindings)
# ============================================================================

"""
    item(arr::InsightArray, idx::Int) -> Number

Extract a scalar value from a 1-D array at 0-based index `idx`.
"""
function item(arr::InsightArray, idx::Int)
    data = to_array(arr)
    return data[idx + 1]
end

"""
    to_data(arr::InsightArray) -> Array

Alias for `to_array`. Copies data from an InsightArray to a Julia Array.
"""
const to_data = to_array

"""
    negative(x::InsightArray) -> InsightArray

Element-wise negation.
"""
function negative(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_negative, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

"""
    abs(x::InsightArray) -> InsightArray

Element-wise absolute value. Alias for `abs_fn`.
"""
abs(x::InsightArray) = abs_fn(x)

"""
    sqrt(x::InsightArray) -> InsightArray

Element-wise square root. Alias for `sqrt_fn`.
"""
sqrt(x::InsightArray) = sqrt_fn(x)

"""
    exp(x::InsightArray) -> InsightArray

Element-wise exponential. Alias for `exp_fn`.
"""
exp(x::InsightArray) = exp_fn(x)

"""
    log(x::InsightArray) -> InsightArray

Element-wise natural logarithm. Alias for `log_fn`.
"""
log(x::InsightArray) = log_fn(x)

"""
    sin(x::InsightArray) -> InsightArray

Element-wise sine. Alias for `sin_fn`.
"""
sin(x::InsightArray) = sin_fn(x)

"""
    cos(x::InsightArray) -> InsightArray

Element-wise cosine. Alias for `cos_fn`.
"""
cos(x::InsightArray) = cos_fn(x)

"""
    tan(x::InsightArray) -> InsightArray

Element-wise tangent. Alias for `tan_fn`.
"""
tan(x::InsightArray) = tan_fn(x)

"""
    floor(x::InsightArray) -> InsightArray

Element-wise floor. Alias for `floor_fn`.
"""
floor(x::InsightArray) = floor_fn(x)

"""
    ceil(x::InsightArray) -> InsightArray

Element-wise ceil. Alias for `ceil_fn`.
"""
ceil(x::InsightArray) = ceil_fn(x)

"""
    round(x::InsightArray) -> InsightArray

Element-wise round. Alias for `round_fn`.
"""
round(x::InsightArray) = round_fn(x)

# --- Comparison operators ---
@jl_binary equal  :insight_jl_equal
@jl_binary greater :insight_jl_greater
@jl_binary less   :insight_jl_less

# --- Additional reductions ---
function max(x::InsightArray; axis::Union{Int,Nothing}=nothing,
             keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(axis) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_max, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function min(x::InsightArray; axis::Union{Int,Nothing}=nothing,
             keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(axis) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_min, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function argmax(x::InsightArray; axis::Union{Int,Nothing}=nothing,
                keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(axis) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_argmax, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function argmin(x::InsightArray; axis::Union{Int,Nothing}=nothing,
                keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(axis) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_argmin, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function prod(x::InsightArray; axis::Union{Int,Nothing}=nothing,
              keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(axis) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_prod, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

# --- Additional linalg ---
"""
    dot(a::InsightArray, b::InsightArray) -> InsightArray

Dot product of two 1-D arrays.
"""
function dot(a::InsightArray, b::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_dot, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}), a, b)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

"""
    outer(a::InsightArray, b::InsightArray) -> InsightArray

Outer product of two 1-D arrays.
"""
function outer(a::InsightArray, b::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_outer, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}), a, b)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

"""
    norm(x::InsightArray; ord::Real=2.0) -> InsightArray

Matrix or vector norm.
"""
function norm(x::InsightArray; ord::Real=2.0)::InsightArray
    ptr = ccall((:insight_jl_norm, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Float64), x, Float64(ord))
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

"""
    trace(x::InsightArray) -> InsightArray

Sum of diagonal elements.
"""
function trace(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_trace, LIB_INSIGHT), Ptr{Cvoid}, (Ptr{Cvoid},), x)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

# ============================================================================
# Display
# ============================================================================

function Base.show(io::IO, a::InsightArray)
    if a.ptr == C_NULL
        print(io, "Array(<freed>)")
        return
    end
    s = shape(a)
    d = dtype(a)
    dt_name = d == float64 ? "float64" :
              d == float32 ? "float32" :
              d == int32   ? "int32" :
              d == int64   ? "int64" :
              d == int8    ? "int8" :
              d == uint8   ? "uint8" :
              d == bool    ? "bool" : "dtype($d)"
    p = device_type(a)
    place_name = p == GPU ? "gpu" : "cpu"
    print(io, "Array(shape=$(s), dtype=$dt_name, place=$place_name)")
end

function Base.show(io::IO, ::MIME"text/plain", a::InsightArray)
    show(io, a)
end

# ============================================================================
# Signal Processing
# ============================================================================

function convolve(a::InsightArray, v::InsightArray;
                  mode::String="full")::InsightArray
    ptr = ccall((:insight_jl_convolve, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Cstring), a, v, mode)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function unwrap(p::InsightArray; axis::Int=-1, discont::Real=Float64(π),
                period::Real=Float64(2π))::InsightArray
    ptr = ccall((:insight_jl_unwrap, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Float64, Float64), p, Int32(axis),
                Float64(discont), Float64(period))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function sinc(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_sinc, LIB_INSIGHT), Ptr{Cvoid}, (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# ============================================================================
# Signal processing (split into sub-module files aligned with C++ structure)
# ============================================================================

# Top-level signal functions (convolve, unwrap, sinc)
# convolve is defined above as ins::convolve wrapper
# unwrap is defined above
# sinc is defined above

# Include signal sub-module files (each contains ccall implementations + docstrings)
const _signal_dir = joinpath(@__DIR__, "modules", "signal")
include(joinpath(_signal_dir, "windows.jl"))
include(joinpath(_signal_dir, "waveforms.jl"))
include(joinpath(_signal_dir, "bsplines.jl"))
include(joinpath(_signal_dir, "filter_design.jl"))
include(joinpath(_signal_dir, "convolution.jl"))
include(joinpath(_signal_dir, "filtering.jl"))
include(joinpath(_signal_dir, "spectral.jl"))
include(joinpath(_signal_dir, "wavelets.jl"))
include(joinpath(_signal_dir, "acoustics.jl"))
include(joinpath(_signal_dir, "demod.jl"))
include(joinpath(_signal_dir, "peak_finding.jl"))
include(joinpath(_signal_dir, "radar.jl"))
include(joinpath(_signal_dir, "estimation.jl"))
include(joinpath(_signal_dir, "io.jl"))

# --- Signal submodule (convenience namespace) ---
module signal
    using ..Insight
    # Re-export all signal functions
    const hann = Insight.hann
    const hamming = Insight.hamming
    const blackman = Insight.blackman
    const kaiser = Insight.kaiser
    const gaussian = Insight.gaussian
    const boxcar = Insight.boxcar
    const triang = Insight.triang
    const bartlett = Insight.bartlett
    const flattop = Insight.flattop
    const nuttall = Insight.nuttall
    const blackmanharris = Insight.blackmanharris
    const tukey = Insight.tukey
    const chebwin = Insight.chebwin
    const taylor = Insight.taylor
    const get_window = Insight.get_window
    const sawtooth = Insight.sawtooth
    const square = Insight.square_wf
    const chirp = Insight.chirp
    const unit_impulse = Insight.unit_impulse
    const gauss_spline = Insight.gauss_spline
    const cubic = Insight.cubic
    const quadratic = Insight.quadratic
    const kaiser_beta = Insight.kaiser_beta
    const kaiser_atten = Insight.kaiser_atten
    const firwin = Insight.firwin
    const cmplx_sort = Insight.cmplx_sort
    const fftconvolve = Insight.fftconvolve
    const correlate = Insight.correlate
    const correlation_lags = Insight.correlation_lags
    const hilbert = Insight.hilbert
    const detrend = Insight.detrend
    const lfilter = Insight.lfilter
    const filtfilt = Insight.filtfilt
    const decimate = Insight.decimate
    const resample = Insight.resample
    const freq_shift = Insight.freq_shift
    const welch = Insight.welch_jl
    const periodogram = Insight.periodogram_jl
    const morlet = Insight.morlet
    const ricker = Insight.ricker
    const mel2hz = Insight.mel2hz
    const hz2mel = Insight.hz2mel
    const mel_frequencies = Insight.mel_frequencies
    const hz2bark = Insight.hz2bark
    const bark2hz = Insight.bark2hz
    const fm_demod = Insight.fm_demod
    const argrelmax = Insight.argrelmax
    const argrelmin = Insight.argrelmin
    const cfar_alpha = Insight.cfar_alpha
    const read_bin = Insight.read_bin
    const write_bin = Insight.write_bin
    const pack_bin = Insight.pack_bin
    const unpack_bin = Insight.unpack_bin
    const convolve = Insight.convolve
    const unwrap = Insight.unwrap
    const sinc_fn = Insight.sinc
    const csd = Insight.csd
    const coherence = Insight.coherence
    const spectrogram = Insight.spectrogram
    const stft = Insight.stft
    const vectorstrength = Insight.vectorstrength
    const choose_conv_method = Insight.choose_conv_method
    const firfilter_zi_state = Insight.firfilter_zi_state
    const pulse_compression = Insight.pulse_compression
    const pulse_doppler = Insight.pulse_doppler
    const mvdr = Insight.mvdr
    const convolve2d = Insight.convolve2d
    const correlate2d = Insight.correlate2d
    const hilbert2 = Insight.hilbert2
    const wiener = Insight.wiener
    const firfilter = Insight.firfilter
    const lfilter_zi = Insight.lfilter_zi
    const resample_poly = Insight.resample_poly
    const morlet2 = Insight.morlet2
    const firwin2 = Insight.firwin2
    const read_sigmf = Insight.read_sigmf
    const write_sigmf = Insight.write_sigmf
    const cosine_win = Insight.cosine_win
    const general_hamming = Insight.general_hamming
    const parzen_win = Insight.parzen_win
    const bohman_win = Insight.bohman_win
    const barthann_win = Insight.barthann_win
    const exponential_win = Insight.exponential_win
    const general_gaussian_win = Insight.general_gaussian_win
    const KalmanFilter = Insight.KalmanFilter
    const kf_predict = Insight.predict
    const kf_update = Insight.update
end

end # module Insight
