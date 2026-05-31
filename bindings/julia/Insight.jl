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

function zeros(dims::Vector{Int64}, dtype_val::Int32=float32,
               device::Int32=CPU)::InsightArray
    ptr = ccall((:insight_jl_zeros, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Int64}, Int32, Int32, Int32),
                dims, Int32(length(dims)), dtype_val, device)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

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
    dims = collect(Int64, size(data))
    ptr = ccall((:insight_jl_from_data, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Int64}, Int32, Int32, Int32),
                pointer(data), dims, Int32(length(dims)), dtype_val, device)
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
    return reshape(dst, Tuple(shape(arr)))
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
# Display
# ============================================================================

function Base.show(io::IO, arr::InsightArray)
    if arr.ptr == C_NULL
        print(io, "<InsightArray (freed)>")
        return
    end
    s = shape(arr)
    dt = dtype(arr)
    dt_str = dt == float32 ? "float32" :
             dt == float64 ? "float64" :
             dt == int32   ? "int32" :
             dt == int64   ? "int64" : "other"
    dev = device_type(arr) == 1 ? "gpu" : "cpu"
    print(io, "<InsightArray shape=($(join(s, ", "))) dtype=$dt_str place=$dev>")
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

function unwrap(p::InsightArray; axis::Int=-1, discont::Float64=π,
                period::Float64=2π)::InsightArray
    ptr = ccall((:insight_jl_unwrap, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Float64, Float64), p, Int32(axis),
                discont, period)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function sinc(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_sinc, LIB_INSIGHT), Ptr{Cvoid}, (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# --- Macro for simple signal functions ---
macro jl_sigfunc(jl_name, c_name, arg_types)
    quote
        function $(esc(jl_name))(args...)::InsightArray
            ptr = ccall(($c_name, LIB_INSIGHT), Ptr{Cvoid},
                        $arg_types, args...)
            arr = InsightArray(ptr)
            finalizer(_free, arr)
            return arr
        end
    end
end

# --- Windows ---
function hann(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_hann, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function hamming(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_hamming, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function blackman(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_blackman, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function kaiser(M::Int, beta::Float64; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_kaiser, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Int32), Int64(M), beta, sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function gaussian(M::Int, std_val::Float64; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_gaussian, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Int32), Int64(M), std_val, sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function boxcar(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_boxcar, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function triang(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_triang, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function bartlett(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_bartlett, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function flattop(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_flattop, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function nuttall(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_nuttall, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function blackmanharris(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_blackmanharris, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function tukey(M::Int; alpha::Float64=0.5, sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_tukey, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Int32), Int64(M), alpha, sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function chebwin(M::Int, at::Float64; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_chebwin, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Int32), Int64(M), at, sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function taylor(M::Int; nbar::Int=4, sll::Float64=-30.0,
                norm::Bool=true, sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_taylor, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int64, Float64, Int32, Int32),
                Int64(M), Int64(nbar), sll, norm ? Int32(1) : Int32(0),
                sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function get_window(window::String, Nx::Int; fftbins::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_get_window, LIB_INSIGHT), Ptr{Cvoid},
                (Cstring, Int64, Int32), window, Int64(Nx),
                fftbins ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# --- Waveforms ---
function sawtooth(t::InsightArray; width::Float64=1.0)::InsightArray
    ptr = ccall((:insight_jl_sawtooth, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Float64), t, width)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function square_wf(t::InsightArray; duty::Float64=0.5)::InsightArray
    ptr = ccall((:insight_jl_square_wf, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Float64), t, duty)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function chirp(t::InsightArray, f0::Float64, t1::Float64, f1::Float64;
               method::Int32=Int32(0), phi::Float64=0.0,
               vertex_zero::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_chirp, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Float64, Float64, Float64, Int32, Float64, Int32),
                t, f0, t1, f1, method, phi, vertex_zero ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function unit_impulse(dims::Vector{Int64}; idx::Int64=Int64(-1))::InsightArray
    ptr = ccall((:insight_jl_unit_impulse, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Int64}, Int32, Int64), dims, Int32(length(dims)), idx)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# --- B-Splines ---
function gauss_spline(x::InsightArray, n::Int)::InsightArray
    ptr = ccall((:insight_jl_gauss_spline, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(n))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function cubic(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_cubic, LIB_INSIGHT), Ptr{Cvoid}, (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function quadratic(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_quadratic, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# --- Filter Design ---
function kaiser_beta(a::Float64)::Float64
    ccall((:insight_jl_kaiser_beta, LIB_INSIGHT), Float64, (Float64,), a)
end

function kaiser_atten(numtaps::Int, width::Float64)::Float64
    ccall((:insight_jl_kaiser_atten, LIB_INSIGHT), Float64,
          (Int64, Float64), Int64(numtaps), width)
end

function firwin(numtaps::Int, cutoff::Vector{Float64};
                window::String="hamming", pass_zero::String="lowpass",
                scale::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_firwin, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Ptr{Float64}, Int32, Cstring, Cstring, Int32),
                Int64(numtaps), cutoff, Int32(length(cutoff)),
                window, pass_zero, scale ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function cmplx_sort(p::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_cmplx_sort, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), p)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# --- Convolution ---
function fftconvolve(in1::InsightArray, in2::InsightArray;
                     mode::String="full")::InsightArray
    ptr = ccall((:insight_jl_fftconvolve, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Cstring), in1, in2, mode)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function correlate(in1::InsightArray, in2::InsightArray;
                   mode::String="full")::InsightArray
    ptr = ccall((:insight_jl_correlate, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Cstring), in1, in2, mode)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function correlation_lags(in1_len::Int, in2_len::Int;
                          mode::String="full")::InsightArray
    ptr = ccall((:insight_jl_correlation_lags, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int64, Cstring), Int64(in1_len), Int64(in2_len), mode)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# --- Filtering ---
function hilbert(x::InsightArray; N::Int=-1)::InsightArray
    ptr = ccall((:insight_jl_hilbert, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int64), x, Int64(N))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function detrend(data::InsightArray; axis::Int=-1,
                 type::String="linear")::InsightArray
    ptr = ccall((:insight_jl_detrend, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Cstring), data, Int32(axis), type)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function lfilter(b::InsightArray, a::InsightArray, x::InsightArray;
                 axis::Int=-1)::InsightArray
    ptr = ccall((:insight_jl_lfilter, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Int32),
                b, a, x, Int32(axis))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function filtfilt(b::InsightArray, a::InsightArray, x::InsightArray;
                  axis::Int=-1)::InsightArray
    ptr = ccall((:insight_jl_filtfilt, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Int32),
                b, a, x, Int32(axis))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function decimate(x::InsightArray, q::Int; axis::Int=-1,
                  zero_phase::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_decimate, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int64, Int32, Int32),
                x, Int64(q), Int32(axis), zero_phase ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function resample(x::InsightArray, num::Int; axis::Int=-1)::InsightArray
    ptr = ccall((:insight_jl_resample, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int64, Int32), x, Int64(num), Int32(axis))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function freq_shift(x::InsightArray, freq::Float64,
                    fs::Float64)::InsightArray
    ptr = ccall((:insight_jl_freq_shift, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Float64, Float64), x, freq, fs)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# --- Spectral Analysis ---
function welch_jl(x::InsightArray; fs::Float64=1.0, window::String="hann",
                  nperseg::Int=256, noverlap::Int=0, nfft::Int=0,
                  detrend::String="constant", return_onesided::Bool=true,
                  scaling::String="density")
    f_ref = Ref{Ptr{Cvoid}}(C_NULL)
    p_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_welch, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Float64, Cstring, Int64, Int64, Int64, Cstring, Int32,
           Cstring, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
          x, fs, window, Int64(nperseg), Int64(noverlap), Int64(nfft),
          detrend, return_onesided ? Int32(1) : Int32(0), scaling,
          f_ref, p_ref)
    f = InsightArray(f_ref[]); finalizer(_free, f)
    pxx = InsightArray(p_ref[]); finalizer(_free, pxx)
    return (f, pxx)
end

function periodogram_jl(x::InsightArray; fs::Float64=1.0,
                        window::String="boxcar", nfft::Int=0,
                        detrend::String="constant",
                        return_onesided::Bool=true,
                        scaling::String="density")
    f_ref = Ref{Ptr{Cvoid}}(C_NULL)
    p_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_periodogram, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Float64, Cstring, Int64, Cstring, Int32, Cstring,
           Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
          x, fs, window, Int64(nfft), detrend,
          return_onesided ? Int32(1) : Int32(0), scaling, f_ref, p_ref)
    f = InsightArray(f_ref[]); finalizer(_free, f)
    pxx = InsightArray(p_ref[]); finalizer(_free, pxx)
    return (f, pxx)
end

# --- Wavelets ---
function morlet(M::Int; w::Float64=5.0, s::Float64=1.0,
                complete::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_morlet, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Float64, Int32),
                Int64(M), w, s, complete ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function ricker(points::Int, a::Float64)::InsightArray
    ptr = ccall((:insight_jl_ricker, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64), Int64(points), a)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# --- Acoustics ---
function mel2hz(mels::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_mel2hz, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), mels)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function hz2mel(hz::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_hz2mel, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), hz)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function mel_frequencies(n_mels::Int; f_low::Float64=0.0,
                         f_high::Float64=11000.0)::InsightArray
    ptr = ccall((:insight_jl_mel_frequencies, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Float64), Int64(n_mels), f_low, f_high)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function hz2bark(hz::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_hz2bark, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), hz)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function bark2hz(bark::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_bark2hz, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), bark)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# --- Demod ---
function fm_demod(x::InsightArray, axis::Int = -1)::InsightArray
    ptr = ccall((:insight_jl_fm_demod, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(axis))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# --- Peak Finding ---
function argrelmax(data::InsightArray, axis::Int = 0, order::Int = 1)::Vector{Int64}
    out = Vector{Int64}(undef, numel(data))
    n = ccall((:insight_jl_argrelmax, LIB_INSIGHT), Int64,
              (Ptr{Cvoid}, Int32, Int32, Ptr{Int64}),
              data, Int32(axis), Int32(order), out)
    return out[1:n]
end

function argrelmin(data::InsightArray, axis::Int = 0, order::Int = 1)::Vector{Int64}
    out = Vector{Int64}(undef, numel(data))
    n = ccall((:insight_jl_argrelmin, LIB_INSIGHT), Int64,
              (Ptr{Cvoid}, Int32, Int32, Ptr{Int64}),
              data, Int32(axis), Int32(order), out)
    return out[1:n]
end

# --- Radar ---
function cfar_alpha(pfa::Float64, N::Int)::Float64
    return ccall((:insight_jl_cfar_alpha, LIB_INSIGHT), Float64,
                 (Float64, Int32), pfa, Int32(N))
end

function pulse_compression(x::InsightArray, tpl::InsightArray,
                           normalize::Bool = false)::InsightArray
    ptr = ccall((:insight_jl_pulse_compression, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Int32), x, tpl, Int32(normalize ? 1 : 0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function pulse_doppler(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_pulse_doppler, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function mvdr(x::InsightArray, sv::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_mvdr, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}), x, sv)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# --- Additional Windows ---
function cosine_win(M::Int, sym::Bool = true)::InsightArray
    ptr = ccall((:insight_jl_cosine_win, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), Int32(sym ? 1 : 0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function general_hamming(M::Int, alpha::Float64, sym::Bool = true)::InsightArray
    ptr = ccall((:insight_jl_general_hamming, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Int32), Int64(M), alpha, Int32(sym ? 1 : 0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function parzen_win(M::Int, sym::Bool = true)::InsightArray
    ptr = ccall((:insight_jl_parzen, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), Int32(sym ? 1 : 0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function bohman_win(M::Int, sym::Bool = true)::InsightArray
    ptr = ccall((:insight_jl_bohman, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), Int32(sym ? 1 : 0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function barthann_win(M::Int, sym::Bool = true)::InsightArray
    ptr = ccall((:insight_jl_barthann, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), Int32(sym ? 1 : 0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function exponential_win(M::Int, center::Float64 = -1.0, tau::Float64 = 1.0,
                         sym::Bool = true)::InsightArray
    ptr = ccall((:insight_jl_exponential_win, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Float64, Int32), Int64(M), center, tau, Int32(sym ? 1 : 0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function general_gaussian_win(M::Int, p::Float64, sig::Float64,
                              sym::Bool = true)::InsightArray
    ptr = ccall((:insight_jl_general_gaussian, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Float64, Int32), Int64(M), p, sig, Int32(sym ? 1 : 0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# --- Additional Filter Design ---
function firwin2(numtaps::Int, freq::Vector{Float64}, gain::Vector{Float64},
                 window::String = "hamming")::InsightArray
    ptr = ccall((:insight_jl_firwin2, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Ptr{Float64}, Int32, Ptr{Float64}, Int32, Cstring),
                Int64(numtaps), freq, Int32(length(freq)), gain, Int32(length(gain)), window)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# --- Additional Convolution ---
function convolve2d(in1::InsightArray, in2::InsightArray,
                   mode::String = "full")::InsightArray
    ptr = ccall((:insight_jl_convolve2d, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Cstring), in1, in2, mode)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function correlate2d(in1::InsightArray, in2::InsightArray,
                     mode::String = "full")::InsightArray
    ptr = ccall((:insight_jl_correlate2d, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Cstring), in1, in2, mode)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# --- Additional Filtering ---
function hilbert2(x::InsightArray, N::Int = -1)::InsightArray
    ptr = ccall((:insight_jl_hilbert2, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int64), x, Int64(N))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function wiener(im::InsightArray, noise::Float64 = -1.0)::InsightArray
    ptr = ccall((:insight_jl_wiener, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Float64), im, noise)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function firfilter(b::InsightArray, x::InsightArray,
                   axis::Int = -1)::InsightArray
    ptr = ccall((:insight_jl_firfilter, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Int32), b, x, Int32(axis))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function lfilter_zi(b::InsightArray, a::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_lfilter_zi, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}), b, a)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function resample_poly(x::InsightArray, up::Int, down::Int,
                       axis::Int = -1)::InsightArray
    ptr = ccall((:insight_jl_resample_poly, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int64, Int64, Int32), x, Int64(up), Int64(down), Int32(axis))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# --- Additional Wavelets ---
function morlet2(M::Int, s::Float64, w::Float64 = 5.0)::InsightArray
    ptr = ccall((:insight_jl_morlet2, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Float64), Int64(M), s, w)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# --- Additional I/O ---
function read_sigmf(data_file::String, meta_file::String = "",
                    num_samples::Int = 0, offset::Int = 0)::InsightArray
    ptr = ccall((:insight_jl_read_sigmf, LIB_INSIGHT), Ptr{Cvoid},
                (Cstring, Cstring, Int64, Int64),
                data_file, meta_file, Int64(num_samples), Int64(offset))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function write_sigmf(data_file::String, data::InsightArray, append::Bool = true)
    ccall((:insight_jl_write_sigmf, LIB_INSIGHT), Cvoid,
          (Cstring, Ptr{Cvoid}, Int32), data_file, data, Int32(append ? 1 : 0))
end

# --- Signal I/O ---
function read_bin(file::String, dtype::DataType = UInt8,
                  num_samples::Int = 0, offset::Int = 0)::InsightArray
    dtype_code = _dtype_code(dtype)
    ptr = ccall((:insight_jl_read_bin, LIB_INSIGHT), Ptr{Cvoid},
                (Cstring, Int32, Int64, Int64),
                file, Int32(dtype_code), Int64(num_samples), Int64(offset))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function write_bin(file::String, data::InsightArray, append::Bool = true)
    ccall((:insight_jl_write_bin, LIB_INSIGHT), Cvoid,
          (Cstring, Ptr{Cvoid}, Int32), file, data, Int32(append ? 1 : 0))
end

function pack_bin(data::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_pack_bin, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), data)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function unpack_bin(binary::InsightArray, dtype::DataType,
                    endianness::String = "L")::InsightArray
    dtype_code = _dtype_code(dtype)
    ptr = ccall((:insight_jl_unpack_bin, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Cstring),
                binary, Int32(dtype_code), endianness)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# --- Spectral Analysis (struct returns) ---
function csd(x::InsightArray, y::InsightArray; fs::Float64 = 1.0,
             window::String = "hann", nperseg::Int = 256,
             noverlap::Int = 0, nfft::Int = 0)
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

function coherence(x::InsightArray, y::InsightArray; fs::Float64 = 1.0,
                   window::String = "hann", nperseg::Int = 256,
                   noverlap::Int = 0, nfft::Int = 0)
    f_ref = Ref{Ptr{Cvoid}}(C_NULL)
    Pxx_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_coherence, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Ptr{Cvoid}, Float64, Cstring, Int64, Int64, Int64,
           Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
          x, y, fs, window, Int64(nperseg), Int64(noverlap), Int64(nfft),
          f_ref, Pxx_ref)
    f_arr = InsightArray(f_ref[]); finalizer(_free, f_arr)
    Pxx_arr = InsightArray(Pxx_ref[]); finalizer(_free, Pxx_arr)
    return (f=f_arr, Pxx=Pxx_arr)
end

function spectrogram(x::InsightArray; fs::Float64 = 1.0,
                     window::String = "hann", nperseg::Int = 256,
                     noverlap::Int = 0, nfft::Int = 0)
    f_ref = Ref{Ptr{Cvoid}}(C_NULL)
    t_ref = Ref{Ptr{Cvoid}}(C_NULL)
    Sxx_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_spectrogram, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Float64, Cstring, Int64, Int64, Int64,
           Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
          x, fs, window, Int64(nperseg), Int64(noverlap), Int64(nfft),
          f_ref, t_ref, Sxx_ref)
    f_arr = InsightArray(f_ref[]); finalizer(_free, f_arr)
    t_arr = InsightArray(t_ref[]); finalizer(_free, t_arr)
    Sxx_arr = InsightArray(Sxx_ref[]); finalizer(_free, Sxx_arr)
    return (f=f_arr, t=t_arr, Sxx=Sxx_arr)
end

function stft(x::InsightArray; fs::Float64 = 1.0,
              window::String = "hann", nperseg::Int = 256,
              noverlap::Int = 0, nfft::Int = 0)
    f_ref = Ref{Ptr{Cvoid}}(C_NULL)
    t_ref = Ref{Ptr{Cvoid}}(C_NULL)
    Sxx_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_stft, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Float64, Cstring, Int64, Int64, Int64,
           Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
          x, fs, window, Int64(nperseg), Int64(noverlap), Int64(nfft),
          f_ref, t_ref, Sxx_ref)
    f_arr = InsightArray(f_ref[]); finalizer(_free, f_arr)
    t_arr = InsightArray(t_ref[]); finalizer(_free, t_arr)
    Sxx_arr = InsightArray(Sxx_ref[]); finalizer(_free, Sxx_arr)
    return (f=f_arr, t=t_arr, Sxx=Sxx_arr)
end

function vectorstrength(events::InsightArray, period::Float64)
    strength_ref = Ref{Float64}(0.0)
    phase_ref = Ref{Float64}(0.0)
    ccall((:insight_jl_vectorstrength, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Float64, Ptr{Float64}, Ptr{Float64}),
          events, period, strength_ref, phase_ref)
    return (strength=strength_ref[], phase=phase_ref[])
end

function choose_conv_method(in1::InsightArray, in2::InsightArray,
                            mode::String = "full")::String
    ptr = ccall((:insight_jl_choose_conv_method, LIB_INSIGHT), Cstring,
                (Ptr{Cvoid}, Ptr{Cvoid}, Cstring), in1, in2, mode)
    return unsafe_string(ptr)
end

function firfilter_zi_state(b::InsightArray, x::InsightArray, zi::InsightArray,
                            axis::Int = -1)
    y_ref = Ref{Ptr{Cvoid}}(C_NULL)
    zf_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_firfilter_zi_state, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Int32,
           Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
          b, x, zi, Int32(axis), y_ref, zf_ref)
    y_arr = InsightArray(y_ref[]); finalizer(_free, y_arr)
    zf_arr = InsightArray(zf_ref[]); finalizer(_free, zf_arr)
    return (y=y_arr, zf=zf_arr)
end

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
end

end # module Insight
