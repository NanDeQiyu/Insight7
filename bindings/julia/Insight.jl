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
       fft, ifft, rand, randn, cast

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
    dims = Int64.(size(data))
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

end # module Insight
