# modules/signal/bsplines.jl
# B-spline functions.

"""
    gauss_spline(x::InsightArray, n::Int) -> InsightArray

Return a Gaussian B-spline of order `n`.
"""
function gauss_spline(x::InsightArray, n::Int)::InsightArray
    ptr = ccall((:insight_jl_gauss_spline, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(n))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    cubic(x::InsightArray) -> InsightArray

Return a cubic B-spline.
"""
function cubic(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_cubic, LIB_INSIGHT), Ptr{Cvoid}, (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    quadratic(x::InsightArray) -> InsightArray

Return a quadratic B-spline.
"""
function quadratic(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_quadratic, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end
