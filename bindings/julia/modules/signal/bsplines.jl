# modules/signal/bsplines.jl
# B-spline basis functions.

"""
    gauss_spline(x::InsightArray, n::Int) -> InsightArray

Return a Gaussian B-spline of order `n` at points `x`.
"""
function gauss_spline end

"""
    cubic(x::InsightArray) -> InsightArray

Return a cubic B-spline at points `x`.
"""
function cubic end

"""
    quadratic(x::InsightArray) -> InsightArray

Return a quadratic B-spline at points `x`.
"""
function quadratic end
