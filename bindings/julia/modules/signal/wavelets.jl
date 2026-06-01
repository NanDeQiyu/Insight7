# modules/signal/wavelets.jl
# Wavelet functions.

"""
    morlet(M::Int; w::Float64=5.0, s::Float64=1.0, complete::Bool=true) -> InsightArray

Return a complex Morlet wavelet of length `M`.
"""
function morlet end

"""
    ricker(points::Int, a::Float64) -> InsightArray

Return a Ricker wavelet (Mexican hat) of the given number of points.
"""
function ricker end

"""
    morlet2(M::Int, s::Float64; w::Float64=5.0) -> InsightArray

Return a complex Morlet wavelet at the specified scale.
"""
function morlet2 end
