# modules/signal/peak_finding.jl
# Peak finding functions.

"""
    argrelmax(data::InsightArray; order::Int=1, axis::Int=-1) -> InsightArray

Return indices of local maxima in `data`.
"""
function argrelmax end

"""
    argrelmin(data::InsightArray; order::Int=1, axis::Int=-1) -> InsightArray

Return indices of local minima in `data`.
"""
function argrelmin end
