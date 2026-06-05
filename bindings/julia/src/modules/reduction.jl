# modules/reduction.jl
# Reduction operations.

"""
    sum(x::InsightArray; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false) -> InsightArray

Sum of array elements over the given axis.
"""
function sum end

"""
    mean(x::InsightArray; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false) -> InsightArray

Mean of array elements over the given axis.
"""
function mean end

"""
    max(x::InsightArray; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false) -> InsightArray

Maximum of array elements over the given axis.
"""
function max end

"""
    min(x::InsightArray; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false) -> InsightArray

Minimum of array elements over the given axis.
"""
function min end

"""
    prod(x::InsightArray; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false) -> InsightArray

Product of array elements over the given axis.
"""
function prod end

"""
    argmax(x::InsightArray; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false) -> InsightArray

Indices of maximum values over the given axis.
"""
function argmax end

"""
    argmin(x::InsightArray; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false) -> InsightArray

Indices of minimum values over the given axis.
"""
function argmin end

"""
    cummax(x::InsightArray, axis::Int) -> InsightArray

Cumulative maximum along the given axis.
"""
function cummax end

"""
    cummin(x::InsightArray, axis::Int) -> InsightArray

Cumulative minimum along the given axis.
"""
function cummin end

"""
    sem(x::InsightArray; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false, ddof::Int=0) -> InsightArray

Standard error of the mean.
"""
function sem end

"""
    count_nonzero(x::InsightArray; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false) -> InsightArray

Count the number of non-zero elements.
"""
function count_nonzero end

"""
    median(x::InsightArray; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false) -> InsightArray

Median of array elements.
"""
function median end

"""
    quantile(x::InsightArray, q::Float64; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false) -> InsightArray

Compute the q-th quantile.
"""
function quantile end

"""
    percentile(x::InsightArray, q::Float64; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false) -> InsightArray

Compute the q-th percentile.
"""
function percentile end

"""
    nansum(x::InsightArray; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false) -> InsightArray

Sum of array elements, ignoring NaN values.
"""
function nansum end

"""
    nanmean(x::InsightArray; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false) -> InsightArray

Mean of array elements, ignoring NaN values.
"""
function nanmean end

"""
    nanmax(x::InsightArray; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false) -> InsightArray

Maximum of array elements, ignoring NaN values.
"""
function nanmax end

"""
    nanmin(x::InsightArray; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false) -> InsightArray

Minimum of array elements, ignoring NaN values.
"""
function nanmin end

"""
    nanstd(x::InsightArray; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false, ddof::Int=0) -> InsightArray

Standard deviation, ignoring NaN values.
"""
function nanstd end

"""
    nanvar(x::InsightArray; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false, ddof::Int=0) -> InsightArray

Variance, ignoring NaN values.
"""
function nanvar end
