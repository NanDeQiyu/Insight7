# modules/indexing.jl
# Indexing and searching functions.

"""
    unique_ins(x::InsightArray; return_indices::Bool=false,
               return_inverse::Bool=false, return_counts::Bool=false) -> NamedTuple

Return the unique elements of `x`, optionally with indices, inverse, and counts.
"""
function unique_ins end

"""
    topk(x::InsightArray, k::Int; axis::Int=-1, largest::Bool=true,
         sorted::Bool=true) -> (values, indices)

Return the k largest (or smallest) elements and their indices.
"""
function topk end

"""
    gather(x::InsightArray, dim::Int, index::InsightArray) -> InsightArray

Gather values along an axis by index.
"""
function gather end

"""
    scatter(x::InsightArray, dim::Int, index::InsightArray, src::InsightArray) -> InsightArray

Scatter values into an array along an axis.
"""
function scatter end

"""
    scatter_add(x::InsightArray, dim::Int, index::InsightArray, src::InsightArray) -> InsightArray

Scatter-add values into an array along an axis.
"""
function scatter_add end

"""
    scatter_reduce(x::InsightArray, dim::Int, index::InsightArray, src::InsightArray;
                   reduce::String="replace") -> InsightArray

Scatter with a reduction operation along an axis.
"""
function scatter_reduce end

"""
    interp(x::InsightArray, xp::InsightArray, fp::InsightArray;
           left::Union{Float64,Nothing}=nothing,
           right::Union{Float64,Nothing}=nothing) -> InsightArray

One-dimensional linear interpolation.
"""
function interp end

"""
    indices_fn(dims::Vector{Int64}; sparse::Bool=false) -> InsightArray

Return an array representing the indices of a grid.
"""
function indices_fn end

"""
    ix_fn(arrays::Vector{InsightArray}) -> Vector{InsightArray}

Construct an open mesh from multiple sequences (for advanced indexing).
"""
function ix_fn end
