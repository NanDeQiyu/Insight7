# modules/cast.jl
# Type casting function.

"""
    cast(x::InsightArray, dtype_val::Int32) -> InsightArray

Cast an array to a different data type.

# Arguments
- `x::InsightArray`: Input array.
- `dtype_val::Int32`: Target dtype code (e.g. `Insight.float32`, `Insight.int64`).

# Returns
- `InsightArray`: Array with the specified dtype.
"""
function cast end
