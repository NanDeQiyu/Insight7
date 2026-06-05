# modules/unary.jl
# Unary math functions.

"""
    abs_fn(x::InsightArray) -> InsightArray

Element-wise absolute value.
"""
function abs_fn end

"""
    negative(x::InsightArray) -> InsightArray

Element-wise negation.
"""
function negative end

"""
    square(x::InsightArray) -> InsightArray

Element-wise square.
"""
function square end

"""
    sqrt_fn(x::InsightArray) -> InsightArray

Element-wise square root.
"""
function sqrt_fn end

"""
    exp_fn(x::InsightArray) -> InsightArray

Element-wise exponential (e^x).
"""
function exp_fn end

"""
    log_fn(x::InsightArray) -> InsightArray

Element-wise natural logarithm.
"""
function log_fn end

"""
    sin_fn(x::InsightArray) -> InsightArray

Element-wise sine.
"""
function sin_fn end

"""
    cos_fn(x::InsightArray) -> InsightArray

Element-wise cosine.
"""
function cos_fn end

"""
    tan_fn(x::InsightArray) -> InsightArray

Element-wise tangent.
"""
function tan_fn end

"""
    floor_fn(x::InsightArray) -> InsightArray

Element-wise floor.
"""
function floor_fn end

"""
    ceil_fn(x::InsightArray) -> InsightArray

Element-wise ceiling.
"""
function ceil_fn end

"""
    round_fn(x::InsightArray) -> InsightArray

Element-wise rounding to nearest integer.
"""
function round_fn end

"""
    sign(x::InsightArray) -> InsightArray

Element-wise sign function (-1, 0, or +1).
"""
function sign end

"""
    exp2_fn(x::InsightArray) -> InsightArray

Element-wise base-2 exponential (2^x).
"""
function exp2_fn end

"""
    expm1_fn(x::InsightArray) -> InsightArray

Element-wise exp(x) - 1.
"""
function expm1_fn end

"""
    log1p_fn(x::InsightArray) -> InsightArray

Element-wise log(1 + x).
"""
function log1p_fn end

"""
    cbrt_fn(x::InsightArray) -> InsightArray

Element-wise cube root.
"""
function cbrt_fn end

"""
    reciprocal_fn(x::InsightArray) -> InsightArray

Element-wise reciprocal (1/x).
"""
function reciprocal_fn end

"""
    asinh_fn(x::InsightArray) -> InsightArray

Element-wise inverse hyperbolic sine.
"""
function asinh_fn end

"""
    acosh_fn(x::InsightArray) -> InsightArray

Element-wise inverse hyperbolic cosine.
"""
function acosh_fn end

"""
    atanh_fn(x::InsightArray) -> InsightArray

Element-wise inverse hyperbolic tangent.
"""
function atanh_fn end

"""
    trunc_fn(x::InsightArray) -> InsightArray

Element-wise truncation toward zero.
"""
function trunc_fn end

"""
    deg2rad_fn(x::InsightArray) -> InsightArray

Element-wise conversion from degrees to radians.
"""
function deg2rad_fn end

"""
    rad2deg_fn(x::InsightArray) -> InsightArray

Element-wise conversion from radians to degrees.
"""
function rad2deg_fn end
