# modules/elementwise.jl
# Elementwise binary operations.

"""
    add(a::InsightArray, b::InsightArray) -> InsightArray

Element-wise addition of two arrays.
"""
function add end

"""
    sub(a::InsightArray, b::InsightArray) -> InsightArray

Element-wise subtraction of two arrays.
"""
function sub end

"""
    mul(a::InsightArray, b::InsightArray) -> InsightArray

Element-wise multiplication of two arrays.
"""
function mul end

"""
    div(a::InsightArray, b::InsightArray) -> InsightArray

Element-wise division of two arrays.
"""
function div end

"""
    pow(a::InsightArray, b::InsightArray) -> InsightArray

Element-wise exponentiation.
"""
function pow end

"""
    mod(a::InsightArray, b::InsightArray) -> InsightArray

Element-wise modulo.
"""
function mod end

"""
    maximum(a::InsightArray, b::InsightArray) -> InsightArray

Element-wise maximum of two arrays.
"""
function maximum end

"""
    minimum(a::InsightArray, b::InsightArray) -> InsightArray

Element-wise minimum of two arrays.
"""
function minimum end

"""
    equal(a::InsightArray, b::InsightArray) -> InsightArray

Element-wise equality comparison. Returns bool array.
"""
function equal end

"""
    not_equal(a::InsightArray, b::InsightArray) -> InsightArray

Element-wise inequality comparison. Returns bool array.
"""
function not_equal end

"""
    greater(a::InsightArray, b::InsightArray) -> InsightArray

Element-wise greater-than comparison. Returns bool array.
"""
function greater end

"""
    less(a::InsightArray, b::InsightArray) -> InsightArray

Element-wise less-than comparison. Returns bool array.
"""
function less end

"""
    greater_equal(a::InsightArray, b::InsightArray) -> InsightArray

Element-wise greater-or-equal comparison. Returns bool array.
"""
function greater_equal end

"""
    less_equal(a::InsightArray, b::InsightArray) -> InsightArray

Element-wise less-or-equal comparison. Returns bool array.
"""
function less_equal end
