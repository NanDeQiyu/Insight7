# modules/complex.jl
# Complex number operations.

"""
    is_complex(x::InsightArray) -> Bool

Check if the array has a complex dtype (complex64 or complex128).
"""
function is_complex end

"""
    has_complex_shape(x::InsightArray) -> Bool

Check if the array uses legacy complex storage (last dimension = 2).
"""
function has_complex_shape end

"""
    to_complex(real::InsightArray) -> InsightArray

Convert a real array to complex (imaginary part = 0).
"""
function to_complex end

"""
    to_complex(real::InsightArray, imag_arr::InsightArray) -> InsightArray

Combine real and imaginary arrays into a complex array.
"""
function to_complex end

"""
    as_complex(x::InsightArray) -> InsightArray

View a real array with last dim=2 as complex (zero-copy).
"""
function as_complex end

"""
    as_real(x::InsightArray) -> InsightArray

View a complex array as real with last dim=2 (zero-copy).
"""
function as_real end

"""
    real_part(z::InsightArray) -> InsightArray

Extract the real part of a complex array.
"""
function real_part end

"""
    imag_part(z::InsightArray) -> InsightArray

Extract the imaginary part of a complex array.
"""
function imag_part end
