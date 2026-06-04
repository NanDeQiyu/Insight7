"""Unary math operations.

Functions:
    abs, negative, square, sqrt, exp, log, log2, log10
    sin, cos, tan, asin, acos, atan
    sinh, cosh, tanh, floor, ceil, round, sign
    isnan, isinf, isfinite, where
    exp2, expm1, log1p, cbrt, reciprocal
    asinh, acosh, atanh, trunc, deg2rad, rad2deg
    conj, angle
"""

from __future__ import annotations

from insight._insight import Array  # noqa: F401
from insight._insight import (
    abs as _native_abs,
    negative as _native_negative,
    square as _native_square,
    sqrt as _native_sqrt,
    exp as _native_exp,
    log as _native_log,
    log2 as _native_log2,
    log10 as _native_log10,
    sin as _native_sin,
    cos as _native_cos,
    tan as _native_tan,
    asin as _native_asin,
    acos as _native_acos,
    atan as _native_atan,
    sinh as _native_sinh,
    cosh as _native_cosh,
    tanh as _native_tanh,
    floor as _native_floor,
    ceil as _native_ceil,
    round as _native_round,
    sign as _native_sign,
    isnan as _native_isnan,
    isinf as _native_isinf,
    isfinite as _native_isfinite,
    where as _native_where,
    exp2 as _native_exp2,
    expm1 as _native_expm1,
    log1p as _native_log1p,
    cbrt as _native_cbrt,
    reciprocal as _native_reciprocal,
    asinh as _native_asinh,
    acosh as _native_acosh,
    atanh as _native_atanh,
    trunc as _native_trunc,
    deg2rad as _native_deg2rad,
    rad2deg as _native_rad2deg,
    conj as _native_conj,
    angle as _native_angle,
)


def abs(x: "Array") -> "Array":
    """Element-wise absolute value.

    Computes the absolute value of each element in the input array.

    Args:
        x: Input array (numeric types).

    Returns:
        Array of the same shape containing |x_i| for each element.
    """
    return _native_abs(x)


def negative(x: "Array") -> "Array":
    """Element-wise negation (unary minus).

    Computes the negation of each element in the input array.

    Args:
        x: Input array.

    Returns:
        Array of the same shape containing -x_i for each element.
    """
    return _native_negative(x)


def square(x: "Array") -> "Array":
    """Element-wise square.

    Computes the square of each element in the input array.

    Args:
        x: Input array.

    Returns:
        Array of the same shape containing x_i**2 for each element.
    """
    return _native_square(x)


def sqrt(x: "Array") -> "Array":
    """Element-wise square root.

    Computes the square root of each element in the input array.

    Args:
        x: Input array.

    Returns:
        Array of the same shape containing sqrt(x_i) for each element.
    """
    return _native_sqrt(x)


def exp(x: "Array") -> "Array":
    """Element-wise natural exponential.

    Computes ``e^x`` for each element in the input array.

    Args:
        x: Input array.

    Returns:
        Array of the same shape containing e^(x_i) for each element.
    """
    return _native_exp(x)


def log(x: "Array") -> "Array":
    """Element-wise natural logarithm.

    Computes the natural logarithm (base e) of each element.

    Args:
        x: Input array.

    Returns:
        Array of the same shape containing ln(x_i) for each element.
    """
    return _native_log(x)


def log2(x: "Array") -> "Array":
    """Element-wise base-2 logarithm.

    Computes the base-2 logarithm of each element.

    Args:
        x: Input array.

    Returns:
        Array of the same shape containing log2(x_i) for each element.
    """
    return _native_log2(x)


def log10(x: "Array") -> "Array":
    """Element-wise base-10 logarithm.

    Computes the base-10 logarithm of each element.

    Args:
        x: Input array.

    Returns:
        Array of the same shape containing log10(x_i) for each element.
    """
    return _native_log10(x)


def sin(x: "Array") -> "Array":
    """Element-wise sine.

    Computes the trigonometric sine of each element (in radians).

    Args:
        x: Input array (values in radians).

    Returns:
        Array of the same shape containing sin(x_i) for each element.
    """
    return _native_sin(x)


def cos(x: "Array") -> "Array":
    """Element-wise cosine.

    Computes the trigonometric cosine of each element (in radians).

    Args:
        x: Input array (values in radians).

    Returns:
        Array of the same shape containing cos(x_i) for each element.
    """
    return _native_cos(x)


def tan(x: "Array") -> "Array":
    """Element-wise tangent.

    Computes the trigonometric tangent of each element (in radians).

    Args:
        x: Input array (values in radians).

    Returns:
        Array of the same shape containing tan(x_i) for each element.
    """
    return _native_tan(x)


def asin(x: "Array") -> "Array":
    """Element-wise arc sine (inverse sine).

    Computes the inverse sine of each element. The result is in radians
    and lies in the range [-pi/2, pi/2].

    Args:
        x: Input array with values in [-1, 1].

    Returns:
        Array of the same shape containing arcsin(x_i) for each element.
    """
    return _native_asin(x)


def acos(x: "Array") -> "Array":
    """Element-wise arc cosine (inverse cosine).

    Computes the inverse cosine of each element. The result is in radians
    and lies in the range [0, pi].

    Args:
        x: Input array with values in [-1, 1].

    Returns:
        Array of the same shape containing arccos(x_i) for each element.
    """
    return _native_acos(x)


def atan(x: "Array") -> "Array":
    """Element-wise arc tangent (inverse tangent).

    Computes the inverse tangent of each element. The result is in
    radians and lies in the range [-pi/2, pi/2].

    Args:
        x: Input array.

    Returns:
        Array of the same shape containing arctan(x_i) for each element.
    """
    return _native_atan(x)


def sinh(x: "Array") -> "Array":
    """Element-wise hyperbolic sine.

    Computes the hyperbolic sine of each element.

    Args:
        x: Input array.

    Returns:
        Array of the same shape containing sinh(x_i) for each element.
    """
    return _native_sinh(x)


def cosh(x: "Array") -> "Array":
    """Element-wise hyperbolic cosine.

    Computes the hyperbolic cosine of each element.

    Args:
        x: Input array.

    Returns:
        Array of the same shape containing cosh(x_i) for each element.
    """
    return _native_cosh(x)


def tanh(x: "Array") -> "Array":
    """Element-wise hyperbolic tangent.

    Computes the hyperbolic tangent of each element.

    Args:
        x: Input array.

    Returns:
        Array of the same shape containing tanh(x_i) for each element.
    """
    return _native_tanh(x)


def floor(x: "Array") -> "Array":
    """Element-wise floor (round toward negative infinity).

    Rounds each element to the largest integer less than or equal to it.

    Args:
        x: Input array.

    Returns:
        Array of the same shape with each element rounded down.
    """
    return _native_floor(x)


def ceil(x: "Array") -> "Array":
    """Element-wise ceiling (round toward positive infinity).

    Rounds each element to the smallest integer greater than or equal to
    it.

    Args:
        x: Input array.

    Returns:
        Array of the same shape with each element rounded up.
    """
    return _native_ceil(x)


def round(x: "Array") -> "Array":
    """Element-wise round to nearest integer.

    Rounds each element to the nearest integer. Halfway cases are rounded
    to the nearest even integer (banker's rounding).

    Args:
        x: Input array.

    Returns:
        Array of the same shape with each element rounded.
    """
    return _native_round(x)


def sign(x: "Array") -> "Array":
    """Element-wise sign function.

    Returns -1 for negative elements, 0 for zero, and +1 for positive
    elements.

    Args:
        x: Input array.

    Returns:
        Array of the same shape containing the sign of each element.
    """
    return _native_sign(x)


def isnan(x: "Array") -> "Array":
    """Element-wise NaN check.

    Returns True for elements that are NaN (Not a Number).

    Args:
        x: Input array.

    Returns:
        Boolean array where each element is True if the corresponding
        input element is NaN.
    """
    return _native_isnan(x)


def isinf(x: "Array") -> "Array":
    """Element-wise infinity check.

    Returns True for elements that are positive or negative infinity.

    Args:
        x: Input array.

    Returns:
        Boolean array where each element is True if the corresponding
        input element is infinite.
    """
    return _native_isinf(x)


def isfinite(x: "Array") -> "Array":
    """Element-wise finite check.

    Returns True for elements that are finite (not NaN and not infinity).

    Args:
        x: Input array.

    Returns:
        Boolean array where each element is True if the corresponding
        input element is finite.
    """
    return _native_isfinite(x)


def where(condition: "Array", x: "Array", y: "Array") -> "Array":
    """Return elements chosen from ``x`` or ``y`` depending on condition.

    For each element, returns the corresponding element from ``x`` where
    ``condition`` is True, and from ``y`` where it is False. Supports
    broadcasting.

    Args:
        condition: Boolean array used as the selection mask.
        x: Values selected where condition is True.
        y: Values selected where condition is False.

    Returns:
        Array with the same shape as the broadcasted inputs, containing
        elements from ``x`` or ``y`` based on ``condition``.
    """
    return _native_where(condition, x, y)


def exp2(x: "Array") -> "Array":
    """Element-wise base-2 exponential.

    Computes ``2^x`` for each element in the input array.

    Args:
        x: Input array.

    Returns:
        Array of the same shape containing 2^(x_i) for each element.
    """
    return _native_exp2(x)


def expm1(x: "Array") -> "Array":
    """Element-wise ``exp(x) - 1``.

    Computes ``e^x - 1`` for each element, which is more accurate than
    ``exp(x) - 1`` for small values of ``x``.

    Args:
        x: Input array.

    Returns:
        Array of the same shape containing e^(x_i) - 1 for each element.
    """
    return _native_expm1(x)


def log1p(x: "Array") -> "Array":
    """Element-wise ``log(1 + x)``.

    Computes the natural logarithm of ``1 + x``, which is more accurate
    than ``log(1 + x)`` for small values of ``x``.

    Args:
        x: Input array.

    Returns:
        Array of the same shape containing ln(1 + x_i) for each element.
    """
    return _native_log1p(x)


def cbrt(x: "Array") -> "Array":
    """Element-wise cube root.

    Computes the cube root of each element in the input array.

    Args:
        x: Input array.

    Returns:
        Array of the same shape containing the cube root of each element.
    """
    return _native_cbrt(x)


def reciprocal(x: "Array") -> "Array":
    """Element-wise reciprocal.

    Computes ``1 / x`` for each element in the input array.

    Args:
        x: Input array.

    Returns:
        Array of the same shape containing 1/x_i for each element.
    """
    return _native_reciprocal(x)


def asinh(x: "Array") -> "Array":
    """Element-wise inverse hyperbolic sine.

    Computes the inverse hyperbolic sine of each element.

    Args:
        x: Input array.

    Returns:
        Array of the same shape containing asinh(x_i) for each element.
    """
    return _native_asinh(x)


def acosh(x: "Array") -> "Array":
    """Element-wise inverse hyperbolic cosine.

    Computes the inverse hyperbolic cosine of each element.

    Args:
        x: Input array with values >= 1.

    Returns:
        Array of the same shape containing acosh(x_i) for each element.
    """
    return _native_acosh(x)


def atanh(x: "Array") -> "Array":
    """Element-wise inverse hyperbolic tangent.

    Computes the inverse hyperbolic tangent of each element.

    Args:
        x: Input array with values in (-1, 1).

    Returns:
        Array of the same shape containing atanh(x_i) for each element.
    """
    return _native_atanh(x)


def trunc(x: "Array") -> "Array":
    """Element-wise truncation toward zero.

    Rounds each element toward zero, discarding the fractional part.

    Args:
        x: Input array.

    Returns:
        Array of the same shape with each element truncated.
    """
    return _native_trunc(x)


def deg2rad(x: "Array") -> "Array":
    """Element-wise conversion from degrees to radians.

    Args:
        x: Input array with angles in degrees.

    Returns:
        Array of the same shape with angles converted to radians.
    """
    return _native_deg2rad(x)


def rad2deg(x: "Array") -> "Array":
    """Element-wise conversion from radians to degrees.

    Args:
        x: Input array with angles in radians.

    Returns:
        Array of the same shape with angles converted to degrees.
    """
    return _native_rad2deg(x)


def conj(x: "Array") -> "Array":
    """Element-wise complex conjugate.

    Computes the complex conjugate of each element. For real arrays,
    returns a copy unchanged.

    Args:
        x: Input array (real or complex).

    Returns:
        Array of the same shape containing the conjugate of each element.
    """
    return _native_conj(x)


def angle(x: "Array") -> "Array":
    """Element-wise angle (phase) of complex or real numbers.

    For complex input, returns the angle in radians in [-pi, pi].
    For real input, returns 0 for non-negative values and pi for negative.

    Args:
        x: Input array (real or complex).

    Returns:
        Float array of the same shape containing the angle of each element.
    """
    return _native_angle(x)


__all__ = [
    "abs",
    "negative",
    "square",
    "sqrt",
    "exp",
    "log",
    "log2",
    "log10",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "floor",
    "ceil",
    "round",
    "sign",
    "isnan",
    "isinf",
    "isfinite",
    "where",
    "exp2",
    "expm1",
    "log1p",
    "cbrt",
    "reciprocal",
    "asinh",
    "acosh",
    "atanh",
    "trunc",
    "deg2rad",
    "rad2deg",
    "conj",
    "angle",
]
