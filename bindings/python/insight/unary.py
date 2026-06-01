"""Unary math operations.

Provides element-wise mathematical functions that operate on a single
array, including trigonometric, hyperbolic, rounding, and special
classification functions.
"""

from insight._insight import (
    abs as _abs,
    negative as _negative,
    square as _square,
    sqrt as _sqrt,
    exp as _exp,
    log as _log,
    log2 as _log2,
    log10 as _log10,
    sin as _sin,
    cos as _cos,
    tan as _tan,
    asin as _asin,
    acos as _acos,
    atan as _atan,
    sinh as _sinh,
    cosh as _cosh,
    tanh as _tanh,
    floor as _floor,
    ceil as _ceil,
    round as _round,
    sign as _sign,
    isnan as _isnan,
    isinf as _isinf,
    isfinite as _isfinite,
    exp2 as _exp2,
    expm1 as _expm1,
    log1p as _log1p,
    cbrt as _cbrt,
    reciprocal as _reciprocal,
    asinh as _asinh,
    acosh as _acosh,
    atanh as _atanh,
    trunc as _trunc,
    deg2rad as _deg2rad,
    rad2deg as _rad2deg,
    where as _where,
)

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
    "where",
]


# ---------------------------------------------------------------------------
# Basic math
# ---------------------------------------------------------------------------


def abs(x):
    """Element-wise absolute value.

    Args:
        x: Input array.

    Returns:
        Array of absolute values with the same shape.
    """
    return _abs(x)


def negative(x):
    """Element-wise negation (``-x``).

    Args:
        x: Input array.

    Returns:
        Array with each element negated.
    """
    return _negative(x)


def square(x):
    """Element-wise square (``x ** 2``).

    Args:
        x: Input array.

    Returns:
        Array of squared values.
    """
    return _square(x)


def sqrt(x):
    """Element-wise square root.

    Args:
        x: Input array.

    Returns:
        Array of square roots.
    """
    return _sqrt(x)


def exp(x):
    """Element-wise natural exponential (``e ** x``).

    Args:
        x: Input array.

    Returns:
        Array of exponentials.
    """
    return _exp(x)


def log(x):
    """Element-wise natural logarithm.

    Args:
        x: Input array.

    Returns:
        Array of natural logarithms.
    """
    return _log(x)


def log2(x):
    """Element-wise base-2 logarithm.

    Args:
        x: Input array.

    Returns:
        Array of base-2 logarithms.
    """
    return _log2(x)


def log10(x):
    """Element-wise base-10 logarithm.

    Args:
        x: Input array.

    Returns:
        Array of base-10 logarithms.
    """
    return _log10(x)


# ---------------------------------------------------------------------------
# Trigonometric
# ---------------------------------------------------------------------------


def sin(x):
    """Element-wise sine.

    Args:
        x: Input array (radians).

    Returns:
        Array of sine values.
    """
    return _sin(x)


def cos(x):
    """Element-wise cosine.

    Args:
        x: Input array (radians).

    Returns:
        Array of cosine values.
    """
    return _cos(x)


def tan(x):
    """Element-wise tangent.

    Args:
        x: Input array (radians).

    Returns:
        Array of tangent values.
    """
    return _tan(x)


def asin(x):
    """Element-wise arc sine.

    Args:
        x: Input array.

    Returns:
        Array of arc sine values in radians.
    """
    return _asin(x)


def acos(x):
    """Element-wise arc cosine.

    Args:
        x: Input array.

    Returns:
        Array of arc cosine values in radians.
    """
    return _acos(x)


def atan(x):
    """Element-wise arc tangent.

    Args:
        x: Input array.

    Returns:
        Array of arc tangent values in radians.
    """
    return _atan(x)


# ---------------------------------------------------------------------------
# Hyperbolic
# ---------------------------------------------------------------------------


def sinh(x):
    """Element-wise hyperbolic sine.

    Args:
        x: Input array.

    Returns:
        Array of hyperbolic sine values.
    """
    return _sinh(x)


def cosh(x):
    """Element-wise hyperbolic cosine.

    Args:
        x: Input array.

    Returns:
        Array of hyperbolic cosine values.
    """
    return _cosh(x)


def tanh(x):
    """Element-wise hyperbolic tangent.

    Args:
        x: Input array.

    Returns:
        Array of hyperbolic tangent values.
    """
    return _tanh(x)


# ---------------------------------------------------------------------------
# Rounding / sign
# ---------------------------------------------------------------------------


def floor(x):
    """Element-wise floor (round toward negative infinity).

    Args:
        x: Input array.

    Returns:
        Array of floored values.
    """
    return _floor(x)


def ceil(x):
    """Element-wise ceiling (round toward positive infinity).

    Args:
        x: Input array.

    Returns:
        Array of ceiling values.
    """
    return _ceil(x)


def round(x):
    """Element-wise rounding to nearest integer (banker's rounding).

    Args:
        x: Input array.

    Returns:
        Array of rounded values.
    """
    return _round(x)


def sign(x):
    """Element-wise sign function.

    Returns ``-1`` for negative, ``0`` for zero, ``1`` for positive.

    Args:
        x: Input array.

    Returns:
        Array of sign values.
    """
    return _sign(x)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def isnan(x):
    """Element-wise NaN check.

    Args:
        x: Input array.

    Returns:
        Boolean array where ``True`` indicates a NaN element.
    """
    return _isnan(x)


def isinf(x):
    """Element-wise infinity check.

    Args:
        x: Input array.

    Returns:
        Boolean array where ``True`` indicates an infinite element.
    """
    return _isinf(x)


def isfinite(x):
    """Element-wise finite check.

    Args:
        x: Input array.

    Returns:
        Boolean array where ``True`` indicates a finite element.
    """
    return _isfinite(x)


# ---------------------------------------------------------------------------
# Extended math
# ---------------------------------------------------------------------------


def exp2(x):
    """Element-wise base-2 exponential (``2 ** x``).

    Args:
        x: Input array.

    Returns:
        Array of base-2 exponentials.
    """
    return _exp2(x)


def expm1(x):
    """Element-wise ``exp(x) - 1``, accurate for small *x*.

    Args:
        x: Input array.

    Returns:
        Array of ``exp(x) - 1`` values.
    """
    return _expm1(x)


def log1p(x):
    """Element-wise ``log(1 + x)``, accurate for small *x*.

    Args:
        x: Input array.

    Returns:
        Array of ``log(1 + x)`` values.
    """
    return _log1p(x)


def cbrt(x):
    """Element-wise cube root.

    Args:
        x: Input array.

    Returns:
        Array of cube roots.
    """
    return _cbrt(x)


def reciprocal(x):
    """Element-wise reciprocal (``1 / x``).

    Args:
        x: Input array.

    Returns:
        Array of reciprocals.
    """
    return _reciprocal(x)


# ---------------------------------------------------------------------------
# Inverse hyperbolic
# ---------------------------------------------------------------------------


def asinh(x):
    """Element-wise inverse hyperbolic sine.

    Args:
        x: Input array.

    Returns:
        Array of inverse hyperbolic sine values.
    """
    return _asinh(x)


def acosh(x):
    """Element-wise inverse hyperbolic cosine.

    Args:
        x: Input array.

    Returns:
        Array of inverse hyperbolic cosine values.
    """
    return _acosh(x)


def atanh(x):
    """Element-wise inverse hyperbolic tangent.

    Args:
        x: Input array.

    Returns:
        Array of inverse hyperbolic tangent values.
    """
    return _atanh(x)


def trunc(x):
    """Element-wise truncation toward zero.

    Args:
        x: Input array.

    Returns:
        Array of truncated values.
    """
    return _trunc(x)


def deg2rad(x):
    """Convert angles from degrees to radians element-wise.

    Args:
        x: Input array (degrees).

    Returns:
        Array of angles in radians.
    """
    return _deg2rad(x)


def rad2deg(x):
    """Convert angles from radians to degrees element-wise.

    Args:
        x: Input array (radians).

    Returns:
        Array of angles in degrees.
    """
    return _rad2deg(x)


# ---------------------------------------------------------------------------
# Conditional
# ---------------------------------------------------------------------------


def where(cond, x, y):
    """Select elements from *x* or *y* depending on *cond*.

    For each element: if the corresponding element in *cond* is truthy
    the result takes from *x*, otherwise from *y*.

    Args:
        cond: Boolean condition array.
        x: Array selected where *cond* is ``True``.
        y: Array selected where *cond* is ``False``.

    Returns:
        Array with elements chosen from *x* or *y*.
    """
    return _where(cond, x, y)
