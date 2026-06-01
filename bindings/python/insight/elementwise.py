"""Element-wise binary operations.

Provides arithmetic, comparison, and logical operations that operate
element-by-element on two arrays (broadcasting as needed).
"""

from insight._insight import (
    add as _add,
    sub as _sub,
    mul as _mul,
    div as _div,
    pow as _pow,
    mod as _mod,
    maximum as _maximum,
    minimum as _minimum,
    equal as _equal,
    not_equal as _not_equal,
    greater as _greater,
    less as _less,
    greater_equal as _greater_equal,
    less_equal as _less_equal,
    logical_and as _logical_and,
    logical_or as _logical_or,
    logical_xor as _logical_xor,
    logical_not as _logical_not,
)

__all__ = [
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "mod",
    "maximum",
    "minimum",
    "equal",
    "not_equal",
    "greater",
    "less",
    "greater_equal",
    "less_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
]


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def add(a, b):
    """Element-wise addition.

    Args:
        a: First input array.
        b: Second input array (broadcast-compatible with *a*).

    Returns:
        Array containing ``a + b``.
    """
    return _add(a, b)


def sub(a, b):
    """Element-wise subtraction.

    Args:
        a: First input array.
        b: Second input array (broadcast-compatible with *a*).

    Returns:
        Array containing ``a - b``.
    """
    return _sub(a, b)


def mul(a, b):
    """Element-wise multiplication.

    Args:
        a: First input array.
        b: Second input array (broadcast-compatible with *a*).

    Returns:
        Array containing ``a * b``.
    """
    return _mul(a, b)


def div(a, b):
    """Element-wise true division.

    Args:
        a: Dividend array.
        b: Divisor array (broadcast-compatible with *a*).

    Returns:
        Array containing ``a / b``.
    """
    return _div(a, b)


def pow(a, b):
    """Element-wise exponentiation.

    Args:
        a: Base array.
        b: Exponent array (broadcast-compatible with *a*).

    Returns:
        Array containing ``a ** b``.
    """
    return _pow(a, b)


def mod(a, b):
    """Element-wise modulus.

    Args:
        a: Dividend array.
        b: Divisor array (broadcast-compatible with *a*).

    Returns:
        Array containing ``a % b``.
    """
    return _mod(a, b)


def maximum(a, b):
    """Element-wise maximum of two arrays.

    Args:
        a: First input array.
        b: Second input array (broadcast-compatible with *a*).

    Returns:
        Array containing the element-wise maximum.
    """
    return _maximum(a, b)


def minimum(a, b):
    """Element-wise minimum of two arrays.

    Args:
        a: First input array.
        b: Second input array (broadcast-compatible with *a*).

    Returns:
        Array containing the element-wise minimum.
    """
    return _minimum(a, b)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def equal(a, b):
    """Element-wise equality comparison.

    Args:
        a: First input array.
        b: Second input array (broadcast-compatible with *a*).

    Returns:
        Boolean array where ``True`` indicates equal elements.
    """
    return _equal(a, b)


def not_equal(a, b):
    """Element-wise inequality comparison.

    Args:
        a: First input array.
        b: Second input array (broadcast-compatible with *a*).

    Returns:
        Boolean array where ``True`` indicates unequal elements.
    """
    return _not_equal(a, b)


def greater(a, b):
    """Element-wise greater-than comparison.

    Args:
        a: First input array.
        b: Second input array (broadcast-compatible with *a*).

    Returns:
        Boolean array where ``True`` indicates ``a > b``.
    """
    return _greater(a, b)


def less(a, b):
    """Element-wise less-than comparison.

    Args:
        a: First input array.
        b: Second input array (broadcast-compatible with *a*).

    Returns:
        Boolean array where ``True`` indicates ``a < b``.
    """
    return _less(a, b)


def greater_equal(a, b):
    """Element-wise greater-or-equal comparison.

    Args:
        a: First input array.
        b: Second input array (broadcast-compatible with *a*).

    Returns:
        Boolean array where ``True`` indicates ``a >= b``.
    """
    return _greater_equal(a, b)


def less_equal(a, b):
    """Element-wise less-or-equal comparison.

    Args:
        a: First input array.
        b: Second input array (broadcast-compatible with *a*).

    Returns:
        Boolean array where ``True`` indicates ``a <= b``.
    """
    return _less_equal(a, b)


# ---------------------------------------------------------------------------
# Logical
# ---------------------------------------------------------------------------


def logical_and(a, b):
    """Element-wise logical AND.

    Args:
        a: First boolean array.
        b: Second boolean array (broadcast-compatible with *a*).

    Returns:
        Boolean array where ``True`` indicates both inputs are truthy.
    """
    return _logical_and(a, b)


def logical_or(a, b):
    """Element-wise logical OR.

    Args:
        a: First boolean array.
        b: Second boolean array (broadcast-compatible with *a*).

    Returns:
        Boolean array where ``True`` indicates at least one input is truthy.
    """
    return _logical_or(a, b)


def logical_xor(a, b):
    """Element-wise logical XOR.

    Args:
        a: First boolean array.
        b: Second boolean array (broadcast-compatible with *a*).

    Returns:
        Boolean array where ``True`` indicates exactly one input is truthy.
    """
    return _logical_xor(a, b)


def logical_not(x):
    """Element-wise logical NOT.

    Args:
        x: Input boolean array.

    Returns:
        Boolean array with each element inverted.
    """
    return _logical_not(x)
