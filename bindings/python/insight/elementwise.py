"""Elementwise binary operations.

Functions:
    add(a, b), sub(a, b), mul(a, b), div(a, b), pow(a, b), mod(a, b)
    maximum(a, b), minimum(a, b)
    equal(a, b), not_equal(a, b), greater(a, b), less(a, b)
    greater_equal(a, b), less_equal(a, b)
    logical_and(a, b), logical_or(a, b), logical_xor(a, b)
    logical_not(x)
"""

from insight._insight import (
    add,
    sub,
    mul,
    div,
    pow,
    mod,
    maximum,
    minimum,
    equal,
    not_equal,
    greater,
    less,
    greater_equal,
    less_equal,
    logical_and,
    logical_or,
    logical_xor,
    logical_not,
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
