"""Array creation functions.

Functions:
    zeros(shape, dtype, place) — Array filled with zeros
    ones(shape, dtype, place) — Array filled with ones
    full(shape, fill_value, dtype, place) — Array filled with scalar
    eye(n, m, k, dtype, place) — 2-D identity matrix
    arange(stop, dtype, place) — Evenly spaced values
    linspace(start, stop, num, dtype, place) — Evenly spaced over interval
    from_numpy(array) — Create from NumPy array
    from_array(data) — Create from Python list or NumPy array
    zeros_like(arr) — Zeros with same shape/dtype
    ones_like(arr) — Ones with same shape/dtype
"""

from insight._insight import zeros as _native_zeros
from insight._insight import ones as _native_ones
from insight._insight import full as _native_full
from insight._insight import Shape, DType, CPUPlace

# Import remaining functions directly (no Shape conversion needed)
from insight._insight import (
    eye,
    arange,
    linspace,
    logspace,
    from_numpy,
    from_array,
    zeros_like,
    ones_like,
)


def _to_shape(s):
    if isinstance(s, (list, tuple)):
        return Shape(list(s))
    return s


def zeros(shape, dtype=DType.float32, place=None):
    """Create an array filled with zeros.

    Args:
        shape: Shape as list of ints or Shape object.
        dtype: Data type (default float32).
        place: Device placement (default CPU).

    Returns:
        Array of zeros.
    """
    s = _to_shape(shape)
    if place is None:
        return _native_zeros(s, dtype)
    return _native_zeros(s, dtype, place)


def ones(shape, dtype=DType.float32, place=None):
    """Create an array filled with ones.

    Args:
        shape: Shape as list of ints or Shape object.
        dtype: Data type (default float32).
        place: Device placement (default CPU).

    Returns:
        Array of ones.
    """
    s = _to_shape(shape)
    if place is None:
        return _native_ones(s, dtype)
    return _native_ones(s, dtype, place)


def full(shape, fill_value, dtype=DType.float32, place=None):
    """Create an array filled with a scalar value.

    Args:
        shape: Shape as list of ints or Shape object.
        fill_value: Scalar fill value.
        dtype: Data type (default float32).
        place: Device placement (default CPU).

    Returns:
        Array filled with fill_value.
    """
    s = _to_shape(shape)
    if place is None:
        return _native_full(s, fill_value, dtype)
    return _native_full(s, fill_value, dtype, place)


__all__ = [
    "zeros", "ones", "full", "eye", "arange", "linspace", "logspace",
    "from_numpy", "from_array", "zeros_like", "ones_like",
]
