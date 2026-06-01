"""Array creation functions.

Provides routines for constructing new arrays filled with zeros, ones,
constants, or generated from ranges and sequences.

Functions:
    zeros(shape, dtype, place) — Array filled with zeros
    ones(shape, dtype, place) — Array filled with ones
    full(shape, fill_value, dtype, place) — Array filled with scalar
    eye(n, m, k, dtype, place) — 2-D identity matrix
    arange(stop/start, end, step, dtype, place) — Evenly spaced values
    linspace(start, stop, num, dtype, place) — Evenly spaced over interval
    logspace(start, stop, num, base, dtype, place) — Log-spaced values
    from_numpy(array) — Create from NumPy array
    from_array(data) — Create from Python list or NumPy array
    zeros_like(arr) — Zeros with same shape/dtype
    ones_like(arr) — Ones with same shape/dtype
"""

from insight._insight import (
    zeros,
    ones,
    full,
    eye,
    arange,
    linspace,
    logspace,
    from_numpy,
    from_array,
    zeros_like,
    ones_like,
)

__all__ = [
    "zeros",
    "ones",
    "full",
    "eye",
    "arange",
    "linspace",
    "logspace",
    "from_numpy",
    "from_array",
    "zeros_like",
    "ones_like",
]
