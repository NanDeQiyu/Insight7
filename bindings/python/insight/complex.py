"""Complex number operations.

Provides functions for creating, querying, and decomposing
complex-valued arrays.
"""

from insight._insight import (
    is_complex as _is_complex,
    has_complex_shape as _has_complex_shape,
    to_complex as _to_complex,
    as_complex as _as_complex,
    as_real as _as_real,
    real as _real,
    imag as _imag,
)

__all__ = [
    "is_complex",
    "has_complex_shape",
    "to_complex",
    "as_complex",
    "as_real",
    "real",
    "imag",
]


def is_complex(x):
    """Check whether an array has a complex dtype.

    Args:
        x: Input array.

    Returns:
        ``True`` if *x* has dtype ``complex64`` or ``complex128``.
    """
    return _is_complex(x)


def has_complex_shape(x):
    """Check whether an array uses legacy complex storage.

    In the legacy layout the last dimension is 2, storing real and
    imaginary parts separately.

    Args:
        x: Input array.

    Returns:
        ``True`` if the last dimension is 2 and the dtype is real.
    """
    return _has_complex_shape(x)


def to_complex(real, imag=None):
    """Convert a real array to a complex array.

    If only *real* is provided, the imaginary part is set to zero.

    Args:
        real: Real-valued input array.
        imag: Optional imaginary-part array.  Must be broadcastable
            to the shape of *real*.

    Returns:
        Complex-valued array.
    """
    if imag is None:
        return _to_complex(real)
    return _to_complex(real, imag)


def as_complex(x):
    """View a real array with last dimension 2 as complex (zero-copy).

    The input array must have a real dtype and its last dimension
    must equal 2.  The result is a complex array of half the element
    count sharing the same memory.

    Args:
        x: Input real array whose last dimension is 2.

    Returns:
        Complex array view.
    """
    return _as_complex(x)


def as_real(x):
    """View a complex array as real with last dimension 2 (zero-copy).

    The result is a real array whose last dimension is 2, containing
    the real and imaginary parts.  Memory is shared with the input.

    Args:
        x: Input complex array.

    Returns:
        Real array view with shape ``(..., 2)``.
    """
    return _as_real(x)


def real(z):
    """Extract the real part of a complex array.

    Args:
        z: Input complex-valued array.

    Returns:
        Real-valued array with the same shape as *z*.
    """
    return _real(z)


def imag(z):
    """Extract the imaginary part of a complex array.

    Args:
        z: Input complex-valued array.

    Returns:
        Real-valued array with the same shape as *z*.
    """
    return _imag(z)
