"""Complex number operations.

Functions:
    is_complex, has_complex_shape, to_complex, as_complex
    as_real, real, imag
"""

from __future__ import annotations

from insight._insight import Array  # noqa: F401
from insight._insight import (
    is_complex as _native_is_complex,
    has_complex_shape as _native_has_complex_shape,
    to_complex as _native_to_complex,
    as_complex as _native_as_complex,
    as_real as _native_as_real,
    real as _native_real,
    imag as _native_imag,
)


def is_complex(x: "Array") -> bool:
    """Check whether an array has a complex data type.

    Args:
        x: Input array.

    Returns:
        True if the array's dtype is complex64 or complex128.
    """
    return _native_is_complex(x)


def has_complex_shape(x: "Array") -> bool:
    """Check whether an array uses legacy complex storage format.

    The legacy format stores real and imaginary parts as the last
    dimension of size 2, using a real-valued dtype (e.g., float32).

    Args:
        x: Input array.

    Returns:
        True if the last dimension exists and has size 2.
    """
    return _native_has_complex_shape(x)


def to_complex(real: "Array", imag: "Array | None" = None) -> "Array":
    """Convert real-valued array(s) to a complex array.

    If only ``real`` is provided, the imaginary part is set to zero.
    If both ``real`` and ``imag`` are provided, the result is
    ``real + imag * j``.

    Args:
        real: Real part array.
        imag: Imaginary part array (optional). If provided, must have
            the same shape as ``real``.

    Returns:
        Complex array with shape ``(*real.shape, 2)`` (interleaved
        real/imag storage).
    """
    if imag is None:
        return _native_to_complex(real)
    return _native_to_complex(real, imag)


def as_complex(x: "Array") -> "Array":
    """View a real array as a complex array (zero-copy).

    The input must have its last dimension equal to 2 (interleaved
    real and imaginary parts). The dtype is promoted from F32 to C32
    or from F64 to C64.

    Args:
        x: Real array with shape ``(..., 2)``.

    Returns:
        Complex array view with shape ``(...)``.
    """
    return _native_as_complex(x)


def as_real(x: "Array") -> "Array":
    """View a complex array as a real array (zero-copy).

    The output gains an extra last dimension of size 2, containing the
    interleaved real and imaginary parts.

    Args:
        x: Complex array (dtype C32 or C64).

    Returns:
        Real array view with shape ``(*x.shape, 2)``.
    """
    return _native_as_real(x)


def real(z: "Array") -> "Array":
    """Extract the real part of a complex array.

    Args:
        z: Complex array with shape ``(..., 2)``.

    Returns:
        Real array view with the last dimension removed, containing
        only the real parts.
    """
    return _native_real(z)


def imag(z: "Array") -> "Array":
    """Extract the imaginary part of a complex array.

    Args:
        z: Complex array with shape ``(..., 2)``.

    Returns:
        Real array view with the last dimension removed, containing
        only the imaginary parts.
    """
    return _native_imag(z)


__all__ = [
    "is_complex",
    "has_complex_shape",
    "to_complex",
    "as_complex",
    "as_real",
    "real",
    "imag",
]
