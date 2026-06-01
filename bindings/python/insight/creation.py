"""Array creation functions.

Provides routines for constructing new arrays filled with zeros, ones,
constants, or generated from ranges and sequences.
"""

from insight._insight import DType, CPUPlace, GPUPlace
from insight._insight import (
    zeros as _zeros,
    ones as _ones,
    full as _full,
    eye as _eye,
    arange as _arange,
    linspace as _linspace,
    logspace as _logspace,
    from_numpy as _from_numpy,
    from_array as _from_array,
    zeros_like as _zeros_like,
    ones_like as _ones_like,
    float32 as _F32,
    int64 as _I64,
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


def zeros(shape, dtype=_F32, place=None):
    """Create an array filled with zeros.

    Args:
        shape: Shape of the new array (list or tuple of ints).
        dtype: Data type of the array.  Default is ``float32``.
        place: Device placement.  Default is the current device.

    Returns:
        Array of the given shape filled with zeros.
    """
    if place is None:
        return _zeros(shape, dtype)
    return _zeros(shape, dtype, place)


def ones(shape, dtype=_F32, place=None):
    """Create an array filled with ones.

    Args:
        shape: Shape of the new array (list or tuple of ints).
        dtype: Data type of the array.  Default is ``float32``.
        place: Device placement.  Default is the current device.

    Returns:
        Array of the given shape filled with ones.
    """
    if place is None:
        return _ones(shape, dtype)
    return _ones(shape, dtype, place)


def full(shape, fill_value, dtype=_F32, place=None):
    """Create an array filled with a scalar value.

    Args:
        shape: Shape of the new array.
        fill_value: Scalar value to fill the array with.
        dtype: Data type of the array.  Default is ``float32``.
        place: Device placement.  Default is the current device.

    Returns:
        Array of the given shape filled with *fill_value*.
    """
    if place is None:
        return _full(shape, fill_value, dtype)
    return _full(shape, fill_value, dtype, place)


def eye(n, m=-1, k=0, dtype=_F32, place=None):
    """Create a 2-D identity matrix or a generalised diagonal matrix.

    Args:
        n: Number of rows.
        m: Number of columns.  Default is *n* (square matrix).
        k: Index of the diagonal.  0 is the main diagonal, positive values
            are above, negative values are below.
        dtype: Data type.  Default is ``float32``.
        place: Device placement.  Default is the current device.

    Returns:
        2-D array with ones on the *k*-th diagonal and zeros elsewhere.
    """
    if place is None:
        return _eye(n, m, k, dtype)
    return _eye(n, m, k, dtype, place)


def arange(*args, dtype=_I64, place=None):
    """Create an array with evenly spaced values within an interval.

    Calling conventions::

        arange(stop)                # start=0, step=1
        arange(start, end, step)    # explicit range

    Args:
        stop: End of the interval (exclusive) when only one positional arg
            is given.
        start: Start of the interval (inclusive).
        end: End of the interval (exclusive).
        step: Spacing between values.  Default is 1.
        dtype: Data type.  Default is ``int64``.
        place: Device placement.  Default is the current device.

    Returns:
        1-D array of evenly spaced values.
    """
    if place is not None:
        if len(args) == 1:
            return _arange(args[0], dtype, place)
        return _arange(args[0], args[1], args[2] if len(args) > 2 else 1.0,
                       dtype, place)
    if len(args) == 1:
        return _arange(args[0], dtype)
    return _arange(args[0], args[1], args[2] if len(args) > 2 else 1.0, dtype)


def linspace(start, stop, num, dtype=_F32, place=None):
    """Create an array of evenly spaced values over an interval.

    Args:
        start: Start of the interval.
        stop: End of the interval (inclusive unless *endpoint* is added).
        num: Number of samples to generate.
        dtype: Data type.  Default is ``float32``.
        place: Device placement.  Default is the current device.

    Returns:
        1-D array of *num* evenly spaced values.
    """
    if place is None:
        return _linspace(start, stop, num, dtype)
    return _linspace(start, stop, num, dtype, place)


def logspace(start, stop, num, base=10.0, dtype=_F32, place=None):
    """Create an array of logarithmically spaced values.

    Args:
        start: ``base ** start`` is the starting value.
        stop: ``base ** stop`` is the final value.
        num: Number of samples to generate.
        base: Logarithmic base.  Default is 10.
        dtype: Data type.  Default is ``float32``.
        place: Device placement.  Default is the current device.

    Returns:
        1-D array of *num* logarithmically spaced values.
    """
    if place is None:
        return _logspace(start, stop, num, base, dtype)
    return _logspace(start, stop, num, base, dtype, place)


def from_numpy(array):
    """Create an Insight Array from a NumPy array.

    Data is copied; the resulting array lives on CPU.

    Args:
        array: A NumPy ndarray.

    Returns:
        Insight Array with the same data, shape, and dtype.
    """
    return _from_numpy(array)


def from_array(data):
    """Create an Insight Array from a Python list or NumPy array.

    Nested lists are supported and will be converted to a 2-D (or higher)
    array.  Element type is inferred as ``float64`` for list inputs.

    Args:
        data: A nested Python list or a NumPy ndarray.

    Returns:
        Insight Array with the inferred shape and data.
    """
    return _from_array(data)


def zeros_like(arr):
    """Create an array of zeros with the same shape and dtype.

    Args:
        arr: Template array whose shape and dtype are copied.

    Returns:
        Array of zeros with the same shape and dtype as *arr*.
    """
    return _zeros_like(arr)


def ones_like(arr):
    """Create an array of ones with the same shape and dtype.

    Args:
        arr: Template array whose shape and dtype are copied.

    Returns:
        Array of ones with the same shape and dtype as *arr*.
    """
    return _ones_like(arr)
