"""Array class wrapper with documented methods.

The Array class is the core data container in Insight7. It supports
multi-dimensional indexing, arithmetic operators, device transfer,
and NumPy interop.
"""

from ._insight import Array as _NativeArray

# Monkey-patch docstrings onto native Array methods and properties.
_NativeArray.shape = property(
    _NativeArray.shape.fget,
    doc="""Shape: Tuple of dimension sizes."""
)
_NativeArray.dtype = property(
    _NativeArray.dtype.fget,
    doc="""DType: Data type of the array."""
)
_NativeArray.place = property(
    _NativeArray.place.fget,
    doc="""Place: Device placement (CPUPlace or GPUPlace)."""
)
_NativeArray.numel = property(
    _NativeArray.numel.fget,
    doc="""int: Total number of elements."""
)
_NativeArray.nbytes = property(
    _NativeArray.nbytes.fget,
    doc="""int: Total bytes of underlying data."""
)
_NativeArray.ndim = property(
    _NativeArray.ndim.fget,
    doc="""int: Number of dimensions."""
)
_NativeArray.is_contiguous = property(
    _NativeArray.is_contiguous.fget,
    doc="""bool: Whether data is contiguous in row-major (C) order."""
)
_NativeArray.defined = property(
    _NativeArray.defined.fget,
    doc="""bool: Whether the array is valid (not null)."""
)

# Method docstrings
_NativeArray.reshape.__doc__ = """Return a view with a new shape.

    Args:
        new_shape: Target shape (list of ints or Shape).

    Returns:
        Array: View with the new shape (shares data).
    """
_NativeArray.transpose.__doc__ = """Return a view with axes transposed.

    Returns:
        Array: Transposed view (reverses axes).
    """
_NativeArray.squeeze.__doc__ = """Remove single-dimensional entries from shape.

    Args:
        axis: Axis to remove (optional, removes all size-1 dims if omitted).

    Returns:
        Array: Squeezed view.
    """
_NativeArray.unsqueeze.__doc__ = """Expand the shape by inserting a dimension of size one.

    Args:
        axis: Position for the new dimension.

    Returns:
        Array: View with expanded dimensions.
    """
_NativeArray.contiguous.__doc__ = """Return a contiguous array in row-major (C) order.

    Returns:
        Array: Contiguous copy if not already contiguous, else self.
    """
_NativeArray.to.__doc__ = """Transfer array to a different device.

    Args:
        place: Target device (CPUPlace or GPUPlace).

    Returns:
        Array: Copy on the target device.
    """
_NativeArray.copy.__doc__ = """Return a deep copy of the array.

    Returns:
        Array: Independent copy with its own data.
    """
_NativeArray.numpy.__doc__ = """Convert to a NumPy array (CPU only).

    Returns:
        numpy.ndarray: Copy of the data as a NumPy array.
    """

Array = _NativeArray

__all__ = ["Array"]
