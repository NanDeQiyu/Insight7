"""Array manipulation operations for Insight arrays.

Provides reshaping, joining, splitting, transposing, padding,
and other structural transformations.

All functions return an :class:`insight.Array`.
"""

from insight._insight import (
    concat as _concat,
    stack as _stack,
    vstack as _vstack,
    hstack as _hstack,
    split as _split,
    tile as _tile,
    repeat as _repeat,
    flip as _flip,
    pad as _pad,
    reshape as _reshape,
    flatten as _flatten,
    ravel as _ravel,
    squeeze as _squeeze,
    unsqueeze as _unsqueeze,
    roll as _roll,
    permute as _permute,
    swapaxes as _swapaxes,
    moveaxis as _moveaxis,
    fliplr as _fliplr,
    flipud as _flipud,
    rot90 as _rot90,
    diag as _diag,
    diagonal as _diagonal,
    tril as _tril,
    triu as _triu,
    diff as _diff,
)

__all__ = [
    "concat",
    "stack",
    "vstack",
    "hstack",
    "split",
    "tile",
    "repeat",
    "flip",
    "pad",
    "reshape",
    "flatten",
    "ravel",
    "squeeze",
    "unsqueeze",
    "roll",
    "permute",
    "swapaxes",
    "moveaxis",
    "fliplr",
    "flipud",
    "rot90",
    "diag",
    "diagonal",
    "tril",
    "triu",
    "diff",
]


def concat(arrays, axis=0):
    """Concatenate arrays along an existing axis.

    Args:
        arrays: Sequence of arrays to concatenate.  All arrays must
            have the same shape except in the concatenation dimension.
        axis: Axis along which the arrays will be joined.
            Default is 0.

    Returns:
        Concatenated array.
    """
    return _concat(arrays, axis=axis)


def stack(arrays, axis=0):
    """Stack arrays along a new axis.

    Args:
        arrays: Sequence of arrays with the same shape.
        axis: Index of the new axis.  Default is 0.

    Returns:
        Stacked array with one more dimension than the inputs.
    """
    return _stack(arrays, axis=axis)


def vstack(arrays):
    """Stack arrays vertically (row-wise).

    Equivalent to ``concat(arrays, axis=0)`` for 1-D and 2-D arrays.
    Arrays with fewer dimensions are prepended with a new axis.

    Args:
        arrays: Sequence of arrays to stack.

    Returns:
        Vertically stacked array.
    """
    return _vstack(arrays)


def hstack(arrays):
    """Stack arrays horizontally (column-wise).

    Equivalent to ``concat(arrays, axis=1)`` for 2-D arrays, or
    ``concat(arrays, axis=0)`` for 1-D arrays.

    Args:
        arrays: Sequence of arrays to stack.

    Returns:
        Horizontally stacked array.
    """
    return _hstack(arrays)


def split(x, sections, axis=0):
    """Split an array into sub-arrays.

    Args:
        x: Input array.
        sections: Number of equal-sized sub-arrays to create, or a
            list of indices at which to split.
        axis: Axis along which to split.  Default is 0.

    Returns:
        List of sub-arrays.
    """
    return _split(x, sections, axis=axis)


def tile(x, reps):
    """Construct an array by repeating *x* according to *reps*.

    Args:
        x: Input array (or scalar).
        reps: Sequence of repetition counts for each dimension.

    Returns:
        Tiled array.
    """
    return _tile(x, reps)


def repeat(x, repeats, axis=None):
    """Repeat elements of an array.

    Args:
        x: Input array.
        repeats: Number of repetitions for each element.
        axis: Axis along which to repeat.  ``None`` (default) repeats
            over the flattened array.

    Returns:
        Array with repeated elements.
    """
    return _repeat(x, repeats, axis=axis)


def flip(x, axis=None):
    """Reverse the order of elements along an axis.

    Args:
        x: Input array.
        axis: Axis or axes along which to flip.  ``None`` (default)
            reverses all axes.

    Returns:
        Flipped array.
    """
    return _flip(x, axis=axis)


def pad(x, pad_width, constant_value=0.0):
    """Pad an array with a constant value.

    Args:
        x: Input array.
        pad_width: Number of values padded to the edges of each axis.
            Specified as ``((before_1, after_1), ..., (before_N, after_N))``.
        constant_value: Value to pad with.  Default is 0.0.

    Returns:
        Padded array.
    """
    return _pad(x, pad_width, constant_value=constant_value)


def reshape(x, new_shape):
    """Reshape an array without changing its data.

    Args:
        x: Input array.
        new_shape: Desired shape.  One dimension may be -1, in which
            case it is inferred from the remaining dimensions.

    Returns:
        Reshaped array.
    """
    return _reshape(x, new_shape)


def flatten(x):
    """Flatten an array into a 1-D contiguous array.

    Args:
        x: Input array.

    Returns:
        1-D contiguous array.
    """
    return _flatten(x)


def ravel(x):
    """Return a contiguous flattened array.

    Args:
        x: Input array.

    Returns:
        1-D contiguous array.
    """
    return _ravel(x)


def squeeze(x, axis=None):
    """Remove single-dimensional entries from array shape.

    Args:
        x: Input array.
        axis: Select a subset of single-dimensional entries to remove.
            ``None`` (default) removes all single-dimensional entries.

    Returns:
        Squeezed array.
    """
    if axis is None:
        return _squeeze(x)
    return _squeeze(x, axis)


def unsqueeze(x, axis):
    """Expand array shape by inserting a new axis.

    Args:
        x: Input array.
        axis: Position where the new axis will be inserted.

    Returns:
        Array with an additional dimension of size one.
    """
    return _unsqueeze(x, axis)


def roll(x, shift, axis=None):
    """Roll array elements along an axis.

    Elements that roll beyond the last position are re-introduced at
    the first position (and vice versa).

    Args:
        x: Input array.
        shift: Number of places by which elements are shifted.  If
            negative, elements are shifted to the left.
        axis: Axis along which to roll.  ``None`` (default) rolls the
            flattened array.

    Returns:
        Rolled array.
    """
    return _roll(x, shift, axis=axis)


def permute(x, axes):
    """Permute the axes of an array.

    Args:
        x: Input array.
        axes: Permutation of ``range(ndim)``.

    Returns:
        Array with permuted axes.
    """
    return _permute(x, axes)


def swapaxes(x, axis1, axis2):
    """Interchange two axes of an array.

    Args:
        x: Input array.
        axis1: First axis.
        axis2: Second axis.

    Returns:
        Array with the two axes swapped.
    """
    return _swapaxes(x, axis1, axis2)


def moveaxis(x, source, destination):
    """Move axes of an array to new positions.

    Args:
        x: Input array.
        source: Original positions of the axes to move.
        destination: Destination positions for each axis.

    Returns:
        Array with moved axes.
    """
    return _moveaxis(x, source, destination)


def fliplr(x):
    """Flip array in the left/right direction.

    Columns are reversed; the row order is preserved.

    Args:
        x: Input array (must have at least 2 dimensions).

    Returns:
        Flipped array.
    """
    return _fliplr(x)


def flipud(x):
    """Flip array in the up/down direction.

    Rows are reversed; the column order is preserved.

    Args:
        x: Input array (must have at least 2 dimensions).

    Returns:
        Flipped array.
    """
    return _flipud(x)


def rot90(x, k=1, axes=None):
    """Rotate array by 90 degrees in the plane specified by *axes*.

    Args:
        x: Input array.
        k: Number of times the array is rotated by 90 degrees.
            Default is 1.
        axes: The two axes that define the plane of rotation.
            Default is ``[0, 1]``.

    Returns:
        Rotated array.
    """
    if axes is None:
        axes = [0, 1]
    return _rot90(x, k=k, axes=axes)


def diag(x, k=0):
    """Extract a diagonal or construct a diagonal array.

    Args:
        x: Input array.  If 1-D, a diagonal matrix is constructed.
            If 2-D, the k-th diagonal is extracted.
        k: Diagonal offset.  ``0`` (default) is the main diagonal.
            Positive values are above, negative below.

    Returns:
        Diagonal array or 1-D array of diagonal elements.
    """
    return _diag(x, k=k)


def diagonal(x, offset=0, axis1=0, axis2=1):
    """Return specified diagonals.

    Args:
        x: Input array.
        offset: Offset of the diagonal from the main diagonal.
            ``0`` (default) is the main diagonal.
        axis1: First axis of the 2-D sub-arrays.  Default is 0.
        axis2: Second axis of the 2-D sub-arrays.  Default is 1.

    Returns:
        Array of diagonals.
    """
    return _diagonal(x, offset=offset, axis1=axis1, axis2=axis2)


def tril(x, k=0):
    """Lower triangular part of an array.

    Args:
        x: Input array.
        k: Diagonal above which to zero elements.  ``0`` (default) is
            the main diagonal.  Positive values include more diagonals.

    Returns:
        Lower triangular array.
    """
    return _tril(x, k=k)


def triu(x, k=0):
    """Upper triangular part of an array.

    Args:
        x: Input array.
        k: Diagonal below which to zero elements.  ``0`` (default) is
            the main diagonal.  Negative values include more diagonals.

    Returns:
        Upper triangular array.
    """
    return _triu(x, k=k)


def diff(x, n=1, axis=-1):
    """Compute the n-th discrete difference along an axis.

    Args:
        x: Input array.
        n: Number of times to apply the difference.  Default is 1.
        axis: Axis along which the difference is taken.  Default is -1
            (last axis).

    Returns:
        Array of differences.
    """
    return _diff(x, n=n, axis=axis)
