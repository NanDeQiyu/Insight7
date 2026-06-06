"""Array manipulation operations.

Functions:
    concat, stack, vstack, hstack, split, tile, repeat
    flip, pad, reshape, flatten, ravel, squeeze, unsqueeze
    roll, permute, swapaxes, moveaxis
    fliplr, flipud, rot90, diag, diagonal, tril, triu, diff

Note: Array.transpose() is available as a method on the Array class.
Use arr.T for quick transpose, or ins.permute(arr, axes) for custom axes.
"""

from __future__ import annotations

from typing import Optional

from insight._insight import Array, Shape  # noqa: F401
from insight._insight import (
    concat as _native_concat,
    stack as _native_stack,
    vstack as _native_vstack,
    hstack as _native_hstack,
    split as _native_split,
    tile as _native_tile,
    repeat as _native_repeat,
    flip as _native_flip,
    pad as _native_pad,
    flatten as _native_flatten,
    ravel as _native_ravel,
    squeeze as _native_squeeze,
    unsqueeze as _native_unsqueeze,
    roll as _native_roll,
    permute as _native_permute,
    swapaxes as _native_swapaxes,
    moveaxis as _native_moveaxis,
    fliplr as _native_fliplr,
    flipud as _native_flipud,
    rot90 as _native_rot90,
    diag as _native_diag,
    diagonal as _native_diagonal,
    tril as _native_tril,
    triu as _native_triu,
    diff as _native_diff,
)
from insight._insight import reshape as _native_reshape


def reshape(x, new_shape):
    """Reshape an array to a new shape.

    Returns a view of the array with the new shape. The total number of
    elements must remain the same.

    Args:
        x: Input array.
        new_shape: Target shape as a list, tuple, or Shape object.

    Returns:
        View of the array with the new shape (shares data with the
        original array).
    """
    if isinstance(new_shape, (list, tuple)):
        new_shape = Shape(list(new_shape))
    return _native_reshape(x, new_shape)


def flatten(x: "Array") -> "Array":
    """Flatten an array to one dimension.

    Returns a copy of the array collapsed into a single dimension.

    Args:
        x: Input array.

    Returns:
        1-D array containing all elements of the input.
    """
    return _native_flatten(x)


def ravel(x: "Array") -> "Array":
    """Flatten an array to one dimension.

    Returns a contiguous flattened array. May return a view if the
    input is already contiguous, otherwise returns a copy.

    Args:
        x: Input array.

    Returns:
        1-D array containing all elements of the input.
    """
    return _native_ravel(x)


def squeeze(x: "Array", axis: Optional[int] = None) -> "Array":
    """Remove dimensions of size 1 from the array shape.

    Args:
        x: Input array.
        axis: If given, only remove the specified axis if its size is 1.
            If None, all dimensions of size 1 are removed.

    Returns:
        View of the array with the size-1 dimensions removed.
    """
    if axis is None:
        return _native_squeeze(x)
    return _native_squeeze(x, axis)


def unsqueeze(x: "Array", axis: int) -> "Array":
    """Insert a new dimension of size 1 at the given axis position.

    Args:
        x: Input array.
        axis: Position at which to insert the new dimension.

    Returns:
        View of the array with the additional dimension.
    """
    return _native_unsqueeze(x, axis)


def transpose(x: "Array") -> "Array":
    """Transpose the array (reverse all axes).

    Returns a view of the array with the axes reversed. For a 2-D
    array, this is the standard matrix transpose.

    Args:
        x: Input array.

    Returns:
        View of the array with reversed axes.
    """
    return x.transpose()


def permute(x: "Array", axes: list) -> "Array":
    """Permute the dimensions of the array.

    Returns a view with the axes reordered according to ``axes``.

    Args:
        x: Input array.
        axes: Desired ordering of axes. Must be a permutation of
            ``range(x.ndim)``.

    Returns:
        View of the array with the permuted axes.
    """
    return _native_permute(x, axes)


def concat(arrays: list, axis: int = 0) -> "Array":
    """Concatenate arrays along an existing axis.

    Joins the given arrays along the specified axis. All arrays must
    have the same shape except along the concatenation axis.

    Args:
        arrays: List of arrays to concatenate.
        axis: Axis along which to concatenate (default: 0).

    Returns:
        Concatenated array.
    """
    return _native_concat(arrays, axis)


def stack(arrays: list, axis: int = 0) -> "Array":
    """Stack arrays along a new axis.

    Joins the given arrays along a new axis. All arrays must have the
    same shape.

    Args:
        arrays: List of arrays to stack.
        axis: Position of the new axis in the result (default: 0).

    Returns:
        Stacked array with one more dimension than the inputs.
    """
    return _native_stack(arrays, axis)


def vstack(arrays: list) -> "Array":
    """Stack arrays vertically (row-wise).

    Equivalent to ``concat(arrays, axis=0)``.

    Args:
        arrays: List of arrays to stack vertically.

    Returns:
        Vertically stacked array.
    """
    return _native_vstack(arrays)


def hstack(arrays: list) -> "Array":
    """Stack arrays horizontally (column-wise).

    Equivalent to ``concat(arrays, axis=1)``.

    Args:
        arrays: List of arrays to stack horizontally.

    Returns:
        Horizontally stacked array.
    """
    return _native_hstack(arrays)


def split(x: "Array", sections: int, axis: int = 0) -> list:
    """Split an array into multiple sub-arrays along the given axis.

    Args:
        x: Input array.
        sections: Number of equal-size sections to split into. If the
            array size along ``axis`` is not divisible by ``sections``,
            the array is split into as many equal parts as possible.
        axis: Axis along which to split (default: 0).

    Returns:
        List of sub-array views.
    """
    return _native_split(x, sections, axis)


def tile(x: "Array", reps: list) -> "Array":
    """Construct an array by tiling the input.

    Repeats the input array according to the repetition pattern in
    ``reps``.

    Args:
        x: Input array.
        reps: Number of repetitions along each axis. If ``reps`` has
            fewer dimensions than ``x``, it is prepended with 1s.

    Returns:
        Tiled array.
    """
    return _native_tile(x, Shape(reps))


def repeat(x: "Array", repeats: int, axis: Optional[int] = None) -> "Array":
    """Repeat elements of an array.

    Args:
        x: Input array.
        repeats: Number of repetitions for each element.
        axis: Axis along which to repeat. If None, the array is
            flattened first.

    Returns:
        Array with repeated elements.
    """
    return _native_repeat(x, repeats, axis)


def flip(x: "Array", axis: Optional[int] = None) -> "Array":
    """Reverse the order of elements along the given axis.

    Args:
        x: Input array.
        axis: Axis along which to flip. If None, all axes are reversed.

    Returns:
        View of the array with the elements reversed.
    """
    return _native_flip(x, axis)


def roll(x: "Array", shift: int, axis: Optional[int] = None) -> "Array":
    """Roll array elements along the given axis.

    Elements that roll beyond the last position are re-introduced at the
    first position (and vice versa).

    Args:
        x: Input array.
        shift: Number of places to shift. Positive values shift to the
            right (or down); negative values shift to the left (or up).
        axis: Axis along which to roll. If None, the array is flattened
            first.

    Returns:
        Rolled array.
    """
    return _native_roll(x, shift, axis)


def pad(
    x: "Array",
    pad_width: list,
    constant_value: float = 0.0,
) -> "Array":
    """Pad an array with constant values.

    Args:
        x: Input array.
        pad_width: Padding widths for each dimension. A flat list of
            ``(before, after)`` pairs, one per dimension.
        constant_value: Value to fill the padding with (default: 0.0).

    Returns:
        Padded array.
    """
    return _native_pad(x, pad_width, constant_value)


def diag(x: "Array", k: int = 0) -> "Array":
    """Extract a diagonal or construct a diagonal array.

    If ``x`` is 1-D, returns a 2-D array with ``x`` on the ``k``-th
    diagonal. If ``x`` is 2-D, returns the ``k``-th diagonal as a 1-D
    array.

    Args:
        x: Input array (1-D or 2-D).
        k: Diagonal offset. 0 is the main diagonal, positive values are
            above, and negative values are below.

    Returns:
        Diagonal array.
    """
    return _native_diag(x, k)


def diagonal(
    x: "Array",
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
) -> "Array":
    """Extract diagonal elements from a multi-dimensional array.

    Returns the specified diagonals defined by ``axis1`` and ``axis2``.

    Args:
        x: Input array.
        offset: Diagonal offset from the main diagonal.
        axis1: First axis of the 2-D sub-arrays from which diagonals
            are extracted.
        axis2: Second axis of the 2-D sub-arrays.

    Returns:
        Array of diagonal elements.
    """
    return _native_diagonal(x, offset, axis1, axis2)


def tril(x: "Array", k: int = 0) -> "Array":
    """Lower triangle of an array.

    Returns a copy with all elements above the ``k``-th diagonal set to
    zero.

    Args:
        x: Input array.
        k: Diagonal above which to zero elements. 0 is the main
            diagonal, positive values are above.

    Returns:
        Array with the lower triangle retained and upper triangle zeroed.
    """
    return _native_tril(x, k)


def triu(x: "Array", k: int = 0) -> "Array":
    """Upper triangle of an array.

    Returns a copy with all elements below the ``k``-th diagonal set to
    zero.

    Args:
        x: Input array.
        k: Diagonal below which to zero elements. 0 is the main
            diagonal, negative values are below.

    Returns:
        Array with the upper triangle retained and lower triangle zeroed.
    """
    return _native_triu(x, k)


def diff(x: "Array", n: int = 1, axis: int = -1) -> "Array":
    """Compute the n-th order discrete difference along the given axis.

    The first-order difference is given by ``out[i] = x[i+1] - x[i]``
    along the specified axis. Higher-order differences are computed by
    recursively applying the difference.

    Args:
        x: Input array.
        n: Number of times to apply the difference (default: 1).
        axis: Axis along which to compute the difference (default: -1).

    Returns:
        Array with the ``n``-th order differences. The size along
        ``axis`` is reduced by ``n``.
    """
    return _native_diff(x, n, axis)


def swapaxes(x: "Array", axis1: int, axis2: int) -> "Array":
    """Swap two axes of an array.

    Args:
        x: Input array.
        axis1: First axis to swap.
        axis2: Second axis to swap.

    Returns:
        View of the array with the two axes swapped.
    """
    return _native_swapaxes(x, axis1, axis2)


def moveaxis(x: "Array", source: int, destination: int) -> "Array":
    """Move an axis of an array to a new position.

    Args:
        x: Input array.
        source: Original position of the axis to move.
        destination: Destination position for the axis.

    Returns:
        View of the array with the axis moved.
    """
    return _native_moveaxis(x, source, destination)


def fliplr(x: "Array") -> "Array":
    """Flip an array left-right (reverse columns).

    Equivalent to ``flip(x, axis=1)``. The input must have at least 2
    dimensions.

    Args:
        x: Input array (at least 2-D).

    Returns:
        View of the array with columns reversed.
    """
    return _native_fliplr(x)


def flipud(x: "Array") -> "Array":
    """Flip an array up-down (reverse rows).

    Equivalent to ``flip(x, axis=0)``. The input must have at least 2
    dimensions.

    Args:
        x: Input array (at least 2-D).

    Returns:
        View of the array with rows reversed.
    """
    return _native_flipud(x)


def rot90(x: "Array", k: int = 1, axes: list = None) -> "Array":
    """Rotate a 2-D array by 90 degrees in the plane specified by axes.

    Args:
        x: Input 2-D array.
        k: Number of times the array is rotated by 90 degrees.
            Positive values rotate counter-clockwise.
        axes: The pair of axes that define the plane of rotation.
            Default is [0, 1].

    Returns:
        Rotated array view.
    """
    if axes is None:
        axes = [0, 1]
    return _native_rot90(x, k, axes)


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
    "transpose",
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
