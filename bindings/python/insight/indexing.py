"""Indexing and searching functions.

Provides functions for gathering, scattering, sorting, searching,
and selecting elements from arrays.
"""

from insight._insight import (
    UniqueResult as UniqueResult,
    argsort as _argsort,
    flatnonzero as _flatnonzero,
    gather as _gather,
    indices as _indices,
    interp as _interp,
    ix_ as _ix_,
    masked_select as _masked_select,
    nonzero as _nonzero,
    scatter as _scatter,
    scatter_add as _scatter_add,
    scatter_reduce as _scatter_reduce,
    searchsorted as _searchsorted,
    sort as _sort,
    take as _take,
    topk as _topk,
    unique as _unique,
)

__all__ = [
    "UniqueResult",
    "take",
    "nonzero",
    "flatnonzero",
    "argsort",
    "sort",
    "masked_select",
    "searchsorted",
    "unique",
    "topk",
    "gather",
    "scatter",
    "scatter_add",
    "scatter_reduce",
    "interp",
    "indices",
    "ix_",
]


def take(x, indices, axis=None):
    """Take elements from an array along an axis.

    Args:
        x: Input array.
        indices: Array of indices to take.
        axis: Axis along which to take values.  If ``None`` (default),
            the input array is flattened first.

    Returns:
        Array with values taken at the given indices.
    """
    return _take(x, indices, axis=axis)


def nonzero(x):
    """Return the indices of non-zero elements.

    Args:
        x: Input array.

    Returns:
        A tuple of arrays, one for each dimension, containing the indices
        of non-zero elements.
    """
    return _nonzero(x)


def flatnonzero(x):
    """Return the indices of non-zero elements in the flattened array.

    Args:
        x: Input array.

    Returns:
        1-D array of indices of non-zero elements.
    """
    return _flatnonzero(x)


def argsort(x, axis=-1, descending=False):
    """Return the indices that would sort an array.

    Args:
        x: Input array.
        axis: Axis along which to sort.  Default is -1 (last axis).
        descending: If True, sort in descending order.  Default is False.

    Returns:
        Array of indices that would sort *x*.
    """
    return _argsort(x, axis=axis, descending=descending)


def sort(x, axis=-1, descending=False):
    """Return a sorted copy of the array.

    Args:
        x: Input array.
        axis: Axis along which to sort.  Default is -1 (last axis).
        descending: If True, sort in descending order.  Default is False.

    Returns:
        Sorted array.
    """
    return _sort(x, axis=axis, descending=descending)


def masked_select(x, mask):
    """Select elements where the mask is True.

    Args:
        x: Input array.
        mask: Boolean mask array.  Must be broadcastable to *x*.

    Returns:
        1-D array of selected elements.
    """
    return _masked_select(x, mask)


def searchsorted(sorted_arr, values, side="left"):
    """Find indices where elements should be inserted to maintain order.

    Args:
        sorted_arr: 1-D sorted input array.
        values: Values to insert.
        side: ``'left'`` (default) or ``'right'``.  If ``'left'``, the
            index of the first suitable location is given.  If ``'right'``,
            the index of the last suitable location is given.

    Returns:
        Array of insertion indices.
    """
    return _searchsorted(sorted_arr, values, side=side)


def unique(x, return_indices=False, return_inverse=False, return_counts=False):
    """Return the unique elements of an array.

    Args:
        x: Input array.
        return_indices: If True, also return the indices of the first
            occurrence of each unique value.  Default is False.
        return_inverse: If True, also return the indices to reconstruct
            the original array.  Default is False.
        return_counts: If True, also return the count of each unique
            value.  Default is False.

    Returns:
        A :class:`UniqueResult` with attribute ``unique`` (the sorted
        unique values).  If any optional flags are True, the result also
        contains ``indices``, ``inverse``, and/or ``counts``.
    """
    return _unique(
        x, return_indices=return_indices, return_inverse=return_inverse, return_counts=return_counts
    )


def topk(x, k, axis=-1, largest=True, sorted=True):
    """Return the k largest (or smallest) elements along an axis.

    Args:
        x: Input array.
        k: Number of top elements to return.
        axis: Axis along which to search.  Default is -1 (last axis).
        largest: If True (default), return the largest elements.
        sorted: If True (default), return the elements in sorted order.

    Returns:
        A tuple ``(values, indices)`` with the top-k values and their
        indices.
    """
    return _topk(x, k, axis=axis, largest=largest, sorted=sorted)


def gather(x, dim, index):
    """Gather values along an axis specified by an index array.

    Args:
        x: Input array (source).
        dim: The axis along which to index.
        index: The indices of elements to gather.

    Returns:
        Array with the same shape as *index*, containing gathered values.
    """
    return _gather(x, dim, index)


def scatter(x, dim, index, src):
    """Scatter *src* values into *x* along *dim* at *index* positions.

    Args:
        x: Destination array.
        dim: The axis along which to index.
        index: The indices of elements to scatter to.
        src: The source array of values to scatter.

    Returns:
        A new array with scattered values.
    """
    return _scatter(x, dim, index, src)


def scatter_add(x, dim, index, src):
    """Add *src* values into *x* at *index* positions along *dim*.

    Args:
        x: Destination array.
        dim: The axis along which to index.
        index: The indices of elements to scatter to.
        src: The source array of values to add.

    Returns:
        A new array with scattered additions.
    """
    return _scatter_add(x, dim, index, src)


def scatter_reduce(x, dim, index, src, reduce="replace"):
    """Scatter *src* into *x* using a reduction operation.

    Args:
        x: Destination array.
        dim: The axis along which to index.
        index: The indices of elements to scatter to.
        src: The source array of values.
        reduce: Reduction operation.  ``'replace'`` (default),
            ``'add'``, ``'multiply'``, ``'mean'``, or ``'amax'``.

    Returns:
        A new array with reduced scattered values.
    """
    return _scatter_reduce(x, dim, index, src, reduce=reduce)


def interp(x, xp, fp, left=None, right=None):
    """One-dimensional linear interpolation.

    Args:
        x: The x-coordinates at which to evaluate the interpolated values.
        xp: The x-coordinates of the data points (must be increasing).
        fp: The y-coordinates of the data points.
        left: Value to return for ``x < xp[0]``.  If ``None`` (default),
            uses ``fp[0]``.
        right: Value to return for ``x > xp[-1]``.  If ``None`` (default),
            uses ``fp[-1]``.

    Returns:
        The interpolated values, same shape as *x*.
    """
    kwargs = {}
    if left is not None:
        kwargs["left"] = left
    if right is not None:
        kwargs["right"] = right
    return _interp(x, xp, fp, **kwargs)


def indices(dimensions, sparse=False):
    """Return an array representing the indices of a grid.

    Args:
        dimensions: Shape of the grid.
        sparse: If True, return a sparse representation.  Default is False.

    Returns:
        Array of indices.
    """
    return _indices(dimensions, sparse=sparse)


def ix_(*args):
    """Construct an open mesh from multiple sequences.

    Args:
        *args: 1-D sequences (arrays or lists).

    Returns:
        A tuple of arrays suitable for open mesh indexing.
    """
    return _ix_(list(args))
