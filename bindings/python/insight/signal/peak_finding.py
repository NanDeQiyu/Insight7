"""Peak finding functions for signal processing.

Provides relative extrema detection (local maxima and minima) for
N-dimensional arrays.
"""

from insight._insight import signal as _signal

__all__ = [
    "argrelextrema",
    "argrelmax",
    "argrelmin",
]


def argrelextrema(data, comparator="greater", axis=0, order=1, mode="clip"):
    """Find indices of relative extrema (peaks or valleys) in data.

    Args:
        data: Input array.
        comparator: Comparison operator.  ``'greater'`` (default) finds
            local maxima; ``'less'`` finds local minima.
        axis: Axis along which to find extrema.  Default is 0.
        order: Number of points on each side to use for comparison.
            Default is 1.
        mode: How to handle boundaries.  ``'clip'`` (default) clips
            indices to the array bounds; ``'wrap'`` wraps around.

    Returns:
        Tuple of index arrays, one per dimension, indicating the positions
        of the relative extrema.
    """
    return _signal.argrelextrema(data, comparator=comparator, axis=axis,
                                 order=order, mode=mode)


def argrelmax(data, axis=0, order=1, mode="clip"):
    """Find indices of relative maxima in data.

    Args:
        data: Input array.
        axis: Axis along which to find maxima.  Default is 0.
        order: Number of points on each side to use for comparison.
            Default is 1.
        mode: How to handle boundaries.  ``'clip'`` (default) clips
            indices to the array bounds; ``'wrap'`` wraps around.

    Returns:
        Tuple of index arrays indicating the positions of the relative
        maxima.
    """
    return _signal.argrelmax(data, axis=axis, order=order, mode=mode)


def argrelmin(data, axis=0, order=1, mode="clip"):
    """Find indices of relative minima in data.

    Args:
        data: Input array.
        axis: Axis along which to find minima.  Default is 0.
        order: Number of points on each side to use for comparison.
            Default is 1.
        mode: How to handle boundaries.  ``'clip'`` (default) clips
            indices to the array bounds; ``'wrap'`` wraps around.

    Returns:
        Tuple of index arrays indicating the positions of the relative
        minima.
    """
    return _signal.argrelmin(data, axis=axis, order=order, mode=mode)
