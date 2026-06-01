"""Reduction operations for Insight arrays.

Provides statistical and logical reductions along specified axes,
including cumulative operations and NaN-safe variants.

All functions return an :class:`insight.Array`.
"""

from insight._insight import (
    sum as _sum,
    mean as _mean,
    max as _max,
    min as _min,
    prod as _prod,
    argmax as _argmax,
    argmin as _argmin,
    any as _any,
    all as _all,
    var as _var,
    std as _std,
    cumsum as _cumsum,
    cumprod as _cumprod,
    cummax as _cummax,
    cummin as _cummin,
    sem as _sem,
    count_nonzero as _count_nonzero,
    median as _median,
    quantile as _quantile,
    percentile as _percentile,
    nansum as _nansum,
    nanmean as _nanmean,
    nanmax as _nanmax,
    nanmin as _nanmin,
    nanstd as _nanstd,
    nanvar as _nanvar,
)

__all__ = [
    "sum",
    "mean",
    "max",
    "min",
    "prod",
    "argmax",
    "argmin",
    "any",
    "all",
    "var",
    "std",
    "cumsum",
    "cumprod",
    "cummax",
    "cummin",
    "sem",
    "count_nonzero",
    "median",
    "quantile",
    "percentile",
    "nansum",
    "nanmean",
    "nanmax",
    "nanmin",
    "nanstd",
    "nanvar",
]


def sum(x, axis=None, keepdims=False):
    """Sum of array elements over a given axis.

    Args:
        x: Input array.
        axis: Axis or axes along which a sum is performed.  ``None``
            (default) sums all elements.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.

    Returns:
        Sum of elements, or an array with the specified axes reduced.
    """
    return _sum(x, axis=axis, keepdims=keepdims)


def mean(x, axis=None, keepdims=False):
    """Mean of array elements over a given axis.

    Args:
        x: Input array.
        axis: Axis or axes along which the mean is computed.
            ``None`` (default) computes the mean of all elements.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.

    Returns:
        Mean value, or an array of means along the specified axis.
    """
    return _mean(x, axis=axis, keepdims=keepdims)


def max(x, axis=None, keepdims=False):
    """Maximum of array elements over a given axis.

    Args:
        x: Input array.
        axis: Axis or axes along which the maximum is computed.
            ``None`` (default) computes the maximum of all elements.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.

    Returns:
        Maximum value, or an array of maxima along the specified axis.
    """
    return _max(x, axis=axis, keepdims=keepdims)


def min(x, axis=None, keepdims=False):
    """Minimum of array elements over a given axis.

    Args:
        x: Input array.
        axis: Axis or axes along which the minimum is computed.
            ``None`` (default) computes the minimum of all elements.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.

    Returns:
        Minimum value, or an array of minima along the specified axis.
    """
    return _min(x, axis=axis, keepdims=keepdims)


def prod(x, axis=None, keepdims=False):
    """Product of array elements over a given axis.

    Args:
        x: Input array.
        axis: Axis or axes along which the product is computed.
            ``None`` (default) computes the product of all elements.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.

    Returns:
        Product of elements, or an array of products along the
        specified axis.
    """
    return _prod(x, axis=axis, keepdims=keepdims)


def argmax(x, axis=None, keepdims=False):
    """Indices of the maximum values along an axis.

    Args:
        x: Input array.
        axis: Axis along which to operate.  ``None`` (default) returns
            the index of the maximum in the flattened array.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.

    Returns:
        Array of indices into the array.
    """
    return _argmax(x, axis=axis, keepdims=keepdims)


def argmin(x, axis=None, keepdims=False):
    """Indices of the minimum values along an axis.

    Args:
        x: Input array.
        axis: Axis along which to operate.  ``None`` (default) returns
            the index of the minimum in the flattened array.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.

    Returns:
        Array of indices into the array.
    """
    return _argmin(x, axis=axis, keepdims=keepdims)


def any(x, axis=None, keepdims=False):
    """Test whether any array element along an axis evaluates to True.

    Args:
        x: Input array.
        axis: Axis or axes along which a logical OR is performed.
            ``None`` (default) reduces over all elements.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.

    Returns:
        Boolean result, or an array of boolean results.
    """
    return _any(x, axis=axis, keepdims=keepdims)


def all(x, axis=None, keepdims=False):
    """Test whether all array elements along an axis evaluate to True.

    Args:
        x: Input array.
        axis: Axis or axes along which a logical AND is performed.
            ``None`` (default) reduces over all elements.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.

    Returns:
        Boolean result, or an array of boolean results.
    """
    return _all(x, axis=axis, keepdims=keepdims)


def var(x, axis=None, keepdims=False, ddof=0):
    """Variance of array elements over a given axis.

    Args:
        x: Input array.
        axis: Axis or axes along which the variance is computed.
            ``None`` (default) computes the variance of all elements.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.
        ddof: Delta degrees of freedom.  The divisor is
            ``N - ddof``, where ``N`` is the number of elements.
            Default is 0.

    Returns:
        Variance, or an array of variances along the specified axis.
    """
    return _var(x, axis=axis, keepdims=keepdims, ddof=ddof)


def std(x, axis=None, keepdims=False, ddof=0):
    """Standard deviation of array elements over a given axis.

    Args:
        x: Input array.
        axis: Axis or axes along which the standard deviation is
            computed.  ``None`` (default) computes it for all elements.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.
        ddof: Delta degrees of freedom.  The divisor is
            ``N - ddof``, where ``N`` is the number of elements.
            Default is 0.

    Returns:
        Standard deviation, or an array of standard deviations along
        the specified axis.
    """
    return _std(x, axis=axis, keepdims=keepdims, ddof=ddof)


def cumsum(x, axis):
    """Cumulative sum of elements along a given axis.

    Args:
        x: Input array.
        axis: Axis along which the cumulative sum is computed.

    Returns:
        Array of the same shape as *x* with cumulative sums along the
        given axis.
    """
    return _cumsum(x, axis)


def cumprod(x, axis):
    """Cumulative product of elements along a given axis.

    Args:
        x: Input array.
        axis: Axis along which the cumulative product is computed.

    Returns:
        Array of the same shape as *x* with cumulative products along
        the given axis.
    """
    return _cumprod(x, axis)


def cummax(x, axis):
    """Cumulative maximum of elements along a given axis.

    Args:
        x: Input array.
        axis: Axis along which the cumulative maximum is computed.

    Returns:
        Array of the same shape as *x* with cumulative maxima along
        the given axis.
    """
    return _cummax(x, axis)


def cummin(x, axis):
    """Cumulative minimum of elements along a given axis.

    Args:
        x: Input array.
        axis: Axis along which the cumulative minimum is computed.

    Returns:
        Array of the same shape as *x* with cumulative minima along
        the given axis.
    """
    return _cummin(x, axis)


def sem(x, axis=None, keepdims=False, ddof=0):
    """Standard error of the mean over a given axis.

    Args:
        x: Input array.
        axis: Axis or axes along which the SEM is computed.
            ``None`` (default) computes it for all elements.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.
        ddof: Delta degrees of freedom.  The divisor for the
            standard deviation is ``N - ddof``.  Default is 0.

    Returns:
        Standard error of the mean, or an array of SEM values.
    """
    return _sem(x, axis=axis, keepdims=keepdims, ddof=ddof)


def count_nonzero(x, axis=None, keepdims=False):
    """Count the number of non-zero values along an axis.

    Args:
        x: Input array.
        axis: Axis or axes along which to count.  ``None`` (default)
            counts over all elements.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.

    Returns:
        Count of non-zero values, or an array of counts.
    """
    return _count_nonzero(x, axis=axis, keepdims=keepdims)


def median(x, axis=None, keepdims=False):
    """Median of array elements over a given axis.

    Args:
        x: Input array.
        axis: Axis or axes along which the median is computed.
            ``None`` (default) computes the median of all elements.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.

    Returns:
        Median value, or an array of medians along the specified axis.
    """
    return _median(x, axis=axis, keepdims=keepdims)


def quantile(x, q, axis=None, keepdims=False):
    """Compute the q-th quantile of array elements.

    Args:
        x: Input array.
        q: Quantile to compute, in the range [0, 1].  May be a
            scalar or an array of quantiles.
        axis: Axis or axes along which the quantile is computed.
            ``None`` (default) computes it over all elements.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.

    Returns:
        Quantile value, or an array of quantiles.
    """
    return _quantile(x, q, axis=axis, keepdims=keepdims)


def percentile(x, q, axis=None, keepdims=False):
    """Compute the q-th percentile of array elements.

    Args:
        x: Input array.
        q: Percentile to compute, in the range [0, 100].
        axis: Axis or axes along which the percentile is computed.
            ``None`` (default) computes it over all elements.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.

    Returns:
        Percentile value, or an array of percentile values.
    """
    return _percentile(x, q, axis=axis, keepdims=keepdims)


def nansum(x, axis=None, keepdims=False):
    """Sum of array elements, treating NaN as zero.

    Args:
        x: Input array.
        axis: Axis or axes along which the sum is computed.
            ``None`` (default) sums all elements.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.

    Returns:
        Sum of non-NaN elements, or an array of sums.
    """
    return _nansum(x, axis=axis, keepdims=keepdims)


def nanmean(x, axis=None, keepdims=False):
    """Mean of array elements, ignoring NaN values.

    Args:
        x: Input array.
        axis: Axis or axes along which the mean is computed.
            ``None`` (default) computes the mean of all elements.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.

    Returns:
        Mean of non-NaN elements, or an array of means.
    """
    return _nanmean(x, axis=axis, keepdims=keepdims)


def nanmax(x, axis=None, keepdims=False):
    """Maximum of array elements, ignoring NaN values.

    Args:
        x: Input array.
        axis: Axis or axes along which the maximum is computed.
            ``None`` (default) computes the maximum of all elements.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.

    Returns:
        Maximum of non-NaN elements, or an array of maxima.
    """
    return _nanmax(x, axis=axis, keepdims=keepdims)


def nanmin(x, axis=None, keepdims=False):
    """Minimum of array elements, ignoring NaN values.

    Args:
        x: Input array.
        axis: Axis or axes along which the minimum is computed.
            ``None`` (default) computes the minimum of all elements.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.

    Returns:
        Minimum of non-NaN elements, or an array of minima.
    """
    return _nanmin(x, axis=axis, keepdims=keepdims)


def nanstd(x, axis=None, keepdims=False, ddof=0):
    """Standard deviation, ignoring NaN values.

    Args:
        x: Input array.
        axis: Axis or axes along which the standard deviation is
            computed.  ``None`` (default) computes it for all elements.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.
        ddof: Delta degrees of freedom.  The divisor is
            ``N - ddof``, where ``N`` is the number of non-NaN
            elements.  Default is 0.

    Returns:
        Standard deviation of non-NaN elements, or an array of
        standard deviations.
    """
    return _nanstd(x, axis=axis, keepdims=keepdims, ddof=ddof)


def nanvar(x, axis=None, keepdims=False, ddof=0):
    """Variance, ignoring NaN values.

    Args:
        x: Input array.
        axis: Axis or axes along which the variance is computed.
            ``None`` (default) computes the variance of all elements.
        keepdims: If ``True``, the reduced axes are retained as
            size-one dimensions.  Default is ``False``.
        ddof: Delta degrees of freedom.  The divisor is
            ``N - ddof``, where ``N`` is the number of non-NaN
            elements.  Default is 0.

    Returns:
        Variance of non-NaN elements, or an array of variances.
    """
    return _nanvar(x, axis=axis, keepdims=keepdims, ddof=ddof)
