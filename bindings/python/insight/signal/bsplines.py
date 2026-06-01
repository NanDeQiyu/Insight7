"""B-spline basis functions for signal processing.

Provides Gaussian, cubic, and quadratic B-spline evaluation functions.
"""

from insight._insight import signal as _signal

__all__ = [
    "gauss_spline",
    "cubic",
    "quadratic",
]


def gauss_spline(x, n):
    """Return a Gaussian B-spline basis function.

    Evaluates the *n*-th order Gaussian B-spline at the points in *x*.

    Args:
        x: Input array of sample positions.
        n: Order of the spline (non-negative integer).

    Returns:
        Gaussian B-spline values at each element of *x*.
    """
    return _signal.gauss_spline(x, n)


def cubic(x):
    """Return a cubic B-spline basis function.

    Evaluates the third-order (cubic) B-spline at the points in *x*.

    Args:
        x: Input array of sample positions.

    Returns:
        Cubic B-spline values at each element of *x*.
    """
    return _signal.cubic(x)


def quadratic(x):
    """Return a quadratic B-spline basis function.

    Evaluates the second-order (quadratic) B-spline at the points in *x*.

    Args:
        x: Input array of sample positions.

    Returns:
        Quadratic B-spline values at each element of *x*.
    """
    return _signal.quadratic(x)
