"""Convolution and correlation functions for signal processing.

Provides FFT-accelerated convolution/correlation for 1-D and 2-D signals,
as well as utilities for choosing the best convolution method.
"""

from insight._insight import signal as _signal

__all__ = [
    "fftconvolve",
    "correlate",
    "convolve2d",
    "correlate2d",
    "choose_conv_method",
    "correlation_lags",
]


def fftconvolve(in1, in2, mode="full"):
    """Convolve two N-D arrays using FFT.

    Args:
        in1: First input array.
        in2: Second input array.
        mode: Output mode.  One of ``'full'`` (default), ``'same'``, or
            ``'valid'``.

    Returns:
        Convolution result array.
    """
    return _signal.fftconvolve(in1, in2, mode=mode)


def correlate(in1, in2, mode="full"):
    """Cross-correlate two N-D arrays.

    Args:
        in1: First input array.
        in2: Second input array.
        mode: Output mode.  One of ``'full'`` (default), ``'same'``, or
            ``'valid'``.

    Returns:
        Cross-correlation result array.
    """
    return _signal.correlate(in1, in2, mode=mode)


def convolve2d(in1, in2, mode="full"):
    """Convolve two 2-D arrays.

    Args:
        in1: First 2-D input array.
        in2: Second 2-D input array.
        mode: Output mode.  One of ``'full'`` (default), ``'same'``, or
            ``'valid'``.

    Returns:
        2-D convolution result array.
    """
    return _signal.convolve2d(in1, in2, mode=mode)


def correlate2d(in1, in2, mode="full"):
    """Cross-correlate two 2-D arrays.

    Args:
        in1: First 2-D input array.
        in2: Second 2-D input array.
        mode: Output mode.  One of ``'full'`` (default), ``'same'``, or
            ``'valid'``.

    Returns:
        2-D cross-correlation result array.
    """
    return _signal.correlate2d(in1, in2, mode=mode)


def choose_conv_method(in1, in2, mode="full"):
    """Choose the fastest convolution method for the given inputs.

    Args:
        in1: First input array.
        in2: Second input array.
        mode: Output mode.  One of ``'full'`` (default), ``'same'``, or
            ``'valid'``.

    Returns:
        A string indicating the recommended method (``'fft'`` or
        ``'direct'``).
    """
    return _signal.choose_conv_method(in1, in2, mode=mode)


def correlation_lags(in1_len, in2_len, mode="full"):
    """Compute the lag vector for cross-correlation.

    Args:
        in1_len: Length of the first input array.
        in2_len: Length of the second input array.
        mode: Correlation mode.  One of ``'full'`` (default), ``'same'``, or
            ``'valid'``.

    Returns:
        1-D array of lag indices.
    """
    return _signal.correlation_lags(in1_len, in2_len, mode=mode)
