"""Wavelet functions for signal processing.

Provides Morlet and Ricker wavelet generators, and the continuous wavelet
transform (CWT).
"""

from insight._insight import signal as _signal

__all__ = [
    "morlet",
    "ricker",
    "morlet2",
    "cwt",
]


def morlet(M, w=5.0, s=1.0, complete=True):
    """Return a complex Morlet wavelet.

    Args:
        M: Length of the wavelet in samples.
        w: Omega0 (center frequency).  Default is 5.0.
        s: Scaling factor.  Default is 1.0.
        complete: Whether to use the complete (with correction term) or
            standard version.  Default is True.

    Returns:
        Complex Morlet wavelet array of length *M*.
    """
    return _signal.morlet(M, w=w, s=s, complete=complete)


def ricker(points, a):
    """Return a Ricker wavelet (Mexican hat wavelet).

    Args:
        points: Number of points in the wavelet.
        a: Width parameter of the wavelet.

    Returns:
        Ricker wavelet array of length *points*.
    """
    return _signal.ricker(points, a)


def morlet2(M, s=1.0, w=5.0):
    """Return a complex Morlet wavelet (alternative parameterization).

    Args:
        M: Length of the wavelet in samples.
        s: Scaling factor.  Default is 1.0.
        w: Omega0 (center frequency).  Default is 5.0.

    Returns:
        Complex Morlet wavelet array of length *M*.
    """
    return _signal.morlet2(M, s=s, w=w)


def cwt(data, wavelet, widths):
    """Compute the continuous wavelet transform.

    Args:
        data: Input signal array (1-D).
        wavelet: Wavelet function.  Should accept ``(points, width)``
            and return a 1-D wavelet array.
        widths: Sequence of widths (scales) at which to compute the CWT.

    Returns:
        2-D array of shape ``(len(widths), len(data))`` containing the
        CWT coefficients.
    """
    return _signal.cwt(data, wavelet, list(widths))
