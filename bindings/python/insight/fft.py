"""Fast Fourier Transform operations.

Provides 1-D, 2-D, and N-D discrete Fourier transforms (real and
complex), inverse transforms, and frequency-space helpers.

All functions operate on :class:`insight.Array` and return results
as :class:`insight.Array`.
"""

from insight._insight import (
    fft as _fft,
    ifft as _ifft,
    fft2 as _fft2,
    ifft2 as _ifft2,
    fftn as _fftn,
    ifftn as _ifftn,
    rfft as _rfft,
    irfft as _irfft,
    fftfreq as _fftfreq,
    rfftfreq as _rfftfreq,
)

__all__ = [
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "fftfreq",
    "rfftfreq",
]


def fft(x, n=-1, axis=-1, norm="backward"):
    """Compute the 1-D discrete Fourier transform.

    Args:
        x: Input array (real or complex).
        n: Length of the transform axis.  If -1 (default), use the
            length of *x* along *axis*.
        axis: Axis along which to compute the FFT.  Default is -1.
        norm: Normalization mode — ``'backward'`` (default),
            ``'ortho'``, or ``'forward'``.

    Returns:
        Complex-valued array of the same shape as *x*.
    """
    return _fft(x, n=n, axis=axis, norm=norm)


def ifft(x, n=-1, axis=-1, norm="backward"):
    """Compute the 1-D inverse discrete Fourier transform.

    Args:
        x: Input complex-valued array.
        n: Length of the transform axis.  If -1 (default), use the
            length of *x* along *axis*.
        axis: Axis along which to compute the IFFT.  Default is -1.
        norm: Normalization mode — ``'backward'`` (default),
            ``'ortho'``, or ``'forward'``.

    Returns:
        Complex-valued array of the same shape as *x*.
    """
    return _ifft(x, n=n, axis=axis, norm=norm)


def fft2(x, s=None, axes=(-2, -1), norm="backward"):
    """Compute the 2-D discrete Fourier transform.

    Args:
        x: Input 2-D (or higher) array.
        s: Shape of the transform along each axis in *axes*.
            If ``None`` (default), use the input shape.
        axes: Axes over which to compute.  Default is (-2, -1).
        norm: Normalization mode — ``'backward'`` (default),
            ``'ortho'``, or ``'forward'``.

    Returns:
        Complex-valued array.
    """
    s = [] if s is None else list(s)
    return _fft2(x, s=list(s), axes=list(axes), norm=norm)


def ifft2(x, s=None, axes=(-2, -1), norm="backward"):
    """Compute the 2-D inverse discrete Fourier transform.

    Args:
        x: Input 2-D (or higher) complex-valued array.
        s: Shape of the transform along each axis in *axes*.
            If ``None`` (default), use the input shape.
        axes: Axes over which to compute.  Default is (-2, -1).
        norm: Normalization mode — ``'backward'`` (default),
            ``'ortho'``, or ``'forward'``.

    Returns:
        Complex-valued array.
    """
    s = [] if s is None else list(s)
    return _ifft2(x, s=list(s), axes=list(axes), norm=norm)


def fftn(x, s=None, axes=None, norm="backward"):
    """Compute the N-D discrete Fourier transform.

    Args:
        x: Input array.
        s: Shape of the transform along each axis in *axes*.
            If ``None`` (default), use the input shape.
        axes: Axes over which to compute.  If ``None`` (default),
            transform all axes.
        norm: Normalization mode — ``'backward'`` (default),
            ``'ortho'``, or ``'forward'``.

    Returns:
        Complex-valued array.
    """
    s = [] if s is None else list(s)
    axes = [] if axes is None else list(axes)
    return _fftn(x, s=list(s), axes=list(axes), norm=norm)


def ifftn(x, s=None, axes=None, norm="backward"):
    """Compute the N-D inverse discrete Fourier transform.

    Args:
        x: Input complex-valued array.
        s: Shape of the transform along each axis in *axes*.
            If ``None`` (default), use the input shape.
        axes: Axes over which to compute.  If ``None`` (default),
            transform all axes.
        norm: Normalization mode — ``'backward'`` (default),
            ``'ortho'``, or ``'forward'``.

    Returns:
        Complex-valued array.
    """
    s = [] if s is None else list(s)
    axes = [] if axes is None else list(axes)
    return _ifftn(x, s=list(s), axes=list(axes), norm=norm)


def rfft(x, n=-1, axis=-1, norm="backward"):
    """Compute the 1-D FFT of a real-valued input.

    Returns only the positive-frequency terms (conjugate symmetry).

    Args:
        x: Input real-valued array.
        n: Length of the transform axis.  If -1 (default), use the
            length of *x* along *axis*.
        axis: Axis along which to compute.  Default is -1.
        norm: Normalization mode — ``'backward'`` (default),
            ``'ortho'``, or ``'forward'``.

    Returns:
        Complex-valued array with ``n//2 + 1`` elements along *axis*.
    """
    return _rfft(x, n=n, axis=axis, norm=norm)


def irfft(x, n=-1, axis=-1, norm="backward"):
    """Compute the inverse of :func:`rfft`.

    The input is assumed to be the positive-frequency half of a
    Hermitian-symmetric spectrum.

    Args:
        x: Input complex-valued array (positive frequencies).
        n: Length of the real-valued output along *axis*.
            If -1 (default), ``n = 2*(m-1)`` where *m* is the input
            length along *axis*.
        axis: Axis along which to compute.  Default is -1.
        norm: Normalization mode — ``'backward'`` (default),
            ``'ortho'``, or ``'forward'``.

    Returns:
        Real-valued array.
    """
    return _irfft(x, n=n, axis=axis, norm=norm)


def fftfreq(n, d=1.0):
    """Return the DFT sample frequencies.

    Args:
        n: Window length (number of samples).
        d: Sample spacing (inverse of sampling rate).  Default is 1.0.

    Returns:
        1-D array of length *n* containing the frequency bins.
    """
    return _fftfreq(n, d=d)


def rfftfreq(n, d=1.0):
    """Return the DFT sample frequencies for :func:`rfft`.

    Args:
        n: Window length (number of samples).
        d: Sample spacing (inverse of sampling rate).  Default is 1.0.

    Returns:
        1-D array of length ``n//2 + 1`` containing the frequency bins.
    """
    return _rfftfreq(n, d=d)
