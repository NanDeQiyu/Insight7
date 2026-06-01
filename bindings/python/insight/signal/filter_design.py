"""Filter design functions for signal processing.

Provides FIR filter design (window method), Kaiser window parameter
estimation, and helper utilities for IIR filter design.
"""

from insight._insight import signal as _signal

__all__ = [
    "kaiser_beta",
    "kaiser_atten",
    "firwin",
    "firwin2",
    "cmplx_sort",
]


def kaiser_beta(a):
    """Compute the Kaiser window beta parameter for a given stopband attenuation.

    Args:
        a: Stopband attenuation in dB (positive value).

    Returns:
        Kaiser window beta parameter.
    """
    return _signal.kaiser_beta(a)


def kaiser_atten(numtaps, width):
    """Compute the Kaiser window stopband attenuation.

    Args:
        numtaps: Number of filter taps (length of the FIR filter).
        width: Transition width as a fraction of the Nyquist frequency.

    Returns:
        Stopband attenuation in dB.
    """
    return _signal.kaiser_atten(numtaps, width)


def firwin(numtaps, cutoff, window="hamming", pass_zero="lowpass", scale=True):
    """Design a FIR filter using the window method.

    Args:
        numtaps: Number of taps in the FIR filter (filter length).
        cutoff: Cutoff frequency or array of cutoff frequencies.
            For a lowpass or highpass filter, a single cutoff is required.
            For bandpass or bandstop, two cutoffs are required.
        window: Window function to use.  Default is ``'hamming'``.
        pass_zero: Filter type.  One of ``'lowpass'``, ``'highpass'``,
            ``'bandpass'``, ``'bandstop'``.  Default is ``'lowpass'``.
        scale: Whether to scale the filter so that the frequency response
            at the first (or only) passband is unity.  Default is True.

    Returns:
        FIR filter coefficients as a 1-D array.
    """
    return _signal.firwin(numtaps, cutoff, window=window, pass_zero=pass_zero, scale=scale)


def firwin2(numtaps, freq, gain, nfreqs=None, window="hamming", antisymmetric=False):
    """Design a FIR filter with arbitrary frequency response.

    Args:
        numtaps: Number of taps in the FIR filter.
        freq: Sequence of frequency breakpoints (normalized to [0, 1] where
            1 is the Nyquist frequency).
        gain: Gain at each frequency breakpoint (same length as *freq*).
        nfreqs: Number of interpolation points for the frequency response.
            If ``None``, uses a default value chosen by the backend.
        window: Window function to use.  Default is ``'hamming'``.
        antisymmetric: If True, design a type III or IV filter with
            antisymmetric impulse response.  Default is False.

    Returns:
        FIR filter coefficients as a 1-D array.
    """
    kwargs = {"window": window, "antisymmetric": antisymmetric}
    if nfreqs is not None:
        kwargs["nfreqs"] = nfreqs
    return _signal.firwin2(numtaps, freq, gain, **kwargs)


def cmplx_sort(p):
    """Sort complex numbers by real part, then by imaginary part.

    Args:
        p: Sequence of complex numbers (array-like).

    Returns:
        A tuple ``(sorted_p, index)`` where *sorted_p* is the sorted array
        and *index* maps from sorted to original order.
    """
    return _signal.cmplx_sort(p)
