"""Spectral analysis functions for signal processing.

Provides power spectral density estimation (Welch, periodogram),
cross-spectral density, coherence, spectrogram, STFT, and vector strength.
"""

from insight._insight import signal as _signal

__all__ = [
    "csd",
    "welch",
    "periodogram",
    "coherence",
    "spectrogram",
    "stft",
    "vectorstrength",
]


def csd(
    x,
    y,
    fs=1.0,
    window="hann",
    nperseg=256,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
):
    """Estimate the cross spectral density using Welch's method.

    Args:
        x: First input signal array.
        y: Second input signal array.
        fs: Sampling frequency in Hz.  Default is 1.0.
        window: Window function name.  Default is ``'hann'``.
        nperseg: Length of each segment.  Default is 256.
        noverlap: Number of overlapping points between segments.  If
            ``None`` (default), defaults to ``nperseg // 2``.
        nfft: FFT size.  If ``None`` (default), uses *nperseg*.
        detrend: Detrending method.  Default is ``'constant'``.
        return_onesided: If True (default), return a one-sided spectrum
            for real inputs.
        scaling: Scaling type.  ``'density'`` (default) for power spectral
            density; ``'spectrum'`` for power spectrum.

    Returns:
        A ``SpectralResult`` with attributes ``f`` (frequency array) and
        ``Pxx`` (cross spectral density).
    """
    kwargs = {}
    if noverlap is not None:
        kwargs["noverlap"] = noverlap
    if nfft is not None:
        kwargs["nfft"] = nfft
    return _signal.csd(
        x,
        y,
        fs=fs,
        window=window,
        nperseg=nperseg,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        **kwargs,
    )


def welch(
    x,
    fs=1.0,
    window="hann",
    nperseg=256,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
):
    """Estimate the power spectral density using Welch's method.

    Args:
        x: Input signal array.
        fs: Sampling frequency in Hz.  Default is 1.0.
        window: Window function name.  Default is ``'hann'``.
        nperseg: Length of each segment.  Default is 256.
        noverlap: Number of overlapping points between segments.  If
            ``None`` (default), defaults to ``nperseg // 2``.
        nfft: FFT size.  If ``None`` (default), uses *nperseg*.
        detrend: Detrending method.  Default is ``'constant'``.
        return_onesided: If True (default), return a one-sided spectrum
            for real inputs.
        scaling: Scaling type.  ``'density'`` (default) for power spectral
            density; ``'spectrum'`` for power spectrum.

    Returns:
        A ``SpectralResult`` with attributes ``f`` (frequency array) and
        ``Pxx`` (power spectral density).
    """
    kwargs = {}
    if noverlap is not None:
        kwargs["noverlap"] = noverlap
    if nfft is not None:
        kwargs["nfft"] = nfft
    return _signal.welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        **kwargs,
    )


def periodogram(
    x, fs=1.0, window="hann", nfft=None, detrend="constant", return_onesided=True, scaling="density"
):
    """Estimate the power spectral density using a periodogram.

    Args:
        x: Input signal array.
        fs: Sampling frequency in Hz.  Default is 1.0.
        window: Window function name.  Default is ``'hann'``.
        nfft: FFT size.  If ``None`` (default), uses the signal length.
        detrend: Detrending method.  Default is ``'constant'``.
        return_onesided: If True (default), return a one-sided spectrum
            for real inputs.
        scaling: Scaling type.  ``'density'`` (default) for power spectral
            density; ``'spectrum'`` for power spectrum.

    Returns:
        A ``SpectralResult`` with attributes ``f`` (frequency array) and
        ``Pxx`` (power spectral density).
    """
    kwargs = {}
    if nfft is not None:
        kwargs["nfft"] = nfft
    return _signal.periodogram(
        x,
        fs=fs,
        window=window,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        **kwargs,
    )


def coherence(
    x, y, fs=1.0, window="hann", nperseg=256, noverlap=None, nfft=None, detrend="constant"
):
    """Estimate the magnitude-squared coherence between two signals.

    Args:
        x: First input signal array.
        y: Second input signal array.
        fs: Sampling frequency in Hz.  Default is 1.0.
        window: Window function name.  Default is ``'hann'``.
        nperseg: Length of each segment.  Default is 256.
        noverlap: Number of overlapping points between segments.  If
            ``None`` (default), defaults to ``nperseg // 2``.
        nfft: FFT size.  If ``None`` (default), uses *nperseg*.
        detrend: Detrending method.  Default is ``'constant'``.

    Returns:
        A ``SpectralResult`` with attributes ``f`` (frequency array) and
        ``Pxx`` (coherence values in [0, 1]).
    """
    kwargs = {}
    if noverlap is not None:
        kwargs["noverlap"] = noverlap
    if nfft is not None:
        kwargs["nfft"] = nfft
    return _signal.coherence(x, y, fs=fs, window=window, nperseg=nperseg, detrend=detrend, **kwargs)


def spectrogram(
    x,
    fs=1.0,
    window="hann",
    nperseg=256,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    mode="psd",
):
    """Compute a spectrogram using Welch's method.

    Args:
        x: Input signal array.
        fs: Sampling frequency in Hz.  Default is 1.0.
        window: Window function name.  Default is ``'hann'``.
        nperseg: Length of each segment.  Default is 256.
        noverlap: Number of overlapping points between segments.  If
            ``None`` (default), defaults to ``nperseg // 2``.
        nfft: FFT size.  If ``None`` (default), uses *nperseg*.
        detrend: Detrending method.  Default is ``'constant'``.
        return_onesided: If True (default), return a one-sided spectrum
            for real inputs.
        mode: Spectrogram mode.  One of ``'psd'`` (default), ``'magnitude'``,
            ``'angle'``, or ``'complex'``.

    Returns:
        A ``SpectrogramResult`` with attributes ``f`` (frequency array),
        ``t`` (time array), and ``Sxx`` (spectrogram array).
    """
    kwargs = {}
    if noverlap is not None:
        kwargs["noverlap"] = noverlap
    if nfft is not None:
        kwargs["nfft"] = nfft
    return _signal.spectrogram(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        detrend=detrend,
        return_onesided=return_onesided,
        mode=mode,
        **kwargs,
    )


def stft(x, fs=1.0, window="hann", nperseg=256, noverlap=None, nfft=None):
    """Compute the Short-Time Fourier Transform.

    Args:
        x: Input signal array.
        fs: Sampling frequency in Hz.  Default is 1.0.
        window: Window function name.  Default is ``'hann'``.
        nperseg: Length of each segment.  Default is 256.
        noverlap: Number of overlapping points between segments.  If
            ``None`` (default), defaults to ``nperseg // 2``.
        nfft: FFT size.  If ``None`` (default), uses *nperseg*.

    Returns:
        A ``SpectrogramResult`` with attributes ``f`` (frequency array),
        ``t`` (time array), and ``Sxx`` (STFT complex array).
    """
    kwargs = {}
    if noverlap is not None:
        kwargs["noverlap"] = noverlap
    if nfft is not None:
        kwargs["nfft"] = nfft
    return _signal.stft(x, fs=fs, window=window, nperseg=nperseg, **kwargs)


def vectorstrength(events, period):
    """Determine how well a set of events are synchronized to a period.

    Args:
        events: Array of event times.
        period: The period to which synchronization is measured.

    Returns:
        A tuple ``(strength, phase)`` where *strength* is the vector
        strength (0 = no synchronization, 1 = perfect) and *phase* is the
        mean phase in radians.
    """
    return _signal.vectorstrength(events, period)
