"""Digital filtering functions for signal processing.

Provides FIR/IIR filtering, Hilbert transform, detrending, resampling,
and Wiener filtering.
"""

from insight._insight import signal as _signal

__all__ = [
    "hilbert",
    "hilbert2",
    "detrend",
    "firfilter",
    "lfilter",
    "lfilter_zi",
    "filtfilt",
    "decimate",
    "resample",
    "resample_poly",
    "freq_shift",
    "wiener",
]


def hilbert(x, N=-1):
    """Compute the analytic signal using the Hilbert transform.

    Args:
        x: Input real-valued signal array.
        N: Number of FFT points.  If -1 (default), uses the length of *x*.

    Returns:
        Complex-valued analytic signal whose real part is the original
        signal and imaginary part is the Hilbert transform.
    """
    return _signal.hilbert(x, N=N)


def hilbert2(x, N=-1):
    """Compute the 2-D analytic signal using the Hilbert transform.

    Args:
        x: Input 2-D real-valued signal array.
        N: Number of FFT points per axis.  If -1 (default), uses the
            array dimensions.

    Returns:
        2-D complex-valued analytic signal.
    """
    return _signal.hilbert2(x, N=N)


def detrend(data, axis=-1, type="linear"):
    """Remove a trend from the data along the given axis.

    Args:
        data: Input array.
        axis: Axis along which to detrend.  Default is -1 (last axis).
        type: Type of trend to remove.  ``'linear'`` (default) removes a
            least-squares fit straight line; ``'constant'`` removes only
            the mean.

    Returns:
        Detrended array with the same shape as *data*.
    """
    return _signal.detrend(data, axis=axis, type=type)


def firfilter(b, x, axis=-1):
    """Apply an FIR filter to a signal.

    Args:
        b: FIR filter coefficients (1-D array).
        x: Input signal array.
        axis: Axis along which to filter.  Default is -1 (last axis).

    Returns:
        Filtered signal array.
    """
    return _signal.firfilter(b, x, axis=axis)


def lfilter(b, a, x, axis=-1):
    """Apply an IIR or FIR filter to a signal using direct-form II.

    Args:
        b: Numerator coefficients of the transfer function.
        a: Denominator coefficients of the transfer function.
        x: Input signal array.
        axis: Axis along which to filter.  Default is -1 (last axis).

    Returns:
        Filtered signal array.
    """
    return _signal.lfilter(b, a, x, axis=axis)


def lfilter_zi(b, a):
    """Compute the initial state for lfilter for step-response steady-state.

    Args:
        b: Numerator coefficients of the transfer function.
        a: Denominator coefficients of the transfer function.

    Returns:
        Initial state array for use with :func:`lfilter`.
    """
    return _signal.lfilter_zi(b, a)


def filtfilt(b, a, x, axis=-1):
    """Apply a forward-backward digital filter (zero-phase filtering).

    Args:
        b: Numerator coefficients of the transfer function.
        a: Denominator coefficients of the transfer function.
        x: Input signal array.
        axis: Axis along which to filter.  Default is -1 (last axis).

    Returns:
        Zero-phase filtered signal array.
    """
    return _signal.filtfilt(b, a, x, axis=axis)


def decimate(x, q, axis=-1, zero_phase=True):
    """Downsample a signal after applying an anti-aliasing filter.

    Args:
        x: Input signal array.
        q: Decimation factor (integer).
        axis: Axis along which to decimate.  Default is -1 (last axis).
        zero_phase: If True (default), use forward-backward filtering to
            prevent phase distortion.

    Returns:
        Decimated signal array.
    """
    return _signal.decimate(x, q, axis=axis, zero_phase=zero_phase)


def resample(x, num, axis=-1):
    """Resample a signal to a given number of samples using FFT.

    Args:
        x: Input signal array.
        num: Number of output samples.
        axis: Axis along which to resample.  Default is -1 (last axis).

    Returns:
        Resampled signal array.
    """
    return _signal.resample(x, num, axis=axis)


def resample_poly(x, up, down, axis=-1):
    """Resample a signal using polyphase filtering.

    Args:
        x: Input signal array.
        up: Upsampling factor.
        down: Downsampling factor.
        axis: Axis along which to resample.  Default is -1 (last axis).

    Returns:
        Resampled signal array.
    """
    return _signal.resample_poly(x, up, down, axis=axis)


def freq_shift(x, freq, fs):
    """Shift the frequency content of a signal.

    Args:
        x: Input signal array.
        freq: Frequency shift in Hz.
        fs: Sampling frequency in Hz.

    Returns:
        Frequency-shifted signal array.
    """
    return _signal.freq_shift(x, freq, fs)


def wiener(im, mysize=None, noise=-1):
    """Apply a Wiener filter to an N-dimensional array.

    Args:
        im: Input array.
        mysize: Size of the local neighbourhood for estimation.  If
            ``None`` (default), uses 3 for each dimension.
        noise: Noise power.  If -1 (default), estimated from the local
            variance.

    Returns:
        Wiener-filtered array with the same shape as *im*.
    """
    kwargs = {}
    if mysize is not None:
        kwargs["mysize"] = list(mysize) if not isinstance(mysize, list) else mysize
    if noise != -1:
        kwargs["noise"] = noise
    return _signal.wiener(im, **kwargs)
