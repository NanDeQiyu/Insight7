"""Radar signal processing functions.

Provides pulse compression, pulse-Doppler processing, CFAR detection,
MVDR beamforming, and ambiguity function computation.
"""

from insight._insight import signal as _signal

__all__ = [
    "pulse_compression",
    "pulse_doppler",
    "cfar_alpha",
    "ca_cfar",
    "mvdr",
    "ambgfun",
]


def pulse_compression(x, template_tx, normalize=False, window="", nfft=0):
    """Perform pulse compression using matched filtering.

    Args:
        x: Received signal array.
        template_tx: Transmit template (reference waveform).
        normalize: If True, normalize the output.  Default is False.
        window: Window function name to apply before compression.
            Empty string disables windowing.  Default is ``""``.
        nfft: FFT size.  If 0 (default), uses the next power of two.

    Returns:
        Pulse-compressed output array.
    """
    return _signal.pulse_compression(x, template_tx, normalize=normalize,
                                     window=window, nfft=nfft)


def pulse_doppler(x, window="", nfft=0):
    """Compute the pulse-Doppler map from a slow-time matrix.

    Args:
        x: Input 2-D array of shape ``(num_pulses, num_range_bins)``.
        window: Window function name to apply along the pulse axis.
            Empty string disables windowing.  Default is ``""``.
        nfft: FFT size for the Doppler dimension.  If 0 (default), uses
            the number of pulses.

    Returns:
        2-D pulse-Doppler map array.
    """
    return _signal.pulse_doppler(x, window=window, nfft=nfft)


def cfar_alpha(pfa, N):
    """Compute the CFAR threshold multiplier for a given PFA and window size.

    Args:
        pfa: Probability of false alarm.
        N: Number of reference cells.

    Returns:
        CFAR scaling factor (alpha).
    """
    return _signal.cfar_alpha(pfa, N)


def ca_cfar(data, guard_cells, reference_cells, pfa=1e-3):
    """Apply Cell-Averaging CFAR (CA-CFAR) detection.

    Args:
        data: Input array (1-D or N-D).
        guard_cells: Number of guard cells on each side per dimension.
        reference_cells: Number of reference cells on each side per
            dimension.
        pfa: Probability of false alarm.  Default is 1e-3.

    Returns:
        Boolean mask array of detected targets (same shape as *data*).
    """
    return _signal.ca_cfar(data, list(guard_cells), list(reference_cells),
                           pfa=pfa)


def mvdr(x, sv, calc_cov=True):
    """Apply MVDR (Capon) beamforming.

    Args:
        x: Input signal matrix of shape ``(num_elements, num_snapshots)``.
        sv: Steering vector.
        calc_cov: If True (default), estimate the covariance matrix from
            *x*.  If False, *x* is used as the covariance matrix directly.

    Returns:
        Beamformed output array.
    """
    return _signal.mvdr(x, sv, calc_cov=calc_cov)


def ambgfun(x, fs, prf=1.0, y=None, cut="2d", cutValue=0.0):
    """Compute the ambiguity function.

    Args:
        x: Input signal array.
        fs: Sampling frequency in Hz.
        prf: Pulse repetition frequency in Hz.  Default is 1.0.
        y: Optional second signal for cross-ambiguity.  If ``None``,
            computes the auto-ambiguity.
        cut: Cut type.  ``'2d'`` (default) returns the full 2-D ambiguity
            surface; ``'delay'`` returns a cut at constant Doppler;
            ``'doppler'`` returns a cut at constant delay.
        cutValue: Value at which to take the cut (in seconds for delay
            cut, in Hz for Doppler cut).  Default is 0.0.

    Returns:
        Ambiguity function array (1-D or 2-D depending on *cut*).
    """
    kwargs = {}
    if y is not None:
        kwargs["y"] = y
    return _signal.ambgfun(x, fs, prf, cut=cut, cutValue=cutValue, **kwargs)
