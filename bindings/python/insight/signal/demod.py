"""FM demodulation functions for signal processing.

Provides frequency demodulation of narrowband FM signals.
"""

from insight._insight import signal as _signal

__all__ = [
    "fm_demod",
]


def fm_demod(x, axis=-1):
    """Demodulate an FM signal.

    Estimates the instantaneous frequency of a narrowband FM signal by
    computing the derivative of the unwrapped phase.

    Args:
        x: Input complex-valued signal array.
        axis: Axis along which to demodulate.  Default is -1 (last axis).

    Returns:
        Real-valued demodulated signal (instantaneous frequency).
    """
    return _signal.fm_demod(x, axis=axis)
