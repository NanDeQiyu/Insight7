"""Acoustic frequency scale functions for signal processing.

Provides conversions between Hz, Mel, and Bark frequency scales,
and Mel frequency bank generation.
"""

from insight._insight import signal as _signal

__all__ = [
    "mel2hz",
    "hz2mel",
    "hz2bark",
    "bark2hz",
    "mel_frequencies",
]


def mel2hz(mel):
    """Convert Mel-scale frequencies to Hz.

    Args:
        mel: Mel-scale frequency value or array.

    Returns:
        Corresponding frequency in Hz.
    """
    return _signal.mel2hz(mel)


def hz2mel(hz):
    """Convert frequencies in Hz to the Mel scale.

    Args:
        hz: Frequency value or array in Hz.

    Returns:
        Corresponding Mel-scale value.
    """
    return _signal.hz2mel(hz)


def hz2bark(hz):
    """Convert frequencies in Hz to the Bark scale.

    Args:
        hz: Frequency value or array in Hz.

    Returns:
        Corresponding Bark-scale value.
    """
    return _signal.hz2bark(hz)


def bark2hz(bark):
    """Convert Bark-scale frequencies to Hz.

    Args:
        bark: Bark-scale frequency value or array.

    Returns:
        Corresponding frequency in Hz.
    """
    return _signal.bark2hz(bark)


def mel_frequencies(num_mel=128, fmin=0.0, fmax=11025.0, htk=False):
    """Compute Mel-spaced frequencies.

    Args:
        num_mel: Number of Mel frequencies to generate.  Default is 128.
        fmin: Minimum frequency in Hz.  Default is 0.0.
        fmax: Maximum frequency in Hz.  Default is 11025.0.
        htk: If True, use the HTK formula instead of Slaney's.  Default is
            False.

    Returns:
        1-D array of *num_mel* equally spaced frequencies on the Mel scale.

    Note:
        The *htk* parameter is accepted for API compatibility but is
        currently ignored by the backend.
    """
    return _signal.mel_frequencies(num_mel, f_low=fmin, f_high=fmax)
