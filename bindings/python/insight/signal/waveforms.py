"""Waveform generation functions for signal processing.

Provides standard signal generators such as sawtooth, square wave,
Gaussian pulse, chirp, and unit impulse.
"""

from insight._insight import signal as _signal

__all__ = [
    "sawtooth",
    "square_wf",
    "gausspulse",
    "gausspulse_full",
    "chirp",
    "unit_impulse",
]

# Map string method names to the C++ ChirpMethod enum values.
_CHIRP_METHOD_MAP = {
    "linear": _signal.ChirpMethod.linear,
    "quadratic": _signal.ChirpMethod.quadratic,
    "logarithmic": _signal.ChirpMethod.logarithmic,
    "hyperbolic": _signal.ChirpMethod.hyperbolic,
}


def sawtooth(t, width=1.0):
    """Return a periodic sawtooth or triangle waveform.

    The sawtooth waveform is periodic with period ``2*pi``, rising from -1
    to 1 on the interval ``[0, 2*pi*width]`` and falling from 1 to -1 on
    ``[2*pi*width, 2*pi]``.

    Args:
        t: Input time array.
        width: Rising segment duty cycle in the range [0, 1].  Default is 1.0
            (a standard sawtooth that rises from -1 to 1).

    Returns:
        Sawtooth waveform evaluated at each element of *t*.
    """
    return _signal.sawtooth(t, width=width)


def square_wf(t, duty=0.5):
    """Return a periodic square-wave waveform.

    The output is 1 for the first ``duty`` fraction of each period and -1
    for the remainder.

    Args:
        t: Input time array (radians).
        duty: Duty cycle in the range [0, 1].  Default is 0.5.

    Returns:
        Square-wave waveform evaluated at each element of *t*.
    """
    return _signal.square_wf(t, duty=duty)


def gausspulse(t, fc=1000, bw=0.5, bwr=-6, tpr=-60):
    """Return a Gaussian-modulated sinusoidal pulse (envelope only).

    Args:
        t: Input time array.
        fc: Center frequency in Hz.  Default is 1000.
        bw: Fractional bandwidth.  Default is 0.5.
        bwr: Reference level at which *bw* is measured in dB.  Default is -6.
        tpr: Threshold in dB below the peak for the pulse envelope.
            Default is -60.

    Returns:
        Gaussian pulse envelope evaluated at each element of *t*.
    """
    return _signal.gausspulse(t, fc=fc, bw=bw, bwr=bwr, tpr=tpr)


def gausspulse_full(t, fc=1000, bw=0.5, bwr=-6, tpr=-60):
    """Return a Gaussian-modulated sinusoidal pulse (I, Q, and envelope).

    Unlike :func:`gausspulse`, this returns the full in-phase, quadrature,
    and envelope components.

    Args:
        t: Input time array.
        fc: Center frequency in Hz.  Default is 1000.
        bw: Fractional bandwidth.  Default is 0.5.
        bwr: Reference level at which *bw* is measured in dB.  Default is -6.
        tpr: Threshold in dB below the peak for the pulse envelope.
            Default is -60.

    Returns:
        A ``GaussPulseResult`` with attributes ``yi`` (in-phase), ``yq``
        (quadrature), and ``ye`` (envelope).
    """
    return _signal.gausspulse_full(t, fc=fc, bw=bw, bwr=bwr, tpr=tpr)


def chirp(t, f0, t1, f1, method="linear", phi=0, vertex_zero=True):
    """Return a frequency-swept cosine generator (chirp signal).

    Args:
        t: Input time array.
        f0: Frequency at time 0 in Hz.
        t1: Time at which *f1* is specified.
        f1: Frequency at time *t1* in Hz.
        method: Sweep method.  One of ``'linear'``, ``'quadratic'``,
            ``'logarithmic'``, or ``'hyperbolic'``.  Default is ``'linear'``.
        phi: Phase offset in degrees.  Default is 0.
        vertex_zero: Whether the vertex of the quadratic sweep is at t=0.
            Only used when *method* is ``'quadratic'``.  Default is True.

    Returns:
        Chirp signal evaluated at each element of *t*.
    """
    enum_method = _CHIRP_METHOD_MAP.get(method, _signal.ChirpMethod.linear)
    return _signal.chirp(t, f0, t1, f1, method=enum_method, phi=phi,
                         vertex_zero=vertex_zero)


def unit_impulse(shape, idx=None, dtype=None, place=None):
    """Return an impulse (delta) function of the given shape.

    Generates an array of zeros with a single 1 at the position specified
    by *idx*.

    Args:
        shape: Output shape (int or tuple of ints).
        idx: Index at which the value 1 is placed.  If ``None``, the
            impulse is at the origin (index 0 for each dimension).
        dtype: Data type of the output array.  If ``None``, defaults to
            ``float64``.
        place: Device placement (CPU or GPU).  If ``None``, uses the
            current default device.

    Returns:
        Impulse array of the specified shape.
    """
    kwargs = {}
    if idx is not None:
        kwargs["idx"] = idx
    if dtype is not None:
        kwargs["dtype"] = dtype
    if place is not None:
        kwargs["place"] = place
    return _signal.unit_impulse(shape, **kwargs)
