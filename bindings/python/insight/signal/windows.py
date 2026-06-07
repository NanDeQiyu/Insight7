"""
Window functions for signal processing.

Provides standard window functions used in spectral analysis,
filter design, and other signal processing tasks.

All functions return an :class:`insight.Array` of the specified
shape and data type.
"""

from insight._insight import signal as _signal

__all__ = [
    "boxcar",
    "triang",
    "parzen",
    "bohman",
    "bartlett",
    "cosine",
    "exponential",
    "blackman",
    "nuttall",
    "blackmanharris",
    "flattop",
    "hann",
    "general_hamming",
    "hamming",
    "tukey",
    "barthann",
    "gaussian",
    "general_gaussian",
    "kaiser",
    "chebwin",
    "taylor",
    "general_cosine",
    "get_window",
    "qmf",
]


def boxcar(M: int):
    """Return a boxcar (rectangular) window.

    Args:
        M: Number of points in the output window.

    Returns:
        Boxcar window of length M.
    """
    return _signal.boxcar(M)


def triang(M: int):
    """Return a triangular window.

    Args:
        M: Number of points in the output window.

    Returns:
        Triangular window of length M.
    """
    return _signal.triang(M)


def parzen(M: int):
    """Return a Parzen window.

    Args:
        M: Number of points in the output window.

    Returns:
        Parzen window of length M.
    """
    return _signal.parzen(M)


def bohman(M: int):
    """Return a Bohman window.

    Args:
        M: Number of points in the output window.

    Returns:
        Bohman window of length M.
    """
    return _signal.bohman(M)


def bartlett(M: int):
    """Return a Bartlett window.

    Args:
        M: Number of points in the output window.

    Returns:
        Bartlett window of length M.
    """
    return _signal.bartlett(M)


def cosine(M: int):
    """Return a cosine window.

    Args:
        M: Number of points in the output window.

    Returns:
        Cosine window of length M.
    """
    return _signal.cosine(M)


def exponential(M: int, tau: float = 1.0, center=None):
    """Return an exponential (Poisson) window.

    Args:
        M: Number of points in the output window.
        tau: Decay parameter. Default is 1.0.
        center: Center of the window. Default is (M-1)/2.

    Returns:
        Exponential window of length M.
    """
    if center is None:
        center = (M - 1) / 2.0
    return _signal.exponential(M, tau, center)


def blackman(M: int):
    """Return a Blackman window.

    Args:
        M: Number of points in the output window.

    Returns:
        Blackman window of length M.
    """
    return _signal.blackman(M)


def nuttall(M: int):
    """Return a Nuttall minimum 4-term Blackman-Harris window.

    Args:
        M: Number of points in the output window.

    Returns:
        Nuttall window of length M.
    """
    return _signal.nuttall(M)


def blackmanharris(M: int):
    """Return a Blackman-Harris window.

    Args:
        M: Number of points in the output window.

    Returns:
        Blackman-Harris window of length M.
    """
    return _signal.blackmanharris(M)


def flattop(M: int):
    """Return a flat-top window.

    Args:
        M: Number of points in the output window.

    Returns:
        Flat-top window of length M.
    """
    return _signal.flattop(M)


def hann(M: int):
    """Return a Hann window.

    Args:
        M: Number of points in the output window.

    Returns:
        Hann window of length M.
    """
    return _signal.hann(M)


def general_hamming(M: int, alpha: float):
    """Return a generalized Hamming window.

    Args:
        M: Number of points in the output window.
        alpha: The window coefficient.

    Returns:
        Generalized Hamming window of length M.
    """
    return _signal.general_hamming(M, alpha)


def hamming(M: int):
    """Return a Hamming window.

    Args:
        M: Number of points in the output window.

    Returns:
        Hamming window of length M.
    """
    return _signal.hamming(M)


def tukey(M: int, alpha: float = 0.5):
    """Return a Tukey (tapered cosine) window.

    Args:
        M: Number of points in the output window.
        alpha: Shape parameter. 0 gives a rectangular window, 1 gives a Hann
            window. Default is 0.5.

    Returns:
        Tukey window of length M.
    """
    return _signal.tukey(M, alpha)


def barthann(M: int):
    """Return a Bartlett-Hann window.

    Args:
        M: Number of points in the output window.

    Returns:
        Bartlett-Hann window of length M.
    """
    return _signal.barthann(M)


def gaussian(M: int, std: float):
    """Return a Gaussian window.

    Args:
        M: Number of points in the output window.
        std: The standard deviation.

    Returns:
        Gaussian window of length M.
    """
    return _signal.gaussian(M, std)


def general_gaussian(M: int, p: float, sig: float):
    """Return a generalized Gaussian window.

    Args:
        M: Number of points in the output window.
        p: Shape parameter.
        sig: Standard deviation.

    Returns:
        Generalized Gaussian window of length M.
    """
    return _signal.general_gaussian(M, p, sig)


def kaiser(M: int, beta: float):
    """Return a Kaiser window.

    Args:
        M: Number of points in the output window.
        beta: Shape parameter.

    Returns:
        Kaiser window of length M.
    """
    return _signal.kaiser(M, beta)


def chebwin(M: int, at: float):
    """Return a Dolph-Chebyshev window.

    Args:
        M: Number of points in the output window.
        at: Attenuation in dB.

    Returns:
        Dolph-Chebyshev window of length M.
    """
    return _signal.chebwin(M, at)


def taylor(M: int, nbar: int = 11, sll: float = -30, norm: bool = True):
    """Return a Taylor window.

    Args:
        M: Number of points in the output window.
        nbar: Number of nearly constant level sidelobes. Default is 11.
        sll: Desired sidelobe level in dB. Default is -30.
        norm: Whether to normalize. Default is True.

    Returns:
        Taylor window of length M.
    """
    return _signal.taylor(M, nbar, sll, norm)


def general_cosine(M: int, coeffs):
    """Return a window with arbitrary cosine coefficients.

    Args:
        M: Number of points in the output window.
        coeffs: Sequence of cosine coefficients.

    Returns:
        Window of length M.
    """
    return _signal.general_cosine(M, list(coeffs))


def get_window(window, Nx: int, fftbins: bool = True):
    """Return a window of a given type and length.

    Args:
        window: Window type string (e.g., 'hann', 'hamming', 'kaiser') or
            a tuple (name, param).
        Nx: Number of points in the output window.
        fftbins: If True, create a periodic window for FFT. Default is True.

    Returns:
        Window array of length Nx.
    """
    if isinstance(window, tuple):
        return _signal.get_window(window[0], Nx, fftbins, window[1])
    return _signal.get_window(window, Nx, fftbins)


def qmf(h_low):
    """Return a Quadrature Mirror Filter (QMF) pair.

    Constructs the lowpass and highpass filters for a quadrature mirror
    filter bank. The highpass filter is a frequency-shifted and
    time-reversed version of the lowpass filter.

    Args:
        h_low: Lowpass filter coefficients (1D array).

    Returns:
        Tuple (lowpass, highpass) of filter arrays.
    """
    return _signal.qmf(h_low)
