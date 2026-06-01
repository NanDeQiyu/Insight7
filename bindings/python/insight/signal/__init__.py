"""
Insight7 Signal Processing Module.

Provides signal processing functions compatible with scipy.signal,
organized into sub-modules matching the C++ source structure.

Sub-modules:
    - windows: Window functions (hann, hamming, kaiser, etc.)
    - waveforms: Signal generators (sawtooth, square, chirp, etc.)
    - bsplines: B-spline basis functions
    - filter_design: FIR/IIR filter design
    - convolution: FFT-based convolution and correlation
    - filtering: Digital filtering (lfilter, filtfilt, etc.)
    - spectral: Spectral analysis (welch, csd, spectrogram, etc.)
    - wavelets: Wavelet functions and CWT
    - acoustics: Acoustic frequency scales (mel, bark, etc.)
    - demod: FM demodulation
    - peak_finding: Peak detection algorithms
    - radar: Radar signal processing (CFAR, pulse compression, etc.)
    - estimation: Kalman filter
    - io: Binary signal I/O
"""

from .windows import *  # noqa: F401,F403
from .waveforms import *  # noqa: F401,F403
from .bsplines import *  # noqa: F401,F403
from .filter_design import *  # noqa: F401,F403
from .convolution import *  # noqa: F401,F403
from .filtering import *  # noqa: F401,F403
from .spectral import *  # noqa: F401,F403
from .wavelets import *  # noqa: F401,F403
from .acoustics import *  # noqa: F401,F403
from .demod import *  # noqa: F401,F403
from .peak_finding import *  # noqa: F401,F403
from .radar import *  # noqa: F401,F403
from .estimation import *  # noqa: F401,F403
from .io import *  # noqa: F401,F403
