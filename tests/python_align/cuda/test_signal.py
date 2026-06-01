"""Signal alignment tests — Insight CUDA vs SciPy.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \\
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \\
    python -m pytest tests/python_align/cuda/test_signal.py -v
"""

import sys
import os
import pytest

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, os.path.join(_root, "bindings", "python"))
sys.path.insert(0, os.path.join(_root, "build", "bindings", "python"))

try:
    import insight as ins
    import numpy as np
    from numpy.testing import assert_allclose
except ImportError:
    pytest.skip("Insight or NumPy not available", allow_module_level=True)

# Skip if CUDA not available
try:
    ins.load_backend("cuda")
except Exception:
    pytest.skip("CUDA backend not available", allow_module_level=True)

GPU = ins.GPUPlace(0)
CPU = ins.CPUPlace()

# Optional SciPy — needed for signal tests
try:
    import scipy.signal as sp_signal
    from scipy.signal.windows import (
        hann as scipy_hann,
        hamming as scipy_hamming,
        kaiser as scipy_kaiser,
        blackman as scipy_blackman,
        boxcar as scipy_boxcar,
        bartlett as scipy_bartlett,
        tukey as scipy_tukey,
        triang as scipy_triang,
    )

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def to_gpu(np_arr):
    return ins.from_numpy(np_arr).to(GPU)


def to_numpy(gpu_arr):
    return gpu_arr.to(CPU).numpy()


# ============================================================================
# Signal windows alignment (GPU)
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestSignalWindowsGPU:
    """Insight signal windows on GPU vs SciPy."""

    def test_hann(self):
        result = ins.signal.hann(64)
        expected = scipy_hann(64)
        assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_hamming(self):
        result = ins.signal.hamming(64)
        expected = scipy_hamming(64)
        assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_kaiser(self):
        result = ins.signal.kaiser(64, 5.0)
        expected = scipy_kaiser(64, 5.0)
        assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_blackman(self):
        result = ins.signal.blackman(64)
        expected = scipy_blackman(64)
        assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_boxcar(self):
        result = ins.signal.boxcar(32)
        expected = scipy_boxcar(32)
        assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_bartlett(self):
        result = ins.signal.bartlett(64)
        expected = scipy_bartlett(64)
        assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_tukey(self):
        result = ins.signal.tukey(64, 0.5)
        expected = scipy_tukey(64, 0.5)
        assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_triang(self):
        result = ins.signal.triang(63)
        expected = scipy_triang(63)
        assert_allclose(result.numpy(), expected, atol=1e-10)


# ============================================================================
# Signal waveforms alignment (GPU)
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestSignalWaveformsGPU:
    """Insight signal waveforms on GPU vs SciPy."""

    def test_sawtooth(self):
        t_np = np.linspace(0, 2 * np.pi, 100, dtype=np.float64)
        t = to_gpu(t_np)
        result = ins.sawtooth(t)
        expected = sp_signal.sawtooth(t_np)
        assert_allclose(to_numpy(result), expected, atol=1e-8)

    def test_square_wf(self):
        t_np = np.linspace(0, 2 * np.pi, 100, dtype=np.float64)
        t = to_gpu(t_np)
        result = ins.square_wf(t)
        expected = sp_signal.square(t_np)
        assert_allclose(to_numpy(result), expected, atol=1e-8)

    def test_gausspulse(self):
        t_np = np.linspace(-0.01, 0.01, 200, dtype=np.float64)
        t = to_gpu(t_np)
        result = ins.gausspulse(t, fc=1000)
        expected = sp_signal.gausspulse(t_np, fc=1000)
        assert_allclose(to_numpy(result), expected, rtol=1e-4)

    def test_chirp_linear(self):
        t_np = np.linspace(0, 10, 1000, dtype=np.float64)
        t = to_gpu(t_np)
        result = ins.chirp(t, f0=1, t1=10, f1=10, method="linear")
        expected = sp_signal.chirp(t_np, f0=1, t1=10, f1=10, method="linear")
        assert_allclose(to_numpy(result), expected, atol=1e-6)

    def test_unit_impulse(self):
        result = ins.unit_impulse(10)
        expected = sp_signal.unit_impulse(10)
        assert_allclose(result.numpy(), expected)


# ============================================================================
# Signal wavelets alignment (GPU)
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestSignalWaveletsGPU:
    """Insight signal wavelets on GPU vs SciPy."""

    def test_ricker(self):
        result = ins.ricker(100, 4.0)
        expected = sp_signal.ricker(100, 4.0)
        assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_morlet(self):
        result = ins.morlet(50, w=5.0, s=1.0, complete=True)
        expected = sp_signal.morlet(50, w=5.0, s=1.0, complete=True)
        assert_allclose(result.numpy(), expected, atol=1e-8)


# ============================================================================
# Signal filter design alignment (GPU)
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestSignalFilterDesignGPU:
    """Insight filter design on GPU vs SciPy."""

    def test_firwin_lowpass(self):
        result = ins.firwin(15, 0.3, window="hamming", pass_zero="lowpass")
        expected = sp_signal.firwin(15, 0.3, window="hamming", pass_zero="lowpass")
        assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_firwin2(self):
        freq = [0, 0.5, 1.0]
        gain = [1, 0.5, 0]
        result = ins.firwin2(15, freq, gain)
        expected = sp_signal.firwin2(15, freq, gain)
        assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_kaiser_beta(self):
        result = ins.kaiser_beta(50)
        expected = sp_signal.kaiser_beta(50)
        assert_allclose(result.numpy(), expected, rtol=1e-6)


# ============================================================================
# Signal convolution alignment (GPU)
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestSignalConvolutionGPU:
    """Insight signal convolution on GPU vs SciPy."""

    def test_fftconvolve_full(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5], dtype=np.float64)
        a, b = to_gpu(a_np), to_gpu(b_np)
        result = ins.fftconvolve(a, b)
        expected = sp_signal.fftconvolve(a_np, b_np)
        assert_allclose(to_numpy(result), expected, atol=1e-8)

    def test_fftconvolve_same(self):
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        b_np = np.array([1, 1, 1], dtype=np.float64)
        a, b = to_gpu(a_np), to_gpu(b_np)
        result = ins.fftconvolve(a, b, mode="same")
        expected = sp_signal.fftconvolve(a_np, b_np, mode="same")
        assert_allclose(to_numpy(result), expected, atol=1e-8)

    def test_correlate(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5, 6], dtype=np.float64)
        a, b = to_gpu(a_np), to_gpu(b_np)
        result = ins.correlate(a, b)
        expected = sp_signal.correlate(a_np, b_np)
        assert_allclose(to_numpy(result), expected, atol=1e-8)


# ============================================================================
# Signal filtering alignment (GPU)
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestSignalFilteringGPU:
    """Insight signal filtering on GPU vs SciPy."""

    def test_hilbert(self):
        x_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.hilbert(x)
        expected = sp_signal.hilbert(x_np)
        assert_allclose(to_numpy(result), expected, atol=1e-8)

    def test_detrend_linear(self):
        x_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.detrend(x, type="linear")
        expected = sp_signal.detrend(x_np, type="linear")
        assert_allclose(to_numpy(result), expected, atol=1e-8)

    def test_detrend_constant(self):
        x_np = np.array([10, 11, 12, 13, 14], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.detrend(x, type="constant")
        expected = sp_signal.detrend(x_np, type="constant")
        assert_allclose(to_numpy(result), expected, atol=1e-8)

    def test_lfilter(self):
        b_np = np.array([1.0, 1.0], dtype=np.float64)
        a_np = np.array([1.0, -0.5], dtype=np.float64)
        x_np = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        b, a, x = to_gpu(b_np), to_gpu(a_np), to_gpu(x_np)
        result = ins.lfilter(b, a, x)
        expected = sp_signal.lfilter(b_np, a_np, x_np)
        assert_allclose(to_numpy(result), expected, atol=1e-8)

    def test_filtfilt(self):
        b_np = np.array([1.0, 1.0], dtype=np.float64)
        a_np = np.array([1.0, -0.5], dtype=np.float64)
        x_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        b, a, x = to_gpu(b_np), to_gpu(a_np), to_gpu(x_np)
        result = ins.filtfilt(b, a, x)
        expected = sp_signal.filtfilt(b_np, a_np, x_np)
        assert_allclose(to_numpy(result), expected, atol=1e-6)

    def test_resample(self):
        x_np = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100, dtype=np.float64))
        x = to_gpu(x_np)
        result = ins.resample(x, 50)
        expected = sp_signal.resample(x_np, 50)
        assert_allclose(to_numpy(result), expected, rtol=1e-4)

    def test_decimate(self):
        x_np = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100, dtype=np.float64))
        x = to_gpu(x_np)
        result = ins.decimate(x, 5)
        expected = sp_signal.decimate(x_np, 5)
        assert_allclose(to_numpy(result), expected, rtol=1e-2)


# ============================================================================
# Signal spectral analysis alignment (GPU)
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestSignalSpectralGPU:
    """Insight signal spectral analysis on GPU vs SciPy."""

    def test_welch(self):
        rng = np.random.RandomState(42)
        x_np = rng.randn(1024).astype(np.float64)
        x = to_gpu(x_np)
        result = ins.welch(x, fs=1000.0, nperseg=256)
        f_expected, Pxx_expected = sp_signal.welch(x_np, fs=1000.0, nperseg=256)
        assert_allclose(result.f.numpy(), f_expected, rtol=1e-6)
        assert_allclose(result.Pxx.numpy(), Pxx_expected, rtol=1e-4)

    def test_periodogram(self):
        rng = np.random.RandomState(42)
        x_np = rng.randn(512).astype(np.float64)
        x = to_gpu(x_np)
        result = ins.periodogram(x, fs=1000.0)
        f_expected, Pxx_expected = sp_signal.periodogram(x_np, fs=1000.0)
        assert_allclose(result.f.numpy(), f_expected, rtol=1e-6)
        assert_allclose(result.Pxx.numpy(), Pxx_expected, rtol=1e-4)

    def test_csd(self):
        rng = np.random.RandomState(42)
        x_np = rng.randn(1024).astype(np.float64)
        y_np = rng.randn(1024).astype(np.float64)
        x, y = to_gpu(x_np), to_gpu(y_np)
        result = ins.csd(x, y, fs=1000.0, nperseg=256)
        f_expected, Pxy_expected = sp_signal.csd(x_np, y_np, fs=1000.0, nperseg=256)
        assert_allclose(result.f.numpy(), f_expected, rtol=1e-6)
        assert_allclose(result.Pxx.numpy(), Pxy_expected, rtol=1e-3)

    def test_coherence(self):
        rng = np.random.RandomState(42)
        x_np = rng.randn(1024).astype(np.float64)
        y_np = rng.randn(1024).astype(np.float64)
        x, y = to_gpu(x_np), to_gpu(y_np)
        result = ins.coherence(x, y, fs=1000.0, nperseg=256)
        f_expected, Cxy_expected = sp_signal.coherence(x_np, y_np, fs=1000.0, nperseg=256)
        assert_allclose(result.f.numpy(), f_expected, rtol=1e-6)
        assert_allclose(result.Pxx.numpy(), Cxy_expected, rtol=1e-3)

    def test_spectrogram(self):
        rng = np.random.RandomState(42)
        x_np = rng.randn(1024).astype(np.float64)
        x = to_gpu(x_np)
        result = ins.spectrogram(x, fs=1000.0, nperseg=256)
        f_exp, t_exp, Sxx_exp = sp_signal.spectrogram(x_np, fs=1000.0, nperseg=256)
        assert_allclose(result.f.numpy(), f_exp, rtol=1e-6)
        assert_allclose(result.Sxx.numpy(), Sxx_exp, rtol=1e-3)

    def test_stft(self):
        rng = np.random.RandomState(42)
        x_np = rng.randn(1024).astype(np.float64)
        x = to_gpu(x_np)
        result = ins.stft(x, fs=1000.0, nperseg=256)
        f_exp, t_exp, Zxx_exp = sp_signal.stft(x_np, fs=1000.0, nperseg=256)
        assert_allclose(result.f.numpy(), f_exp, rtol=1e-6)
        assert_allclose(result.Sxx.numpy(), Zxx_exp, rtol=1e-3)


# ============================================================================
# B-spline alignment (GPU)
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestSignalBSplinesGPU:
    """Insight B-spline functions on GPU vs SciPy."""

    def test_cubic(self):
        x_np = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.cubic(x)
        from scipy.signal import cubic as scipy_cubic

        expected = scipy_cubic(x_np)
        assert_allclose(to_numpy(result), expected, atol=1e-10)

    def test_quadratic(self):
        x_np = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.quadratic(x)
        from scipy.signal import quadratic as scipy_quadratic

        expected = scipy_quadratic(x_np)
        assert_allclose(to_numpy(result), expected, atol=1e-10)


# ============================================================================
# Acoustics alignment (GPU)
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestSignalAcousticsGPU:
    """Insight acoustic functions on GPU vs SciPy."""

    def test_hz2mel(self):
        x_np = np.array([0, 100, 1000, 10000], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.hz2mel(x)
        expected = sp_signal.hz2mel(x_np)
        assert_allclose(to_numpy(result), expected, rtol=1e-6)

    def test_mel2hz(self):
        mel_np = np.array([0, 100, 500, 1000], dtype=np.float64)
        mel = to_gpu(mel_np)
        result = ins.mel2hz(mel)
        expected = sp_signal.mel2hz(mel_np)
        assert_allclose(to_numpy(result), expected, rtol=1e-6)

    def test_hz2bark(self):
        x_np = np.array([0, 100, 1000, 10000], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.hz2bark(x)
        expected = sp_signal.hz2bark(x_np)
        assert_allclose(to_numpy(result), expected, rtol=1e-6)

    def test_bark2hz(self):
        bark_np = np.array([0, 1, 5, 10, 20], dtype=np.float64)
        bark = to_gpu(bark_np)
        result = ins.bark2hz(bark)
        expected = sp_signal.bark2hz(bark_np)
        assert_allclose(to_numpy(result), expected, rtol=1e-6)

    def test_mel_frequencies(self):
        result = ins.mel_frequencies(num_mel=16, fmin=0.0, fmax=8000.0)
        expected = sp_signal.mel_frequencies(num_mel=16, fmin=0.0, fmax=8000.0)
        assert_allclose(result.numpy(), expected, rtol=1e-6)
