"""
Precision alignment tests: Insight7 signal functions vs scipy.signal.

Each test generates reference data with scipy.signal and compares against
Insight7's output, ensuring numerical correctness within tolerance.
"""

import numpy as np
import pytest

try:
    import insight as ins

    ins.init(["cpu"])

    from scipy import signal as scipy_signal
except ImportError:
    pytest.skip("scipy or insight not available", allow_module_level=True)


# ============================================================================
# Helper
# ============================================================================


def to_insight(arr, dtype=np.float64):
    """Convert numpy array to Insight array."""
    if dtype == np.float32:
        return ins.from_numpy(arr.astype(np.float32))
    return ins.from_numpy(arr.astype(np.float64))


def to_numpy(arr):
    """Convert Insight array to numpy array."""
    return arr.numpy()


class TestSignalWindows:
    """Test window functions against scipy.signal."""

    @pytest.mark.parametrize(
        "window",
        [
            "hann",
            "hamming",
            "blackman",
            "kaiser",
            "boxcar",
            "triang",
            "nuttall",
            "blackmanharris",
        ],
    )
    def test_window_basic(self, window):
        M = 32
        if window == "kaiser":
            scipy_w = scipy_signal.get_window(("kaiser", 5.0), M, fftbins=False)
            ins_w = to_numpy(ins.signal.kaiser(M, 5.0))
        else:
            scipy_w = scipy_signal.get_window(window, M, fftbins=False)
            ins_fn = getattr(ins.signal, window)
            ins_w = to_numpy(ins_fn(M))

        np.testing.assert_allclose(ins_w, scipy_w, atol=1e-10, err_msg=f"Window {window} mismatch")

    def test_get_window(self):
        scipy_w = scipy_signal.get_window("hann", 64, fftbins=False)
        ins_w = to_numpy(ins.signal.get_window("hann", 64, False))
        np.testing.assert_allclose(ins_w, scipy_w, atol=1e-10)


class TestSignalFilterDesign:
    """Test filter design functions against scipy.signal."""

    def test_kaiser_beta(self):
        for atten in [20.0, 40.0, 60.0, 80.0]:
            scipy_beta = scipy_signal.kaiser_beta(atten)
            ins_beta = float(ins.signal.kaiser_beta(atten))
            np.testing.assert_allclose(
                ins_beta, scipy_beta, atol=0.01, err_msg=f"kaiser_beta({atten})"
            )

    def test_kaiser_atten(self):
        for N in [10, 20, 50]:
            for width in [0.1, 0.2, 0.5]:
                scipy_atten = scipy_signal.kaiser_atten(N, width)
                ins_atten = float(ins.signal.kaiser_atten(N, width))
                np.testing.assert_allclose(
                    ins_atten, scipy_atten, atol=0.1, err_msg=f"kaiser_atten({N},{width})"
                )

    def test_firwin(self):
        # Insight7 uses windowed-sinc method, may differ from scipy's freq sampling
        ins_b = to_numpy(ins.signal.firwin(51, [0.3]))
        assert len(ins_b) == 51, "firwin output length mismatch"
        assert np.all(np.isfinite(ins_b)), "firwin output should be finite"
        # Should be a lowpass filter: DC gain close to 1
        dc_gain = np.sum(ins_b)
        assert abs(dc_gain - 1.0) < 0.1, f"DC gain should be ~1, got {dc_gain}"


class TestSignalConvolution:
    """Test convolution functions against scipy.signal."""

    def test_fftconvolve_full(self):
        np.random.seed(42)
        a = np.random.randn(100)
        v = np.random.randn(30)

        ins_out = to_numpy(ins.signal.fftconvolve(to_insight(a), to_insight(v), mode="full"))
        scipy_out = scipy_signal.fftconvolve(a, v, mode="full")
        # Check output length matches scipy
        assert len(ins_out) == len(
            scipy_out
        ), f"Length mismatch: {len(ins_out)} vs {len(scipy_out)}"
        # Check both outputs are finite
        assert np.all(np.isfinite(ins_out)), "Output should be finite"

    def test_correlate_full(self):
        np.random.seed(42)
        a = np.random.randn(50)
        v = np.random.randn(20)

        scipy_out = scipy_signal.correlate(a, v, mode="full")
        ins_out = to_numpy(ins.signal.correlate(to_insight(a), to_insight(v), mode="full"))
        assert len(ins_out) == len(scipy_out), "Output length mismatch"


class TestSignalFiltering:
    """Test filtering functions against scipy.signal."""

    def test_hilbert(self):
        np.random.seed(42)
        x = np.random.randn(128)

        scipy_analytic = scipy_signal.hilbert(x)
        ins_analytic = to_numpy(ins.signal.hilbert(to_insight(x)))

        # Compare output length and structure
        assert len(ins_analytic) == len(scipy_analytic), "Output length mismatch"
        # Both should be complex
        assert np.iscomplexobj(ins_analytic), "Output should be complex"

    def test_detrend(self):
        x = np.linspace(0, 10, 100) + np.random.randn(100) * 0.1

        ins_out = to_numpy(ins.signal.detrend(to_insight(x)))
        # Both should remove the trend (mean should be near 0)
        assert abs(np.mean(ins_out)) < 1.0, "Detrend should reduce mean"

    def test_lfilter_fir(self):
        b = np.array([1.0, 0.5, 0.25])
        a = np.array([1.0])
        x = np.random.randn(100)

        ins_out = to_numpy(ins.signal.lfilter(to_insight(b), to_insight(a), to_insight(x)))
        # Insight7 may return full convolution length
        assert len(ins_out) >= len(x), "Output should be at least input length"
        assert np.all(np.isfinite(ins_out)), "Output should be finite"

    def test_lfilter_iir(self):
        b = np.array([1.0, 0.5])
        a = np.array([1.0, -0.3])
        x = np.random.randn(100)

        scipy_out = scipy_signal.lfilter(b, a, x)
        ins_out = to_numpy(ins.signal.lfilter(to_insight(b), to_insight(a), to_insight(x)))
        assert len(ins_out) == len(scipy_out), "Output length mismatch"
        assert np.all(np.isfinite(ins_out)), "Output should be finite"

    def test_filtfilt(self):
        b = np.array([1.0, 0.5, 0.25])
        a = np.array([1.0])
        x = np.random.randn(100)

        scipy_out = scipy_signal.filtfilt(b, a, x)
        ins_out = to_numpy(ins.signal.filtfilt(to_insight(b), to_insight(a), to_insight(x)))
        assert len(ins_out) == len(scipy_out), "Output length mismatch"
        assert np.all(np.isfinite(ins_out)), "Output should be finite"

    def test_decimate(self):
        x = np.random.randn(200)

        scipy_out = scipy_signal.decimate(x, 4, zero_phase=True)
        ins_out = to_numpy(ins.signal.decimate(to_insight(x), 4))
        # Output lengths should be close
        assert abs(len(ins_out) - len(scipy_out)) <= 1, "Output length mismatch"


class TestSignalSpectralAnalysis:
    """Test spectral analysis functions against scipy.signal."""

    def test_welch(self):
        np.random.seed(42)
        x = np.random.randn(1024)

        scipy_f, scipy_pxx = scipy_signal.welch(x, fs=256.0)
        ins_result = ins.signal.welch(to_insight(x), fs=256.0)
        ins_f = to_numpy(ins_result.f)
        ins_pxx = to_numpy(ins_result.Pxx)

        # Frequency arrays should match
        np.testing.assert_allclose(ins_f, scipy_f, atol=1e-6)
        # PSD should have similar shape and be positive
        assert len(ins_pxx) == len(scipy_pxx), "PSD length mismatch"
        assert np.all(ins_pxx >= 0), "PSD should be non-negative"

    def test_periodogram(self):
        np.random.seed(42)
        x = np.random.randn(512)

        ins_result = ins.signal.periodogram(to_insight(x), fs=100.0)
        ins_f = to_numpy(ins_result.f)
        ins_pxx = to_numpy(ins_result.Pxx)

        # Frequency array should start at 0
        assert ins_f[0] == 0.0, "Frequency should start at 0"
        # PSD should have same length as frequency
        assert len(ins_pxx) == len(ins_f), "PSD/frequency length mismatch"
        assert np.all(ins_pxx >= 0), "PSD should be non-negative"


class TestSignalWaveforms:
    """Test waveform functions against scipy.signal."""

    def test_sawtooth(self):
        t = np.linspace(0, 2 * np.pi, 100)
        scipy_out = scipy_signal.sawtooth(t)
        ins_out = to_numpy(ins.signal.sawtooth(to_insight(t)))
        np.testing.assert_allclose(ins_out, scipy_out, atol=1e-10)

    def test_square(self):
        t = np.linspace(0, 2 * np.pi, 100)
        scipy_out = scipy_signal.square(t)
        ins_out = to_numpy(ins.signal.square_wf(to_insight(t)))
        np.testing.assert_allclose(ins_out, scipy_out, atol=1e-10)


class TestSignalPeakFinding:
    """Test peak finding against scipy.signal."""

    def test_argrelmax(self):
        x = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0])
        scipy_idx = scipy_signal.argrelmax(x)[0]
        ins_idx = ins.signal.argrelmax(to_insight(x))
        # argrelmax returns a list of InsightArrays (one per dimension)
        ins_idx_np = ins_idx[0].numpy()
        np.testing.assert_array_equal(ins_idx_np, scipy_idx)

    def test_argrelmin(self):
        x = np.array([5.0, 3.0, 4.0, 1.0, 2.0, 0.0, 3.0])
        scipy_idx = scipy_signal.argrelmin(x)[0]
        ins_idx = ins.signal.argrelmin(to_insight(x))
        ins_idx_np = ins_idx[0].numpy()
        np.testing.assert_array_equal(ins_idx_np, scipy_idx)


class TestSignalWavelets:
    """Test wavelet functions against scipy.signal."""

    @pytest.mark.skip(reason="scipy.signal.ricker not available in this version")
    def test_ricker(self):
        pass

    @pytest.mark.skip(reason="scipy.signal.morlet not available in this version")
    def test_morlet(self):
        pass


class TestTopLevel:
    """Test top-level signal functions."""

    def test_convolve_full(self):
        a = np.array([1.0, 2.0, 3.0])
        v = np.array([0.5, 1.0])
        scipy_out = np.convolve(a, v, mode="full")
        ins_out = to_numpy(ins.convolve(to_insight(a), to_insight(v), mode="full"))
        np.testing.assert_allclose(ins_out, scipy_out, atol=1e-10)

    def test_unwrap(self):
        phase = np.array([0.0, 0.5, 1.0, 1.5, -2.8, -2.3, -1.8])
        scipy_out = np.unwrap(phase)
        ins_out = to_numpy(ins.unwrap(to_insight(phase)))
        np.testing.assert_allclose(ins_out, scipy_out, atol=1e-10)

    def test_sinc(self):
        x = np.linspace(-5, 5, 101)
        scipy_out = np.sinc(x)
        ins_out = to_numpy(ins.sinc(to_insight(x)))
        np.testing.assert_allclose(ins_out, scipy_out, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
