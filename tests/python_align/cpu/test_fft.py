"""FFT alignment tests — Insight vs NumPy.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python python -m pytest tests/python_align/cpu/test_fft.py -v
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
    pytest.skip("Insight, NumPy, or SciPy not available", allow_module_level=True)


class TestFFTAlignment:
    """Insight FFT vs NumPy FFT."""

    def test_fft(self):
        x_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        x = ins.from_numpy(x_np)
        result = ins.fft(x)
        assert_allclose(result.numpy(), np.fft.fft(x_np), atol=1e-8)

    def test_ifft(self):
        x_np = np.array([1, 2, 3, 4], dtype=np.complex128)
        x = ins.from_numpy(x_np)
        result = ins.ifft(x)
        assert_allclose(result.numpy(), np.fft.ifft(x_np), atol=1e-8)

    def test_fft_ifft_roundtrip(self):
        x_np = np.random.randn(16).astype(np.float64)
        x = ins.from_numpy(x_np)
        result = ins.ifft(ins.fft(x))
        assert_allclose(result.numpy().real, x_np, atol=1e-8)

    def test_rfft(self):
        x_np = np.array([1, 2, 3, 4], dtype=np.float64)
        x = ins.from_numpy(x_np)
        result = ins.rfft(x)
        assert_allclose(result.numpy(), np.fft.rfft(x_np), atol=1e-8)

    def test_fftfreq(self):
        result = ins.fftfreq(8, 0.1)
        assert_allclose(result.numpy(), np.fft.fftfreq(8, 0.1))

    def test_rfftfreq(self):
        result = ins.rfftfreq(8, 0.1)
        assert_allclose(result.numpy(), np.fft.rfftfreq(8, 0.1))


class TestFFTExtended:
    """Extended FFT ops vs NumPy FFT."""

    def test_fft2(self):
        x_np = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        # fft2 requires complex input in Insight
        x_complex = x_np.astype(np.complex128)
        x = ins.from_numpy(x_complex)
        result = ins.fft2(x)
        assert_allclose(result.numpy(), np.fft.fft2(x_np), atol=1e-6)

    def test_ifft2(self):
        x_np = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.complex128)
        x = ins.from_numpy(x_np)
        result = ins.ifft2(x)
        assert_allclose(result.numpy(), np.fft.ifft2(x_np), atol=1e-6)

    def test_fftn(self):
        x_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        x = ins.from_numpy(x_np)
        result = ins.fftn(x)
        assert_allclose(result.numpy(), np.fft.fftn(x_np), atol=1e-6)

    def test_ifftn(self):
        x_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        x = ins.from_numpy(x_np)
        result = ins.ifftn(x)
        assert_allclose(result.numpy(), np.fft.ifftn(x_np), atol=1e-6)

    def test_rfft_irfft_roundtrip(self):
        x_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        x = ins.from_numpy(x_np)
        freq = ins.rfft(x)
        result = ins.irfft(freq, 8)
        assert_allclose(result.numpy(), x_np, atol=1e-8)

    def test_fft2_roundtrip(self):
        x_np = np.random.RandomState(42).randn(4, 4).astype(np.float64)
        # fft2 requires complex input in Insight
        x_complex = x_np.astype(np.complex128)
        x = ins.from_numpy(x_complex)
        result = ins.ifft2(ins.fft2(x))
        assert_allclose(result.numpy().real, x_np, atol=1e-8)


class TestFFTShift:
    """fftshift / ifftshift alignment against NumPy."""

    def test_fftshift_1d(self):
        x = ins.from_numpy(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64))
        result = ins.fftshift(x)
        assert_allclose(
            result.numpy(), np.fft.fftshift(np.array([1, 2, 3, 4, 5, 6, 7, 8])), atol=1e-10
        )

    def test_fftshift_fftfreq(self):
        x = ins.fftfreq(16, 100e-6)
        result = ins.fftshift(x)
        assert_allclose(result.numpy(), np.fft.fftshift(np.fft.fftfreq(16, 100e-6)), atol=1e-10)

    def test_fftshift_2d(self):
        x = ins.from_numpy(np.arange(24, dtype=np.float64).reshape(4, 6))
        result = ins.fftshift(x)
        assert_allclose(result.numpy(), np.fft.fftshift(np.arange(24).reshape(4, 6)), atol=1e-10)

    def test_fftshift_axis(self):
        x = ins.from_numpy(np.arange(24, dtype=np.float64).reshape(4, 6))
        result = ins.fftshift(x, axis=1)
        assert_allclose(
            result.numpy(), np.fft.fftshift(np.arange(24).reshape(4, 6), axes=1), atol=1e-10
        )

    def test_ifftshift_1d(self):
        x = ins.from_numpy(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64))
        result = ins.ifftshift(x)
        assert_allclose(
            result.numpy(), np.fft.ifftshift(np.array([1, 2, 3, 4, 5, 6, 7, 8])), atol=1e-10
        )

    def test_ifftshift_roundtrip(self):
        x_np = np.random.randn(16).astype(np.float64)
        x = ins.from_numpy(x_np)
        result = ins.fftshift(ins.ifftshift(x))
        assert_allclose(result.numpy(), x_np, atol=1e-10)
