"""CPU basic operations tests — aligned with C++ test suite.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python python -m pytest tests/cpu/python/test_cpu_basic.py -v
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
except ImportError:
    pytest.skip("Insight or NumPy not available", allow_module_level=True)


# ============================================================================
# Creation
# ============================================================================

class TestCreationCPU:
    """Array creation — aligned with test_creation.cpp."""

    def test_zeros(self):
        a = ins.zeros([2, 3], ins.float32)
        assert a.numel() == 6
        assert a.numpy().sum() == 0.0

    def test_ones(self):
        a = ins.ones([2, 3], ins.float32)
        assert a.numel() == 6
        np.testing.assert_allclose(a.numpy(), np.ones([2, 3]))

    def test_full(self):
        a = ins.full([2, 3], 3.14, ins.float32)
        np.testing.assert_allclose(a.numpy(), np.full([2, 3], 3.14), rtol=1e-5)

    def test_eye(self):
        a = ins.eye(3)
        assert a.numel() == 9
        np.testing.assert_allclose(a.numpy(), np.eye(3))

    def test_arange(self):
        a = ins.arange(10, ins.float32)
        assert a.numel() == 10
        np.testing.assert_allclose(a.numpy(), np.arange(10, dtype=np.float32))

    def test_linspace(self):
        a = ins.linspace(0.0, 1.0, 5, ins.float64)
        assert a.numel() == 5
        np.testing.assert_allclose(a.numpy(), np.linspace(0, 1, 5))

    def test_zeros_like(self):
        a = ins.ones([3, 3], ins.float32)
        b = ins.zeros_like(a)
        assert b.numel() == 9
        assert b.numpy().sum() == 0.0

    def test_ones_like(self):
        a = ins.zeros([2, 4], ins.float32)
        b = ins.ones_like(a)
        np.testing.assert_allclose(b.numpy(), np.ones([2, 4]))


# ============================================================================
# Elementwise
# ============================================================================

class TestElementwiseCPU:
    """Elementwise arithmetic — aligned with test_elementwise.cpp."""

    def test_add(self):
        a_np = np.array([1, 2, 3], dtype=np.float32)
        b_np = np.array([4, 5, 6], dtype=np.float32)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        result = ins.add(a, b)
        np.testing.assert_allclose(result.numpy(), a_np + b_np)

    def test_sub(self):
        a_np = np.array([10, 20, 30], dtype=np.float32)
        b_np = np.array([1, 2, 3], dtype=np.float32)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        result = ins.sub(a, b)
        np.testing.assert_allclose(result.numpy(), a_np - b_np)

    def test_mul(self):
        a_np = np.array([2, 3, 4], dtype=np.float32)
        b_np = np.array([5, 6, 7], dtype=np.float32)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        result = ins.mul(a, b)
        np.testing.assert_allclose(result.numpy(), a_np * b_np)

    def test_div(self):
        a_np = np.array([10, 20, 30], dtype=np.float32)
        b_np = np.array([2, 4, 5], dtype=np.float32)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        result = ins.div(a, b)
        np.testing.assert_allclose(result.numpy(), a_np / b_np, rtol=1e-5)

    def test_equal(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float32))
        b = ins.from_numpy(np.array([1, 0, 3], dtype=np.float32))
        result = ins.equal(a, b)
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_logical_and(self):
        a = ins.from_numpy(np.array([True, True, False], dtype=np.bool_))
        b = ins.from_numpy(np.array([True, False, False], dtype=np.bool_))
        result = ins.logical_and(a, b)
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(result.numpy(), expected)


# ============================================================================
# Unary
# ============================================================================

class TestUnaryCPU:
    """Unary math — aligned with test_unary.cpp."""

    def test_abs(self):
        a = ins.from_numpy(np.array([-3, -1, 0, 2, 4], dtype=np.float32))
        result = ins.abs(a)
        np.testing.assert_allclose(result.numpy(), np.abs(a.numpy()))

    def test_sqrt(self):
        a = ins.from_numpy(np.array([1, 4, 9, 16], dtype=np.float32))
        result = ins.sqrt(a)
        np.testing.assert_allclose(result.numpy(), np.sqrt(a.numpy()), rtol=1e-5)

    def test_exp(self):
        a = ins.from_numpy(np.array([0, 1, 2], dtype=np.float32))
        result = ins.exp(a)
        np.testing.assert_allclose(result.numpy(), np.exp(a.numpy()), rtol=1e-5)

    def test_sin(self):
        a = ins.from_numpy(np.array([0, 0.5, 1.0], dtype=np.float32))
        result = ins.sin(a)
        np.testing.assert_allclose(result.numpy(), np.sin(a.numpy()), rtol=1e-5)

    def test_sign(self):
        a = ins.from_numpy(np.array([-2, -1, 0, 1, 2], dtype=np.float32))
        result = ins.sign(a)
        np.testing.assert_allclose(result.numpy(), np.sign(a.numpy()))

    def test_where(self):
        cond = ins.from_numpy(np.array([True, False, True], dtype=np.bool_))
        x = ins.from_numpy(np.array([1, 2, 3], dtype=np.float32))
        y = ins.from_numpy(np.array([4, 5, 6], dtype=np.float32))
        result = ins.where(cond, x, y)
        expected = np.array([1, 5, 3], dtype=np.float32)
        np.testing.assert_allclose(result.numpy(), expected)


# ============================================================================
# Reduction
# ============================================================================

class TestReductionCPU:
    """Reduction ops — aligned with test_reduction.cpp."""

    def test_sum(self):
        a = ins.from_numpy(np.array([1, 2, 3, 4], dtype=np.float32))
        result = ins.sum(a)
        assert result.numpy().item() == pytest.approx(10.0)

    def test_mean(self):
        a = ins.from_numpy(np.array([1, 2, 3, 4], dtype=np.float32))
        result = ins.mean(a)
        assert result.numpy().item() == pytest.approx(2.5)

    def test_max(self):
        a = ins.from_numpy(np.array([3, 1, 4, 1, 5], dtype=np.float32))
        result = ins.max(a)
        assert result.numpy().item() == pytest.approx(5.0)

    def test_min(self):
        a = ins.from_numpy(np.array([3, 1, 4, 1, 5], dtype=np.float32))
        result = ins.min(a)
        assert result.numpy().item() == pytest.approx(1.0)

    def test_sum_axis(self):
        a = ins.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32))
        result = ins.sum(a, axis=0)
        np.testing.assert_allclose(result.numpy(), np.array([4, 6], dtype=np.float32))

    def test_argmax(self):
        a = ins.from_numpy(np.array([1, 5, 3, 2], dtype=np.float32))
        result = ins.argmax(a)
        assert result.numpy().item() == 1


# ============================================================================
# Linalg
# ============================================================================

class TestLinalgCPU:
    """Linear algebra — aligned with test_linalg.cpp."""

    def test_matmul(self):
        a = ins.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32))
        b = ins.from_numpy(np.array([[5, 6], [7, 8]], dtype=np.float32))
        result = ins.matmul(a, b)
        expected = np.matmul(a.numpy(), b.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_det(self):
        a = ins.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float64))
        result = ins.det(a)
        assert result.numpy().item() == pytest.approx(-2.0, abs=1e-8)

    def test_inv(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        a = ins.from_numpy(a_np)
        result = ins.inv(a)
        expected = np.linalg.inv(a_np)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-8)

    def test_trace(self):
        a = ins.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32))
        result = ins.trace(a)
        assert result.numpy().item() == pytest.approx(5.0)

    def test_norm(self):
        a = ins.from_numpy(np.array([3, 4], dtype=np.float32))
        result = ins.norm(a)
        assert result.numpy().item() == pytest.approx(5.0, abs=1e-5)


# ============================================================================
# FFT
# ============================================================================

class TestFFTCPU:
    """FFT — aligned with test_fft.cpp."""

    def test_fft(self):
        x_np = np.array([1, 2, 3, 4], dtype=np.float64)
        x = ins.from_numpy(x_np)
        result = ins.fft(x)
        expected = np.fft.fft(x_np)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-8)

    def test_ifft(self):
        x_np = np.array([1+0j, 2+0j, 3+0j, 4+0j], dtype=np.complex128)
        x = ins.from_numpy(x_np)
        result = ins.ifft(x)
        expected = np.fft.ifft(x_np)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-8)

    def test_fftfreq(self):
        result = ins.fftfreq(8, 1.0)
        expected = np.fft.fftfreq(8, 1.0)
        np.testing.assert_allclose(result.numpy(), expected)


# ============================================================================
# Signal
# ============================================================================

class TestSignalCPU:
    """Signal processing — aligned with test_signal_*.cpp."""

    def test_hann_window(self):
        w = ins.signal.hann(16)
        assert w.numel() == 16
        assert w.numpy()[0] == pytest.approx(0.0, abs=1e-5)
        assert w.numpy()[8] == pytest.approx(1.0, abs=1e-1)

    def test_hamming_window(self):
        w = ins.signal.hamming(16)
        assert w.numel() == 16
        assert w.numpy()[0] > 0  # Hamming doesn't go to 0

    def test_blackman_window(self):
        w = ins.signal.blackman(32)
        assert w.numel() == 32

    def test_fftconvolve(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float64))
        b = ins.from_numpy(np.array([1, 1], dtype=np.float64))
        result = ins.signal.fftconvolve(a, b)
        expected = np.convolve([1, 2, 3], [1, 1])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-8)
