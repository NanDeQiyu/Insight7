"""CPU binding tests — 35 tests aligned with Lua and Julia.

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
# Creation (7 tests)
# ============================================================================


class TestCreation:
    """Array creation — tests 1-7."""

    def test_zeros(self):
        a = ins.zeros([2, 3], ins.float32)
        assert a.numel() == 6
        assert np.allclose(a.numpy(), 0)

    def test_ones(self):
        a = ins.ones([2, 3], ins.float32)
        assert a.numel() == 6
        np.testing.assert_allclose(a.numpy(), np.ones([2, 3]))

    def test_full(self):
        a = ins.full([2, 3], 7.0, ins.float32)
        assert a.numel() == 6
        np.testing.assert_allclose(a.numpy(), np.full([2, 3], 7.0))

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
        np.testing.assert_allclose(a.numpy(), np.linspace(0, 1, 5), rtol=1e-6)

    def test_from_numpy(self):
        data = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        a = ins.from_numpy(data)
        assert a.numel() == 3
        np.testing.assert_allclose(a.numpy(), data)


# ============================================================================
# Arithmetic (5 tests)
# ============================================================================


class TestArithmetic:
    """Elementwise arithmetic — tests 8-12."""

    def test_add(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5, 6], dtype=np.float64)
        result = ins.add(ins.from_numpy(a_np), ins.from_numpy(b_np))
        np.testing.assert_allclose(result.numpy(), a_np + b_np, rtol=1e-10)

    def test_sub(self):
        a_np = np.array([10, 20, 30], dtype=np.float64)
        b_np = np.array([1, 2, 3], dtype=np.float64)
        result = ins.sub(ins.from_numpy(a_np), ins.from_numpy(b_np))
        np.testing.assert_allclose(result.numpy(), a_np - b_np, rtol=1e-10)

    def test_mul(self):
        a_np = np.array([2, 3, 4], dtype=np.float64)
        b_np = np.array([5, 6, 7], dtype=np.float64)
        result = ins.mul(ins.from_numpy(a_np), ins.from_numpy(b_np))
        np.testing.assert_allclose(result.numpy(), a_np * b_np, rtol=1e-10)

    def test_div(self):
        a_np = np.array([10, 20, 30], dtype=np.float64)
        b_np = np.array([2, 4, 5], dtype=np.float64)
        result = ins.div(ins.from_numpy(a_np), ins.from_numpy(b_np))
        np.testing.assert_allclose(result.numpy(), a_np / b_np, rtol=1e-10)

    def test_neg(self):
        a_np = np.array([1, -2, 3], dtype=np.float64)
        result = ins.negative(ins.from_numpy(a_np))
        np.testing.assert_allclose(result.numpy(), -a_np, rtol=1e-10)


# ============================================================================
# Unary (5 tests)
# ============================================================================


class TestUnary:
    """Unary math — tests 13-17."""

    def test_abs(self):
        a_np = np.array([-3, -1, 0, 2, 4], dtype=np.float64)
        result = ins.abs(ins.from_numpy(a_np))
        np.testing.assert_allclose(result.numpy(), np.abs(a_np), rtol=1e-10)

    def test_sqrt(self):
        a_np = np.array([1, 4, 9, 16], dtype=np.float64)
        result = ins.sqrt(ins.from_numpy(a_np))
        np.testing.assert_allclose(result.numpy(), np.sqrt(a_np), rtol=1e-10)

    def test_exp(self):
        a_np = np.array([0, 1, 2], dtype=np.float64)
        result = ins.exp(ins.from_numpy(a_np))
        np.testing.assert_allclose(result.numpy(), np.exp(a_np), rtol=1e-10)

    def test_log(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        result = ins.log(ins.from_numpy(a_np))
        np.testing.assert_allclose(result.numpy(), np.log(a_np), rtol=1e-10)

    def test_sin(self):
        a_np = np.array([0, 0.5, 1.0], dtype=np.float64)
        result = ins.sin(ins.from_numpy(a_np))
        np.testing.assert_allclose(result.numpy(), np.sin(a_np), rtol=1e-10)


# ============================================================================
# Reduction (5 tests)
# ============================================================================


class TestReduction:
    """Reduction ops — tests 18-22."""

    def test_sum(self):
        a = ins.from_numpy(np.array([1, 2, 3, 4], dtype=np.float64))
        result = ins.sum(a)
        assert result.numpy().item() == pytest.approx(10.0)

    def test_mean(self):
        a = ins.from_numpy(np.array([1, 2, 3, 4], dtype=np.float64))
        result = ins.mean(a)
        assert result.numpy().item() == pytest.approx(2.5)

    def test_max(self):
        a = ins.from_numpy(np.array([3, 1, 4, 1, 5], dtype=np.float64))
        result = ins.max(a)
        assert result.numpy().item() == pytest.approx(5.0)

    def test_min(self):
        a = ins.from_numpy(np.array([3, 1, 4, 1, 5], dtype=np.float64))
        result = ins.min(a)
        assert result.numpy().item() == pytest.approx(1.0)

    def test_argmax(self):
        a = ins.from_numpy(np.array([1, 5, 3, 2], dtype=np.float64))
        result = ins.argmax(a)
        assert result.numpy().item() == 1


# ============================================================================
# Comparison (4 tests)
# ============================================================================


class TestComparison:
    """Comparison ops — tests 23-26."""

    def test_equal(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float64))
        b = ins.from_numpy(np.array([1, 0, 3], dtype=np.float64))
        result = ins.equal(a, b)
        np.testing.assert_array_equal(result.numpy(), np.array([True, False, True]))

    def test_greater(self):
        a = ins.from_numpy(np.array([3, 1, 5], dtype=np.float64))
        b = ins.from_numpy(np.array([2, 4, 5], dtype=np.float64))
        result = ins.greater(a, b)
        np.testing.assert_array_equal(result.numpy(), np.array([True, False, False]))

    def test_less(self):
        a = ins.from_numpy(np.array([1, 5, 3], dtype=np.float64))
        b = ins.from_numpy(np.array([2, 4, 5], dtype=np.float64))
        result = ins.less(a, b)
        np.testing.assert_array_equal(result.numpy(), np.array([True, False, True]))

    def test_logical_and(self):
        a = ins.from_numpy(np.array([True, True, False], dtype=np.bool_))
        b = ins.from_numpy(np.array([True, False, False], dtype=np.bool_))
        result = ins.logical_and(a, b)
        np.testing.assert_array_equal(result.numpy(), np.array([True, False, False]))


# ============================================================================
# Manipulation (3 tests)
# ============================================================================


class TestManipulation:
    """Manipulation ops — tests 27-29."""

    def test_reshape(self):
        a = ins.from_numpy(np.arange(6, dtype=np.float64))
        b = ins.reshape(a, [2, 3])
        assert b.numel() == 6

    def test_transpose(self):
        a = ins.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64))
        b = ins.permute(a, [1, 0])
        np.testing.assert_allclose(b.numpy(), np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float64))

    def test_squeeze(self):
        a = ins.from_numpy(np.zeros([1, 3, 1], dtype=np.float64))
        b = ins.squeeze(a)
        assert b.numel() == 3


# ============================================================================
# Linalg (3 tests)
# ============================================================================


class TestLinalg:
    """Linear algebra — tests 30-32."""

    def test_matmul(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        b_np = np.array([[5, 6], [7, 8]], dtype=np.float64)
        result = ins.matmul(ins.from_numpy(a_np), ins.from_numpy(b_np))
        np.testing.assert_allclose(result.numpy(), a_np @ b_np, rtol=1e-10)

    def test_det(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
        try:
            result = ins.det(ins.from_numpy(a_np))
            assert result.numpy().item() == pytest.approx(-2.0, abs=1e-3)
        except RuntimeError:
            pytest.skip("det requires OpenBLAS")

    def test_inv(self):
        a_np = np.array([[4, 7], [2, 6]], dtype=np.float32)
        try:
            result = ins.inv(ins.from_numpy(a_np))
            np.testing.assert_allclose(result.numpy(), np.linalg.inv(a_np), atol=1e-3)
        except RuntimeError:
            pytest.skip("inv requires OpenBLAS")


# ============================================================================
# FFT (2 tests)
# ============================================================================


class TestFFT:
    """FFT — tests 33-34."""

    def test_fft(self):
        x_np = np.array([1, 2, 3, 4], dtype=np.float64)
        result = ins.fft(ins.from_numpy(x_np))
        expected = np.fft.fft(x_np)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-8)

    def test_ifft(self):
        x_np = np.array([1, 2, 3, 4], dtype=np.float64)
        fft_np = np.fft.fft(x_np)
        result = ins.ifft(ins.from_numpy(fft_np.astype(np.complex128)))
        expected = np.fft.ifft(fft_np)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-8)


# ============================================================================
# Signal (1 test)
# ============================================================================


class TestSignal:
    """Signal — test 35."""

    def test_hann(self):
        w = ins.signal.hann(16)
        assert w.numel() == 16
        assert w.numpy()[0] == pytest.approx(0.0, abs=1e-5)
        assert w.numpy()[8] == pytest.approx(1.0, abs=1e-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
