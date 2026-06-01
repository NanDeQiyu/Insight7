"""Numerical precision alignment tests — Insight vs NumPy/SciPy on CPU.

These tests verify that Insight7 produces results identical (or within
acceptable tolerance) to NumPy/SciPy reference implementations.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python python -m pytest tests/python_align/cpu/test_numerical_cpu.py -v
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


# ============================================================================
# Creation alignment
# ============================================================================


class TestCreationAlignment:
    """Insight creation functions vs NumPy equivalents."""

    def test_zeros(self):
        a = ins.zeros([3, 4], ins.float32)
        assert_allclose(a.numpy(), np.zeros([3, 4], dtype=np.float32))

    def test_ones(self):
        a = ins.ones([3, 4], ins.float32)
        assert_allclose(a.numpy(), np.ones([3, 4], dtype=np.float32))

    def test_full(self):
        a = ins.full([2, 3], 3.14, ins.float64)
        assert_allclose(a.numpy(), np.full([2, 3], 3.14))

    def test_eye(self):
        a = ins.eye(4)
        assert_allclose(a.numpy(), np.eye(4, dtype=np.float32))

    def test_arange(self):
        a = ins.arange(20, ins.float64)
        assert_allclose(a.numpy(), np.arange(20, dtype=np.float64))

    def test_linspace(self):
        a = ins.linspace(0.0, 1.0, 50, ins.float64)
        assert_allclose(a.numpy(), np.linspace(0, 1, 50), atol=1e-10)

    def test_logspace(self):
        a = ins.logspace(0, 3, 4, ins.float64)
        assert_allclose(a.numpy(), np.logspace(0, 3, 4), rtol=1e-5)


# ============================================================================
# Elementwise alignment
# ============================================================================


class TestElementwiseAlignment:
    """Insight elementwise ops vs NumPy."""

    @pytest.fixture
    def random_arrays(self):
        rng = np.random.RandomState(42)
        a = rng.randn(5, 6).astype(np.float64)
        b = rng.randn(5, 6).astype(np.float64)
        return a, b

    def test_add(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.add(a, b).numpy(), a_np + b_np)

    def test_sub(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.sub(a, b).numpy(), a_np - b_np)

    def test_mul(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.mul(a, b).numpy(), a_np * b_np)

    def test_div(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.div(a, b).numpy(), a_np / b_np, rtol=1e-6)

    def test_pow(self, random_arrays):
        a_np, b_np = random_arrays
        a_np = np.abs(a_np) + 0.1  # avoid negative base
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.pow(a, b).numpy(), a_np**b_np, rtol=1e-4)

    def test_maximum(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.maximum(a, b).numpy(), np.maximum(a_np, b_np))

    def test_minimum(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.minimum(a, b).numpy(), np.minimum(a_np, b_np))

    def test_equal(self, random_arrays):
        a_np, _b_np = random_arrays
        a = ins.from_numpy(a_np)
        assert_allclose(ins.equal(a, a).numpy(), np.ones_like(a_np, dtype=bool))

    def test_logical_and(self):
        a_np = np.array([True, True, False, False])
        b_np = np.array([True, False, True, False])
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.logical_and(a, b).numpy(), np.logical_and(a_np, b_np))

    def test_logical_or(self):
        a_np = np.array([True, True, False, False])
        b_np = np.array([True, False, True, False])
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.logical_or(a, b).numpy(), np.logical_or(a_np, b_np))


# ============================================================================
# Unary alignment
# ============================================================================


class TestUnaryAlignment:
    """Insight unary ops vs NumPy."""

    @pytest.fixture
    def data(self):
        return np.linspace(0.1, 5.0, 20).astype(np.float64)

    def test_abs(self):
        x_np = np.array([-3, -1, 0, 2, 4], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.abs(x).numpy(), np.abs(x_np))

    def test_sqrt(self, data):
        x = ins.from_numpy(data)
        assert_allclose(ins.sqrt(x).numpy(), np.sqrt(data), rtol=1e-8)

    def test_exp(self, data):
        x = ins.from_numpy(data[:5])  # small values to avoid overflow
        assert_allclose(ins.exp(x).numpy(), np.exp(data[:5]), rtol=1e-6)

    def test_log(self, data):
        x = ins.from_numpy(data)
        assert_allclose(ins.log(x).numpy(), np.log(data), rtol=1e-8)

    def test_sin(self, data):
        x = ins.from_numpy(data)
        assert_allclose(ins.sin(x).numpy(), np.sin(data), rtol=1e-8)

    def test_cos(self, data):
        x = ins.from_numpy(data)
        assert_allclose(ins.cos(x).numpy(), np.cos(data), rtol=1e-8)

    def test_tan(self):
        x_np = np.array([0.1, 0.5, 1.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.tan(x).numpy(), np.tan(x_np), rtol=1e-6)

    def test_floor(self):
        x_np = np.array([1.1, 2.7, -0.3, 3.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.floor(x).numpy(), np.floor(x_np))

    def test_ceil(self):
        x_np = np.array([1.1, 2.7, -0.3, 3.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.ceil(x).numpy(), np.ceil(x_np))

    def test_round(self):
        x_np = np.array([1.1, 2.5, 3.7, -1.3], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.round(x).numpy(), np.round(x_np))

    def test_sign(self):
        x_np = np.array([-3, -1, 0, 1, 3], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.sign(x).numpy(), np.sign(x_np))

    def test_exp2(self):
        x_np = np.array([0, 1, 2, 3], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.exp2(x).numpy(), np.exp2(x_np), rtol=1e-8)

    def test_expm1(self):
        x_np = np.array([0, 1e-10, 0.5], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.expm1(x).numpy(), np.expm1(x_np), rtol=1e-8)

    def test_log1p(self):
        x_np = np.array([0, 1e-10, 0.5], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.log1p(x).numpy(), np.log1p(x_np), rtol=1e-8)

    def test_cbrt(self):
        x_np = np.array([1, 8, 27, 64], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.cbrt(x).numpy(), np.cbrt(x_np), rtol=1e-8)

    def test_deg2rad(self):
        x_np = np.array([0, 90, 180, 360], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.deg2rad(x).numpy(), np.deg2rad(x_np), rtol=1e-10)

    def test_rad2deg(self):
        x_np = np.array([0, np.pi / 4, np.pi / 2, np.pi], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.rad2deg(x).numpy(), np.rad2deg(x_np), rtol=1e-10)

    def test_where(self):
        cond_np = np.array([True, False, True, False])
        x_np = np.array([1, 2, 3, 4], dtype=np.float64)
        y_np = np.array([5, 6, 7, 8], dtype=np.float64)
        cond = ins.from_numpy(cond_np)
        x = ins.from_numpy(x_np)
        y = ins.from_numpy(y_np)
        assert_allclose(ins.where(cond, x, y).numpy(), np.where(cond_np, x_np, y_np))


# ============================================================================
# Reduction alignment
# ============================================================================


class TestReductionAlignment:
    """Insight reduction ops vs NumPy."""

    @pytest.fixture
    def data_1d(self):
        return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)

    @pytest.fixture
    def data_2d(self):
        return np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

    def test_sum(self, data_1d):
        x = ins.from_numpy(data_1d)
        assert_allclose(ins.sum(x).numpy(), np.sum(data_1d))

    def test_mean(self, data_1d):
        x = ins.from_numpy(data_1d)
        assert_allclose(ins.mean(x).numpy(), np.mean(data_1d))

    def test_max(self, data_1d):
        x = ins.from_numpy(data_1d)
        assert_allclose(ins.max(x).numpy(), np.max(data_1d))

    def test_min(self, data_1d):
        x = ins.from_numpy(data_1d)
        assert_allclose(ins.min(x).numpy(), np.min(data_1d))

    def test_sum_axis(self, data_2d):
        x = ins.from_numpy(data_2d)
        assert_allclose(ins.sum(x, axis=0).numpy(), np.sum(data_2d, axis=0))
        assert_allclose(ins.sum(x, axis=1).numpy(), np.sum(data_2d, axis=1))

    def test_mean_axis(self, data_2d):
        x = ins.from_numpy(data_2d)
        assert_allclose(ins.mean(x, axis=0).numpy(), np.mean(data_2d, axis=0))

    def test_argmax(self, data_1d):
        x = ins.from_numpy(data_1d)
        assert_allclose(ins.argmax(x).numpy(), np.argmax(data_1d))

    def test_argmin(self, data_1d):
        x = ins.from_numpy(data_1d)
        assert_allclose(ins.argmin(x).numpy(), np.argmin(data_1d))

    def test_prod(self, data_1d):
        x = ins.from_numpy(data_1d)
        assert_allclose(ins.prod(x).numpy(), np.prod(data_1d), rtol=1e-5)

    def test_var(self, data_1d):
        x = ins.from_numpy(data_1d)
        assert_allclose(ins.var(x).numpy(), np.var(data_1d), rtol=1e-5)

    def test_std(self, data_1d):
        x = ins.from_numpy(data_1d)
        assert_allclose(ins.std(x).numpy(), np.std(data_1d), rtol=1e-5)

    def test_cumsum(self, data_1d):
        x = ins.from_numpy(data_1d)
        assert_allclose(ins.cumsum(x, 0).numpy(), np.cumsum(data_1d))


# ============================================================================
# Linalg alignment
# ============================================================================


class TestLinalgAlignment:
    """Insight linalg ops vs NumPy."""

    @pytest.fixture
    def matrix_2x2(self):
        return np.array([[1, 2], [3, 4]], dtype=np.float64)

    @pytest.fixture
    def matrix_3x3(self):
        return np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=np.float64)

    def test_matmul(self, matrix_2x2):
        a = ins.from_numpy(matrix_2x2)
        b = ins.from_numpy(matrix_2x2)
        result = ins.matmul(a, b)
        assert_allclose(result.numpy(), np.matmul(matrix_2x2, matrix_2x2), rtol=1e-8)

    def test_det(self, matrix_2x2):
        a = ins.from_numpy(matrix_2x2)
        assert_allclose(ins.det(a).numpy(), np.linalg.det(matrix_2x2), rtol=1e-8)

    def test_inv(self, matrix_2x2):
        a = ins.from_numpy(matrix_2x2)
        result = ins.inv(a)
        assert_allclose(result.numpy(), np.linalg.inv(matrix_2x2), atol=1e-8)

    def test_trace(self, matrix_2x2):
        a = ins.from_numpy(matrix_2x2)
        assert_allclose(ins.trace(a).numpy(), np.trace(matrix_2x2))

    def test_norm(self):
        x_np = np.array([3, 4], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.norm(x).numpy(), np.linalg.norm(x_np), rtol=1e-8)

    def test_dot(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5, 6], dtype=np.float64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        assert_allclose(ins.dot(a, b).numpy(), np.dot(a_np, b_np))

    def test_outer(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5], dtype=np.float64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        assert_allclose(ins.outer(a, b).numpy(), np.outer(a_np, b_np))


# ============================================================================
# FFT alignment
# ============================================================================


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


# ============================================================================
# Manipulation alignment
# ============================================================================


class TestManipulationAlignment:
    """Insight manipulation ops vs NumPy."""

    def test_reshape(self):
        a_np = np.arange(12, dtype=np.float64)
        a = ins.from_numpy(a_np)
        result = ins.reshape(a, [3, 4])
        assert_allclose(result.numpy(), a_np.reshape(3, 4))

    def test_flatten(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        a = ins.from_numpy(a_np)
        result = ins.flatten(a)
        assert_allclose(result.numpy(), a_np.flatten())

    def test_concat(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5, 6], dtype=np.float64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        result = ins.concat([a, b])
        assert_allclose(result.numpy(), np.concatenate([a_np, b_np]))

    def test_flip(self):
        a_np = np.array([1, 2, 3, 4], dtype=np.float64)
        a = ins.from_numpy(a_np)
        result = ins.flip(a)
        assert_allclose(result.numpy(), np.flip(a_np))

    def test_squeeze(self):
        a_np = np.array([[[1, 2, 3]]], dtype=np.float64)  # shape (1,1,3)
        a = ins.from_numpy(a_np)
        result = ins.squeeze(a)
        assert_allclose(result.numpy(), np.squeeze(a_np))

    def test_transpose(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        a = ins.from_numpy(a_np)
        result = ins.transpose(a)
        assert_allclose(result.numpy(), np.transpose(a_np))
