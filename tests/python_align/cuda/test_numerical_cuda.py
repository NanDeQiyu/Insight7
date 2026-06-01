"""Numerical precision alignment tests — Insight CUDA vs NumPy/SciPy.

These tests verify that Insight7 GPU operations produce results identical
(or within acceptable tolerance) to NumPy/SciPy reference implementations.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \\
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \\
    python -m pytest tests/python_align/cuda/test_numerical_cuda.py -v
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


# ============================================================================
# Helpers
# ============================================================================


def to_gpu(np_arr):
    return ins.from_numpy(np_arr).to(GPU)


def to_numpy(gpu_arr):
    return gpu_arr.to(CPU).numpy()


# ============================================================================
# Creation alignment (GPU)
# ============================================================================


class TestCreationAlignmentGPU:
    """Insight creation on GPU vs NumPy."""

    def test_zeros(self):
        a = ins.zeros([3, 4], ins.float32).to(GPU)
        assert_allclose(to_numpy(a), np.zeros([3, 4], dtype=np.float32))

    def test_ones(self):
        a = ins.ones([3, 4], ins.float32).to(GPU)
        assert_allclose(to_numpy(a), np.ones([3, 4], dtype=np.float32))

    def test_full(self):
        a = ins.full([2, 3], 3.14, ins.float64).to(GPU)
        assert_allclose(to_numpy(a), np.full([2, 3], 3.14))

    def test_eye(self):
        a = ins.eye(4).to(GPU)
        assert_allclose(to_numpy(a), np.eye(4, dtype=np.float32))

    def test_arange(self):
        a = ins.arange(20, ins.float64).to(GPU)
        assert_allclose(to_numpy(a), np.arange(20, dtype=np.float64))

    def test_linspace(self):
        a = ins.linspace(0.0, 1.0, 50, ins.float64).to(GPU)
        assert_allclose(to_numpy(a), np.linspace(0, 1, 50), atol=1e-10)


# ============================================================================
# Elementwise alignment (GPU)
# ============================================================================


class TestElementwiseAlignmentGPU:
    """Insight elementwise ops on GPU vs NumPy."""

    @pytest.fixture
    def random_arrays(self):
        rng = np.random.RandomState(42)
        a = rng.randn(5, 6).astype(np.float64)
        b = rng.randn(5, 6).astype(np.float64)
        return a, b

    def test_add(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.add(a, b)), a_np + b_np, rtol=1e-6)

    def test_sub(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.sub(a, b)), a_np - b_np, rtol=1e-6)

    def test_mul(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.mul(a, b)), a_np * b_np, rtol=1e-6)

    def test_div(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.div(a, b)), a_np / b_np, rtol=1e-6)

    def test_maximum(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.maximum(a, b)), np.maximum(a_np, b_np), rtol=1e-6)

    def test_minimum(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.minimum(a, b)), np.minimum(a_np, b_np), rtol=1e-6)

    def test_equal(self, random_arrays):
        a_np, _ = random_arrays
        a = to_gpu(a_np)
        assert_allclose(to_numpy(ins.equal(a, a)), np.ones_like(a_np, dtype=bool))

    def test_logical_and(self):
        a_np = np.array([True, True, False, False])
        b_np = np.array([True, False, True, False])
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.logical_and(a, b)), np.logical_and(a_np, b_np))

    def test_logical_or(self):
        a_np = np.array([True, True, False, False])
        b_np = np.array([True, False, True, False])
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.logical_or(a, b)), np.logical_or(a_np, b_np))


# ============================================================================
# Unary alignment (GPU)
# ============================================================================


class TestUnaryAlignmentGPU:
    """Insight unary ops on GPU vs NumPy."""

    @pytest.fixture
    def data(self):
        return np.linspace(0.1, 5.0, 20).astype(np.float64)

    def test_abs(self):
        x_np = np.array([-3, -1, 0, 2, 4], dtype=np.float64)
        assert_allclose(to_numpy(ins.abs(to_gpu(x_np))), np.abs(x_np), rtol=1e-6)

    def test_sqrt(self, data):
        assert_allclose(to_numpy(ins.sqrt(to_gpu(data))), np.sqrt(data), rtol=1e-6)

    def test_exp(self, data):
        x = data[:5]  # small values
        assert_allclose(to_numpy(ins.exp(to_gpu(x))), np.exp(x), rtol=1e-6)

    def test_log(self, data):
        assert_allclose(to_numpy(ins.log(to_gpu(data))), np.log(data), rtol=1e-6)

    def test_sin(self, data):
        assert_allclose(to_numpy(ins.sin(to_gpu(data))), np.sin(data), rtol=1e-6)

    def test_cos(self, data):
        assert_allclose(to_numpy(ins.cos(to_gpu(data))), np.cos(data), rtol=1e-6)

    def test_tan(self):
        x_np = np.array([0.1, 0.5, 1.0], dtype=np.float64)
        assert_allclose(to_numpy(ins.tan(to_gpu(x_np))), np.tan(x_np), rtol=1e-6)

    def test_floor(self):
        x_np = np.array([1.1, 2.7, -0.3, 3.0], dtype=np.float64)
        assert_allclose(to_numpy(ins.floor(to_gpu(x_np))), np.floor(x_np), rtol=1e-6)

    def test_ceil(self):
        x_np = np.array([1.1, 2.7, -0.3, 3.0], dtype=np.float64)
        assert_allclose(to_numpy(ins.ceil(to_gpu(x_np))), np.ceil(x_np), rtol=1e-6)

    def test_round(self):
        x_np = np.array([1.1, 2.5, 3.7, -1.3], dtype=np.float64)
        assert_allclose(to_numpy(ins.round(to_gpu(x_np))), np.round(x_np), rtol=1e-6)

    def test_sign(self):
        x_np = np.array([-3, -1, 0, 1, 3], dtype=np.float64)
        assert_allclose(to_numpy(ins.sign(to_gpu(x_np))), np.sign(x_np), rtol=1e-6)

    def test_exp2(self):
        x_np = np.array([0, 1, 2, 3], dtype=np.float64)
        assert_allclose(to_numpy(ins.exp2(to_gpu(x_np))), np.exp2(x_np), rtol=1e-6)

    def test_expm1(self):
        x_np = np.array([0, 1e-10, 0.5], dtype=np.float64)
        assert_allclose(to_numpy(ins.expm1(to_gpu(x_np))), np.expm1(x_np), rtol=1e-6)

    def test_log1p(self):
        x_np = np.array([0, 1e-10, 0.5], dtype=np.float64)
        assert_allclose(to_numpy(ins.log1p(to_gpu(x_np))), np.log1p(x_np), rtol=1e-6)

    def test_cbrt(self):
        x_np = np.array([1, 8, 27, 64], dtype=np.float64)
        assert_allclose(to_numpy(ins.cbrt(to_gpu(x_np))), np.cbrt(x_np), rtol=1e-6)

    def test_deg2rad(self):
        x_np = np.array([0, 90, 180, 360], dtype=np.float64)
        assert_allclose(to_numpy(ins.deg2rad(to_gpu(x_np))), np.deg2rad(x_np), rtol=1e-6)

    def test_rad2deg(self):
        x_np = np.array([0, np.pi / 4, np.pi / 2, np.pi], dtype=np.float64)
        assert_allclose(to_numpy(ins.rad2deg(to_gpu(x_np))), np.rad2deg(x_np), rtol=1e-6)

    def test_where(self):
        cond_np = np.array([True, False, True, False])
        x_np = np.array([1, 2, 3, 4], dtype=np.float64)
        y_np = np.array([5, 6, 7, 8], dtype=np.float64)
        cond, x, y = to_gpu(cond_np), to_gpu(x_np), to_gpu(y_np)
        assert_allclose(to_numpy(ins.where(cond, x, y)), np.where(cond_np, x_np, y_np), rtol=1e-6)


# ============================================================================
# Reduction alignment (GPU)
# ============================================================================


class TestReductionAlignmentGPU:
    """Insight reduction on GPU vs NumPy."""

    @pytest.fixture
    def data_1d(self):
        return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)

    @pytest.fixture
    def data_2d(self):
        return np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

    def test_sum(self, data_1d):
        assert_allclose(to_numpy(ins.sum(to_gpu(data_1d))), np.sum(data_1d), rtol=1e-6)

    def test_mean(self, data_1d):
        assert_allclose(to_numpy(ins.mean(to_gpu(data_1d))), np.mean(data_1d), rtol=1e-6)

    def test_max(self, data_1d):
        assert_allclose(to_numpy(ins.max(to_gpu(data_1d))), np.max(data_1d), rtol=1e-6)

    def test_min(self, data_1d):
        assert_allclose(to_numpy(ins.min(to_gpu(data_1d))), np.min(data_1d), rtol=1e-6)

    def test_sum_axis(self, data_2d):
        x = to_gpu(data_2d)
        assert_allclose(to_numpy(ins.sum(x, axis=0)), np.sum(data_2d, axis=0), rtol=1e-6)
        assert_allclose(to_numpy(ins.sum(x, axis=1)), np.sum(data_2d, axis=1), rtol=1e-6)

    def test_mean_axis(self, data_2d):
        x = to_gpu(data_2d)
        assert_allclose(to_numpy(ins.mean(x, axis=0)), np.mean(data_2d, axis=0), rtol=1e-6)

    def test_argmax(self, data_1d):
        assert_allclose(to_numpy(ins.argmax(to_gpu(data_1d))), np.argmax(data_1d), rtol=1e-6)

    def test_argmin(self, data_1d):
        assert_allclose(to_numpy(ins.argmin(to_gpu(data_1d))), np.argmin(data_1d), rtol=1e-6)

    def test_prod(self, data_1d):
        assert_allclose(to_numpy(ins.prod(to_gpu(data_1d))), np.prod(data_1d), rtol=1e-4)

    def test_var(self, data_1d):
        assert_allclose(to_numpy(ins.var(to_gpu(data_1d))), np.var(data_1d), rtol=1e-4)

    def test_std(self, data_1d):
        assert_allclose(to_numpy(ins.std(to_gpu(data_1d))), np.std(data_1d), rtol=1e-4)

    def test_cumsum(self, data_1d):
        assert_allclose(to_numpy(ins.cumsum(to_gpu(data_1d), 0)), np.cumsum(data_1d), rtol=1e-6)


# ============================================================================
# Linalg alignment (GPU)
# ============================================================================


class TestLinalgAlignmentGPU:
    """Insight linalg on GPU vs NumPy."""

    @pytest.fixture
    def matrix_2x2(self):
        return np.array([[1, 2], [3, 4]], dtype=np.float64)

    def test_matmul(self, matrix_2x2):
        a, b = to_gpu(matrix_2x2), to_gpu(matrix_2x2)
        result = ins.matmul(a, b)
        assert_allclose(to_numpy(result), np.matmul(matrix_2x2, matrix_2x2), rtol=1e-6)

    def test_det(self, matrix_2x2):
        a = to_gpu(matrix_2x2)
        assert_allclose(to_numpy(ins.det(a)), np.linalg.det(matrix_2x2), rtol=1e-6)

    def test_inv(self, matrix_2x2):
        a = to_gpu(matrix_2x2)
        assert_allclose(to_numpy(ins.inv(a)), np.linalg.inv(matrix_2x2), atol=1e-6)

    def test_trace(self, matrix_2x2):
        a = to_gpu(matrix_2x2)
        assert_allclose(to_numpy(ins.trace(a)), np.trace(matrix_2x2), rtol=1e-6)

    def test_norm(self):
        x_np = np.array([3, 4], dtype=np.float64)
        x = to_gpu(x_np)
        assert_allclose(to_numpy(ins.norm(x)), np.linalg.norm(x_np), rtol=1e-6)

    def test_dot(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5, 6], dtype=np.float64)
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.dot(a, b)), np.dot(a_np, b_np), rtol=1e-6)

    def test_outer(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5], dtype=np.float64)
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.outer(a, b)), np.outer(a_np, b_np), rtol=1e-6)


# ============================================================================
# FFT alignment (GPU)
# ============================================================================


class TestFFTAlignmentGPU:
    """Insight FFT on GPU vs NumPy FFT."""

    def test_fft(self):
        x_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.fft(x)
        assert_allclose(to_numpy(result), np.fft.fft(x_np), atol=1e-6)

    def test_ifft(self):
        x_np = np.array([1, 2, 3, 4], dtype=np.complex128)
        x = to_gpu(x_np)
        result = ins.ifft(x)
        assert_allclose(to_numpy(result), np.fft.ifft(x_np), atol=1e-6)

    def test_fft_ifft_roundtrip(self):
        x_np = np.random.randn(16).astype(np.float64)
        x = to_gpu(x_np)
        result = ins.ifft(ins.fft(x))
        assert_allclose(to_numpy(result).real, x_np, atol=1e-6)

    def test_rfft(self):
        x_np = np.array([1, 2, 3, 4], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.rfft(x)
        assert_allclose(to_numpy(result), np.fft.rfft(x_np), atol=1e-6)

    def test_fftfreq(self):
        result = ins.fftfreq(8, 0.1)
        assert_allclose(result.numpy(), np.fft.fftfreq(8, 0.1))

    def test_rfftfreq(self):
        result = ins.rfftfreq(8, 0.1)
        assert_allclose(result.numpy(), np.fft.rfftfreq(8, 0.1))


# ============================================================================
# Manipulation alignment (GPU)
# ============================================================================


class TestManipulationAlignmentGPU:
    """Insight manipulation on GPU vs NumPy."""

    def test_reshape(self):
        a_np = np.arange(12, dtype=np.float64)
        a = to_gpu(a_np)
        result = ins.reshape(a, [3, 4])
        assert_allclose(to_numpy(result), a_np.reshape(3, 4), rtol=1e-6)

    def test_flatten(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        a = to_gpu(a_np)
        result = ins.flatten(a)
        assert_allclose(to_numpy(result), a_np.flatten(), rtol=1e-6)

    def test_concat(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5, 6], dtype=np.float64)
        a, b = to_gpu(a_np), to_gpu(b_np)
        result = ins.concat([a, b])
        assert_allclose(to_numpy(result), np.concatenate([a_np, b_np]), rtol=1e-6)

    def test_flip(self):
        a_np = np.array([1, 2, 3, 4], dtype=np.float64)
        a = to_gpu(a_np)
        result = ins.flip(a)
        assert_allclose(to_numpy(result), np.flip(a_np), rtol=1e-6)

    def test_squeeze(self):
        a_np = np.array([[[1, 2, 3]]], dtype=np.float64)
        a = to_gpu(a_np)
        result = ins.squeeze(a)
        assert_allclose(to_numpy(result), np.squeeze(a_np), rtol=1e-6)

    def test_transpose(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        a = to_gpu(a_np)
        result = ins.transpose(a)
        assert_allclose(to_numpy(result), np.transpose(a_np), rtol=1e-6)


# ============================================================================
# Signal alignment (GPU)
# ============================================================================


class TestSignalAlignmentGPU:
    """Insight signal ops on GPU vs SciPy."""

    def test_hann_window(self):
        try:
            from scipy.signal.windows import hann as scipy_hann
        except ImportError:
            pytest.skip("SciPy not available")
        result = ins.signal.hann(64)
        expected = scipy_hann(64)
        assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_hamming_window(self):
        try:
            from scipy.signal.windows import hamming as scipy_hamming
        except ImportError:
            pytest.skip("SciPy not available")
        result = ins.signal.hamming(64)
        expected = scipy_hamming(64)
        assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_kaiser_window(self):
        try:
            from scipy.signal.windows import kaiser as scipy_kaiser
        except ImportError:
            pytest.skip("SciPy not available")
        result = ins.signal.kaiser(64, 5.0)
        expected = scipy_kaiser(64, 5.0)
        assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_fftconvolve(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5], dtype=np.float64)
        a, b = to_gpu(a_np), to_gpu(b_np)
        result = ins.signal.fftconvolve(a, b)
        try:
            from scipy.signal import fftconvolve as scipy_fftconvolve

            expected = scipy_fftconvolve(a_np, b_np)
            assert_allclose(to_numpy(result), expected, atol=1e-8)
        except ImportError:
            expected = np.array([4, 13, 23, 15], dtype=np.float64)
            assert_allclose(to_numpy(result), expected, atol=1e-8)


# ============================================================================
# Extended elementwise alignment (GPU)
# ============================================================================


class TestElementwiseExtendedGPU:
    """Extended elementwise comparison ops on GPU vs NumPy."""

    @pytest.fixture
    def random_arrays(self):
        rng = np.random.RandomState(123)
        a = rng.randn(4, 5).astype(np.float64)
        b = rng.randn(4, 5).astype(np.float64)
        return a, b

    def test_greater(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.greater(a, b)), np.greater(a_np, b_np))

    def test_less(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.less(a, b)), np.less(a_np, b_np))

    def test_greater_equal(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.greater_equal(a, b)), np.greater_equal(a_np, b_np))

    def test_less_equal(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.less_equal(a, b)), np.less_equal(a_np, b_np))

    def test_not_equal(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.not_equal(a, b)), np.not_equal(a_np, b_np))

    def test_equal_same(self):
        a_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        a = to_gpu(a_np)
        assert_allclose(to_numpy(ins.equal(a, a)), np.equal(a_np, a_np))

    def test_logical_xor(self):
        a_np = np.array([True, True, False, False])
        b_np = np.array([True, False, True, False])
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.logical_xor(a, b)), np.logical_xor(a_np, b_np))

    def test_mod(self):
        a_np = np.array([10.0, 7.5, -3.0, 11.0], dtype=np.float64)
        b_np = np.array([3.0, 2.0, 2.0, 5.0], dtype=np.float64)
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.mod(a, b)), np.mod(a_np, b_np), rtol=1e-6)

    def test_bitwise_and(self):
        a_np = np.array([0b1100, 0b1010], dtype=np.int32)
        b_np = np.array([0b1010, 0b1100], dtype=np.int32)
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.bitwise_and(a, b)), np.bitwise_and(a_np, b_np))

    def test_bitwise_or(self):
        a_np = np.array([0b1100, 0b1010], dtype=np.int32)
        b_np = np.array([0b1010, 0b1100], dtype=np.int32)
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.bitwise_or(a, b)), np.bitwise_or(a_np, b_np))

    def test_bitwise_xor(self):
        a_np = np.array([0b1100, 0b1010], dtype=np.int32)
        b_np = np.array([0b1010, 0b1100], dtype=np.int32)
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.bitwise_xor(a, b)), np.bitwise_xor(a_np, b_np))


# ============================================================================
# Extended unary alignment (GPU)
# ============================================================================


class TestUnaryExtendedGPU:
    """Extended unary ops on GPU vs NumPy."""

    def test_negative(self):
        x_np = np.array([1.5, -2.3, 0.0, 4.7], dtype=np.float64)
        assert_allclose(to_numpy(ins.negative(to_gpu(x_np))), np.negative(x_np), rtol=1e-6)

    def test_reciprocal(self):
        x_np = np.array([1.0, 2.0, 4.0, 0.5], dtype=np.float64)
        assert_allclose(to_numpy(ins.reciprocal(to_gpu(x_np))), np.reciprocal(x_np), rtol=1e-6)

    def test_asin(self):
        x_np = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
        assert_allclose(to_numpy(ins.asin(to_gpu(x_np))), np.arcsin(x_np), rtol=1e-6)

    def test_acos(self):
        x_np = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
        assert_allclose(to_numpy(ins.acos(to_gpu(x_np))), np.arccos(x_np), rtol=1e-6)

    def test_atan(self):
        x_np = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float64)
        assert_allclose(to_numpy(ins.atan(to_gpu(x_np))), np.arctan(x_np), rtol=1e-6)

    def test_sinh(self):
        x_np = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float64)
        assert_allclose(to_numpy(ins.sinh(to_gpu(x_np))), np.sinh(x_np), rtol=1e-6)

    def test_cosh(self):
        x_np = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float64)
        assert_allclose(to_numpy(ins.cosh(to_gpu(x_np))), np.cosh(x_np), rtol=1e-6)

    def test_tanh(self):
        x_np = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float64)
        assert_allclose(to_numpy(ins.tanh(to_gpu(x_np))), np.tanh(x_np), rtol=1e-6)

    def test_asinh(self):
        x_np = np.array([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float64)
        assert_allclose(to_numpy(ins.asinh(to_gpu(x_np))), np.arcsinh(x_np), rtol=1e-6)

    def test_acosh(self):
        x_np = np.array([1.0, 1.5, 2.0, 5.0, 10.0], dtype=np.float64)
        assert_allclose(to_numpy(ins.acosh(to_gpu(x_np))), np.arccosh(x_np), rtol=1e-6)

    def test_atanh(self):
        x_np = np.array([-0.9, -0.5, 0.0, 0.5, 0.9], dtype=np.float64)
        assert_allclose(to_numpy(ins.atanh(to_gpu(x_np))), np.arctanh(x_np), rtol=1e-6)

    def test_trunc(self):
        x_np = np.array([1.7, -2.3, 0.5, 3.0, -0.1], dtype=np.float64)
        assert_allclose(to_numpy(ins.trunc(to_gpu(x_np))), np.trunc(x_np), rtol=1e-6)

    def test_isnan(self):
        x_np = np.array([1.0, np.nan, 3.0, np.nan], dtype=np.float64)
        assert_allclose(to_numpy(ins.isnan(to_gpu(x_np))), np.isnan(x_np))

    def test_isinf(self):
        x_np = np.array([1.0, np.inf, 3.0, -np.inf], dtype=np.float64)
        assert_allclose(to_numpy(ins.isinf(to_gpu(x_np))), np.isinf(x_np))

    def test_isfinite(self):
        x_np = np.array([1.0, np.nan, np.inf, -np.inf, 3.0], dtype=np.float64)
        assert_allclose(to_numpy(ins.isfinite(to_gpu(x_np))), np.isfinite(x_np))

    def test_logical_not(self):
        a_np = np.array([True, False, True, False])
        assert_allclose(to_numpy(ins.logical_not(to_gpu(a_np))), np.logical_not(a_np))

    def test_square(self):
        x_np = np.array([-3.0, -1.0, 0.0, 2.0, 4.0], dtype=np.float64)
        assert_allclose(to_numpy(ins.square(to_gpu(x_np))), np.square(x_np), rtol=1e-6)

    def test_log2(self):
        x_np = np.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float64)
        assert_allclose(to_numpy(ins.log2(to_gpu(x_np))), np.log2(x_np), rtol=1e-6)

    def test_log10(self):
        x_np = np.array([1.0, 10.0, 100.0, 1000.0], dtype=np.float64)
        assert_allclose(to_numpy(ins.log10(to_gpu(x_np))), np.log10(x_np), rtol=1e-6)


# ============================================================================
# Extended reduction alignment (GPU)
# ============================================================================


class TestReductionExtendedGPU:
    """Extended reduction ops on GPU vs NumPy."""

    @pytest.fixture
    def data_1d(self):
        return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)

    def test_cumprod(self, data_1d):
        assert_allclose(to_numpy(ins.cumprod(to_gpu(data_1d), 0)), np.cumprod(data_1d), rtol=1e-4)

    def test_cummax(self, data_1d):
        result = to_numpy(ins.cummax(to_gpu(data_1d), 0))
        expected = np.maximum.accumulate(data_1d)
        assert_allclose(result, expected, rtol=1e-6)

    def test_cummin(self, data_1d):
        result = to_numpy(ins.cummin(to_gpu(data_1d), 0))
        expected = np.minimum.accumulate(data_1d)
        assert_allclose(result, expected, rtol=1e-6)

    def test_median_odd(self):
        x_np = np.array([3, 1, 4, 1, 5], dtype=np.float64)
        assert_allclose(to_numpy(ins.median(to_gpu(x_np))), np.median(x_np), rtol=1e-6)

    def test_median_even(self):
        x_np = np.array([3, 1, 4, 1, 5, 9], dtype=np.float64)
        assert_allclose(to_numpy(ins.median(to_gpu(x_np))), np.median(x_np), rtol=1e-6)

    def test_quantile(self, data_1d):
        result = to_numpy(ins.quantile(to_gpu(data_1d), 0.5))
        assert_allclose(result, np.quantile(data_1d, 0.5), rtol=1e-4)

    def test_quantile_75(self, data_1d):
        result = to_numpy(ins.quantile(to_gpu(data_1d), 0.75))
        assert_allclose(result, np.quantile(data_1d, 0.75), rtol=1e-4)

    def test_count_nonzero(self):
        x_np = np.array([0, 1, 0, 3, 0, 5], dtype=np.float64)
        result = to_numpy(ins.count_nonzero(to_gpu(x_np)))
        assert_allclose(result, np.count_nonzero(x_np))

    def test_nansum(self):
        x_np = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float64)
        assert_allclose(to_numpy(ins.nansum(to_gpu(x_np))), np.nansum(x_np), rtol=1e-6)

    def test_nanmean(self):
        x_np = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float64)
        assert_allclose(to_numpy(ins.nanmean(to_gpu(x_np))), np.nanmean(x_np), rtol=1e-6)

    def test_nanstd(self):
        x_np = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float64)
        assert_allclose(to_numpy(ins.nanstd(to_gpu(x_np))), np.nanstd(x_np), rtol=1e-4)

    def test_nanvar(self):
        x_np = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float64)
        assert_allclose(to_numpy(ins.nanvar(to_gpu(x_np))), np.nanvar(x_np), rtol=1e-4)

    def test_any(self):
        x_np = np.array([False, False, True, False])
        assert_allclose(to_numpy(ins.any(to_gpu(x_np))), np.any(x_np))

    def test_all(self):
        x_np = np.array([True, True, True, False])
        assert_allclose(to_numpy(ins.all(to_gpu(x_np))), np.all(x_np))

    def test_sem(self, data_1d):
        expected = np.std(data_1d) / np.sqrt(len(data_1d))
        assert_allclose(to_numpy(ins.sem(to_gpu(data_1d))), expected, rtol=1e-4)


# ============================================================================
# Extended linalg alignment (GPU)
# ============================================================================


class TestLinalgExtendedGPU:
    """Extended linalg ops on GPU vs NumPy."""

    @pytest.fixture
    def spd_matrix_3x3(self):
        """Symmetric positive-definite 3x3 matrix."""
        return np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]], dtype=np.float64)

    @pytest.fixture
    def nonsingular_3x3(self):
        return np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=np.float64)

    @pytest.fixture
    def matrix_3x3(self):
        return np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=np.float64)

    def test_solve(self, nonsingular_3x3):
        A_np = nonsingular_3x3
        b_np = np.array([1, 2, 3], dtype=np.float64)
        A, b = to_gpu(A_np), to_gpu(b_np)
        x = ins.solve(A, b)
        expected = np.linalg.solve(A_np, b_np)
        assert_allclose(to_numpy(x), expected, atol=1e-6)

    def test_svd(self, matrix_3x3):
        A = to_gpu(matrix_3x3)
        U, S, Vt = ins.svd(A)
        _, S_np, _ = np.linalg.svd(matrix_3x3)
        assert_allclose(to_numpy(S), S_np, rtol=1e-6)

    def test_eigh(self, spd_matrix_3x3):
        A = to_gpu(spd_matrix_3x3)
        w, v = ins.eigh(A)
        w_np, _ = np.linalg.eigh(spd_matrix_3x3)
        assert_allclose(to_numpy(w), w_np, rtol=1e-6)

    def test_cholesky(self, spd_matrix_3x3):
        A = to_gpu(spd_matrix_3x3)
        L = ins.cholesky(A)
        expected = np.linalg.cholesky(spd_matrix_3x3)
        assert_allclose(to_numpy(L), expected, atol=1e-6)

    def test_qr(self, matrix_3x3):
        A = to_gpu(matrix_3x3)
        Q, R = ins.qr(A)
        # Check reconstruction
        assert_allclose(to_numpy(Q) @ to_numpy(R), matrix_3x3, atol=1e-6)

    def test_lu(self, matrix_3x3):
        A = to_gpu(matrix_3x3)
        P, L, U = ins.lu(A)
        recon = to_numpy(L) @ to_numpy(U)
        assert_allclose(to_numpy(P) @ matrix_3x3, recon, atol=1e-6)

    def test_lstsq(self):
        A_np = np.array([[1, 1], [1, 2], [1, 3], [1, 4]], dtype=np.float64)
        b_np = np.array([2.1, 3.9, 6.2, 7.8], dtype=np.float64)
        A, b = to_gpu(A_np), to_gpu(b_np)
        result = ins.lstsq(A, b)
        x_np, _, _, _ = np.linalg.lstsq(A_np, b_np, rcond=None)
        if isinstance(result, (list, tuple)):
            x_ins = result[0]
        else:
            x_ins = result
        assert_allclose(to_numpy(x_ins), x_np, atol=1e-6)

    def test_cond(self, matrix_3x3):
        A = to_gpu(matrix_3x3)
        result = ins.cond(A)
        expected = np.linalg.cond(matrix_3x3)
        assert_allclose(to_numpy(result), expected, rtol=1e-4)

    def test_matrix_rank(self, matrix_3x3):
        A = to_gpu(matrix_3x3)
        result = ins.matrix_rank(A)
        expected = np.linalg.matrix_rank(matrix_3x3)
        assert_allclose(to_numpy(result), expected)

    def test_pinv(self, matrix_3x3):
        A = to_gpu(matrix_3x3)
        result = ins.pinv(A)
        expected = np.linalg.pinv(matrix_3x3)
        assert_allclose(to_numpy(result), expected, atol=1e-6)

    def test_slogdet(self, matrix_3x3):
        A = to_gpu(matrix_3x3)
        sign, logabsdet = ins.slogdet(A)
        sign_np, logabsdet_np = np.linalg.slogdet(matrix_3x3)
        assert_allclose(to_numpy(sign), sign_np)
        assert_allclose(to_numpy(logabsdet), logabsdet_np, rtol=1e-6)

    def test_matrix_power(self, matrix_3x3):
        A = to_gpu(matrix_3x3)
        result = ins.matrix_power(A, 3)
        expected = np.linalg.matrix_power(matrix_3x3, 3)
        assert_allclose(to_numpy(result), expected, atol=1e-6)

    def test_eigvalsh(self, spd_matrix_3x3):
        A = to_gpu(spd_matrix_3x3)
        w = ins.eigvalsh(A)
        w_np = np.linalg.eigvalsh(spd_matrix_3x3)
        assert_allclose(to_numpy(w), w_np, rtol=1e-6)

    def test_svdvals(self, matrix_3x3):
        A = to_gpu(matrix_3x3)
        s = ins.svdvals(A)
        _, s_np, _ = np.linalg.svd(matrix_3x3)
        assert_allclose(to_numpy(s), s_np, rtol=1e-6)

    def test_cov_1d(self):
        x_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.cov(x)
        expected = np.cov(x_np)
        assert_allclose(to_numpy(result), expected, rtol=1e-6)

    def test_inner(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5, 6], dtype=np.float64)
        a, b = to_gpu(a_np), to_gpu(b_np)
        result = ins.inner(a, b)
        expected = np.inner(a_np, b_np)
        assert_allclose(to_numpy(result), expected, rtol=1e-6)


# ============================================================================
# Extended FFT alignment (GPU)
# ============================================================================


class TestFFTExtendedGPU:
    """Extended FFT ops on GPU vs NumPy FFT."""

    def test_fft2(self):
        x_np = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.fft2(x)
        assert_allclose(to_numpy(result), np.fft.fft2(x_np), atol=1e-6)

    def test_ifft2(self):
        x_np = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.complex128)
        x = to_gpu(x_np)
        result = ins.ifft2(x)
        assert_allclose(to_numpy(result), np.fft.ifft2(x_np), atol=1e-6)

    def test_fftn(self):
        x_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        x = to_gpu(x_np)
        result = ins.fftn(x)
        assert_allclose(to_numpy(result), np.fft.fftn(x_np), atol=1e-6)

    def test_ifftn(self):
        x_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        x = to_gpu(x_np)
        result = ins.ifftn(x)
        assert_allclose(to_numpy(result), np.fft.ifftn(x_np), atol=1e-6)

    def test_rfft_irfft_roundtrip(self):
        x_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        x = to_gpu(x_np)
        freq = ins.rfft(x)
        result = ins.irfft(freq, 8)
        assert_allclose(to_numpy(result), x_np, atol=1e-6)


# ============================================================================
# Cast alignment (GPU)
# ============================================================================


class TestCastAlignmentGPU:
    """Insight cast on GPU vs NumPy dtype conversion."""

    def test_float32_to_float64(self):
        x_np = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        x = to_gpu(x_np)
        result = ins.cast(x, ins.float64)
        assert_allclose(to_numpy(result), x_np.astype(np.float64))

    def test_float64_to_float32(self):
        x_np = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.cast(x, ins.float32)
        assert_allclose(to_numpy(result), x_np.astype(np.float32))

    def test_float64_to_int32(self):
        x_np = np.array([1.9, 2.1, -3.7, 0.0], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.cast(x, ins.int32)
        expected = x_np.astype(np.int32)
        assert_allclose(to_numpy(result), expected)

    def test_float64_to_int64(self):
        x_np = np.array([1.9, 2.1, -3.7, 0.0], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.cast(x, ins.int64)
        expected = x_np.astype(np.int64)
        assert_allclose(to_numpy(result), expected)

    def test_float64_to_bool(self):
        x_np = np.array([0.0, 1.0, -1.0, 0.0, 3.14], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.cast(x, ins.bool)
        expected = x_np.astype(bool)
        assert_allclose(to_numpy(result), expected)

    def test_int32_to_float64(self):
        x_np = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        x = to_gpu(x_np)
        result = ins.cast(x, ins.float64)
        assert_allclose(to_numpy(result), x_np.astype(np.float64))

    def test_bool_to_float64(self):
        x_np = np.array([True, False, True, False], dtype=bool)
        x = to_gpu(x_np)
        result = ins.cast(x, ins.float64)
        assert_allclose(to_numpy(result), x_np.astype(np.float64))


# ============================================================================
# Complex alignment (GPU)
# ============================================================================


class TestComplexAlignmentGPU:
    """Insight complex ops on GPU vs NumPy."""

    def test_to_complex(self):
        real_np = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        imag_np = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        real, imag = to_gpu(real_np), to_gpu(imag_np)
        result = ins.to_complex(real, imag)
        expected = real_np + 1j * imag_np
        assert_allclose(to_numpy(result), expected)

    def test_as_complex(self):
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.as_complex(x)
        expected = x_np.astype(np.complex128)
        assert_allclose(to_numpy(result), expected)

    def test_real(self):
        x_np = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
        x = to_gpu(x_np)
        result = ins.real(x)
        assert_allclose(to_numpy(result), np.real(x_np))

    def test_imag(self):
        x_np = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
        x = to_gpu(x_np)
        result = ins.imag(x)
        assert_allclose(to_numpy(result), np.imag(x_np))

    def test_as_real(self):
        x_np = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
        x = to_gpu(x_np)
        result = ins.as_real(x)
        expected = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
        assert_allclose(to_numpy(result), expected)

    def test_is_complex(self):
        x_real = to_gpu(np.array([1.0], dtype=np.float64))
        x_cplx = to_gpu(np.array([1 + 2j], dtype=np.complex128))
        assert ins.is_complex(x_cplx) == True  # noqa: E712
        assert ins.is_complex(x_real) == False  # noqa: E712

    def test_conj(self):
        """Test conjugate if available."""
        try:
            from insight._insight import conj
        except ImportError:
            pytest.skip("conj not available in Python bindings")
        x_np = np.array([1 + 2j, 3 - 4j], dtype=np.complex128)
        x = to_gpu(x_np)
        result = conj(x)
        assert_allclose(to_numpy(result), np.conj(x_np))

    def test_angle(self):
        """Test angle (phase) if available."""
        try:
            from insight._insight import angle
        except ImportError:
            pytest.skip("angle not available in Python bindings")
        x_np = np.array([1 + 0j, 0 + 1j, -1 + 0j, 1 + 1j], dtype=np.complex128)
        x = to_gpu(x_np)
        result = angle(x)
        assert_allclose(to_numpy(result), np.angle(x_np), rtol=1e-6)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
