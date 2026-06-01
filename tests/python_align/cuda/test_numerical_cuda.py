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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
