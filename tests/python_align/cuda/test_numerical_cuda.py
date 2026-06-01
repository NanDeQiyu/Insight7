"""Numerical precision alignment tests — Insight vs NumPy/SciPy on CUDA.

These tests verify that Insight7 GPU operations produce results identical
(or within acceptable tolerance) to NumPy/SciPy reference implementations.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python python -m pytest tests/python_align/cuda/test_numerical_cuda.py -v
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
        a = ins.from_numpy(a_np).to(GPU)
        b = ins.from_numpy(b_np).to(GPU)
        result = ins.add(a, b).to(CPU)
        assert_allclose(result.numpy(), a_np + b_np)

    def test_sub(self, random_arrays):
        a_np, b_np = random_arrays
        a = ins.from_numpy(a_np).to(GPU)
        b = ins.from_numpy(b_np).to(GPU)
        result = ins.sub(a, b).to(CPU)
        assert_allclose(result.numpy(), a_np - b_np)

    def test_mul(self, random_arrays):
        a_np, b_np = random_arrays
        a = ins.from_numpy(a_np).to(GPU)
        b = ins.from_numpy(b_np).to(GPU)
        result = ins.mul(a, b).to(CPU)
        assert_allclose(result.numpy(), a_np * b_np)

    def test_div(self, random_arrays):
        a_np, b_np = random_arrays
        a = ins.from_numpy(a_np).to(GPU)
        b = ins.from_numpy(b_np).to(GPU)
        result = ins.div(a, b).to(CPU)
        assert_allclose(result.numpy(), a_np / b_np, rtol=1e-6)

    def test_maximum(self, random_arrays):
        a_np, b_np = random_arrays
        a = ins.from_numpy(a_np).to(GPU)
        b = ins.from_numpy(b_np).to(GPU)
        result = ins.maximum(a, b).to(CPU)
        assert_allclose(result.numpy(), np.maximum(a_np, b_np))

    def test_minimum(self, random_arrays):
        a_np, b_np = random_arrays
        a = ins.from_numpy(a_np).to(GPU)
        b = ins.from_numpy(b_np).to(GPU)
        result = ins.minimum(a, b).to(CPU)
        assert_allclose(result.numpy(), np.minimum(a_np, b_np))


# ============================================================================
# Unary alignment (GPU)
# ============================================================================

class TestUnaryAlignmentGPU:
    """Insight unary ops on GPU vs NumPy."""

    def test_abs(self):
        x_np = np.array([-3, -1, 0, 2, 4], dtype=np.float64)
        x = ins.from_numpy(x_np).to(GPU)
        result = ins.abs(x).to(CPU)
        assert_allclose(result.numpy(), np.abs(x_np))

    def test_sqrt(self):
        x_np = np.array([1, 4, 9, 16], dtype=np.float64)
        x = ins.from_numpy(x_np).to(GPU)
        result = ins.sqrt(x).to(CPU)
        assert_allclose(result.numpy(), np.sqrt(x_np), rtol=1e-8)

    def test_exp(self):
        x_np = np.array([0, 1, 2], dtype=np.float64)
        x = ins.from_numpy(x_np).to(GPU)
        result = ins.exp(x).to(CPU)
        assert_allclose(result.numpy(), np.exp(x_np), rtol=1e-6)

    def test_sin(self):
        x_np = np.array([0, 0.5, 1.0], dtype=np.float64)
        x = ins.from_numpy(x_np).to(GPU)
        result = ins.sin(x).to(CPU)
        assert_allclose(result.numpy(), np.sin(x_np), rtol=1e-8)

    def test_cos(self):
        x_np = np.array([0, 0.5, 1.0], dtype=np.float64)
        x = ins.from_numpy(x_np).to(GPU)
        result = ins.cos(x).to(CPU)
        assert_allclose(result.numpy(), np.cos(x_np), rtol=1e-8)

    def test_floor(self):
        x_np = np.array([1.1, 2.7, -0.3, 3.0], dtype=np.float64)
        x = ins.from_numpy(x_np).to(GPU)
        result = ins.floor(x).to(CPU)
        assert_allclose(result.numpy(), np.floor(x_np))

    def test_ceil(self):
        x_np = np.array([1.1, 2.7, -0.3, 3.0], dtype=np.float64)
        x = ins.from_numpy(x_np).to(GPU)
        result = ins.ceil(x).to(CPU)
        assert_allclose(result.numpy(), np.ceil(x_np))


# ============================================================================
# Reduction alignment (GPU)
# ============================================================================

class TestReductionAlignmentGPU:
    """Insight reduction on GPU vs NumPy."""

    def test_sum(self):
        x_np = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        x = ins.from_numpy(x_np).to(GPU)
        result = ins.sum(x).to(CPU)
        assert_allclose(result.numpy(), np.sum(x_np))

    def test_mean(self):
        x_np = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        x = ins.from_numpy(x_np).to(GPU)
        result = ins.mean(x).to(CPU)
        assert_allclose(result.numpy(), np.mean(x_np))

    def test_max(self):
        x_np = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.float64)
        x = ins.from_numpy(x_np).to(GPU)
        result = ins.max(x).to(CPU)
        assert_allclose(result.numpy(), np.max(x_np))

    def test_min(self):
        x_np = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.float64)
        x = ins.from_numpy(x_np).to(GPU)
        result = ins.min(x).to(CPU)
        assert_allclose(result.numpy(), np.min(x_np))

    def test_sum_axis(self):
        x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        x = ins.from_numpy(x_np).to(GPU)
        result = ins.sum(x, axis=0).to(CPU)
        assert_allclose(result.numpy(), np.sum(x_np, axis=0))


# ============================================================================
# Linalg alignment (GPU)
# ============================================================================

class TestLinalgAlignmentGPU:
    """Insight linalg on GPU vs NumPy."""

    def test_matmul(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        b_np = np.array([[5, 6], [7, 8]], dtype=np.float64)
        a = ins.from_numpy(a_np).to(GPU)
        b = ins.from_numpy(b_np).to(GPU)
        result = ins.matmul(a, b).to(CPU)
        assert_allclose(result.numpy(), np.matmul(a_np, b_np), rtol=1e-8)

    def test_det(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        a = ins.from_numpy(a_np).to(GPU)
        result = ins.det(a).to(CPU)
        assert_allclose(result.numpy(), np.linalg.det(a_np), rtol=1e-8)

    def test_inv(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        a = ins.from_numpy(a_np).to(GPU)
        result = ins.inv(a).to(CPU)
        assert_allclose(result.numpy(), np.linalg.inv(a_np), atol=1e-8)

    def test_trace(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        a = ins.from_numpy(a_np).to(GPU)
        result = ins.trace(a).to(CPU)
        assert_allclose(result.numpy(), np.trace(a_np))

    def test_norm(self):
        x_np = np.array([3, 4], dtype=np.float64)
        x = ins.from_numpy(x_np).to(GPU)
        result = ins.norm(x).to(CPU)
        assert_allclose(result.numpy(), np.linalg.norm(x_np), rtol=1e-8)


# ============================================================================
# FFT alignment (GPU)
# ============================================================================

class TestFFTAlignmentGPU:
    """Insight FFT on GPU vs NumPy FFT."""

    def test_fft(self):
        x_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        x = ins.from_numpy(x_np).to(GPU)
        result = ins.fft(x).to(CPU)
        assert_allclose(result.numpy(), np.fft.fft(x_np), atol=1e-8)

    def test_ifft(self):
        x_np = np.array([1, 2, 3, 4], dtype=np.complex128)
        x = ins.from_numpy(x_np).to(GPU)
        result = ins.ifft(x).to(CPU)
        assert_allclose(result.numpy(), np.fft.ifft(x_np), atol=1e-8)

    def test_fft_ifft_roundtrip(self):
        x_np = np.random.randn(16).astype(np.float64)
        x = ins.from_numpy(x_np).to(GPU)
        result = ins.ifft(ins.fft(x)).to(CPU)
        assert_allclose(result.numpy().real, x_np, atol=1e-8)
