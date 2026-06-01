"""CUDA basic operations tests — aligned with C++ test suite.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python python -m pytest tests/cuda/python/test_cuda_basic.py -v
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

# Skip if CUDA not available
try:
    ins.load_backend("cuda")
except Exception:
    pytest.skip("CUDA backend not available", allow_module_level=True)

GPU = ins.GPUPlace(0)
CPU = ins.CPUPlace()


# ============================================================================
# Creation
# ============================================================================


class TestCreationGPU:
    """Array creation on GPU — aligned with test_creation.cpp (CUDA)."""

    def test_zeros(self):
        a = ins.zeros([2, 3], ins.float32, GPU)
        assert a.numel() == 6
        cpu_a = a.to(CPU)
        assert cpu_a.numpy().sum() == 0.0

    def test_ones(self):
        a = ins.ones([2, 3], ins.float32, GPU)
        cpu_a = a.to(CPU)
        np.testing.assert_allclose(cpu_a.numpy(), np.ones([2, 3]))

    def test_full(self):
        a = ins.full([2, 3], 3.14, ins.float32, GPU)
        cpu_a = a.to(CPU)
        np.testing.assert_allclose(cpu_a.numpy(), np.full([2, 3], 3.14), rtol=1e-5)

    def test_eye(self):
        a = ins.eye(3, place=GPU)
        cpu_a = a.to(CPU)
        np.testing.assert_allclose(cpu_a.numpy(), np.eye(3))


# ============================================================================
# Elementwise
# ============================================================================


class TestElementwiseGPU:
    """Elementwise ops on GPU — aligned with test_elementwise.cpp (CUDA)."""

    def test_add(self):
        a_np = np.array([1, 2, 3], dtype=np.float32)
        b_np = np.array([4, 5, 6], dtype=np.float32)
        a = ins.from_numpy(a_np).to(GPU)
        b = ins.from_numpy(b_np).to(GPU)
        result = ins.add(a, b).to(CPU)
        np.testing.assert_allclose(result.numpy(), a_np + b_np)

    def test_sub(self):
        a_np = np.array([10, 20, 30], dtype=np.float32)
        b_np = np.array([1, 2, 3], dtype=np.float32)
        a = ins.from_numpy(a_np).to(GPU)
        b = ins.from_numpy(b_np).to(GPU)
        result = ins.sub(a, b).to(CPU)
        np.testing.assert_allclose(result.numpy(), a_np - b_np)

    def test_mul(self):
        a_np = np.array([2, 3, 4], dtype=np.float32)
        b_np = np.array([5, 6, 7], dtype=np.float32)
        a = ins.from_numpy(a_np).to(GPU)
        b = ins.from_numpy(b_np).to(GPU)
        result = ins.mul(a, b).to(CPU)
        np.testing.assert_allclose(result.numpy(), a_np * b_np)

    def test_div(self):
        a_np = np.array([10, 20, 30], dtype=np.float32)
        b_np = np.array([2, 4, 5], dtype=np.float32)
        a = ins.from_numpy(a_np).to(GPU)
        b = ins.from_numpy(b_np).to(GPU)
        result = ins.div(a, b).to(CPU)
        np.testing.assert_allclose(result.numpy(), a_np / b_np, rtol=1e-5)


# ============================================================================
# Unary
# ============================================================================


class TestUnaryGPU:
    """Unary ops on GPU — aligned with test_unary.cpp (CUDA)."""

    def test_abs(self):
        a_np = np.array([-3, -1, 0, 2, 4], dtype=np.float32)
        a = ins.from_numpy(a_np).to(GPU)
        result = ins.abs(a).to(CPU)
        np.testing.assert_allclose(result.numpy(), np.abs(a_np))

    def test_sqrt(self):
        a_np = np.array([1, 4, 9, 16], dtype=np.float32)
        a = ins.from_numpy(a_np).to(GPU)
        result = ins.sqrt(a).to(CPU)
        np.testing.assert_allclose(result.numpy(), np.sqrt(a_np), rtol=1e-5)

    def test_exp(self):
        a_np = np.array([0, 1, 2], dtype=np.float32)
        a = ins.from_numpy(a_np).to(GPU)
        result = ins.exp(a).to(CPU)
        np.testing.assert_allclose(result.numpy(), np.exp(a_np), rtol=1e-5)

    def test_sin(self):
        a_np = np.array([0, 0.5, 1.0], dtype=np.float32)
        a = ins.from_numpy(a_np).to(GPU)
        result = ins.sin(a).to(CPU)
        np.testing.assert_allclose(result.numpy(), np.sin(a_np), rtol=1e-5)


# ============================================================================
# Reduction
# ============================================================================


class TestReductionGPU:
    """Reduction on GPU — aligned with test_reduction.cpp (CUDA)."""

    def test_sum(self):
        a_np = np.array([1, 2, 3, 4], dtype=np.float32)
        a = ins.from_numpy(a_np).to(GPU)
        result = ins.sum(a).to(CPU)
        assert result.numpy().item() == pytest.approx(10.0)

    def test_mean(self):
        a_np = np.array([1, 2, 3, 4], dtype=np.float32)
        a = ins.from_numpy(a_np).to(GPU)
        result = ins.mean(a).to(CPU)
        assert result.numpy().item() == pytest.approx(2.5)

    def test_max(self):
        a_np = np.array([3, 1, 4, 1, 5], dtype=np.float32)
        a = ins.from_numpy(a_np).to(GPU)
        result = ins.max(a).to(CPU)
        assert result.numpy().item() == pytest.approx(5.0)


# ============================================================================
# Linalg
# ============================================================================


class TestLinalgGPU:
    """Linalg on GPU — aligned with test_linalg.cpp (CUDA)."""

    def test_matmul(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b_np = np.array([[5, 6], [7, 8]], dtype=np.float32)
        a = ins.from_numpy(a_np).to(GPU)
        b = ins.from_numpy(b_np).to(GPU)
        result = ins.matmul(a, b).to(CPU)
        expected = np.matmul(a_np, b_np)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_det(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        a = ins.from_numpy(a_np).to(GPU)
        result = ins.det(a).to(CPU)
        assert result.numpy().item() == pytest.approx(-2.0, abs=1e-8)


# ============================================================================
# Signal
# ============================================================================


class TestSignalGPU:
    """Signal processing on GPU — aligned with test_signal_*.cpp (CUDA)."""

    def test_hann_window(self):
        w = ins.signal.hann(16)
        assert w.numel() == 16

    def test_fftconvolve(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float64)).to(GPU)
        b = ins.from_numpy(np.array([1, 1], dtype=np.float64)).to(GPU)
        result = ins.signal.fftconvolve(a, b).to(CPU)
        expected = np.convolve([1, 2, 3], [1, 1])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-8)
