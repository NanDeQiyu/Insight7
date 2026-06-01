"""Elementwise alignment tests — Insight CUDA vs NumPy.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \\
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \\
    python -m pytest tests/python_align/cuda/test_elementwise.py -v
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


def to_gpu(np_arr):
    return ins.from_numpy(np_arr).to(GPU)


def to_numpy(gpu_arr):
    return gpu_arr.to(CPU).numpy()


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
