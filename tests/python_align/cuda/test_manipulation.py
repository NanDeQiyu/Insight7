"""Manipulation alignment tests — Insight CUDA vs NumPy.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \\
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \\
    python -m pytest tests/python_align/cuda/test_manipulation.py -v
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
