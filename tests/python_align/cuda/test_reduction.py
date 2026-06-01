"""Reduction alignment tests — Insight CUDA vs NumPy.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \\
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \\
    python -m pytest tests/python_align/cuda/test_reduction.py -v
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
