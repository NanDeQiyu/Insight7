"""Unary alignment tests — Insight CUDA vs NumPy.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \\
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \\
    python -m pytest tests/python_align/cuda/test_unary.py -v
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
