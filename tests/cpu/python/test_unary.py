"""Unary CPU binding tests — aligned with C++ test_unary.cpp.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python
    LD_LIBRARY_PATH=build/backends/cpu
    python -m pytest tests/cpu/python/test_unary.py -v
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


class TestUnaryCPU:
    """Unary math operations — CPU backend."""

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

    def test_sin_cos_tan(self):
        a_np = np.array([0, 0.5, 1.0], dtype=np.float64)
        np.testing.assert_allclose(ins.sin(ins.from_numpy(a_np)).numpy(), np.sin(a_np), rtol=1e-10)
        np.testing.assert_allclose(ins.cos(ins.from_numpy(a_np)).numpy(), np.cos(a_np), rtol=1e-10)
        np.testing.assert_allclose(ins.tan(ins.from_numpy(a_np)).numpy(), np.tan(a_np), rtol=1e-10)

    def test_floor_ceil_round(self):
        a_np = np.array([1.2, 2.5, 3.7, -1.3], dtype=np.float64)
        np.testing.assert_allclose(
            ins.floor(ins.from_numpy(a_np)).numpy(), np.floor(a_np), rtol=1e-10
        )
        np.testing.assert_allclose(
            ins.ceil(ins.from_numpy(a_np)).numpy(), np.ceil(a_np), rtol=1e-10
        )
        np.testing.assert_allclose(
            ins.round(ins.from_numpy(a_np)).numpy(), np.round(a_np), rtol=1e-10
        )

    def test_sign(self):
        a_np = np.array([-3, -1, 0, 2, 4], dtype=np.float64)
        result = ins.sign(ins.from_numpy(a_np))
        np.testing.assert_allclose(result.numpy(), np.sign(a_np), rtol=1e-10)

    def test_negative(self):
        a_np = np.array([1, -2, 3], dtype=np.float64)
        result = ins.negative(ins.from_numpy(a_np))
        np.testing.assert_allclose(result.numpy(), -a_np, rtol=1e-10)

    def test_square(self):
        a_np = np.array([1, 2, 3, 4], dtype=np.float64)
        result = ins.square(ins.from_numpy(a_np))
        np.testing.assert_allclose(result.numpy(), a_np**2, rtol=1e-10)

    def test_reciprocal(self):
        a_np = np.array([1, 2, 4, 5], dtype=np.float64)
        result = ins.reciprocal(ins.from_numpy(a_np))
        np.testing.assert_allclose(result.numpy(), 1.0 / a_np, rtol=1e-10)

    def test_log2_log10(self):
        a_np = np.array([1, 2, 8, 100], dtype=np.float64)
        np.testing.assert_allclose(
            ins.log2(ins.from_numpy(a_np)).numpy(), np.log2(a_np), rtol=1e-10
        )
        np.testing.assert_allclose(
            ins.log10(ins.from_numpy(a_np)).numpy(), np.log10(a_np), rtol=1e-10
        )

    def test_asin_acos_atan(self):
        a_np = np.array([-0.5, 0, 0.5], dtype=np.float64)
        np.testing.assert_allclose(
            ins.asin(ins.from_numpy(a_np)).numpy(), np.arcsin(a_np), rtol=1e-10
        )
        np.testing.assert_allclose(
            ins.acos(ins.from_numpy(a_np)).numpy(), np.arccos(a_np), rtol=1e-10
        )
        np.testing.assert_allclose(
            ins.atan(ins.from_numpy(a_np)).numpy(), np.arctan(a_np), rtol=1e-10
        )

    def test_sinh_cosh_tanh(self):
        a_np = np.array([-1, 0, 1], dtype=np.float64)
        np.testing.assert_allclose(
            ins.sinh(ins.from_numpy(a_np)).numpy(), np.sinh(a_np), rtol=1e-10
        )
        np.testing.assert_allclose(
            ins.cosh(ins.from_numpy(a_np)).numpy(), np.cosh(a_np), rtol=1e-10
        )
        np.testing.assert_allclose(
            ins.tanh(ins.from_numpy(a_np)).numpy(), np.tanh(a_np), rtol=1e-10
        )

    def test_deg2rad_rad2deg(self):
        a_np = np.array([0, 90, 180, 360], dtype=np.float64)
        np.testing.assert_allclose(
            ins.deg2rad(ins.from_numpy(a_np)).numpy(), np.deg2rad(a_np), rtol=1e-10
        )
        b_np = np.array([0, np.pi / 2, np.pi, 2 * np.pi], dtype=np.float64)
        np.testing.assert_allclose(
            ins.rad2deg(ins.from_numpy(b_np)).numpy(), np.rad2deg(b_np), rtol=1e-10
        )

    def test_exp2_expm1_log1p(self):
        a_np = np.array([0, 1, 2, 3], dtype=np.float64)
        np.testing.assert_allclose(
            ins.exp2(ins.from_numpy(a_np)).numpy(), np.exp2(a_np), rtol=1e-10
        )
        np.testing.assert_allclose(
            ins.expm1(ins.from_numpy(a_np)).numpy(), np.expm1(a_np), rtol=1e-10
        )
        np.testing.assert_allclose(
            ins.log1p(ins.from_numpy(a_np)).numpy(), np.log1p(a_np), rtol=1e-10
        )

    def test_cbrt(self):
        a_np = np.array([1, 8, 27], dtype=np.float64)
        result = ins.cbrt(ins.from_numpy(a_np))
        np.testing.assert_allclose(result.numpy(), np.cbrt(a_np), rtol=1e-10)

    def test_trunc(self):
        a_np = np.array([1.2, -2.7, 3.5], dtype=np.float64)
        result = ins.trunc(ins.from_numpy(a_np))
        np.testing.assert_allclose(result.numpy(), np.trunc(a_np), rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
