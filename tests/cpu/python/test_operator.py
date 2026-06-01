"""Operator CPU binding tests — aligned with C++ test_operator.cpp.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python
    LD_LIBRARY_PATH=build/backends/cpu
    python -m pytest tests/cpu/python/test_operator.py -v
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


class TestOperatorCPU:
    """Arithmetic, comparison, and indexing operators — CPU backend."""

    def test_add_operator(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5, 6], dtype=np.float64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        np.testing.assert_allclose((a + b).numpy(), a_np + b_np, rtol=1e-10)

    def test_sub_operator(self):
        a_np = np.array([10, 20, 30], dtype=np.float64)
        b_np = np.array([1, 2, 3], dtype=np.float64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        np.testing.assert_allclose((a - b).numpy(), a_np - b_np, rtol=1e-10)

    def test_mul_operator(self):
        a_np = np.array([2, 3, 4], dtype=np.float64)
        b_np = np.array([5, 6, 7], dtype=np.float64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        np.testing.assert_allclose((a * b).numpy(), a_np * b_np, rtol=1e-10)

    def test_div_operator(self):
        a_np = np.array([10, 20, 30], dtype=np.float64)
        b_np = np.array([2, 4, 5], dtype=np.float64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        np.testing.assert_allclose((a / b).numpy(), a_np / b_np, rtol=1e-10)

    def test_negation(self):
        a_np = np.array([1, -2, 3], dtype=np.float64)
        a = ins.from_numpy(a_np)
        np.testing.assert_allclose((-a).numpy(), -a_np, rtol=1e-10)

    def test_scalar_add(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        a = ins.from_numpy(a_np)
        np.testing.assert_allclose((a + 10.0).numpy(), a_np + 10.0, rtol=1e-10)

    def test_scalar_mul(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        a = ins.from_numpy(a_np)
        np.testing.assert_allclose((a * 2.0).numpy(), a_np * 2.0, rtol=1e-10)

    def test_eq_operator(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float64))
        b = ins.from_numpy(np.array([1, 0, 3], dtype=np.float64))
        result = a == b
        np.testing.assert_array_equal(result.numpy(), [True, False, True])

    def test_lt_operator(self):
        a = ins.from_numpy(np.array([1, 5, 3], dtype=np.float64))
        b = ins.from_numpy(np.array([2, 4, 5], dtype=np.float64))
        result = a < b
        np.testing.assert_array_equal(result.numpy(), [True, False, True])

    def test_le_operator(self):
        a = ins.from_numpy(np.array([1, 5, 3], dtype=np.float64))
        b = ins.from_numpy(np.array([2, 4, 5], dtype=np.float64))
        result = a <= b
        np.testing.assert_array_equal(result.numpy(), [True, False, True])

    def test_gt_operator(self):
        a = ins.from_numpy(np.array([3, 1, 5], dtype=np.float64))
        b = ins.from_numpy(np.array([2, 4, 5], dtype=np.float64))
        result = a > b
        np.testing.assert_array_equal(result.numpy(), [True, False, False])

    def test_indexing(self):
        a = ins.from_numpy(np.array([10, 20, 30, 40], dtype=np.float64))
        b = a["0"]
        assert b.numel() == 1

    def test_string_slice(self):
        a = ins.ones([4, 4], ins.float32)
        b = a[":,1:-1"]
        assert b.defined()
        assert b.numel() == 8

    def test_inplace_add(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        a = ins.from_numpy(a_np)
        a += ins.from_numpy(np.array([10, 20, 30], dtype=np.float64))
        np.testing.assert_allclose(a.numpy(), [11, 22, 33], rtol=1e-10)

    def test_inplace_mul(self):
        a_np = np.array([2, 3, 4], dtype=np.float64)
        a = ins.from_numpy(a_np)
        a *= ins.from_numpy(np.array([5, 6, 7], dtype=np.float64))
        np.testing.assert_allclose(a.numpy(), [10, 18, 28], rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
