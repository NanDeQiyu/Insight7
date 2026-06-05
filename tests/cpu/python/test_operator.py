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

    # ========== New operators ==========

    def test_matmul_operator(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        b_np = np.array([[5, 6], [7, 8]], dtype=np.float64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        np.testing.assert_allclose((a @ b).numpy(), a_np @ b_np, rtol=1e-10)

    def test_pow_operator(self):
        a_np = np.array([2, 3, 4], dtype=np.float64)
        b_np = np.array([3, 2, 1], dtype=np.float64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        np.testing.assert_allclose((a**b).numpy(), a_np**b_np, rtol=1e-10)

    def test_pow_scalar(self):
        a_np = np.array([2, 3, 4], dtype=np.float64)
        a = ins.from_numpy(a_np)
        np.testing.assert_allclose((a**2).numpy(), a_np**2, rtol=1e-10)

    def test_floordiv_operator(self):
        a_np = np.array([7, 10, 15], dtype=np.float64)
        b_np = np.array([2, 3, 4], dtype=np.float64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        np.testing.assert_allclose((a // b).numpy(), a_np // b_np, rtol=1e-10)

    def test_floordiv_scalar(self):
        a_np = np.array([7, 10, 15], dtype=np.float64)
        a = ins.from_numpy(a_np)
        np.testing.assert_allclose((a // 3).numpy(), a_np // 3, rtol=1e-10)

    def test_abs_operator(self):
        a_np = np.array([-1, 2, -3], dtype=np.float64)
        a = ins.from_numpy(a_np)
        np.testing.assert_allclose(abs(a).numpy(), np.abs(a_np), rtol=1e-10)

    def test_len_operator(self):
        a = ins.zeros([3, 4, 5], ins.float32)
        assert len(a) == 3

    def test_len_1d(self):
        a = ins.ones([10], ins.float32)
        assert len(a) == 10

    def test_int_scalar(self):
        a = ins.from_numpy(np.array(42.0, dtype=np.float64))
        assert int(a) == 42

    def test_float_scalar(self):
        a = ins.from_numpy(np.array(42, dtype=np.int64))
        assert float(a) == 42.0

    def test_and_operator(self):
        a_np = np.array([0b1010, 0b1100], dtype=np.int64)
        b_np = np.array([0b1100, 0b1010], dtype=np.int64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        np.testing.assert_array_equal((a & b).numpy(), a_np & b_np)

    def test_or_operator(self):
        a_np = np.array([0b1010, 0b1100], dtype=np.int64)
        b_np = np.array([0b1100, 0b1010], dtype=np.int64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        np.testing.assert_array_equal((a | b).numpy(), a_np | b_np)

    def test_xor_operator(self):
        a_np = np.array([0b1010, 0b1100], dtype=np.int64)
        b_np = np.array([0b1100, 0b1010], dtype=np.int64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        np.testing.assert_array_equal((a ^ b).numpy(), a_np ^ b_np)

    def test_lshift_operator(self):
        a_np = np.array([1, 2, 3], dtype=np.int64)
        a = ins.from_numpy(a_np)
        np.testing.assert_array_equal((a << 2).numpy(), a_np << 2)

    def test_rshift_operator(self):
        a_np = np.array([8, 16, 32], dtype=np.int64)
        a = ins.from_numpy(a_np)
        np.testing.assert_array_equal((a >> 2).numpy(), a_np >> 2)

    def test_invert_operator(self):
        a_np = np.array([0b1010, 0b1100], dtype=np.int64)
        a = ins.from_numpy(a_np)
        np.testing.assert_array_equal((~a).numpy(), ~a_np)

    def test_pos_operator(self):
        a_np = np.array([-1, 2, -3], dtype=np.float64)
        a = ins.from_numpy(a_np)
        np.testing.assert_allclose((+a).numpy(), (+a_np), rtol=1e-10)

    def test_mod_operator(self):
        a_np = np.array([7, 10, 15], dtype=np.float64)
        b_np = np.array([3, 4, 6], dtype=np.float64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        np.testing.assert_allclose((a % b).numpy(), a_np % b_np, rtol=1e-10)

    def test_ne_operator(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float64))
        b = ins.from_numpy(np.array([1, 0, 3], dtype=np.float64))
        np.testing.assert_array_equal((a != b).numpy(), [False, True, False])

    def test_ge_operator(self):
        a = ins.from_numpy(np.array([3, 1, 5], dtype=np.float64))
        b = ins.from_numpy(np.array([2, 4, 5], dtype=np.float64))
        np.testing.assert_array_equal((a >= b).numpy(), [True, False, True])

    def test_bool_scalar(self):
        a = ins.from_numpy(np.array(1.0, dtype=np.float64))
        assert bool(a) is True
        b = ins.from_numpy(np.array(0.0, dtype=np.float64))
        assert bool(b) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
