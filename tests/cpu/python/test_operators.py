"""Comprehensive Python operator tests — all dunder operators, CPU + GPU.

Covers: Array×Array, Array×scalar, scalar×Array, type errors, edge cases.
Run with:
    PYTHONPATH=bindings/python LD_LIBRARY_PATH=build/backends/cpu
    python -m pytest tests/cpu/python/test_operators.py -v
"""

import sys
import os
import pytest
import numpy as np

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, os.path.join(_root, "bindings", "python"))
sys.path.insert(0, os.path.join(_root, "build", "bindings", "python"))

try:
    import insight as ins
except ImportError:
    pytest.skip("Insight not available", allow_module_level=True)


# ============================================================================
# Helpers
# ============================================================================


def _place_ids():
    """Return available places for parametrize."""
    places = [("CPU", ins.CPUPlace())]
    if ins.has_device("gpu"):
        places.append(("GPU", ins.GPUPlace(0)))
    return places


def _np_to_ins(arr, place=None):
    """Convert numpy array to Insight array on given place."""
    a = ins.from_numpy(arr)
    if place is not None:
        a = a.to(place)
    return a


# ============================================================================
# Arithmetic Operators: +, -, *, /, //, %, **, @
# ============================================================================


class TestArithmetic:
    """Test all arithmetic operators with Array and scalar operands."""

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_add_array_array(self, place_name, place):
        a = _np_to_ins(np.array([1, 2, 3], dtype=np.float64), place)
        b = _np_to_ins(np.array([4, 5, 6], dtype=np.float64), place)
        np.testing.assert_allclose((a + b).numpy(), [5, 7, 9], rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_add_array_scalar(self, place_name, place):
        a = _np_to_ins(np.array([1, 2, 3], dtype=np.float64), place)
        np.testing.assert_allclose((a + 10.0).numpy(), [11, 12, 13], rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_radd_scalar_array(self, place_name, place):
        a = _np_to_ins(np.array([1, 2, 3], dtype=np.float64), place)
        np.testing.assert_allclose((10.0 + a).numpy(), [11, 12, 13], rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_sub_array_array(self, place_name, place):
        a = _np_to_ins(np.array([10, 20, 30], dtype=np.float64), place)
        b = _np_to_ins(np.array([1, 2, 3], dtype=np.float64), place)
        np.testing.assert_allclose((a - b).numpy(), [9, 18, 27], rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_rsub_scalar_array(self, place_name, place):
        a = _np_to_ins(np.array([1, 2, 3], dtype=np.float64), place)
        np.testing.assert_allclose((10.0 - a).numpy(), [9, 8, 7], rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_mul_array_array(self, place_name, place):
        a = _np_to_ins(np.array([2, 3, 4], dtype=np.float64), place)
        b = _np_to_ins(np.array([5, 6, 7], dtype=np.float64), place)
        np.testing.assert_allclose((a * b).numpy(), [10, 18, 28], rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_rmul_scalar_array(self, place_name, place):
        a = _np_to_ins(np.array([2, 3, 4], dtype=np.float64), place)
        np.testing.assert_allclose((3.0 * a).numpy(), [6, 9, 12], rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_truediv_array_array(self, place_name, place):
        a = _np_to_ins(np.array([10, 20, 30], dtype=np.float64), place)
        b = _np_to_ins(np.array([2, 4, 5], dtype=np.float64), place)
        np.testing.assert_allclose((a / b).numpy(), [5, 5, 6], rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_truediv_scalar(self, place_name, place):
        a = _np_to_ins(np.array([10, 20, 30], dtype=np.float64), place)
        np.testing.assert_allclose((a / 2.0).numpy(), [5, 10, 15], rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_rtruediv_scalar(self, place_name, place):
        a = _np_to_ins(np.array([2, 4, 5], dtype=np.float64), place)
        np.testing.assert_allclose((60.0 / a).numpy(), [30, 15, 12], rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_floordiv_array_array(self, place_name, place):
        a_np = np.array([7, 10, 15], dtype=np.float64)
        b_np = np.array([2, 3, 4], dtype=np.float64)
        a = _np_to_ins(a_np, place)
        b = _np_to_ins(b_np, place)
        np.testing.assert_allclose((a // b).numpy(), a_np // b_np, rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_floordiv_scalar(self, place_name, place):
        a_np = np.array([7, 10, 15], dtype=np.float64)
        a = _np_to_ins(a_np, place)
        np.testing.assert_allclose((a // 3).numpy(), a_np // 3, rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_rfloordiv_scalar(self, place_name, place):
        a_np = np.array([2, 3, 4], dtype=np.float64)
        a = _np_to_ins(a_np, place)
        np.testing.assert_allclose((10.0 // a).numpy(), 10.0 // a_np, rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_mod_array_array(self, place_name, place):
        a_np = np.array([7, 10, 15], dtype=np.float64)
        b_np = np.array([3, 4, 6], dtype=np.float64)
        a = _np_to_ins(a_np, place)
        b = _np_to_ins(b_np, place)
        np.testing.assert_allclose((a % b).numpy(), a_np % b_np, rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_mod_scalar(self, place_name, place):
        a_np = np.array([7, 10, 15], dtype=np.float64)
        a = _np_to_ins(a_np, place)
        np.testing.assert_allclose((a % 3).numpy(), a_np % 3, rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_pow_array_array(self, place_name, place):
        a_np = np.array([2, 3, 4], dtype=np.float64)
        b_np = np.array([3, 2, 1], dtype=np.float64)
        a = _np_to_ins(a_np, place)
        b = _np_to_ins(b_np, place)
        np.testing.assert_allclose((a**b).numpy(), a_np**b_np, rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_pow_scalar(self, place_name, place):
        a_np = np.array([2, 3, 4], dtype=np.float64)
        a = _np_to_ins(a_np, place)
        np.testing.assert_allclose((a**2).numpy(), a_np**2, rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_rpow_scalar(self, place_name, place):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        a = _np_to_ins(a_np, place)
        np.testing.assert_allclose((2.0**a).numpy(), 2.0**a_np, rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_matmul_array_array(self, place_name, place):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        b_np = np.array([[5, 6], [7, 8]], dtype=np.float64)
        a = _np_to_ins(a_np, place)
        b = _np_to_ins(b_np, place)
        np.testing.assert_allclose((a @ b).numpy(), a_np @ b_np, rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_matmul_identity(self, place_name, place):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        eye = np.eye(3, dtype=np.float64)
        a = _np_to_ins(a_np, place)
        e = _np_to_ins(eye, place)
        np.testing.assert_allclose((a @ e).numpy(), a_np, rtol=1e-10)


# ============================================================================
# Unary Operators: -, +, abs, ~
# ============================================================================


class TestUnary:
    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_neg(self, place_name, place):
        a_np = np.array([1, -2, 3], dtype=np.float64)
        a = _np_to_ins(a_np, place)
        np.testing.assert_allclose((-a).numpy(), -a_np, rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_pos(self, place_name, place):
        a_np = np.array([-1, 2, -3], dtype=np.float64)
        a = _np_to_ins(a_np, place)
        np.testing.assert_allclose((+a).numpy(), (+a_np), rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_abs(self, place_name, place):
        a_np = np.array([-1, 2, -3], dtype=np.float64)
        a = _np_to_ins(a_np, place)
        np.testing.assert_allclose(abs(a).numpy(), np.abs(a_np), rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_invert(self, place_name, place):
        a_np = np.array([0b1010, 0b1100, 0], dtype=np.int64)
        a = _np_to_ins(a_np, place)
        np.testing.assert_array_equal((~a).numpy(), ~a_np)


# ============================================================================
# Comparison Operators: ==, !=, <, <=, >, >=
# ============================================================================


class TestComparison:
    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_eq(self, place_name, place):
        a = _np_to_ins(np.array([1, 2, 3], dtype=np.float64), place)
        b = _np_to_ins(np.array([1, 0, 3], dtype=np.float64), place)
        np.testing.assert_array_equal((a == b).numpy(), [True, False, True])

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_ne(self, place_name, place):
        a = _np_to_ins(np.array([1, 2, 3], dtype=np.float64), place)
        b = _np_to_ins(np.array([1, 0, 3], dtype=np.float64), place)
        np.testing.assert_array_equal((a != b).numpy(), [False, True, False])

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_lt(self, place_name, place):
        a = _np_to_ins(np.array([1, 5, 3], dtype=np.float64), place)
        b = _np_to_ins(np.array([2, 4, 5], dtype=np.float64), place)
        np.testing.assert_array_equal((a < b).numpy(), [True, False, True])

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_le(self, place_name, place):
        a = _np_to_ins(np.array([1, 5, 3], dtype=np.float64), place)
        b = _np_to_ins(np.array([2, 4, 5], dtype=np.float64), place)
        np.testing.assert_array_equal((a <= b).numpy(), [True, False, True])

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_gt(self, place_name, place):
        a = _np_to_ins(np.array([3, 1, 5], dtype=np.float64), place)
        b = _np_to_ins(np.array([2, 4, 5], dtype=np.float64), place)
        np.testing.assert_array_equal((a > b).numpy(), [True, False, False])

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_ge(self, place_name, place):
        a = _np_to_ins(np.array([3, 1, 5], dtype=np.float64), place)
        b = _np_to_ins(np.array([2, 4, 5], dtype=np.float64), place)
        np.testing.assert_array_equal((a >= b).numpy(), [True, False, True])


# ============================================================================
# Bitwise Operators: &, |, ^, <<, >>
# ============================================================================


class TestBitwise:
    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_and(self, place_name, place):
        a_np = np.array([0b1010, 0b1100], dtype=np.int64)
        b_np = np.array([0b1100, 0b1010], dtype=np.int64)
        a = _np_to_ins(a_np, place)
        b = _np_to_ins(b_np, place)
        np.testing.assert_array_equal((a & b).numpy(), a_np & b_np)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_and_scalar(self, place_name, place):
        a_np = np.array([0b1010, 0b1100], dtype=np.int64)
        a = _np_to_ins(a_np, place)
        np.testing.assert_array_equal((a & 0b1100).numpy(), a_np & 0b1100)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_or(self, place_name, place):
        a_np = np.array([0b1010, 0b1100], dtype=np.int64)
        b_np = np.array([0b1100, 0b1010], dtype=np.int64)
        a = _np_to_ins(a_np, place)
        b = _np_to_ins(b_np, place)
        np.testing.assert_array_equal((a | b).numpy(), a_np | b_np)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_or_scalar(self, place_name, place):
        a_np = np.array([0b1010, 0b1100], dtype=np.int64)
        a = _np_to_ins(a_np, place)
        np.testing.assert_array_equal((a | 0b0001).numpy(), a_np | 0b0001)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_xor(self, place_name, place):
        a_np = np.array([0b1010, 0b1100], dtype=np.int64)
        b_np = np.array([0b1100, 0b1010], dtype=np.int64)
        a = _np_to_ins(a_np, place)
        b = _np_to_ins(b_np, place)
        np.testing.assert_array_equal((a ^ b).numpy(), a_np ^ b_np)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_lshift(self, place_name, place):
        a_np = np.array([1, 2, 3], dtype=np.int64)
        a = _np_to_ins(a_np, place)
        np.testing.assert_array_equal((a << 2).numpy(), a_np << 2)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_rshift(self, place_name, place):
        a_np = np.array([8, 16, 32], dtype=np.int64)
        a = _np_to_ins(a_np, place)
        np.testing.assert_array_equal((a >> 2).numpy(), a_np >> 2)


# ============================================================================
# Scalar Conversion: len, int, float, bool
# ============================================================================


class TestScalarConversion:
    def test_len_2d(self):
        a = ins.zeros([3, 4, 5], ins.float32)
        assert len(a) == 3

    def test_len_1d(self):
        a = ins.ones([10], ins.float32)
        assert len(a) == 10

    def test_len_scalar_raises(self):
        a = ins.from_numpy(np.array(42.0, dtype=np.float64))
        with pytest.raises(TypeError):
            len(a)

    def test_int_from_float(self):
        a = ins.from_numpy(np.array(42.7, dtype=np.float64))
        assert int(a) == 42

    def test_int_from_int(self):
        a = ins.from_numpy(np.array(42, dtype=np.int64))
        assert int(a) == 42

    def test_float_from_float(self):
        a = ins.from_numpy(np.array(3.14, dtype=np.float64))
        assert abs(float(a) - 3.14) < 1e-10

    def test_float_from_int(self):
        a = ins.from_numpy(np.array(42, dtype=np.int64))
        assert float(a) == 42.0

    def test_bool_true(self):
        a = ins.from_numpy(np.array(1.0, dtype=np.float64))
        assert bool(a) is True

    def test_bool_false(self):
        a = ins.from_numpy(np.array(0.0, dtype=np.float64))
        assert bool(a) is False

    def test_int_from_array_raises(self):
        a = ins.ones([3], ins.float32)
        with pytest.raises(Exception):
            int(a)

    def test_float_from_array_raises(self):
        a = ins.ones([3], ins.float32)
        with pytest.raises(Exception):
            float(a)


# ============================================================================
# Broadcasting
# ============================================================================


class TestBroadcasting:
    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_add_broadcast(self, place_name, place):
        a_np = np.array([[1, 2, 3]], dtype=np.float64)  # (1, 3)
        b_np = np.array([[1], [2]], dtype=np.float64)  # (2, 1)
        a = _np_to_ins(a_np, place)
        b = _np_to_ins(b_np, place)
        np.testing.assert_allclose((a + b).numpy(), a_np + b_np, rtol=1e-10)

    @pytest.mark.parametrize("place_name,place", _place_ids())
    def test_mul_broadcast(self, place_name, place):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        b_np = np.array([[10], [20]], dtype=np.float64)
        a = _np_to_ins(a_np, place)
        b = _np_to_ins(b_np, place)
        np.testing.assert_allclose((a * b).numpy(), a_np * b_np, rtol=1e-10)


# ============================================================================
# Error Cases
# ============================================================================


class TestErrorCases:
    def test_matmul_string_raises(self):
        a = ins.ones([2, 2], ins.float32)
        with pytest.raises(Exception):
            a @ "hello"

    def test_add_string_raises(self):
        a = ins.ones([3], ins.float32)
        with pytest.raises(Exception):
            a + "hello"

    def test_div_zero(self):
        a = ins.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        b = ins.from_numpy(np.array([0.0, 1.0, 0.0], dtype=np.float64))
        result = (a / b).numpy()
        assert np.isinf(result[0])
        assert result[1] == 2.0
        assert np.isinf(result[2])


# ============================================================================
# DType Coverage: f32, f64, i32, i64
# ============================================================================


class TestDtypeCoverage:
    @pytest.mark.parametrize(
        "dtype,np_dtype",
        [
            (ins.float32, np.float32),
            (ins.float64, np.float64),
            (ins.int32, np.int32),
            (ins.int64, np.int64),
        ],
    )
    def test_add_dtypes(self, dtype, np_dtype):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np_dtype))
        b = ins.from_numpy(np.array([4, 5, 6], dtype=np_dtype))
        result = a + b
        np.testing.assert_array_equal(result.numpy(), np.array([5, 7, 9], dtype=np_dtype))

    def test_mixed_dtype_promotion(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float32))
        b = ins.from_numpy(np.array([4, 5, 6], dtype=np.float64))
        result = a + b
        assert result.numpy().dtype == np.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
