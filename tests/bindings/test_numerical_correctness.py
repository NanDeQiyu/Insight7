"""Numerical correctness tests: Insight vs NumPy.

Verifies that Insight operations produce results identical (or within
floating-point tolerance) to NumPy for the same inputs.

Run with:
    LD_LIBRARY_PATH=build/backends/cpu python3 -m pytest tests/bindings/test_numerical_correctness.py -v
"""
import sys
import os
import math

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(_root, "bindings", "python"))
sys.path.insert(0, os.path.join(_root, "build", "bindings", "python"))

import pytest

try:
    import insight as ins
except ImportError:
    pytest.skip("insight module not built yet", allow_module_level=True)

np = pytest.importorskip("numpy")
np.random.seed(42)


# ============================================================================
# Helpers
# ============================================================================

def to_insight(np_arr, dtype=None):
    """Convert a NumPy array to Insight array."""
    if dtype is not None:
        np_arr = np_arr.astype(dtype)
    return ins.from_numpy(np_arr)


def to_numpy(ins_arr):
    """Convert an Insight array to NumPy array."""
    return ins_arr.numpy()


def assert_close(actual, expected, rtol=1e-5, atol=1e-6, label=""):
    """Assert two NumPy arrays are close."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if np.iscomplexobj(actual) or np.iscomplexobj(expected):
        # For complex arrays, compare real and imaginary parts separately
        np.testing.assert_allclose(actual.real, expected.real, rtol=rtol,
                                   atol=atol,
                                   err_msg=f"Real part mismatch in {label}")
        np.testing.assert_allclose(actual.imag, expected.imag, rtol=rtol,
                                   atol=atol,
                                   err_msg=f"Imag part mismatch in {label}")
    else:
        actual = np.asarray(actual, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol,
                                   err_msg=f"Mismatch in {label}")


def assert_equal(actual, expected, label=""):
    """Assert two NumPy arrays are exactly equal."""
    np.testing.assert_array_equal(actual, expected,
                                  err_msg=f"Mismatch in {label}")


# ============================================================================
# Creation
# ============================================================================

class TestCreation:
    """Verify creation functions match NumPy."""

    def test_zeros(self):
        a = to_numpy(ins.zeros(ins.Shape([3, 4]), ins.float32))
        assert_equal(a, np.zeros([3, 4], dtype=np.float32), "zeros")

    def test_ones(self):
        a = to_numpy(ins.ones(ins.Shape([2, 5]), ins.float64))
        assert_equal(a, np.ones([2, 5], dtype=np.float64), "ones")

    def test_full(self):
        a = to_numpy(ins.full(ins.Shape([3, 3]), 3.14, ins.float32))
        assert_close(a, np.full([3, 3], 3.14, dtype=np.float32), label="full")

    def test_eye(self):
        a = to_numpy(ins.eye(4))
        assert_equal(a, np.eye(4, dtype=np.float32), "eye")

    def test_arange(self):
        a = to_numpy(ins.arange(0.0, 10.0, 1.0, ins.float64))
        assert_equal(a, np.arange(0, 10, 1, dtype=np.float64), "arange")

    def test_linspace(self):
        a = to_numpy(ins.linspace(0.0, 1.0, 11, ins.float64))
        b = np.linspace(0.0, 1.0, 11, dtype=np.float64)
        assert_close(a, b, rtol=1e-6, label="linspace")


# ============================================================================
# Elementwise arithmetic
# ============================================================================

class TestElementwiseArithmetic:
    """Verify elementwise operations match NumPy."""

    @pytest.fixture
    def ab(self):
        np_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        np_b = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        return to_insight(np_a), to_insight(np_b), np_a, np_b

    def test_add(self, ab):
        ia, ib, na, nb = ab
        assert_close(to_numpy(ia + ib), na + nb, label="add")

    def test_sub(self, ab):
        ia, ib, na, nb = ab
        assert_close(to_numpy(ia - ib), na - nb, label="sub")

    def test_mul(self, ab):
        ia, ib, na, nb = ab
        assert_close(to_numpy(ia * ib), na * nb, label="mul")

    def test_div(self, ab):
        ia, ib, na, nb = ab
        assert_close(to_numpy(ia / ib), na / nb, label="div")

    def test_neg(self, ab):
        ia, _, na, _ = ab
        assert_close(to_numpy(-ia), -na, label="neg")

    def test_scalar_add(self, ab):
        ia, _, na, _ = ab
        assert_close(to_numpy(ia + 10.0), na + 10.0, label="scalar_add")

    def test_scalar_mul(self, ab):
        ia, _, na, _ = ab
        assert_close(to_numpy(ia * 2.5), na * 2.5, label="scalar_mul")


# ============================================================================
# Unary math
# ============================================================================

class TestUnaryMath:
    """Verify unary math functions match NumPy."""

    @pytest.fixture
    def x(self):
        np_x = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)
        return to_insight(np_x), np_x

    def test_exp(self, x):
        ix, nx = x
        assert_close(to_numpy(ins.exp(ix)), np.exp(nx), label="exp")

    def test_log(self, x):
        ix, nx = x
        assert_close(to_numpy(ins.log(ix)), np.log(nx), label="log")

    def test_log2(self, x):
        ix, nx = x
        assert_close(to_numpy(ins.log2(ix)), np.log2(nx), label="log2")

    def test_log10(self, x):
        ix, nx = x
        assert_close(to_numpy(ins.log10(ix)), np.log10(nx), label="log10")

    def test_sqrt(self, x):
        ix, nx = x
        assert_close(to_numpy(ins.sqrt(ix)), np.sqrt(nx), label="sqrt")

    def test_abs(self, x):
        nx_neg = np.array([-1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float64)
        ix_neg = to_insight(nx_neg)
        assert_close(to_numpy(ins.abs(ix_neg)), np.abs(nx_neg), label="abs")

    def test_sin(self, x):
        ix, nx = x
        assert_close(to_numpy(ins.sin(ix)), np.sin(nx), label="sin")

    def test_cos(self, x):
        ix, nx = x
        assert_close(to_numpy(ins.cos(ix)), np.cos(nx), label="cos")

    def test_tan(self, x):
        ix, nx = x
        assert_close(to_numpy(ins.tan(ix)), np.tan(nx), label="tan")

    def test_floor(self, x):
        ix, nx = x
        assert_close(to_numpy(ins.floor(ix)), np.floor(nx), label="floor")

    def test_ceil(self, x):
        ix, nx = x
        assert_close(to_numpy(ins.ceil(ix)), np.ceil(nx), label="ceil")

    def test_square(self, x):
        ix, nx = x
        assert_close(to_numpy(ins.square(ix)), np.square(nx), label="square")


# ============================================================================
# Reduction
# ============================================================================

class TestReduction:
    """Verify reduction operations match NumPy."""

    @pytest.fixture
    def x2d(self):
        np_x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        return to_insight(np_x), np_x

    def test_sum_all(self, x2d):
        ix, nx = x2d
        assert_close(to_numpy(ins.sum(ix)), np.sum(nx), label="sum_all")

    def test_sum_axis0(self, x2d):
        ix, nx = x2d
        assert_close(to_numpy(ins.sum(ix, axis=0)),
                     np.sum(nx, axis=0), label="sum_axis0")

    def test_sum_axis1(self, x2d):
        ix, nx = x2d
        assert_close(to_numpy(ins.sum(ix, axis=1)),
                     np.sum(nx, axis=1), label="sum_axis1")

    def test_mean_all(self, x2d):
        ix, nx = x2d
        assert_close(to_numpy(ins.mean(ix)), np.mean(nx), label="mean_all")

    def test_mean_axis0(self, x2d):
        ix, nx = x2d
        result = to_numpy(ins.mean(ix, axis=0)).squeeze()
        assert_close(result, np.mean(nx, axis=0), label="mean_axis0")

    def test_max_all(self, x2d):
        ix, nx = x2d
        assert_close(to_numpy(ins.max(ix)), np.max(nx), label="max_all")

    def test_min_all(self, x2d):
        ix, nx = x2d
        assert_close(to_numpy(ins.min(ix)), np.min(nx), label="min_all")

    def test_argmax(self, x2d):
        ix, nx = x2d
        assert_equal(to_numpy(ins.argmax(ix)), np.argmax(nx), "argmax")

    def test_argmin(self, x2d):
        ix, nx = x2d
        assert_equal(to_numpy(ins.argmin(ix)), np.argmin(nx), "argmin")


# ============================================================================
# Linear Algebra
# ============================================================================

class TestLinalg:
    """Verify linear algebra operations match NumPy."""

    def test_matmul(self):
        np_a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        np_b = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)
        ia, ib = to_insight(np_a), to_insight(np_b)
        assert_close(to_numpy(ins.matmul(ia, ib)),
                     np.matmul(np_a, np_b), label="matmul")

    def test_dot_1d(self):
        np_a = np.array([1, 2, 3], dtype=np.float64)
        np_b = np.array([4, 5, 6], dtype=np.float64)
        ia, ib = to_insight(np_a), to_insight(np_b)
        assert_close(to_numpy(ins.dot(ia, ib)),
                     np.dot(np_a, np_b), label="dot_1d")

    def test_det_2x2(self):
        np_a = np.array([[1, 2], [3, 4]], dtype=np.float64)
        ia = to_insight(np_a)
        assert_close(to_numpy(ins.det(ia)),
                     np.linalg.det(np_a), rtol=1e-5, label="det_2x2")

    def test_det_3x3(self):
        np_a = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]], dtype=np.float64)
        ia = to_insight(np_a)
        assert_close(to_numpy(ins.det(ia)),
                     np.linalg.det(np_a), rtol=1e-5, label="det_3x3")

    def test_inv(self):
        np_a = np.array([[1, 2], [3, 4]], dtype=np.float64)
        ia = to_insight(np_a)
        assert_close(to_numpy(ins.inv(ia)),
                     np.linalg.inv(np_a), rtol=1e-5, label="inv")

    def test_trace(self):
        np_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        dtype=np.float32)
        ia = to_insight(np_a)
        assert_close(to_numpy(ins.trace(ia)),
                     np.trace(np_a), label="trace")

    def test_norm_1d(self):
        np_a = np.array([3.0, 4.0], dtype=np.float64)
        ia = to_insight(np_a)
        assert_close(to_numpy(ins.norm(ia)),
                     np.linalg.norm(np_a), rtol=1e-5, label="norm_1d")

    def test_solve(self):
        np_a = np.array([[2, 1], [1, 3]], dtype=np.float64)
        np_b = np.array([5, 7], dtype=np.float64)
        ia, ib = to_insight(np_a), to_insight(np_b)
        assert_close(to_numpy(ins.solve(ia, ib)),
                     np.linalg.solve(np_a, np_b), rtol=1e-5, label="solve")

    def test_svd(self):
        np_a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        ia = to_insight(np_a)
        u, s, vt = ins.svd(ia)
        np_u, np_s, np_vt = np.linalg.svd(np_a)
        assert_close(to_numpy(s), np_s, rtol=1e-5, label="svd_s")


# ============================================================================
# FFT
# ============================================================================

class TestFFT:
    """Verify FFT operations match NumPy."""

    def test_fft(self):
        np_x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        ix = to_insight(np_x)
        result = to_numpy(ins.fft(ix))
        expected = np.fft.fft(np_x)
        assert_close(result, expected, rtol=1e-5, label="fft")

    def test_ifft_roundtrip(self):
        np_x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        ix = to_insight(np_x)
        result = to_numpy(ins.ifft(ins.fft(ix)))
        assert_close(result, np_x, rtol=1e-5, label="ifft_roundtrip")

    def test_rfft(self):
        np_x = np.array([1, 2, 3, 4], dtype=np.float64)
        ix = to_insight(np_x)
        result = to_numpy(ins.rfft(ix))
        expected = np.fft.rfft(np_x)
        assert_close(result, expected, rtol=1e-5, label="rfft")

    def test_fftfreq(self):
        result = to_numpy(ins.fftfreq(8, 1.0))
        expected = np.fft.fftfreq(8, 1.0)
        assert_close(result, expected, label="fftfreq")


# ============================================================================
# Signal
# ============================================================================

class TestSignal:
    """Verify signal operations."""

    def test_convolve_full(self):
        np_a = np.array([1, 2, 3], dtype=np.float64)
        np_b = np.array([1, 1, 1], dtype=np.float64)
        ia, ib = to_insight(np_a), to_insight(np_b)
        result = to_numpy(ins.convolve(ia, ib, "full"))
        expected = np.convolve(np_a, np_b, "full")
        assert_close(result, expected, label="convolve_full")

    def test_convolve_valid(self):
        np_a = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        np_b = np.array([1, 1, 1], dtype=np.float64)
        ia, ib = to_insight(np_a), to_insight(np_b)
        result = to_numpy(ins.convolve(ia, ib, "valid"))
        expected = np.convolve(np_a, np_b, "valid")
        assert_close(result, expected, label="convolve_valid")


# ============================================================================
# Comparison
# ============================================================================

class TestComparison:
    """Verify comparison operations."""

    @pytest.fixture
    def ab(self):
        na = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        nb = np.array([5, 4, 3, 2, 1], dtype=np.float32)
        return to_insight(na), to_insight(nb), na, nb

    def test_equal(self, ab):
        ia, ib, na, nb = ab
        assert_equal(to_numpy(ins.equal(ia, ib)),
                     np.equal(na, nb).astype(np.float32), "equal")

    def test_greater(self, ab):
        ia, ib, na, nb = ab
        assert_equal(to_numpy(ins.greater(ia, ib)),
                     np.greater(na, nb).astype(np.float32), "greater")

    def test_less(self, ab):
        ia, ib, na, nb = ab
        assert_equal(to_numpy(ins.less(ia, ib)),
                     np.less(na, nb).astype(np.float32), "less")


# ============================================================================
# Casting
# ============================================================================

class TestCast:
    """Verify dtype casting."""

    def test_float32_to_float64(self):
        na = np.array([1, 2, 3], dtype=np.float32)
        ia = to_insight(na)
        result = to_numpy(ins.cast(ia, ins.float64))
        expected = na.astype(np.float64)
        assert_equal(result, expected, "f32->f64")

    def test_float64_to_int32(self):
        na = np.array([1.7, 2.3, 3.9], dtype=np.float64)
        ia = to_insight(na)
        result = to_numpy(ins.cast(ia, ins.int32))
        expected = na.astype(np.int32)
        assert_equal(result, expected, "f64->i32")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
