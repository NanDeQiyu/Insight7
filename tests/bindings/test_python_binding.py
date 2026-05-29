"""Smoke tests for the Insight7 Python binding.

Run with:
    cd build && python -m pytest tests/bindings/test_python_binding.py -v
    # or with wrapper:
    PYTHONPATH=bindings/python:build/bindings/python python -m pytest tests/bindings/test_python_binding.py -v
"""
import sys
import os
import math

# Add wrapper package and native module directories to the path
_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_root, "bindings", "python"))
sys.path.insert(0, os.path.join(_root, "build", "bindings", "python"))

import pytest

try:
    import insight as ins
except ImportError:
    pytest.skip("insight module not built yet", allow_module_level=True)


class TestDType:
    """Module-level dtype access (Paddle-style)."""

    def test_dtype_values(self):
        assert ins.float32 is not None
        assert ins.float64 is not None
        assert ins.int32 is not None
        assert ins.int64 is not None
        assert ins.bool is not None

    def test_dtype_enum(self):
        assert ins.DType.float32 == ins.float32
        assert ins.DType.int64 == ins.int64


class TestPlace:
    """Place constructors."""

    def test_cpu_place(self):
        p = ins.CPUPlace()
        assert p.is_cpu()
        assert not p.is_gpu()

    def test_place_repr(self):
        p = ins.CPUPlace()
        assert "CPU" in repr(p)


class TestArrayCreation:
    """Array creation functions."""

    def test_zeros(self):
        a = ins.zeros(ins.Shape([2, 3]), ins.float32)
        assert a.defined()
        assert a.numel() == 6
        assert a.ndim() == 2

    def test_ones(self):
        a = ins.ones(ins.Shape([4]), ins.float64)
        assert a.defined()
        assert a.numel() == 4

    def test_full(self):
        a = ins.full(ins.Shape([2, 2]), 3.14, ins.float32)
        assert a.defined()
        assert a.numel() == 4

    def test_eye(self):
        a = ins.eye(3)
        assert a.defined()
        assert a.numel() == 9

    def test_arange(self):
        a = ins.arange(10)
        assert a.defined()
        assert a.numel() == 10

    def test_linspace(self):
        a = ins.linspace(0.0, 1.0, 5)
        assert a.defined()
        assert a.numel() == 5

    def test_zeros_like(self):
        a = ins.ones(ins.Shape([3, 3]), ins.float32)
        b = ins.zeros_like(a)
        assert b.numel() == 9


class TestArrayProperties:
    """Array metadata and properties."""

    def test_shape(self):
        a = ins.zeros(ins.Shape([2, 3, 4]))
        s = a.shape()
        assert s.ndim() == 3
        assert s.numel() == 24

    def test_dtype(self):
        a = ins.zeros(ins.Shape([2]), ins.float64)
        assert a.dtype() == ins.float64

    def test_contiguous(self):
        a = ins.ones(ins.Shape([3, 4]))
        assert a.is_contiguous()

    def test_repr(self):
        a = ins.zeros(ins.Shape([2, 3]), ins.float32)
        r = repr(a)
        assert "insight.Array" in r
        assert "float32" in r
        assert "2" in r
        assert "3" in r


class TestArrayOperators:
    """Arithmetic and comparison operators."""

    def test_add(self):
        a = ins.ones(ins.Shape([3]), ins.float32)
        b = ins.ones(ins.Shape([3]), ins.float32)
        c = a + b
        assert c.defined()
        assert c.numel() == 3

    def test_sub(self):
        a = ins.ones(ins.Shape([3]), ins.float32)
        b = ins.ones(ins.Shape([3]), ins.float32)
        c = a - b
        assert c.defined()

    def test_mul(self):
        a = ins.full(ins.Shape([3]), 2.0, ins.float32)
        b = ins.full(ins.Shape([3]), 3.0, ins.float32)
        c = a * b
        assert c.defined()

    def test_scalar_ops(self):
        a = ins.ones(ins.Shape([3]), ins.float32)
        b = a + 1.0
        assert b.defined()
        c = 2.0 * a
        assert c.defined()

    def test_neg(self):
        a = ins.ones(ins.Shape([3]), ins.float32)
        b = -a
        assert b.defined()


class TestElementwise:
    """Elementwise operations."""

    def test_add_function(self):
        a = ins.ones(ins.Shape([3]), ins.float32)
        b = ins.ones(ins.Shape([3]), ins.float32)
        c = ins.add(a, b)
        assert c.defined()

    def test_mul_function(self):
        a = ins.full(ins.Shape([3]), 2.0, ins.float32)
        b = ins.full(ins.Shape([3]), 3.0, ins.float32)
        c = ins.mul(a, b)
        assert c.defined()

    def test_exp(self):
        a = ins.zeros(ins.Shape([3]), ins.float32)
        b = ins.exp(a)
        assert b.defined()

    def test_log(self):
        a = ins.ones(ins.Shape([3]), ins.float32)
        b = ins.log(a)
        assert b.defined()

    def test_sin(self):
        a = ins.zeros(ins.Shape([3]), ins.float32)
        b = ins.sin(a)
        assert b.defined()


class TestReduction:
    """Reduction operations."""

    def test_sum(self):
        a = ins.ones(ins.Shape([3, 4]), ins.float32)
        s = ins.sum(a)
        assert s.defined()
        assert s.numel() == 1

    def test_sum_axis(self):
        a = ins.ones(ins.Shape([3, 4]), ins.float32)
        s = ins.sum(a, axis=0)
        assert s.defined()

    def test_mean(self):
        a = ins.ones(ins.Shape([3, 4]), ins.float32)
        m = ins.mean(a)
        assert m.defined()

    def test_max_min(self):
        a = ins.ones(ins.Shape([3, 4]), ins.float32)
        assert ins.max(a).defined()
        assert ins.min(a).defined()


class TestLinalg:
    """Linear algebra operations."""

    def test_matmul(self):
        a = ins.ones(ins.Shape([2, 3]), ins.float32)
        b = ins.ones(ins.Shape([3, 4]), ins.float32)
        c = ins.matmul(a, b)
        assert c.defined()
        assert c.numel() == 8  # 2*4

    def test_det(self):
        a = ins.eye(3)
        d = ins.det(a)
        assert d.defined()

    def test_inv(self):
        a = ins.eye(3)
        b = ins.inv(a)
        assert b.defined()


class TestNumPyInterop:
    """NumPy interop (if numpy is available)."""

    def test_to_numpy(self):
        np = pytest.importorskip("numpy")
        a = ins.ones(ins.Shape([2, 3]), ins.float32)
        arr = a.numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 3)
        assert arr.dtype == np.float32

    def test_from_numpy(self):
        np = pytest.importorskip("numpy")
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        a = ins.from_numpy(arr)
        assert a.defined()
        assert a.numel() == 6

    def test_roundtrip(self):
        np = pytest.importorskip("numpy")
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        a = ins.from_numpy(arr)
        result = a.numpy()
        np.testing.assert_array_equal(result, arr)


class TestFFT:
    """FFT operations (ins::fft namespace)."""

    def test_fft(self):
        a = ins.ones(ins.Shape([8]), ins.float64)
        b = ins.fft(a)
        assert b.defined()

    def test_ifft(self):
        a = ins.ones(ins.Shape([8]), ins.float64)
        b = ins.fft(a)
        c = ins.ifft(b)
        assert c.defined()


class TestSlicing:
    """String-based slicing."""

    def test_string_slice(self):
        a = ins.ones(ins.Shape([4, 4]), ins.float32)
        b = a[":,1:-1"]
        assert b.defined()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
