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
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
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


class TestComplex:
    """Complex number operations."""

    def test_is_complex(self):
        a = ins.ones(ins.Shape([3]), ins.float32)
        assert not ins.is_complex(a)

    def test_to_complex(self):
        a = ins.ones(ins.Shape([3]), ins.float32)
        c = ins.to_complex(a)
        assert c.defined()
        assert ins.is_complex(c)

    def test_to_complex_two_args(self):
        r = ins.ones(ins.Shape([3]), ins.float32)
        i = ins.zeros(ins.Shape([3]), ins.float32)
        c = ins.to_complex(r, i)
        assert c.defined()

    def test_real_imag(self):
        a = ins.ones(ins.Shape([3]), ins.float32)
        c = ins.to_complex(a)
        assert ins.real(c).defined()
        assert ins.imag(c).defined()


class TestPhaseDUnary:
    """Additional unary operations (Phase D)."""

    def test_exp2(self):
        a = ins.ones(ins.Shape([4]), ins.float32)
        assert ins.exp2(a).defined()

    def test_expm1(self):
        a = ins.ones(ins.Shape([4]), ins.float32)
        assert ins.expm1(a).defined()

    def test_log1p(self):
        a = ins.ones(ins.Shape([4]), ins.float32)
        assert ins.log1p(a).defined()

    def test_cbrt(self):
        a = ins.ones(ins.Shape([4]), ins.float32)
        assert ins.cbrt(a).defined()

    def test_reciprocal(self):
        a = ins.ones(ins.Shape([4]), ins.float32)
        assert ins.reciprocal(a).defined()

    def test_trunc(self):
        a = ins.ones(ins.Shape([4]), ins.float32)
        assert ins.trunc(a).defined()

    def test_deg2rad(self):
        a = ins.ones(ins.Shape([4]), ins.float32)
        assert ins.deg2rad(a).defined()

    def test_rad2deg(self):
        a = ins.ones(ins.Shape([4]), ins.float32)
        assert ins.rad2deg(a).defined()


class TestPhaseDReduction:
    """Additional reduction operations (Phase D)."""

    def test_cummax(self):
        a = ins.ones(ins.Shape([3, 4]), ins.float32)
        assert ins.cummax(a, 0).defined()

    def test_cummin(self):
        a = ins.ones(ins.Shape([3, 4]), ins.float32)
        assert ins.cummin(a, 0).defined()

    def test_count_nonzero(self):
        a = ins.ones(ins.Shape([3, 4]), ins.float32)
        assert ins.count_nonzero(a).defined()

    def test_median(self):
        a = ins.ones(ins.Shape([3, 4]), ins.float32)
        assert ins.median(a).defined()

    def test_quantile(self):
        a = ins.ones(ins.Shape([3, 4]), ins.float32)
        assert ins.quantile(a, 0.5).defined()

    def test_nansum(self):
        a = ins.ones(ins.Shape([3, 4]), ins.float32)
        assert ins.nansum(a).defined()

    def test_nanmean(self):
        a = ins.ones(ins.Shape([3, 4]), ins.float32)
        assert ins.nanmean(a).defined()


class TestPhaseDManipulation:
    """Additional manipulation operations (Phase D)."""

    def test_permute(self):
        a = ins.ones(ins.Shape([2, 3, 4]), ins.float32)
        b = ins.permute(a, [2, 0, 1])
        assert b.defined()
        assert b.numel() == 24

    def test_swapaxes(self):
        a = ins.ones(ins.Shape([2, 3]), ins.float32)
        b = ins.swapaxes(a, 0, 1)
        assert b.defined()

    def test_fliplr(self):
        a = ins.ones(ins.Shape([2, 3]), ins.float32)
        assert ins.fliplr(a).defined()

    def test_flipud(self):
        a = ins.ones(ins.Shape([2, 3]), ins.float32)
        assert ins.flipud(a).defined()

    def test_tril(self):
        a = ins.ones(ins.Shape([3, 3]), ins.float32)
        assert ins.tril(a).defined()

    def test_triu(self):
        a = ins.ones(ins.Shape([3, 3]), ins.float32)
        assert ins.triu(a).defined()

    def test_diff(self):
        a = ins.ones(ins.Shape([5]), ins.float32)
        assert ins.diff(a).defined()


class TestPhaseDIndexing:
    """Additional indexing operations (Phase D)."""

    def test_unique(self):
        np = pytest.importorskip("numpy")
        a = ins.from_numpy(np.array([1, 2, 2, 3, 3, 3], dtype=np.float32))
        r = ins.unique(a)
        assert r.unique.defined()
        assert r.unique.numel() == 3

    def test_topk(self):
        np = pytest.importorskip("numpy")
        a = ins.from_numpy(np.array([1, 5, 3, 4, 2], dtype=np.float32))
        vals, idx = ins.topk(a, 3)
        assert vals.defined()
        assert vals.numel() == 3


class TestPhaseDRandom:
    """Additional random operations (Phase D)."""

    def test_seed(self):
        ins.seed(42)

    def test_get_seed(self):
        s = ins.get_seed()
        assert isinstance(s, int)

    def test_rand_like(self):
        a = ins.zeros(ins.Shape([3, 4]), ins.float32)
        r = ins.rand_like(a)
        assert r.defined()
        assert r.numel() == 12

    def test_exponential(self):
        r = ins.exponential(1.0, ins.Shape([3, 4]))
        assert r.defined()
        assert r.numel() == 12


class TestDeviceInfo:
    """Device information queries."""

    def test_device_name_cpu(self):
        name = ins.device_name("cpu")
        assert name == "CPU"

    def test_cuda_version(self):
        v = ins.cuda_version()
        assert isinstance(v, int)

    def test_driver_version(self):
        v = ins.driver_version()
        assert isinstance(v, int)

    def test_compute_capability(self):
        v = ins.compute_capability()
        assert isinstance(v, int)

    def test_device_memory(self):
        total, free = ins.device_memory()
        assert isinstance(total, int)
        assert isinstance(free, int)

    def test_device_count(self):
        c = ins.device_count()
        assert isinstance(c, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
