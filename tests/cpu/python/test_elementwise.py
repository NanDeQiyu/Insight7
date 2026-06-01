"""Elementwise CPU binding tests — aligned with C++ test_elementwise.cpp.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python
    LD_LIBRARY_PATH=build/backends/cpu
    python -m pytest tests/cpu/python/test_elementwise.py -v
"""

import sys, os, pytest

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, os.path.join(_root, "bindings", "python"))
sys.path.insert(0, os.path.join(_root, "build", "bindings", "python"))

try:
    import insight as ins
    import numpy as np
except ImportError:
    pytest.skip("Insight or NumPy not available", allow_module_level=True)


class TestElementwiseCPU:
    """Elementwise binary operations — CPU backend."""

    def test_add(self):
        a_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        b_np = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
        result = ins.add(ins.from_numpy(a_np), ins.from_numpy(b_np))
        np.testing.assert_allclose(result.numpy(), a_np + b_np, rtol=1e-10)

    def test_sub(self):
        a_np = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        b_np = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = ins.sub(ins.from_numpy(a_np), ins.from_numpy(b_np))
        np.testing.assert_allclose(result.numpy(), a_np - b_np, rtol=1e-10)

    def test_mul(self):
        a_np = np.array([2.0, 3.0, 4.0], dtype=np.float64)
        b_np = np.array([5.0, 6.0, 7.0], dtype=np.float64)
        result = ins.mul(ins.from_numpy(a_np), ins.from_numpy(b_np))
        np.testing.assert_allclose(result.numpy(), a_np * b_np, rtol=1e-10)

    def test_div(self):
        a_np = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        b_np = np.array([2.0, 4.0, 5.0], dtype=np.float64)
        result = ins.div(ins.from_numpy(a_np), ins.from_numpy(b_np))
        np.testing.assert_allclose(result.numpy(), a_np / b_np, rtol=1e-10)

    def test_pow(self):
        a_np = np.array([2.0, 3.0, 4.0], dtype=np.float64)
        b_np = np.array([3.0, 2.0, 0.5], dtype=np.float64)
        result = ins.pow(ins.from_numpy(a_np), ins.from_numpy(b_np))
        np.testing.assert_allclose(result.numpy(), a_np ** b_np, rtol=1e-6)

    def test_mod(self):
        a_np = np.array([10.0, 7.0, 5.0], dtype=np.float64)
        b_np = np.array([3.0, 2.0, 3.0], dtype=np.float64)
        result = ins.mod(ins.from_numpy(a_np), ins.from_numpy(b_np))
        np.testing.assert_allclose(result.numpy(), a_np % b_np, rtol=1e-10)

    def test_maximum(self):
        a_np = np.array([1.0, 5.0, 3.0], dtype=np.float64)
        b_np = np.array([4.0, 2.0, 6.0], dtype=np.float64)
        result = ins.maximum(ins.from_numpy(a_np), ins.from_numpy(b_np))
        np.testing.assert_allclose(result.numpy(), np.maximum(a_np, b_np))

    def test_minimum(self):
        a_np = np.array([1.0, 5.0, 3.0], dtype=np.float64)
        b_np = np.array([4.0, 2.0, 6.0], dtype=np.float64)
        result = ins.minimum(ins.from_numpy(a_np), ins.from_numpy(b_np))
        np.testing.assert_allclose(result.numpy(), np.minimum(a_np, b_np))

    def test_equal(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float64))
        b = ins.from_numpy(np.array([1, 0, 3], dtype=np.float64))
        result = ins.equal(a, b)
        np.testing.assert_array_equal(result.numpy(), np.array([True, False, True]))

    def test_not_equal(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float64))
        b = ins.from_numpy(np.array([1, 0, 3], dtype=np.float64))
        result = ins.not_equal(a, b)
        np.testing.assert_array_equal(result.numpy(), np.array([False, True, False]))

    def test_greater(self):
        a = ins.from_numpy(np.array([3, 1, 5], dtype=np.float64))
        b = ins.from_numpy(np.array([2, 4, 5], dtype=np.float64))
        result = ins.greater(a, b)
        np.testing.assert_array_equal(result.numpy(), np.array([True, False, False]))

    def test_less(self):
        a = ins.from_numpy(np.array([1, 5, 3], dtype=np.float64))
        b = ins.from_numpy(np.array([2, 4, 5], dtype=np.float64))
        result = ins.less(a, b)
        np.testing.assert_array_equal(result.numpy(), np.array([True, False, True]))

    def test_greater_equal(self):
        a = ins.from_numpy(np.array([3, 1, 5], dtype=np.float64))
        b = ins.from_numpy(np.array([2, 4, 5], dtype=np.float64))
        result = ins.greater_equal(a, b)
        np.testing.assert_array_equal(result.numpy(), np.array([True, False, True]))

    def test_less_equal(self):
        a = ins.from_numpy(np.array([1, 5, 3], dtype=np.float64))
        b = ins.from_numpy(np.array([2, 4, 5], dtype=np.float64))
        result = ins.less_equal(a, b)
        np.testing.assert_array_equal(result.numpy(), np.array([True, False, True]))

    def test_logical_and(self):
        a = ins.from_numpy(np.array([True, True, False], dtype=np.bool_))
        b = ins.from_numpy(np.array([True, False, False], dtype=np.bool_))
        result = ins.logical_and(a, b)
        np.testing.assert_array_equal(result.numpy(), np.array([True, False, False]))

    def test_logical_or(self):
        a = ins.from_numpy(np.array([True, True, False], dtype=np.bool_))
        b = ins.from_numpy(np.array([True, False, False], dtype=np.bool_))
        result = ins.logical_or(a, b)
        np.testing.assert_array_equal(result.numpy(), np.array([True, True, False]))

    def test_logical_xor(self):
        a = ins.from_numpy(np.array([True, True, False], dtype=np.bool_))
        b = ins.from_numpy(np.array([True, False, False], dtype=np.bool_))
        result = ins.logical_xor(a, b)
        np.testing.assert_array_equal(result.numpy(), np.array([False, True, False]))

    def test_logical_not(self):
        a = ins.from_numpy(np.array([True, False, True], dtype=np.bool_))
        result = ins.logical_not(a)
        np.testing.assert_array_equal(result.numpy(), np.array([False, True, False]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
