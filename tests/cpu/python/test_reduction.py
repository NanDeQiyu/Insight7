"""Reduction CPU binding tests — aligned with C++ test_reduction.cpp.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python
    LD_LIBRARY_PATH=build/backends/cpu
    python -m pytest tests/cpu/python/test_reduction.py -v
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


class TestReductionCPU:
    """Reduction operations — CPU backend."""

    def test_sum(self):
        a = ins.from_numpy(np.array([1, 2, 3, 4], dtype=np.float64))
        result = ins.sum(a)
        assert result.numpy().item() == pytest.approx(10.0)

    def test_sum_axis(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        result = ins.sum(ins.from_numpy(a_np), axis=0)
        np.testing.assert_allclose(result.numpy(), np.sum(a_np, axis=0))

    def test_mean(self):
        a = ins.from_numpy(np.array([1, 2, 3, 4], dtype=np.float64))
        result = ins.mean(a)
        assert result.numpy().item() == pytest.approx(2.5)

    def test_mean_axis(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        result = ins.mean(ins.from_numpy(a_np), axis=1)
        np.testing.assert_allclose(result.numpy(), np.mean(a_np, axis=1))

    def test_max(self):
        a = ins.from_numpy(np.array([3, 1, 4, 1, 5], dtype=np.float64))
        result = ins.max(a)
        assert result.numpy().item() == pytest.approx(5.0)

    def test_min(self):
        a = ins.from_numpy(np.array([3, 1, 4, 1, 5], dtype=np.float64))
        result = ins.min(a)
        assert result.numpy().item() == pytest.approx(1.0)

    def test_prod(self):
        a = ins.from_numpy(np.array([1, 2, 3, 4], dtype=np.float64))
        result = ins.prod(a)
        assert result.numpy().item() == pytest.approx(24.0)

    def test_argmax(self):
        a = ins.from_numpy(np.array([1, 5, 3, 2], dtype=np.float64))
        result = ins.argmax(a)
        assert result.numpy().item() == 1

    def test_argmin(self):
        a = ins.from_numpy(np.array([5, 1, 3, 2], dtype=np.float64))
        result = ins.argmin(a)
        assert result.numpy().item() == 1

    def test_cumsum(self):
        a_np = np.array([1, 2, 3, 4], dtype=np.float64)
        result = ins.cumsum(ins.from_numpy(a_np), axis=0)
        np.testing.assert_allclose(result.numpy(), np.cumsum(a_np))

    def test_cumprod(self):
        a_np = np.array([1, 2, 3, 4], dtype=np.float64)
        result = ins.cumprod(ins.from_numpy(a_np), axis=0)
        np.testing.assert_allclose(result.numpy(), np.cumprod(a_np))

    def test_cummax(self):
        a_np = np.array([3, 1, 4, 1, 5], dtype=np.float64)
        result = ins.cummax(ins.from_numpy(a_np), axis=0)
        np.testing.assert_allclose(result.numpy(), np.maximum.accumulate(a_np))

    def test_cummin(self):
        a_np = np.array([3, 1, 4, 1, 5], dtype=np.float64)
        result = ins.cummin(ins.from_numpy(a_np), axis=0)
        np.testing.assert_allclose(result.numpy(), np.minimum.accumulate(a_np))

    def test_any(self):
        a = ins.from_numpy(np.array([1.0, 0.0, 1.0], dtype=np.float64))
        result = ins.any(a)
        assert result.numpy().item() == 1.0

    def test_all(self):
        a = ins.from_numpy(np.array([1.0, 0.0, 1.0], dtype=np.float64))
        result = ins.all(a)
        assert result.numpy().item() == 0.0

    def test_var(self):
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        result = ins.var(ins.from_numpy(a_np))
        assert result.numpy().item() == pytest.approx(np.var(a_np), rel=1e-5)

    def test_std(self):
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        result = ins.std(ins.from_numpy(a_np))
        assert result.numpy().item() == pytest.approx(np.std(a_np), rel=1e-5)

    def test_count_nonzero(self):
        a = ins.from_numpy(np.array([0, 1, 0, 3, 0], dtype=np.float64))
        result = ins.count_nonzero(a)
        assert result.numpy().item() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
