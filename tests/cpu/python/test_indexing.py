"""Indexing and sorting CPU binding tests."""

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


class TestIndexingCPU:
    def test_take_1d(self):
        a = ins.from_numpy(np.array([10, 20, 30, 40, 50], dtype=np.float64))
        idx = ins.from_numpy(np.array([0, 2, 4], dtype=np.int64))
        result = ins.take(a, idx)
        np.testing.assert_allclose(result.numpy(), np.array([10, 30, 50]))

    def test_nonzero(self):
        a = ins.from_numpy(np.array([0, 1, 0, 3, 0], dtype=np.float64))
        result = ins.nonzero(a)
        assert result.numel() > 0

    def test_flatnonzero(self):
        a = ins.from_numpy(np.array([[0, 1], [2, 0]], dtype=np.float64))
        result = ins.flatnonzero(a)
        assert result.numel() == 2

    def test_unique_1d(self):
        a = ins.from_numpy(np.array([3, 1, 2, 1, 3, 2], dtype=np.float64))
        result = ins.unique(a)
        assert hasattr(result, "unique")
        vals = sorted(result.unique.numpy().tolist())
        assert vals == [1.0, 2.0, 3.0]

    def test_unique_sorted(self):
        a = ins.from_numpy(np.array([5, 3, 1, 3, 5], dtype=np.float64))
        result = ins.unique(a)
        data = result.unique.numpy()
        assert np.all(np.diff(data) >= 0)

    def test_argsort_1d(self):
        a = ins.from_numpy(np.array([3, 1, 2], dtype=np.float64))
        result = ins.argsort(a)
        expected = np.argsort(np.array([3, 1, 2]))
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_sort_1d(self):
        a = ins.from_numpy(np.array([3, 1, 4, 1, 5], dtype=np.float64))
        result = ins.sort(a)
        expected = np.sort(np.array([3, 1, 4, 1, 5]))
        np.testing.assert_allclose(result.numpy(), expected)

    def test_topk(self):
        a = ins.from_numpy(np.array([1, 5, 3, 2, 4], dtype=np.float64))
        result = ins.topk(a, 3)
        data = sorted(result[0].numpy().tolist(), reverse=True)
        assert data == [5.0, 4.0, 3.0]

    def test_gather(self):
        a = ins.from_numpy(np.array([[10, 20], [30, 40]], dtype=np.float64))
        idx = ins.from_numpy(np.array([[0, 1], [1, 0]], dtype=np.int64))
        result = ins.gather(a, 0, idx)
        assert result.numel() == 4

    def test_scatter(self):
        a = ins.from_numpy(np.zeros([4], dtype=np.float64))
        idx = ins.from_numpy(np.array([0, 2], dtype=np.int64))
        src = ins.from_numpy(np.array([10, 20], dtype=np.float64))
        result = ins.scatter(a, 0, idx, src)
        assert result.numel() == 4

    def test_scatter_add(self):
        a = ins.from_numpy(np.zeros([4], dtype=np.float64))
        idx = ins.from_numpy(np.array([0, 0, 2], dtype=np.int64))
        src = ins.from_numpy(np.array([1, 2, 3], dtype=np.float64))
        result = ins.scatter_add(a, 0, idx, src)
        assert result.numel() == 4
        data = result.numpy()
        assert data[0] == pytest.approx(3.0)

    def test_searchsorted(self):
        a = ins.from_numpy(np.array([1, 3, 5, 7], dtype=np.float64))
        v = ins.from_numpy(np.array([0, 2, 5, 8], dtype=np.float64))
        result = ins.searchsorted(a, v)
        expected = np.searchsorted(np.array([1, 3, 5, 7]), np.array([0, 2, 5, 8]))
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_interp(self):
        xp = ins.from_numpy(np.array([0, 1, 2, 3], dtype=np.float64))
        fp = ins.from_numpy(np.array([0, 2, 4, 6], dtype=np.float64))
        x = ins.from_numpy(np.array([0.5, 1.5, 2.5], dtype=np.float64))
        result = ins.interp(x, xp, fp)
        expected = np.interp([0.5, 1.5, 2.5], [0, 1, 2, 3], [0, 2, 4, 6])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_masked_select(self):
        a = ins.from_numpy(np.array([1, 2, 3, 4, 5], dtype=np.float64))
        mask = ins.from_numpy(np.array([True, False, True, False, True], dtype=np.bool_))
        result = ins.masked_select(a, mask)
        assert result.numel() == 3

    def test_where(self):
        cond = ins.from_numpy(np.array([True, False, True], dtype=np.bool_))
        x = ins.from_numpy(np.array([1, 2, 3], dtype=np.float64))
        y = ins.from_numpy(np.array([4, 5, 6], dtype=np.float64))
        result = ins.where(cond, x, y)
        np.testing.assert_allclose(result.numpy(), np.array([1, 5, 3]))

    def test_indices(self):
        result = ins.indices(ins.Shape([2, 3]))
        assert result.numel() > 0

    def test_sort_2d(self):
        a = ins.from_numpy(np.array([[3, 1], [2, 4]], dtype=np.float64))
        result = ins.sort(a)
        assert result.numel() == 4

    def test_argsort_2d(self):
        a = ins.from_numpy(np.array([[3, 1], [2, 4]], dtype=np.float64))
        result = ins.argsort(a)
        assert result.numel() == 4

    def test_topk_smallest(self):
        a = ins.from_numpy(np.array([1, 5, 3, 2, 4], dtype=np.float64))
        result = ins.topk(a, 3, largest=False)
        data = sorted(result[0].numpy().tolist())
        assert data == [1.0, 2.0, 3.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
