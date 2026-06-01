"""Manipulation CPU binding tests — aligned with C++ test_manipulation.cpp.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python
    LD_LIBRARY_PATH=build/backends/cpu
    python -m pytest tests/cpu/python/test_manipulation.py -v
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


class TestManipulationCPU:
    """Array manipulation operations — CPU backend."""

    def test_reshape(self):
        a = ins.from_numpy(np.arange(6, dtype=np.float64))
        b = ins.reshape(a, [2, 3])
        assert b.numel() == 6
        assert b.ndim() == 2

    def test_transpose(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = ins.permute(ins.from_numpy(a_np), [1, 0])
        np.testing.assert_allclose(b.numpy(), a_np.T)

    def test_squeeze(self):
        a = ins.from_numpy(np.zeros([1, 3, 1], dtype=np.float64))
        b = ins.squeeze(a)
        assert b.numel() == 3

    def test_unsqueeze(self):
        a = ins.from_numpy(np.zeros([3], dtype=np.float64))
        b = ins.unsqueeze(a, 0)
        assert b.ndim() == 2

    def test_concat(self):
        a = ins.from_numpy(np.array([1, 2], dtype=np.float64))
        b = ins.from_numpy(np.array([3, 4], dtype=np.float64))
        c = ins.concat([a, b])
        assert c.numel() == 4
        np.testing.assert_allclose(c.numpy(), [1, 2, 3, 4])

    def test_concat_axis(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        b_np = np.array([[5, 6]], dtype=np.float64)
        c = ins.concat([ins.from_numpy(a_np), ins.from_numpy(b_np)], axis=0)
        np.testing.assert_allclose(c.numpy(), np.concatenate([a_np, b_np], axis=0))

    def test_split(self):
        a = ins.from_numpy(np.arange(6, dtype=np.float64))
        parts = ins.split(a, 3, axis=0)
        assert len(parts) == 3
        assert parts[0].numel() == 2

    def test_tile(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float64))
        b = ins.tile(a, [2])
        assert b.numel() == 6

    def test_repeat(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float64))
        b = ins.repeat(a, 2)
        assert b.numel() == 6

    def test_flip(self):
        a_np = np.array([1, 2, 3, 4], dtype=np.float64)
        result = ins.flip(ins.from_numpy(a_np))
        np.testing.assert_allclose(result.numpy(), np.flip(a_np))

    def test_roll(self):
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        result = ins.roll(ins.from_numpy(a_np), 2)
        np.testing.assert_allclose(result.numpy(), np.roll(a_np, 2))

    def test_pad(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float64))
        b = ins.pad(a, [2, 3])
        assert b.numel() == 8

    def test_diag(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float64))
        b = ins.diag(a)
        assert b.numel() == 9
        assert b.ndim() == 2

    def test_tril(self):
        a = ins.from_numpy(np.ones([3, 3], dtype=np.float64))
        b = ins.tril(a)
        np.testing.assert_allclose(b.numpy(), np.tril(np.ones([3, 3])))

    def test_triu(self):
        a = ins.from_numpy(np.ones([3, 3], dtype=np.float64))
        b = ins.triu(a)
        np.testing.assert_allclose(b.numpy(), np.triu(np.ones([3, 3])))

    def test_diff(self):
        a_np = np.array([1, 3, 6, 10], dtype=np.float64)
        result = ins.diff(ins.from_numpy(a_np))
        np.testing.assert_allclose(result.numpy(), np.diff(a_np))

    def test_swapaxes(self):
        a = ins.from_numpy(np.ones([2, 3, 4], dtype=np.float64))
        b = ins.swapaxes(a, 0, 2)
        assert b.numel() == 24

    def test_moveaxis(self):
        a = ins.from_numpy(np.ones([2, 3, 4], dtype=np.float64))
        b = ins.moveaxis(a, 0, 2)
        assert b.numel() == 24

    def test_fliplr(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        result = ins.fliplr(ins.from_numpy(a_np))
        np.testing.assert_allclose(result.numpy(), np.fliplr(a_np))

    def test_flipud(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        result = ins.flipud(ins.from_numpy(a_np))
        np.testing.assert_allclose(result.numpy(), np.flipud(a_np))

    def test_rot90(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        result = ins.rot90(ins.from_numpy(a_np))
        np.testing.assert_allclose(result.numpy(), np.rot90(a_np))

    def test_flatten(self):
        a = ins.from_numpy(np.ones([2, 3], dtype=np.float64))
        b = ins.flatten(a)
        assert b.numel() == 6
        assert b.ndim() == 1

    def test_ravel(self):
        a = ins.from_numpy(np.ones([2, 3], dtype=np.float64))
        b = ins.ravel(a)
        assert b.numel() == 6
        assert b.ndim() == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
