"""Print CPU binding tests — aligned with C++ test_print.cpp.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python
    LD_LIBRARY_PATH=build/backends/cpu
    python -m pytest tests/cpu/python/test_print.py -v
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


class TestPrintCPU:
    """Print / repr / str operations — CPU backend."""

    def test_repr_1d(self):
        a = ins.zeros([5], ins.float32)
        r = repr(a)
        assert "insight" in r.lower() or "array" in r.lower() or "float32" in r

    def test_repr_2d(self):
        a = ins.ones([3, 4], ins.float64)
        r = repr(a)
        assert isinstance(r, str) and len(r) > 0

    def test_repr_int(self):
        a = ins.zeros([3], ins.int32)
        r = repr(a)
        assert isinstance(r, str) and len(r) > 0

    def test_str_1d(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float32))
        s = str(a)
        assert isinstance(s, str) and len(s) > 0

    def test_str_2d(self):
        a = ins.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float64))
        s = str(a)
        assert isinstance(s, str) and len(s) > 0

    def test_print_does_not_crash_scalar(self):
        a = ins.zeros([], ins.float32)
        # Should not raise
        _ = repr(a)

    def test_print_does_not_crash_large(self):
        a = ins.ones([100, 100], ins.float32)
        _ = repr(a)

    def test_print_empty(self):
        a = ins.zeros([0], ins.float32)
        _ = repr(a)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
