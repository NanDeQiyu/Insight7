"""CSV CPU binding tests — aligned with C++ test_csv.cpp.

Note: CSV read/write is not exposed in the Python bindings (it is a C++ internal
utility for exporting operator support matrices). This test verifies the closest
available I/O interop: from_numpy/to_numpy roundtrip.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python
    LD_LIBRARY_PATH=build/backends/cpu
    python -m pytest tests/cpu/python/test_csv.py -v
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


class TestCSVCPU:
    """CSV / I/O roundtrip — CPU backend.

    CSV write/read is not exposed in bindings; these tests verify the
    NumPy interop roundtrip which is the binding-level equivalent.
    """

    def test_roundtrip_1d(self):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        a = ins.from_numpy(data)
        result = a.numpy()
        np.testing.assert_array_equal(result, data)

    def test_roundtrip_2d(self):
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        a = ins.from_numpy(data)
        result = a.numpy()
        np.testing.assert_array_equal(result, data)

    def test_roundtrip_int(self):
        data = np.array([10, 20, 30], dtype=np.int32)
        a = ins.from_numpy(data)
        result = a.numpy()
        np.testing.assert_array_equal(result, data)

    def test_roundtrip_float64(self):
        data = np.array([1.1, 2.2, 3.3], dtype=np.float64)
        a = ins.from_numpy(data)
        result = a.numpy()
        np.testing.assert_allclose(result, data, rtol=1e-10)

    def test_roundtrip_after_ops(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        a = ins.from_numpy(data)
        b = ins.add(a, a)
        result = b.numpy()
        np.testing.assert_allclose(result, data * 2, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
