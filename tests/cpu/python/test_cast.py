"""Type casting CPU binding tests."""

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


class TestCastCPU:
    def test_cast_float64_to_float32(self):
        a = ins.from_numpy(np.array([1.5, 2.5, 3.5], dtype=np.float64))
        b = ins.cast(a, ins.float32)
        np.testing.assert_allclose(
            b.numpy(), np.array([1.5, 2.5, 3.5], dtype=np.float32), rtol=1e-5
        )

    def test_cast_float32_to_float64(self):
        a = ins.from_numpy(np.array([1.5, 2.5, 3.5], dtype=np.float32))
        b = ins.cast(a, ins.float64)
        np.testing.assert_allclose(
            b.numpy(), np.array([1.5, 2.5, 3.5], dtype=np.float64), rtol=1e-10
        )

    def test_cast_float64_to_int32(self):
        a = ins.from_numpy(np.array([1.9, 2.5, 3.1], dtype=np.float64))
        b = ins.cast(a, ins.int32)
        np.testing.assert_array_equal(b.numpy(), np.array([1, 2, 3], dtype=np.int32))

    def test_cast_float64_to_int64(self):
        a = ins.from_numpy(np.array([1.9, 2.5, 3.1], dtype=np.float64))
        b = ins.cast(a, ins.int64)
        np.testing.assert_array_equal(b.numpy(), np.array([1, 2, 3], dtype=np.int64))

    def test_cast_float64_to_uint8(self):
        a = ins.from_numpy(np.array([0, 128, 255], dtype=np.float64))
        b = ins.cast(a, ins.uint8)
        np.testing.assert_array_equal(b.numpy(), np.array([0, 128, 255], dtype=np.uint8))

    def test_cast_float64_to_bool(self):
        a = ins.from_numpy(np.array([0.0, 1.0, 0.5], dtype=np.float64))
        b = ins.cast(a, ins.bool)
        expected = np.array([False, True, True])
        np.testing.assert_array_equal(b.numpy(), expected)

    def test_cast_bool_to_float64(self):
        a = ins.from_numpy(np.array([True, False, True], dtype=np.bool_))
        b = ins.cast(a, ins.float64)
        np.testing.assert_array_equal(b.numpy(), np.array([1.0, 0.0, 1.0]))

    def test_cast_int32_to_float64(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.int32))
        b = ins.cast(a, ins.float64)
        np.testing.assert_allclose(b.numpy(), np.array([1.0, 2.0, 3.0]), rtol=1e-10)

    def test_cast_preserves_shape(self):
        a = ins.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float64))
        b = ins.cast(a, ins.float32)
        assert b.numel() == 4

    def test_cast_idempotent(self):
        a = ins.from_numpy(np.array([1.5, 2.5], dtype=np.float64))
        b = ins.cast(a, ins.float64)
        np.testing.assert_allclose(b.numpy(), a.numpy(), rtol=1e-10)

    def test_cast_int_to_int(self):
        a = ins.from_numpy(np.array([100, 200, 300], dtype=np.int32))
        b = ins.cast(a, ins.int64)
        np.testing.assert_array_equal(b.numpy(), np.array([100, 200, 300], dtype=np.int64))

    def test_cast_uint8_to_float32(self):
        a = ins.from_numpy(np.array([0, 128, 255], dtype=np.uint8))
        b = ins.cast(a, ins.float32)
        np.testing.assert_allclose(b.numpy(), np.array([0, 128, 255], dtype=np.float32), rtol=1e-5)

    def test_cast_2d_array(self):
        a = ins.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64))
        b = ins.cast(a, ins.int32)
        np.testing.assert_array_equal(b.numpy(), np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))

    def test_cast_roundtrip(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.int32))
        b = ins.cast(a, ins.float64)
        c = ins.cast(b, ins.int32)
        np.testing.assert_array_equal(c.numpy(), a.numpy())

    def test_cast_negative_values(self):
        a = ins.from_numpy(np.array([-1.5, -2.5, 3.5], dtype=np.float64))
        b = ins.cast(a, ins.int32)
        np.testing.assert_array_equal(b.numpy(), np.array([-1, -2, 3], dtype=np.int32))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
