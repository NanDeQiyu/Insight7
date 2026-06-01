"""Complex number CPU binding tests."""
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


class TestComplexCPU:
    def test_is_complex_real(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float64))
        assert not ins.is_complex(a)

    def test_is_complex_complex(self):
        a = ins.from_numpy(np.array([1 + 1j, 2 + 2j], dtype=np.complex128))
        assert ins.is_complex(a)

    def test_to_complex(self):
        real = ins.from_numpy(np.array([1, 2, 3], dtype=np.float64))
        result = ins.to_complex(real)
        assert ins.is_complex(result)
        assert result.numel() == 3

    def test_to_complex_with_imag(self):
        real = ins.from_numpy(np.array([1, 2, 3], dtype=np.float64))
        imag = ins.from_numpy(np.array([4, 5, 6], dtype=np.float64))
        result = ins.to_complex(real, imag)
        assert ins.is_complex(result)
        assert result.numel() == 3

    def test_real(self):
        a = ins.from_numpy(np.array([1 + 2j, 3 + 4j], dtype=np.complex128))
        r = ins.real(a)
        np.testing.assert_allclose(r.numpy(), np.array([1.0, 3.0]), rtol=1e-10)

    def test_imag(self):
        a = ins.from_numpy(np.array([1 + 2j, 3 + 4j], dtype=np.complex128))
        i = ins.imag(a)
        np.testing.assert_allclose(i.numpy(), np.array([2.0, 4.0]), rtol=1e-10)

    def test_as_complex(self):
        a = ins.from_numpy(np.array([1, 2, 3, 4], dtype=np.float64).reshape(2, 2))
        result = ins.as_complex(a)
        assert ins.is_complex(result)

    def test_as_real(self):
        a = ins.from_numpy(np.array([1 + 2j, 3 + 4j], dtype=np.complex128))
        result = ins.as_real(a)
        assert not ins.is_complex(result)
        assert result.numel() == 4

    def test_has_complex_shape(self):
        a = ins.from_numpy(np.array([1, 2, 3, 4], dtype=np.float64).reshape(2, 2))
        assert ins.has_complex_shape(a)

    def test_complex_abs(self):
        a_np = np.array([3 + 4j, 1 + 0j], dtype=np.complex128)
        a = ins.from_numpy(a_np)
        result = ins.abs(a)
        expected = np.abs(a_np)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_complex_conj(self):
        a_np = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
        a = ins.from_numpy(a_np)
        result = ins.conj(a)
        expected = np.conj(a_np)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_complex_angle(self):
        a_np = np.array([1 + 0j, 0 + 1j, -1 + 0j], dtype=np.complex128)
        a = ins.from_numpy(a_np)
        result = ins.angle(a)
        expected = np.angle(a_np)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_complex_add(self):
        a = ins.from_numpy(np.array([1 + 2j, 3 + 4j], dtype=np.complex128))
        b = ins.from_numpy(np.array([5 + 6j, 7 + 8j], dtype=np.complex128))
        result = ins.add(a, b)
        expected = np.array([6 + 8j, 10 + 12j])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_complex_mul(self):
        a = ins.from_numpy(np.array([1 + 2j, 3 + 4j], dtype=np.complex128))
        b = ins.from_numpy(np.array([5 + 6j, 7 + 8j], dtype=np.complex128))
        result = ins.mul(a, b)
        expected = np.array([1 + 2j, 3 + 4j]) * np.array([5 + 6j, 7 + 8j])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_complex_exp(self):
        a_np = np.array([0 + 0j, 0 + 1j], dtype=np.complex128)
        a = ins.from_numpy(a_np)
        result = ins.exp(a)
        expected = np.exp(a_np)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-8)

    def test_complex64(self):
        a = ins.from_numpy(np.array([1 + 2j, 3 + 4j], dtype=np.complex64))
        assert ins.is_complex(a)
        r = ins.real(a)
        assert r.numel() == 2

    def test_to_complex_creates_complex128(self):
        real = ins.from_numpy(np.array([1, 2], dtype=np.float64))
        result = ins.to_complex(real)
        assert ins.is_complex(result)

    def test_as_complex_as_real_roundtrip(self):
        a = ins.from_numpy(np.array([1, 2, 3, 4], dtype=np.float64).reshape(2, 2))
        c = ins.as_complex(a)
        assert ins.is_complex(c)
        r = ins.as_real(c)
        assert not ins.is_complex(r)
        assert r.numel() == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
