"""Complex alignment tests — Insight CUDA vs NumPy.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \\
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \\
    python -m pytest tests/python_align/cuda/test_complex.py -v
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
    from numpy.testing import assert_allclose
except ImportError:
    pytest.skip("Insight or NumPy not available", allow_module_level=True)

# Skip if CUDA not available
try:
    ins.load_backend("cuda")
except Exception:
    pytest.skip("CUDA backend not available", allow_module_level=True)

GPU = ins.GPUPlace(0)
CPU = ins.CPUPlace()


def to_gpu(np_arr):
    return ins.from_numpy(np_arr).to(GPU)


def to_numpy(gpu_arr):
    return gpu_arr.to(CPU).numpy()


class TestComplexAlignmentGPU:
    """Insight complex ops on GPU vs NumPy."""

    def test_to_complex(self):
        real_np = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        imag_np = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        real, imag = to_gpu(real_np), to_gpu(imag_np)
        result = ins.to_complex(real, imag)
        expected = real_np + 1j * imag_np
        assert_allclose(to_numpy(result), expected)

    def test_as_complex(self):
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.as_complex(x)
        expected = x_np.astype(np.complex128)
        assert_allclose(to_numpy(result), expected)

    def test_real(self):
        x_np = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
        x = to_gpu(x_np)
        result = ins.real(x)
        assert_allclose(to_numpy(result), np.real(x_np))

    def test_imag(self):
        x_np = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
        x = to_gpu(x_np)
        result = ins.imag(x)
        assert_allclose(to_numpy(result), np.imag(x_np))

    def test_as_real(self):
        x_np = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
        x = to_gpu(x_np)
        result = ins.as_real(x)
        expected = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
        assert_allclose(to_numpy(result), expected)

    def test_is_complex(self):
        x_real = to_gpu(np.array([1.0], dtype=np.float64))
        x_cplx = to_gpu(np.array([1 + 2j], dtype=np.complex128))
        assert ins.is_complex(x_cplx) == True  # noqa: E712
        assert ins.is_complex(x_real) == False  # noqa: E712

    def test_conj(self):
        """Test conjugate if available."""
        try:
            from insight._insight import conj
        except ImportError:
            pytest.skip("conj not available in Python bindings")
        x_np = np.array([1 + 2j, 3 - 4j], dtype=np.complex128)
        x = to_gpu(x_np)
        result = conj(x)
        assert_allclose(to_numpy(result), np.conj(x_np))

    def test_angle(self):
        """Test angle (phase) if available."""
        try:
            from insight._insight import angle
        except ImportError:
            pytest.skip("angle not available in Python bindings")
        x_np = np.array([1 + 0j, 0 + 1j, -1 + 0j, 1 + 1j], dtype=np.complex128)
        x = to_gpu(x_np)
        result = angle(x)
        assert_allclose(to_numpy(result), np.angle(x_np), rtol=1e-6)
