"""Creation alignment tests — Insight vs NumPy.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python python -m pytest tests/python_align/cpu/test_creation.py -v
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
    pytest.skip("Insight, NumPy, or SciPy not available", allow_module_level=True)


class TestCreationAlignment:
    """Insight creation functions vs NumPy equivalents."""

    def test_zeros(self):
        a = ins.zeros([3, 4], ins.float32)
        assert_allclose(a.numpy(), np.zeros([3, 4], dtype=np.float32))

    def test_ones(self):
        a = ins.ones([3, 4], ins.float32)
        assert_allclose(a.numpy(), np.ones([3, 4], dtype=np.float32))

    def test_full(self):
        a = ins.full([2, 3], 3.14, ins.float64)
        assert_allclose(a.numpy(), np.full([2, 3], 3.14))

    def test_eye(self):
        a = ins.eye(4)
        assert_allclose(a.numpy(), np.eye(4, dtype=np.float32))

    def test_arange(self):
        a = ins.arange(20, ins.float64)
        assert_allclose(a.numpy(), np.arange(20, dtype=np.float64))

    def test_linspace(self):
        a = ins.linspace(0.0, 1.0, 50, ins.float64)
        assert_allclose(a.numpy(), np.linspace(0, 1, 50), atol=1e-10)

    def test_logspace(self):
        a = ins.logspace(0, 3, 4, ins.float64)
        assert_allclose(a.numpy(), np.logspace(0, 3, 4), rtol=1e-5)
