"""Linalg alignment tests — Insight CUDA vs NumPy.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \\
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \\
    python -m pytest tests/python_align/cuda/test_linalg.py -v
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


class TestLinalgAlignmentGPU:
    """Insight linalg on GPU vs NumPy."""

    @pytest.fixture
    def matrix_2x2(self):
        return np.array([[1, 2], [3, 4]], dtype=np.float64)

    def test_matmul(self, matrix_2x2):
        a, b = to_gpu(matrix_2x2), to_gpu(matrix_2x2)
        result = ins.matmul(a, b)
        assert_allclose(to_numpy(result), np.matmul(matrix_2x2, matrix_2x2), rtol=1e-6)

    def test_det(self, matrix_2x2):
        a = to_gpu(matrix_2x2)
        assert_allclose(to_numpy(ins.det(a)), np.linalg.det(matrix_2x2), rtol=1e-6)

    def test_inv(self, matrix_2x2):
        a = to_gpu(matrix_2x2)
        assert_allclose(to_numpy(ins.inv(a)), np.linalg.inv(matrix_2x2), atol=1e-6)

    def test_trace(self, matrix_2x2):
        a = to_gpu(matrix_2x2)
        assert_allclose(to_numpy(ins.trace(a)), np.trace(matrix_2x2), rtol=1e-6)

    def test_norm(self):
        x_np = np.array([3, 4], dtype=np.float64)
        x = to_gpu(x_np)
        assert_allclose(to_numpy(ins.norm(x)), np.linalg.norm(x_np), rtol=1e-6)

    def test_dot(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5, 6], dtype=np.float64)
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.dot(a, b)), np.dot(a_np, b_np), rtol=1e-6)

    def test_outer(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5], dtype=np.float64)
        a, b = to_gpu(a_np), to_gpu(b_np)
        assert_allclose(to_numpy(ins.outer(a, b)), np.outer(a_np, b_np), rtol=1e-6)


class TestLinalgExtendedGPU:
    """Extended linalg ops on GPU vs NumPy."""

    @pytest.fixture
    def spd_matrix_3x3(self):
        """Symmetric positive-definite 3x3 matrix."""
        return np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]], dtype=np.float64)

    @pytest.fixture
    def nonsingular_3x3(self):
        return np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=np.float64)

    @pytest.fixture
    def matrix_3x3(self):
        return np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=np.float64)

    def test_solve(self, nonsingular_3x3):
        A_np = nonsingular_3x3
        b_np = np.array([1, 2, 3], dtype=np.float64)
        A, b = to_gpu(A_np), to_gpu(b_np)
        x = ins.solve(A, b)
        expected = np.linalg.solve(A_np, b_np)
        assert_allclose(to_numpy(x), expected, atol=1e-6)

    def test_svd(self, matrix_3x3):
        A = to_gpu(matrix_3x3)
        U, S, Vt = ins.svd(A)
        _, S_np, _ = np.linalg.svd(matrix_3x3)
        assert_allclose(to_numpy(S), S_np, rtol=1e-6)

    def test_eigh(self, spd_matrix_3x3):
        A = to_gpu(spd_matrix_3x3)
        w, v = ins.eigh(A)
        w_np, _ = np.linalg.eigh(spd_matrix_3x3)
        assert_allclose(to_numpy(w), w_np, rtol=1e-6)

    def test_cholesky(self, spd_matrix_3x3):
        A = to_gpu(spd_matrix_3x3)
        L = ins.cholesky(A)
        expected = np.linalg.cholesky(spd_matrix_3x3)
        assert_allclose(to_numpy(L), expected, atol=1e-6)

    def test_qr(self, matrix_3x3):
        A = to_gpu(matrix_3x3)
        Q, R = ins.qr(A)
        # Check reconstruction
        assert_allclose(to_numpy(Q) @ to_numpy(R), matrix_3x3, atol=1e-6)

    def test_lu(self, matrix_3x3):
        A = to_gpu(matrix_3x3)
        P, L, U = ins.lu(A)
        recon = to_numpy(L) @ to_numpy(U)
        assert_allclose(to_numpy(P) @ matrix_3x3, recon, atol=1e-6)

    def test_lstsq(self):
        A_np = np.array([[1, 1], [1, 2], [1, 3], [1, 4]], dtype=np.float64)
        b_np = np.array([2.1, 3.9, 6.2, 7.8], dtype=np.float64)
        A, b = to_gpu(A_np), to_gpu(b_np)
        result = ins.lstsq(A, b)
        x_np, _, _, _ = np.linalg.lstsq(A_np, b_np, rcond=None)
        if isinstance(result, (list, tuple)):
            x_ins = result[0]
        else:
            x_ins = result
        assert_allclose(to_numpy(x_ins), x_np, atol=1e-6)

    def test_cond(self, matrix_3x3):
        A = to_gpu(matrix_3x3)
        result = ins.cond(A)
        expected = np.linalg.cond(matrix_3x3)
        assert_allclose(to_numpy(result), expected, rtol=1e-4)

    def test_matrix_rank(self, matrix_3x3):
        A = to_gpu(matrix_3x3)
        result = ins.matrix_rank(A)
        expected = np.linalg.matrix_rank(matrix_3x3)
        assert_allclose(to_numpy(result), expected)

    def test_pinv(self, matrix_3x3):
        A = to_gpu(matrix_3x3)
        result = ins.pinv(A)
        expected = np.linalg.pinv(matrix_3x3)
        assert_allclose(to_numpy(result), expected, atol=1e-6)

    def test_slogdet(self, matrix_3x3):
        A = to_gpu(matrix_3x3)
        sign, logabsdet = ins.slogdet(A)
        sign_np, logabsdet_np = np.linalg.slogdet(matrix_3x3)
        assert_allclose(to_numpy(sign), sign_np)
        assert_allclose(to_numpy(logabsdet), logabsdet_np, rtol=1e-6)

    def test_matrix_power(self, matrix_3x3):
        A = to_gpu(matrix_3x3)
        result = ins.matrix_power(A, 3)
        expected = np.linalg.matrix_power(matrix_3x3, 3)
        assert_allclose(to_numpy(result), expected, atol=1e-6)

    def test_eigvalsh(self, spd_matrix_3x3):
        A = to_gpu(spd_matrix_3x3)
        w = ins.eigvalsh(A)
        w_np = np.linalg.eigvalsh(spd_matrix_3x3)
        assert_allclose(to_numpy(w), w_np, rtol=1e-6)

    def test_svdvals(self, matrix_3x3):
        A = to_gpu(matrix_3x3)
        s = ins.svdvals(A)
        _, s_np, _ = np.linalg.svd(matrix_3x3)
        assert_allclose(to_numpy(s), s_np, rtol=1e-6)

    def test_cov_1d(self):
        x_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.cov(x)
        expected = np.cov(x_np)
        assert_allclose(to_numpy(result), expected, rtol=1e-6)

    def test_inner(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5, 6], dtype=np.float64)
        a, b = to_gpu(a_np), to_gpu(b_np)
        result = ins.inner(a, b)
        expected = np.inner(a_np, b_np)
        assert_allclose(to_numpy(result), expected, rtol=1e-6)
