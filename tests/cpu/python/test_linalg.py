"""Linalg CPU binding tests."""

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


class TestLinalgCPU:
    def test_matmul(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        b_np = np.array([[5, 6], [7, 8]], dtype=np.float64)
        result = ins.matmul(ins.from_numpy(a_np), ins.from_numpy(b_np))
        np.testing.assert_allclose(result.numpy(), a_np @ b_np, rtol=1e-10)

    def test_dot(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5, 6], dtype=np.float64)
        result = ins.dot(ins.from_numpy(a_np), ins.from_numpy(b_np))
        expected = np.dot(a_np, b_np)
        assert result.numpy().item() == pytest.approx(expected, rel=1e-10)

    def test_outer(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5], dtype=np.float64)
        result = ins.outer(ins.from_numpy(a_np), ins.from_numpy(b_np))
        np.testing.assert_allclose(result.numpy(), np.outer(a_np, b_np), rtol=1e-10)

    def test_norm_vector(self):
        a_np = np.array([3, 4], dtype=np.float64)
        result = ins.norm(ins.from_numpy(a_np))
        assert result.numpy().item() == pytest.approx(5.0, rel=1e-5)

    def test_norm_matrix(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        result = ins.norm(ins.from_numpy(a_np))
        expected = np.linalg.norm(a_np, "fro")
        assert result.numpy().item() == pytest.approx(expected, rel=1e-5)

    def test_trace(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        result = ins.trace(ins.from_numpy(a_np))
        assert result.numpy().item() == pytest.approx(5.0, rel=1e-10)

    def test_det(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        result = ins.det(ins.from_numpy(a_np))
        assert result.numpy().item() == pytest.approx(-2.0, rel=1e-5)

    def test_det_3x3(self):
        a_np = np.array([[6, 1, 1], [4, -2, 5], [2, 8, 7]], dtype=np.float64)
        result = ins.det(ins.from_numpy(a_np))
        expected = np.linalg.det(a_np)
        assert result.numpy().item() == pytest.approx(expected, rel=1e-5)

    def test_inv(self):
        a_np = np.array([[4, 7], [2, 6]], dtype=np.float64)
        result = ins.inv(ins.from_numpy(a_np))
        np.testing.assert_allclose(result.numpy(), np.linalg.inv(a_np), rtol=1e-5)

    def test_solve(self):
        a_np = np.array([[3, 1], [1, 2]], dtype=np.float64)
        b_np = np.array([9, 8], dtype=np.float64)
        result = ins.solve(ins.from_numpy(a_np), ins.from_numpy(b_np))
        expected = np.linalg.solve(a_np, b_np)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_svd(self):
        a_np = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        u, s, vt = ins.svd(ins.from_numpy(a_np))
        u_np, s_np, vt_np = np.linalg.svd(a_np, full_matrices=False)
        np.testing.assert_allclose(s.numpy(), s_np, rtol=1e-5)

    def test_eigh(self):
        a_np = np.array([[4, 2], [2, 3]], dtype=np.float64)
        w, v = ins.eigh(ins.from_numpy(a_np))
        w_np, v_np = np.linalg.eigh(a_np)
        np.testing.assert_allclose(np.sort(w.numpy()), np.sort(w_np), rtol=1e-5)

    def test_cholesky(self):
        a_np = np.array([[4, 2], [2, 3]], dtype=np.float64)
        result = ins.cholesky(ins.from_numpy(a_np))
        expected = np.linalg.cholesky(a_np)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_qr(self):
        a_np = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        q, r = ins.qr(ins.from_numpy(a_np))
        q_np, r_np = np.linalg.qr(a_np)
        np.testing.assert_allclose(r.numpy(), r_np, atol=1e-5)

    def test_matrix_power(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        result = ins.matrix_power(ins.from_numpy(a_np), 2)
        expected = np.linalg.matrix_power(a_np, 2)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_slogdet(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        sign, logabsdet = ins.slogdet(ins.from_numpy(a_np))
        sign_np, logabsdet_np = np.linalg.slogdet(a_np)
        assert sign.numpy().item() == pytest.approx(sign_np, abs=1e-5)
        assert logabsdet.numpy().item() == pytest.approx(logabsdet_np, rel=1e-5)

    def test_eigvalsh(self):
        a_np = np.array([[4, 2], [2, 3]], dtype=np.float64)
        w = ins.eigvalsh(ins.from_numpy(a_np))
        w_np = np.linalg.eigvalsh(a_np)
        np.testing.assert_allclose(np.sort(w.numpy()), np.sort(w_np), rtol=1e-5)

    def test_pinv(self):
        a_np = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        result = ins.pinv(ins.from_numpy(a_np))
        expected = np.linalg.pinv(a_np)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_lstsq(self):
        a_np = np.array([[1, 1], [1, 2], [1, 3]], dtype=np.float64)
        b_np = np.array([1, 2, 2], dtype=np.float64)
        result = ins.lstsq(ins.from_numpy(a_np), ins.from_numpy(b_np))
        expected, _, _, _ = np.linalg.lstsq(a_np, b_np, rcond=None)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_cond(self):
        a_np = np.array([[1, 0], [0, 2]], dtype=np.float64)
        result = ins.cond(ins.from_numpy(a_np))
        expected = np.linalg.cond(a_np)
        assert result.numpy().item() == pytest.approx(expected, rel=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
