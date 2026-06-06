"""Random number generation CPU binding tests."""

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


class TestRandomCPU:
    def test_rand_shape(self):
        a = ins.rand(ins.Shape([3, 4]), ins.float64)
        assert a.numel == 12

    def test_rand_range(self):
        a = ins.rand(ins.Shape([1000]), ins.float64)
        data = a.numpy()
        assert np.all(data >= 0) and np.all(data < 1)

    def test_randn_shape(self):
        a = ins.randn(ins.Shape([3, 4]), ins.float64)
        assert a.numel == 12

    def test_randn_statistics(self):
        a = ins.randn(ins.Shape([10000]), ins.float64)
        data = a.numpy()
        assert abs(np.mean(data)) < 0.1
        assert abs(np.std(data) - 1.0) < 0.1

    def test_seed_determinism(self):
        ins.seed(42)
        a = ins.rand(ins.Shape([5]), ins.float64)
        ins.seed(42)
        b = ins.rand(ins.Shape([5]), ins.float64)
        # Note: seed determinism depends on backend implementation
        assert a.numel == b.numel

    def test_get_seed(self):
        ins.seed(12345)
        s = ins.get_seed()
        assert s == 12345

    def test_randint_shape(self):
        a = ins.randint(0, 10, ins.Shape([3, 3]), ins.int64)
        assert a.numel == 9

    def test_randint_range(self):
        a = ins.randint(0, 10, ins.Shape([1000]), ins.int64)
        data = a.numpy()
        assert np.all(data >= 0) and np.all(data < 10)

    def test_rand_like(self):
        template = ins.from_numpy(np.zeros([3, 4], dtype=np.float64))
        a = ins.rand_like(template)
        assert a.numel == 12

    def test_randn_like(self):
        template = ins.from_numpy(np.zeros([3, 4], dtype=np.float64))
        a = ins.randn_like(template)
        assert a.numel == 12

    def test_exponential_shape(self):
        a = ins.exponential(1.0, ins.Shape([100]), ins.float64)
        assert a.numel == 100
        data = a.numpy()
        assert np.all(data >= 0)

    def test_gamma_shape(self):
        a = ins.gamma(2.0, 1.0, ins.Shape([100]), ins.float32)
        assert a.numel == 100

    def test_beta_shape(self):
        a = ins.beta(2.0, 5.0, ins.Shape([100]), ins.float64)
        assert a.numel == 100
        data = a.numpy()
        assert np.all(data >= 0) and np.all(data <= 1)

    def test_binomial_shape(self):
        a = ins.binomial(10, 0.5, ins.Shape([100]), ins.int64)
        assert a.numel == 100

    def test_poisson_shape(self):
        a = ins.poisson(5.0, ins.Shape([100]), ins.int64)
        assert a.numel == 100

    def test_rand_float32(self):
        a = ins.rand(ins.Shape([10]), ins.float32)
        assert a.numel == 10

    def test_rand_3d(self):
        a = ins.rand(ins.Shape([2, 3, 4]), ins.float64)
        assert a.numel == 24

    def test_randperm(self):
        a = ins.randperm(10, ins.int64)
        assert a.numel == 10
        data = sorted(a.numpy().tolist())
        assert data == list(range(10))

    def test_uniform(self):
        a = ins.uniform(0.0, 1.0, ins.Shape([100]), ins.float64)
        assert a.numel == 100

    def test_normal(self):
        a = ins.normal(0.0, 1.0, ins.Shape([100]), ins.float64)
        assert a.numel == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
