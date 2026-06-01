"""Device information CUDA binding tests."""
import sys
import os
import pytest

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, os.path.join(_root, "bindings", "python"))
sys.path.insert(0, os.path.join(_root, "build", "bindings", "python"))

try:
    import insight as ins
except ImportError:
    pytest.skip("Insight not available", allow_module_level=True)

try:
    ins.load_backend("cuda")
    if ins.device_count() == 0:
        pytest.skip("No GPU available", allow_module_level=True)
except Exception:
    pytest.skip("CUDA backend not available", allow_module_level=True)


class TestDeviceInfoCUDA:
    def test_device_name_gpu(self):
        name = ins.device_name("gpu", 0)
        assert isinstance(name, str)
        assert len(name) > 0

    def test_device_name_cuda_alias(self):
        name = ins.device_name("cuda", 0)
        assert isinstance(name, str)
        assert len(name) > 0

    def test_cuda_version_positive(self):
        ver = ins.cuda_version()
        assert ver > 0

    def test_driver_version_positive(self):
        ver = ins.driver_version()
        assert ver > 0

    def test_device_count_positive(self):
        count = ins.device_count()
        assert count > 0

    def test_compute_capability_positive(self):
        cc = ins.compute_capability(0)
        assert cc > 0

    def test_device_memory(self):
        mem = ins.device_memory(0)
        assert isinstance(mem, tuple)
        assert len(mem) == 2
        assert mem[0] > 0  # total > 0
        assert mem[1] > 0  # free > 0
        assert mem[0] >= mem[1]  # total >= free

    def test_compute_capability_range(self):
        cc = ins.compute_capability(0)
        assert 30 <= cc <= 100  # reasonable range for modern GPUs

    def test_cuda_version_format(self):
        ver = ins.cuda_version()
        major = ver // 1000
        minor = (ver % 1000) // 10
        assert major >= 11  # CUDA 11+

    def test_driver_version_format(self):
        ver = ins.driver_version()
        major = ver // 1000
        assert major >= 11

    def test_device_memory_total_reasonable(self):
        total, free = ins.device_memory(0)
        # At least 1 GB
        assert total >= 1024 * 1024 * 1024

    def test_device_name_multiple_calls(self):
        # Should be stable across calls
        name1 = ins.device_name("gpu", 0)
        name2 = ins.device_name("gpu", 0)
        assert name1 == name2

    def test_compute_capability_multiple_calls(self):
        cc1 = ins.compute_capability(0)
        cc2 = ins.compute_capability(0)
        assert cc1 == cc2

    def test_is_initialized(self):
        assert ins.is_initialized()

    def test_device_name_gpu_string(self):
        name = ins.device_name("gpu", 0)
        # Should contain GPU-related info
        assert len(name) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
