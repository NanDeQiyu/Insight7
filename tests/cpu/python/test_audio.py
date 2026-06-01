"""Audio/I/O CPU binding tests.

Tests signal I/O functions (read_bin/write_bin roundtrip).
Note: Audio WAV read/write is C++ only and tested in test_audio.cpp.
"""
import sys
import os
import tempfile
import pytest

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, os.path.join(_root, "bindings", "python"))
sys.path.insert(0, os.path.join(_root, "build", "bindings", "python"))

try:
    import insight as ins
    import numpy as np
except ImportError:
    pytest.skip("Insight or NumPy not available", allow_module_level=True)


class TestAudioCPU:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp(prefix="insight_audio_test_")

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_write_read_bin_float64_roundtrip(self):
        data_np = np.array([1.0, 2.5, 3.7, -1.2, 0.0], dtype=np.float64)
        arr = ins.from_numpy(data_np)
        path = os.path.join(self.tmpdir, "test_f64.bin")
        ins.write_bin(path, arr)
        result = ins.read_bin(path, "float64")
        np.testing.assert_allclose(result.numpy(), data_np, rtol=1e-10)

    def test_write_read_bin_float32_roundtrip(self):
        data_np = np.array([1.0, 2.5, 3.7, -1.2], dtype=np.float32)
        arr = ins.from_numpy(data_np)
        path = os.path.join(self.tmpdir, "test_f32.bin")
        ins.write_bin(path, arr)
        result = ins.read_bin(path, "float32")
        np.testing.assert_allclose(result.numpy(), data_np, rtol=1e-5)

    def test_write_read_bin_int32_roundtrip(self):
        data_np = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        arr = ins.from_numpy(data_np)
        path = os.path.join(self.tmpdir, "test_i32.bin")
        ins.write_bin(path, arr)
        result = ins.read_bin(path, "int32")
        np.testing.assert_array_equal(result.numpy(), data_np)

    def test_write_read_bin_int16_roundtrip(self):
        data_np = np.array([100, 200, 300, -100], dtype=np.int16)
        arr = ins.from_numpy(data_np)
        path = os.path.join(self.tmpdir, "test_i16.bin")
        ins.write_bin(path, arr)
        result = ins.read_bin(path, "int16")
        np.testing.assert_array_equal(result.numpy(), data_np)

    def test_write_read_bin_large_signal(self):
        np.random.seed(42)
        data_np = np.random.randn(10000).astype(np.float64)
        arr = ins.from_numpy(data_np)
        path = os.path.join(self.tmpdir, "test_large.bin")
        ins.write_bin(path, arr)
        result = ins.read_bin(path, "float64")
        np.testing.assert_allclose(result.numpy(), data_np, rtol=1e-10)

    def test_write_read_bin_zeros(self):
        data_np = np.zeros(100, dtype=np.float64)
        arr = ins.from_numpy(data_np)
        path = os.path.join(self.tmpdir, "test_zeros.bin")
        ins.write_bin(path, arr)
        result = ins.read_bin(path, "float64")
        np.testing.assert_allclose(result.numpy(), data_np)

    def test_write_read_bin_ones(self):
        data_np = np.ones(50, dtype=np.float64)
        arr = ins.from_numpy(data_np)
        path = os.path.join(self.tmpdir, "test_ones.bin")
        ins.write_bin(path, arr)
        result = ins.read_bin(path, "float64")
        np.testing.assert_allclose(result.numpy(), data_np)

    def test_write_read_bin_negative_values(self):
        data_np = np.array([-100.5, -200.3, -300.1], dtype=np.float64)
        arr = ins.from_numpy(data_np)
        path = os.path.join(self.tmpdir, "test_neg.bin")
        ins.write_bin(path, arr)
        result = ins.read_bin(path, "float64")
        np.testing.assert_allclose(result.numpy(), data_np, rtol=1e-10)

    def test_write_read_bin_single_element(self):
        data_np = np.array([42.0], dtype=np.float64)
        arr = ins.from_numpy(data_np)
        path = os.path.join(self.tmpdir, "test_single.bin")
        ins.write_bin(path, arr)
        result = ins.read_bin(path, "float64")
        assert result.numpy().item() == pytest.approx(42.0)

    def test_read_bin_dtype_string(self):
        data_np = np.array([1, 2, 3], dtype=np.float64)
        arr = ins.from_numpy(data_np)
        path = os.path.join(self.tmpdir, "test_dtype.bin")
        ins.write_bin(path, arr)
        result = ins.read_bin(path, dtype="float64")
        np.testing.assert_allclose(result.numpy(), data_np, rtol=1e-10)

    def test_write_read_bin_uint8(self):
        data_np = np.array([0, 128, 255, 64], dtype=np.uint8)
        arr = ins.from_numpy(data_np)
        path = os.path.join(self.tmpdir, "test_u8.bin")
        ins.write_bin(path, arr)
        result = ins.read_bin(path, "uint8")
        np.testing.assert_array_equal(result.numpy(), data_np)

    def test_write_read_bin_multiple_files(self):
        for i in range(5):
            data_np = np.array([i * 1.0, i * 2.0], dtype=np.float64)
            arr = ins.from_numpy(data_np)
            path = os.path.join(self.tmpdir, f"test_{i}.bin")
            ins.write_bin(path, arr)
            result = ins.read_bin(path, "float64")
            np.testing.assert_allclose(result.numpy(), data_np, rtol=1e-10)

    def test_write_read_bin_overwrite(self):
        data1 = np.array([1, 2, 3], dtype=np.float64)
        data2 = np.array([4, 5, 6, 7], dtype=np.float64)
        path = os.path.join(self.tmpdir, "test_overwrite.bin")
        ins.write_bin(path, ins.from_numpy(data1))
        ins.write_bin(path, ins.from_numpy(data2))
        result = ins.read_bin(path, "float64")
        np.testing.assert_allclose(result.numpy(), data2, rtol=1e-10)

    def test_write_creates_file(self):
        data_np = np.array([1, 2, 3], dtype=np.float64)
        arr = ins.from_numpy(data_np)
        path = os.path.join(self.tmpdir, "test_create.bin")
        ins.write_bin(path, arr)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_read_bin_file_size_correct(self):
        data_np = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        arr = ins.from_numpy(data_np)
        path = os.path.join(self.tmpdir, "test_size.bin")
        ins.write_bin(path, arr)
        size = os.path.getsize(path)
        assert size == 5 * 8  # 5 doubles * 8 bytes each


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
