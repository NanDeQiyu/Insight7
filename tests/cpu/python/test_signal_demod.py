"""Signal demod CPU binding tests."""

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


class TestSignalDemodCPU:
    """FM demodulation — signal_demod CPU tests."""

    def _make_fm_signal(self, n=512, fs=256.0, fc=20.0, fm=2.0, deviation=5.0):
        t = np.arange(n) / fs
        message = np.sin(2 * np.pi * fm * t)
        phase = 2 * np.pi * fc * t + 2 * np.pi * deviation * np.cumsum(message) / fs
        return np.exp(1j * phase).astype(np.complex128)

    def test_fm_demod(self):
        x_np = self._make_fm_signal()
        x = ins.from_numpy(x_np)
        result = ins.signal.fm_demod(x)
        assert result is not None
        assert result.numel > 0

    def test_fm_demod_output_length(self):
        x_np = self._make_fm_signal(n=256)
        x = ins.from_numpy(x_np)
        result = ins.signal.fm_demod(x)
        assert result is not None
        # diff reduces length by 1: 256 -> 255
        assert result.numel == 255

    def test_fm_demod_bounded(self):
        x_np = self._make_fm_signal(n=1024, deviation=10.0)
        x = ins.from_numpy(x_np)
        result = ins.signal.fm_demod(x)
        result_np = result.numpy()
        assert np.all(np.isfinite(result_np))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
