"""Signal radar CPU binding tests."""

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


class TestSignalRadarCPU:
    """Radar signal processing — signal_radar CPU tests."""

    def test_pulse_compression(self):
        n_pulses = 4
        n_samples = 64
        template = np.random.randn(n_samples).astype(np.float64)
        # pulse_compression expects 2D input [num_pulses, samples_per_pulse]
        data = np.zeros((n_pulses, n_samples), dtype=np.float64)
        data[0, :n_samples] += template
        x = ins.from_numpy(data)
        tpl = ins.from_numpy(template)
        result = ins.signal.pulse_compression(x, tpl)
        assert result is not None
        assert result.numel > 0

    def test_pulse_compression_normalized(self):
        n_pulses = 4
        n_samples = 64
        template = np.random.randn(n_samples).astype(np.float64)
        data = np.zeros((n_pulses, n_samples), dtype=np.float64)
        data[0, :n_samples] += template
        x = ins.from_numpy(data)
        tpl = ins.from_numpy(template)
        result = ins.signal.pulse_compression(x, tpl, normalize=True)
        assert result is not None
        assert result.numel > 0

    def test_pulse_doppler(self):
        n_pulses = 16
        n_range = 32
        data = np.random.randn(n_pulses, n_range).astype(np.float64)
        x = ins.from_numpy(data)
        result = ins.signal.pulse_doppler(x)
        assert result is not None
        assert result.numel > 0

    def test_cfar_alpha(self):
        alpha = ins.signal.cfar_alpha(1e-3, 20)
        assert alpha is not None
        assert alpha > 0

    def test_cfar_alpha_different_pfa(self):
        alpha1 = ins.signal.cfar_alpha(1e-2, 20)
        alpha2 = ins.signal.cfar_alpha(1e-6, 20)
        assert alpha1 < alpha2

    def test_ca_cfar(self):
        data = np.zeros(100, dtype=np.float64)
        data[30] = 10.0
        data[70] = 10.0
        x = ins.from_numpy(data)
        result = ins.signal.ca_cfar(x, [2], [10], pfa=1e-2)
        assert result is not None
        # ca_cfar returns a tuple (threshold, detections)
        assert len(result) == 2
        assert result[0].numel == 100

    def test_mvdr(self):
        n_elements = 4
        n_snapshots = 32
        x_np = np.random.randn(n_elements, n_snapshots).astype(np.float64)
        sv_np = np.ones(n_elements, dtype=np.float64)
        x = ins.from_numpy(x_np)
        sv = ins.from_numpy(sv_np)
        result = ins.signal.mvdr(x, sv)
        assert result is not None
        assert result.numel > 0

    def test_ambgfun(self):
        n = 64
        x_np = np.random.randn(n).astype(np.float64)
        x = ins.from_numpy(x_np)
        result = ins.signal.ambgfun(x, fs=1000.0, prf=100.0)
        assert result is not None
        assert result.numel > 0

    def test_ambgfun_delay_cut(self):
        n = 32
        x_np = np.random.randn(n).astype(np.float64)
        x = ins.from_numpy(x_np)
        result = ins.signal.ambgfun(x, fs=1000.0, prf=100.0, cut="delay", cutValue=0)
        assert result is not None
        assert result.numel > 0

    def test_ambgfun_doppler_cut(self):
        n = 32
        x_np = np.random.randn(n).astype(np.float64)
        x = ins.from_numpy(x_np)
        result = ins.signal.ambgfun(x, fs=1000.0, prf=100.0, cut="doppler", cutValue=0)
        assert result is not None
        assert result.numel > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
