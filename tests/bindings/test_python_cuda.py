"""
CUDA binding tests for Insight7 signal processing functions.

Run with:
    cd build && LD_LIBRARY_PATH="$(pwd):$(pwd)/backends/cpu:$(pwd)/backends/cuda:$LD_LIBRARY_PATH" \
    PYTHONPATH="/path/to/bindings/python:$PYTHONPATH" \
    python3 ../../tests/bindings/test_python_cuda.py
"""

import sys
import numpy as np

import insight as ins


def test_signal_cuda():
    """Test signal functions with CUDA backend loaded."""
    ins.load_backend("cuda")
    gpu = ins.GPUPlace(0)

    # Windows
    w = ins.signal.hann(16)
    assert w.numel() == 16, f"hann numel: {w.numel()}"

    w2 = ins.signal.hamming(32)
    assert w2.numel() == 32

    w3 = ins.signal.kaiser(64, 5.0)
    assert w3.numel() == 64

    w4 = ins.signal.blackman(48)
    assert w4.numel() == 48

    w5 = ins.signal.gaussian(32, 1.0)
    assert w5.numel() == 32

    w6 = ins.signal.get_window("hann", 20)
    assert w6.numel() == 20

    # Filter design (scalar returns)
    beta = ins.signal.kaiser_beta(50)
    assert isinstance(beta, float) and beta > 0

    atten = ins.signal.kaiser_atten(100, 0.1)
    assert isinstance(atten, float)

    # Convolution on GPU
    a = ins.from_numpy(np.array([1.0, 2.0, 3.0, 4.0])).to(gpu)
    v = ins.from_numpy(np.array([1.0, 1.0])).to(gpu)
    c = ins.signal.fftconvolve(a, v)
    assert c.numel() == 5
    assert c.place().is_gpu(), "fftconvolve result should be on GPU"

    c2 = ins.signal.correlate(a, v)
    assert c2.numel() == 5

    # Filtering on GPU
    x = ins.from_numpy(np.random.randn(100)).to(gpu)
    h = ins.signal.hilbert(x)
    assert h.numel() == 100

    d = ins.signal.detrend(x)
    assert d.numel() == 100

    # B-splines on GPU
    t = ins.from_numpy(np.array([-1.0, -0.5, 0.0, 0.5, 1.0])).to(gpu)
    cs = ins.signal.cubic(t)
    assert cs.numel() == 5

    qs = ins.signal.quadratic(t)
    assert qs.numel() == 5

    # Acoustics on GPU
    hz = ins.from_numpy(np.array([0.0, 1000.0, 4000.0])).to(gpu)
    mel = ins.signal.hz2mel(hz)
    assert mel.numel() == 3

    mel_back = ins.signal.mel2hz(mel)
    assert mel_back.numel() == 3

    bark = ins.signal.hz2bark(hz)
    assert bark.numel() == 3

    bark_back = ins.signal.bark2hz(bark)
    assert bark_back.numel() == 3

    # Wavelets
    m = ins.signal.morlet(16)
    assert m.numel() == 16

    r = ins.signal.ricker(20, 2.0)
    assert r.numel() == 20

    # Spectral analysis (CPU-only)
    x_cpu = ins.from_numpy(np.random.randn(1024))
    res = ins.signal.welch(x_cpu, fs=256.0)
    assert res.f.numel() > 0
    assert res.Pxx.numel() > 0

    res2 = ins.signal.periodogram(x_cpu, fs=256.0)
    assert res2.f.numel() > 0

    # Unit impulse
    imp = ins.signal.unit_impulse(ins.Shape([10]), 3)
    assert imp.numel() == 10
    assert float(ins.sum(imp).numpy()) == 1.0

    # ChirpMethod enum
    assert ins.signal.ChirpMethod.linear is not None
    assert ins.signal.ChirpMethod.quadratic is not None

    # Top-level aliases
    assert hasattr(ins, 'convolve')
    assert hasattr(ins, 'unwrap')
    assert hasattr(ins, 'sinc')

    print("=== ALL PYTHON CUDA SIGNAL TESTS PASSED ===")


if __name__ == "__main__":
    test_signal_cuda()
