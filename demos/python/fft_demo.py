"""FFT demo for Insight7 Python bindings.

Demonstrates: FFT signal processing (RFFT/IRFFT roundtrip) on CPU
and GPU (if available), with F32 and F64 precision.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "build", "bindings", "python")
)

import insight as ins
import numpy as np


def separator(title):
    print(f"\n{'=' * 40}")
    print(f"  {title}")
    print(f"{'=' * 40}")


def gpu_available():
    try:
        ins.load_backend("cuda")
        return True
    except Exception:
        return False


def run_fft_cpu():
    separator("CPU FFT")

    # Generate a signal: sum of two sinusoids
    n = 64
    t = np.arange(n) / n
    signal = np.sin(2 * math.pi * 5 * t) + 0.5 * np.sin(2 * math.pi * 12 * t)
    signal = signal.astype(np.float32)

    x = ins.from_numpy(signal)
    print(f"Input signal (first 8): {x[:8]}")

    # RFFT -> IRFFT roundtrip
    X = ins.rfft(x)
    print(f"RFFT output length: {X.numel()} (complex)")

    x_recon = ins.irfft(X, n=n)
    print(f"Reconstructed signal (first 8): {x_recon[:8]}")

    # Check reconstruction error
    max_err = np.max(np.abs(signal - x_recon.numpy()))
    print(f"Max reconstruction error: {max_err:.6e}")

    # F64 FFT
    separator("CPU FFT (F64)")
    signal_f64 = np.cos(2 * math.pi * 3 * np.arange(n) / n).astype(np.float64)
    x64 = ins.from_numpy(signal_f64)
    X64 = ins.rfft(x64)
    x64_recon = ins.irfft(X64, n=n)
    max_err_f64 = np.max(np.abs(signal_f64 - x64_recon.numpy()))
    print(f"F64 FFT roundtrip max error: {max_err_f64:.6e}")

    # next_fast_len
    separator("FFT Length Optimization")
    print(f"next_fast_len(64)  = {ins.next_fast_len(64)}")
    print(f"next_fast_len(100) = {ins.next_fast_len(100)}")
    print(f"next_fast_len(127) = {ins.next_fast_len(127)}")


def run_fft_gpu():
    separator("GPU FFT")

    n = 64
    t = np.arange(n) / n
    signal = (np.sin(2 * math.pi * 5 * t) + 0.5 * np.sin(2 * math.pi * 12 * t)).astype(np.float32)

    x = ins.from_numpy(signal).to(ins.GPUPlace(0))
    X = ins.rfft(x)
    x_recon = ins.irfft(X, n=n).to(ins.CPUPlace())

    print(f"GPU RFFT->IRFFT roundtrip (first 8): {x_recon[:8]}")
    max_err = np.max(np.abs(signal - x_recon.numpy()))
    print(f"GPU max reconstruction error: {max_err:.6e}")

    # GPU F64 FFT
    separator("GPU FFT (F64)")
    signal_f64 = np.cos(2 * math.pi * 3 * np.arange(n) / n).astype(np.float64)
    x64 = ins.from_numpy(signal_f64).to(ins.GPUPlace(0))
    X64 = ins.rfft(x64)
    x64_recon = ins.irfft(X64, n=n).to(ins.CPUPlace())
    max_err_f64 = np.max(np.abs(signal_f64 - x64_recon.numpy()))
    print(f"GPU F64 FFT roundtrip max error: {max_err_f64:.6e}")


def main():
    ins.init()

    print("Insight7 FFT Demo (Python)")

    run_fft_cpu()

    if gpu_available():
        try:
            run_fft_gpu()
        except Exception:
            pass

    print("\nDone!")


if __name__ == "__main__":
    main()
