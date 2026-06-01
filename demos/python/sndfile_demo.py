"""Insight7 Signal Processing Demo (Python).

Generate a multi-frequency test signal, apply a low-pass filter via FFT,
and save results. Mirrors the C++ sndfile_demo without libsndfile dependency.
"""

import math
import numpy as np
import insight as ins


def separator(title: str) -> None:
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    ins.init(["cpu"])

    sample_rate = 44100
    duration = 2.0
    frames = int(sample_rate * duration)

    print("=== Insight Signal Processing Demo (Python) ===")
    print(f"    Sample rate: {sample_rate} Hz")
    print(f"    Duration:    {duration:.1f} s ({frames} frames)")
    print()

    # Step 1: Generate multi-frequency test signal
    separator("[1] Generate test signal")
    print("    Components: 440Hz (0.5), 880Hz (0.3), 3141Hz (0.1)")

    t = np.arange(frames, dtype=np.float64) / sample_rate
    signal_data = (
        0.5 * np.sin(2 * math.pi * 440 * t)
        + 0.3 * np.sin(2 * math.pi * 880 * t)
        + 0.1 * np.sin(2 * math.pi * 3141 * t)
    )
    signal = ins.from_numpy(signal_data.astype(np.float32))
    print(f"    Signal: [{signal.numel}] elements")
    print()

    # Step 2: Write signal to binary file
    separator("[2] Save signal to binary file")
    tmp = "/tmp/insight_demo_"

    ins.signal.write_bin(tmp + "original.bin", signal)
    print(f"    Written: {tmp}original.bin")

    read_back = ins.signal.read_bin(tmp + "original.bin")
    print(f"    Roundtrip read: {read_back.numel} elements")
    print()

    # Step 3: FFT analysis
    separator("[3] FFT analysis")

    spectrum = ins.rfft(signal)
    freq_bins = spectrum.numel
    print(f"    FFT bins: {freq_bins}")
    print(f"    Nyquist: {sample_rate // 2} Hz")
    print(f"    Bin resolution: {sample_rate / (2 * freq_bins):.2f} Hz")
    print()

    # Step 4: Low-pass filter via FFT (cutoff 1000Hz)
    separator("[4] Low-pass filter (cutoff 1000Hz)")

    cutoff_bin = int(1000 / (sample_rate / 2.0) * freq_bins)
    taper_width = freq_bins // 100
    print(f"    Cutoff bin: {cutoff_bin} (1000Hz)")
    print(f"    Taper width: {taper_width} bins")

    spec_real = ins.real(spectrum)
    spec_imag = ins.imag(spectrum)

    mask_data = np.zeros(freq_bins, dtype=np.float32)
    for i in range(freq_bins):
        if i < cutoff_bin - taper_width // 2:
            mask_data[i] = 1.0
        elif i < cutoff_bin + taper_width // 2:
            t_val = 1.0 - (i - cutoff_bin + taper_width // 2.0) / taper_width
            mask_data[i] = max(0.0, min(1.0, t_val))
        else:
            mask_data[i] = 0.0

    mask = ins.from_numpy(mask_data)

    filtered_real = ins.mul(spec_real, mask)
    filtered_imag = ins.mul(spec_imag, mask)

    filtered_spectrum = ins.to_complex(filtered_real, filtered_imag)
    filtered = ins.irfft(filtered_spectrum, frames)
    print(f"    Filtered signal: [{filtered.numel}] elements")

    # Step 5: Energy comparison
    print()
    separator("[5] Signal statistics")

    energy_orig = float(ins.sum(ins.mul(signal, signal)).item())
    energy_filt = float(ins.sum(ins.mul(filtered, filtered)).item())

    print(f"    Energy (original): {energy_orig:.4f}")
    print(f"    Energy (filtered): {energy_filt:.4f}")
    if energy_orig > 0:
        print(f"    Retained: {(energy_filt / energy_orig) * 100:.1f}%")

    ins.signal.write_bin(tmp + "filtered.bin", filtered)
    print()
    print("Output files:")
    print(f"  {tmp}original.bin  - Original multi-freq signal")
    print(f"  {tmp}filtered.bin  - Low-pass filtered (<1000Hz)")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
