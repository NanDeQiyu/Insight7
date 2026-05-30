---
name: fix-fft-domain-manipulation
description: FFT-based signal processing (hilbert, resample, spectral) must work on CPU and manually manipulate frequency-domain arrays
source: auto-skill
extracted_at: '2026-05-30T17:30:00.000Z'
---

# Fix FFT-Domain Manipulation

## Problem

Functions that perform FFT → modify in frequency domain → IFFT fail on GPU because:
1. `fft::fft()` returns complex arrays whose `.data<char>()` can't be read from CPU
2. `mul(complex_array, complex_array)` may not work correctly for all dtype combinations
3. The result needs to be read element-by-element to apply frequency-domain operations

## Symptoms

- Hilbert transform produces wrong magnitude (not ~1 for analytic signal of cos)
- `resample` segfaults when called from CUDA tests
- Spectral analysis functions produce incorrect scaling

## Solution

### Pattern 1: Manual frequency-domain multiplication (hilbert)

```cpp
// Transfer to CPU first
Place cpu = CPUPlace();
Array x_cpu = (x.place().kind() == DeviceKind::CPU) ? x : x.to(cpu);
Array Xf = fft::fft(x_cpu, N);

// Read FFT output directly
const std::complex<double> *X_data =
    reinterpret_cast<const std::complex<double> *>(Xf.data<char>());

// Apply filter manually
std::vector<std::complex<double>> result_data(N);
result_data[0] = X_data[0] * 1.0;  // DC: h[0] = 1
for (int64_t i = 1; i < (N + 1) / 2; ++i) {
    result_data[i] = X_data[i] * 2.0;  // positive freq: h[i] = 2
}
// h[N/2+1..N-1] = 0 (already zero-initialized)

Array result_arr = to_array(result_data, DType::C64, cpu);
Array result = fft::ifft(result_arr, N);
if (x.place().kind() != DeviceKind::CPU) result = result.to(x.place());
```

### Pattern 2: Frequency-domain zero-padding (resample)

```cpp
// Read FFT output
const std::complex<double> *X_data =
    reinterpret_cast<const std::complex<double> *>(Xf.data<char>());

// Create new frequency array with zeros inserted
std::vector<std::complex<double>> new_X(target_len, {0.0, 0.0});
new_X[0] = X_data[0];  // DC
for (int64_t i = 1; i <= half_n; ++i) new_X[i] = X_data[i];  // positive
for (int64_t i = 1; i < half_n; ++i) new_X[target_len - i] = X_data[n - i];  // negative

Array new_arr = to_array(new_X, DType::C64, cpu);
Array result = fft::ifft(new_arr, target_len);
```

### Pattern 3: Spectral helper (welch/csd/spectrogram)

```cpp
// For each segment:
// 1. Extract and window on CPU
// 2. FFT on CPU
// 3. Read complex output directly
// 4. Accumulate |X|^2 or conj(X)*Y

Array seg_arr = to_array(seg_data, DType::C64, cpu);
Array Xf = fft::fft(seg_arr, nfft);
const std::complex<double> *X_data =
    reinterpret_cast<const std::complex<double> *>(Xf.data<char>());

for (int64_t i = 0; i < freq_len; ++i) {
    Pxx_accum[i] += std::norm(X_data[i]);  // |X|^2
}
```

## Key Rules

1. **Always transfer to CPU before FFT** if the function needs to read the output
2. **Never use `mul(complex, complex)` for frequency-domain filtering** — use manual loops
3. **FFT normalization**: `fft` has no scaling (backward), `ifft` divides by N
4. **One-sided spectrum**: Double non-DC, non-Nyquist components for real signals
5. **Transfer back** to original device after IFFT if needed

## Affected Functions

- `hilbert` / `hilbert2` — frequency-domain filter application
- `resample` — frequency-domain zero-padding/truncation
- `welch` / `csd` / `spectrogram` / `stft` — segment FFT processing
- `freq_shift` — complex exponential multiplication
