---
name: guard-optional-dependency-tests
description: Guard tests that depend on optional libraries (FFTW3, OpenBLAS) with preprocessor checks
source: auto-skill
extracted_at: '2026-06-02T00:00:00.000Z'
---

# Guard Tests for Optional Dependencies

## Problem

Tests that use FFT (via `signal::cwt`, `signal::welch`, `fft::fft`, etc.) fail in CI environments where FFTW3 is not installed:
```
Kernel 'fft' failed on cpu:0 with status 2: insight_kernel_launch: kernel not found for 'fft'
```

Similarly, tests using `ins::inv`, `ins::solve`, `ins::matmul` fail without OpenBLAS.

## Fix Pattern

### C++ tests: Use `#ifdef` guards

```cpp
// At the top of the test file, in SetUpTestSuite:
class SignalWaveletsTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu"});
    set_device(CPUPlace());
#ifndef INSIGHT_USE_FFTW3
    GTEST_SKIP() << "FFTW3 not available, skipping FFT-dependent tests";
#endif
  }
};

// Or guard individual test cases:
#ifdef INSIGHT_USE_FFTW3
TEST_F(SignalWaveletsTestCPU, CwtBasic) {
  // ... uses FFT internally via cwt() ...
}
#endif // INSIGHT_USE_FFTW3
```

### Which tests need FFTW3 guards?

Any test that calls functions using FFT internally:
- `signal::cwt` (uses convolution → FFT)
- `signal::welch`, `signal::periodogram`, `signal::csd`, `signal::coherence`
- `signal::spectrogram`, `signal::stft`
- `signal::fftconvolve`, `signal::correlate`
- `signal::hilbert`, `signal::hilbert2`
- `signal::resample`, `signal::resample_poly`
- `fft::fft`, `fft::ifft`, `fft::rfft`, `fft::irfft`

### Which tests need OpenBLAS guards?

Any test that calls linear algebra functions:
- `linalg::inv`, `linalg::solve`, `linalg::det`
- `linalg::svd`, `linalg::eigh`, `linalg::cholesky`
- `linalg::matmul` (for large matrices)

Use `#ifdef INSIGHT_USE_OPENBLAS` or `GTEST_SKIP()`.

### Python/Lua/Julia tests: Use try/catch

```python
def test_cwt(self):
    try:
        result = ins.signal.cwt(x, widths)
    except RuntimeError as e:
        if "kernel not found" in str(e):
            pytest.skip("FFT kernel not available")
        raise
```

## CI Matrix

The project has multiple CI jobs:
- **Full** (FFTW3 + OpenBLAS): all tests run
- **No-FFTW3**: FFT-dependent tests skipped
- **No-OpenBLAS**: LAPACK-dependent tests skipped
- **Minimal**: only core tests

Ensure each test is properly guarded for the appropriate CI job.
