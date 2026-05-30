---
name: workflow-test-no-optional-deps
description: Test Insight7 builds without optional dependencies (FFTW, OpenBLAS, CUDA) to ensure CI compatibility
source: auto-skill
extracted_at: '2026-05-31T03:00:00.000Z'
---

# Workflow: Test Without Optional Dependencies

## Purpose

GitHub CI runners may not have FFTW, OpenBLAS, or CUDA. Verify that the project compiles and non-optional tests pass.

## Commands

```bash
# Create separate build directory (don't pollute main build)
mkdir -p build-no-optional && cd build-no-optional

# Configure without optional dependencies
cmake .. \
  -DINSIGHT_USE_FFTW3=OFF \
  -DINSIGHT_USE_OPENBLAS=OFF \
  -DINSIGHT_WITH_CUDA=OFF \
  -DINSIGHT_BUILD_TESTS=ON

# Build
cmake --build . -j$(nproc)

# Run tests — expect FFT-dependent tests to fail
./tests/insight_tests_cpu --gtest_print_time=0
```

## Expected Results

| Test Category | Without FFTW | Without OpenBLAS | Without CUDA |
|---------------|-------------|-----------------|-------------|
| Elementwise, Unary, Operator | ✅ Pass | ✅ Pass | ✅ Pass |
| Creation, Cast, Complex | ✅ Pass | ✅ Pass | ✅ Pass |
| Reduction, Manipulation, Indexing | ✅ Pass | ✅ Pass | ✅ Pass |
| Random, Print, DType | ✅ Pass | ✅ Pass | ✅ Pass |
| Audio, Device Info | ✅ Pass | ✅ Pass | ✅ Pass |
| FFT (fft, rfft, fftshift) | ❌ Fail | ✅ Pass | ✅ Pass |
| Linalg (matmul, inv, svd) | ✅ Pass | ⚠️ May differ | ✅ Pass |
| Signal (convolution, filtering) | ❌ Fail | ✅ Pass | ✅ Pass |
| Signal (spectral, welch, stft) | ❌ Fail | ✅ Pass | ✅ Pass |
| Signal (windows, waveforms, bsplines) | ✅ Pass | ✅ Pass | ✅ Pass |
| Plot | ✅ Pass | ✅ Pass | ✅ Pass |

## Failure Mode

Without FFTW, the `rfft` kernel is not registered:
```
Kernel 'rfft' failed on cpu:0 with status 2: kernel not found for 'rfft'
```

This is expected — 46 FFT-dependent tests will fail. All other tests (517/627) should pass.

## CI Strategy

The main CI workflow should:
1. Build with all optional deps (FFTW, OpenBLAS) → full test suite
2. Optionally: build without FFTW as a smoke test (accept FFT test failures)

## Cleanup

```bash
rm -rf build-no-optional  # After verification
```
