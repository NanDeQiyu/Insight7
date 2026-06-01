---
name: migrate-signal-to-backend-kernels
description: Move signal frontend raw pointer operations into registered backend kernels following project HAL conventions — COMPLETE as of 2026-06-01
source: auto-skill
extracted_at: '2026-06-01T15:54:00.295Z'
---

# Migrate Signal Functions to Backend Kernels — COMPLETE

> **STATUS: COMPLETE (2026-06-01)** — All 14 signal submodules have been migrated to
> registered backend kernels. 50 total kernels with `signal_` prefix convention.
> dtype support: CPU (F64, F32), CUDA (F64, F32, F16, BF16).
> OpenMP parallelism enabled for CPU kernels when M > 1000.
> CUDA kernels use 256 threads/block.

## Problem

Signal frontend files (`src/ops/signal/*.cpp`) contain raw pointer operations
(`.data<T>()`, `reinterpret_cast`, pointer loops) that violate the project
convention: all computation must go through registered backend kernels via
the HAL dispatch mechanism (`ops().launch()`).

## Strategy Decision: Backend Kernel vs Composite Op

Not every signal function needs a dedicated backend kernel. Classify each function:

| Category | Approach | Examples |
|----------|----------|---------|
| **Sequential algorithm** (can't be expressed as composites) | Backend kernel | lfilter, ca_cfar, ambgfun |
| **Pure composite** (calls other HAL-dispatched ops) | Keep as frontend, remove raw pointers | hann, fftconvolve, welch |
| **Data extraction** (reading complex arrays) | Replace with `real()`/`imag()`/`abs()` | csd, spectrogram magnitude |
| **I/O boundary** (file read/write) | Exception — raw pointers acceptable | read_bin, write_bin |

## Step 1: Scan for raw pointers

```bash
for f in src/ops/signal/*.cpp; do
  count=$(grep -c '\.data<\|reinterpret_cast\|memcpy\|memset' "$f")
  echo "$(basename $f): $count"
done
```

## Step 2: Create backend kernel directory structure

```bash
mkdir -p backends/cpu/kernels/signal/{filtering,radar,wavelets,spectral,estimation,io,windows,...}
mkdir -p backends/cuda/kernels/signal/{filtering,radar,wavelets,spectral,estimation,io,windows,...}
```

No CMakeLists.txt changes needed — `GLOB_RECURSE` picks up new files automatically.

## Step 3: Create CPU backend kernel

Include path from `backends/cpu/kernels/signal/<module>/`:
```cpp
#include "../../../registry/cpu_registry.h"  // 3 levels up to cpu/
#include "insight/c_api/array.h"
```

Kernel pattern:
```cpp
extern "C" {
C_Status lfilter_kernel_cpu(void **inputs, void **outputs) {
    InsightArray *b = (InsightArray *)inputs[0];   // numerator coeffs
    InsightArray *a = (InsightArray *)inputs[1];   // denominator coeffs
    InsightArray *x = (InsightArray *)inputs[2];   // input signal
    InsightArray *y = (InsightArray *)outputs[0];  // output signal

    if (!b || !a || !x || !y) {
        cpu_set_last_error("lfilter: null array pointer");
        return C_FAILED;
    }

    switch (x->dtype) {
        case INSIGHT_DTYPE_F64: {
            const double *bp = (const double *)b->data;
            const double *ap = (const double *)a->data;
            const double *xp = (const double *)x->data;
            double *yp = (double *)y->data;
            int64_t n = x->numel;
            int order = (int)b->numel - 1;

            // Direct Form II Transposed
            std::vector<double> z(order, 0.0);
            for (int64_t i = 0; i < n; ++i) {
                yp[i] = bp[0] * xp[i] + z[0];
                for (int j = 0; j < order - 1; ++j)
                    z[j] = bp[j+1] * xp[i] - ap[j+1] * yp[i] + z[j+1];
                z[order-1] = bp[order] * xp[i] - ap[order] * yp[i];
            }
            break;
        }
        case INSIGHT_DTYPE_F32: { /* same with float */ break; }
        default:
            cpu_set_last_error("lfilter: unsupported dtype");
            return C_FAILED;
    }
    return C_SUCCESS;
}
}
REGISTER_CPU_KERNEL(lfilter, INSIGHT_DTYPE_F32, lfilter_kernel_cpu);
REGISTER_CPU_KERNEL(lfilter, INSIGHT_DTYPE_F64, lfilter_kernel_cpu);
```

### OMP parallelism

For CPU kernels, add OpenMP when element count is large:
```cpp
#pragma omp parallel for if (out->numel > 1000)
for (int64_t i = 0; i < out->numel; ++i) { ... }
```

**Exception**: Sequential algorithms (IIR filter, cumsum-based CFAR) cannot be
parallelized. Only use OMP for independent per-element computations.

## Step 4: Rewrite frontend to eliminate raw pointers

### For sequential algorithms — wrap scalar params in arrays

The kernel interface only takes `InsightArray*`. Scalar parameters (guard_cells,
train_cells, alpha) must be wrapped:

```cpp
// In frontend:
Array alpha_arr = full({1}, alpha, work_dtype, cpu);
Array guard_arr = full({1}, (double)guard_cells, DType::F64, cpu);

ops().launch("ca_cfar", cpu, work_dtype,
    {(void*)data.layout_ptr(), (void*)alpha_arr.layout_ptr(),
     (void*)guard_arr.layout_ptr()},
    {(void*)threshold.layout_ptr(), (void*)detections.layout_ptr()});
```

### For data extraction — use composite ops

Replace complex array raw pointer access:
```cpp
// BEFORE (raw pointer):
const std::complex<double> *d =
    reinterpret_cast<const std::complex<double>*>(arr.data<char>());
double mag = std::abs(d[i]);

// AFTER (composite):
Array mag_arr = abs(arr);
double mag = slice(mag_arr, 0, i, i+1).item<double>();
```

### For array building — use to_array()

Replace raw pointer loops that build output arrays:
```cpp
// BEFORE:
double *out = result.data<double>();
for (int64_t i = 0; i < n; ++i) out[i] = computed_value(i);

// AFTER:
std::vector<double> buf(n);
for (int64_t i = 0; i < n; ++i) buf[i] = computed_value(i);
Array result = to_array(buf, work_dtype, cpu);
```

### For FFT-based algorithms — use composite ops

Replace raw FFT result extraction:
```cpp
// BEFORE:
Array Xf = fft::rfft(x);
const std::complex<double> *d =
    reinterpret_cast<const std::complex<double>*>(Xf.data<char>());

// AFTER:
Array Xf = fft::rfft(x);
Array mag = abs(Xf);           // or real(Xf), imag(Xf)
```

### For circular shift — use slice+concat (not roll)

`roll()` may not support all dtypes (e.g., C64). Use:
```cpp
Array part1 = slice(arr, 0, start, n);
Array part2 = slice(arr, 0, 0, start);
Array shifted = concat({part1, part2}, 0);
```

### slice() syntax — critical

The free function `slice` takes `(arr, dim, start, stop, step)`:
```cpp
slice(arr, 0, i, i+1)         // ✅ correct
slice(arr, {0}, {i}, {i+1})   // ❌ WRONG — initializer list mismatch
```

For multi-dimensional access, use `at()`:
```cpp
Array val = arr.at(r, c);     // ✅ returns scalar array
```

## Step 5: Verify

```bash
# Build
cmake --build build -j$(nproc)

# Run signal tests
cd build/tests && ./insight_tests_cpu --gtest_filter="Signal*.*"

# Count remaining raw pointers
for f in src/ops/signal/*.cpp; do
  count=$(grep -c '\.data<\|reinterpret_cast' "$f")
  [ "$count" -gt "0" ] && echo "❌ $(basename $f): $count"
done
```

## Key Pitfalls

1. **slice syntax**: `slice(arr, dim, start, stop)` NOT `slice(arr, {dim}, {start}, {stop})`
2. **Include path**: From `kernels/signal/module/`, use `../../../registry/cpu_registry.h`
3. **Scalar params**: Must be wrapped in 1-element arrays for kernel dispatch
4. **I/O exception**: `io.cpp` raw pointers are acceptable (file I/O boundary)
5. **roll() dtype limit**: May not support C64 — use slice+concat instead
6. **item() only for scalars**: `arr.item<double>()` requires `numel==1`
   — use `slice(arr, 0, i, i+1).item<double>()` for element extraction
7. **conj() declaration**: In `unary.h`, not `complex.h` — include both
8. **signal_ prefix REQUIRED**: All signal kernel names must use `signal_` prefix
   to avoid collisions with non-signal kernels (e.g., `signal_exponential` not
   `exponential`, which collides with `random/exponential`). Use prefix for both
   REGISTER macro name and C function name.
9. **CUDA window/waveform kernels**: These generate data (no input array). The
   output array is `outputs[0]`. Scalar parameters come via `inputs[]` as
   1-element arrays. On CUDA, copy param to host with `cudaMemcpy` before launch.

## Complete Module Coverage (2026-06-01)

All 14 signal submodules now have backend kernels. Kernel naming convention:
all signal kernels use `signal_` prefix to avoid collisions with non-signal kernels
(e.g., `signal_exponential` not `exponential`, which collides with `random/exponential`).

**dtype support**: CPU (F64, F32), CUDA (F64, F32, F16, BF16)
**CPU parallelism**: OpenMP `#pragma omp parallel for if (numel > 1000)` for element-wise kernels
**CUDA config**: 256 threads/block, `elementwise_blocks(numel)` for grid dimensions

| Module | Kernel Count | Key Kernels |
|--------|-------------|-------------|
| windows | 12 | hann, hamming, blackman, bartlett, boxcar, triang, cosine, gaussian, kaiser, signal_exponential, tukey, general_cosine |
| waveforms | 5 | sawtooth, signal_square, unit_impulse, chirp, gausspulse |
| bsplines | 3 | cubic, quadratic, gauss_spline |
| convolution | 3 | convolve1d, correlate1d, convolve2d |
| peak_finding | 2 | boolrelextrema_1d, boolrelextrema_2d |
| filtering | 8 | lfilter, lfilter_zi, firfilter, firfilter_zi_state, signal_wiener, signal_hilbert, signal_detrend, signal_freq_shift |
| spectral | 3 | spectrogram_accum, spectrogram_power, signal_lombscargle |
| wavelets | 3 | signal_morlet, signal_ricker, signal_morlet2 |
| filter_design | 1 | signal_firwin |
| acoustics | 5 | signal_mel2hz, signal_hz2mel, signal_mel_frequencies, signal_hz2bark, signal_bark2hz |
| io | 2 | signal_pack_bin, signal_unpack_bin |
| estimation | 1 | simple_inv |
| radar | 2 | ca_cfar, ambgfun |
| demod | 0 | (pure composite — no kernel needed) |
| **Total** | **50** | |
