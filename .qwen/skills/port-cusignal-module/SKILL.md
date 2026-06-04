---
name: port-cusignal-module
description: How to port a cusignal Python/CUDA module to ins::signal C++ using Insight7 primitives and HAL
source: auto-skill
extracted_at: '2026-05-29T23:24:25.262Z'
---

# Porting a cusignal Module to ins::signal

This skill captures the end-to-end workflow for porting a cuSignal Python/CUDA module to Insight7's `ins::signal` namespace. cuSignal is scipy.signal's GPU-accelerated twin; Insight7 replaces CuPy with a HAL-based C++ framework supporting CPU, CUDA, and future accelerators.

## Prerequisites

- cusignal source at `~/plum/cusignal/` (read-only reference)
- Insight7 project at `~/plum/Insight7/`
- Existing skills: `add-api`, `add-cpu-op`, `add-cuda-op`, `test-op`

## Key Design Decisions

### Namespace: `ins::signal`
All cuSignal functions move from `cusignal.xxx` to `ins::signal::xxx`. No sub-namespaces — flat namespace like scipy.signal.

### File Structure
```
include/insight/ops/signal.h              # Master include (all sub-headers)
include/insight/ops/signal/<module>.h     # Per-module declarations
src/ops/signal/<module>.cpp               # Per-module implementations
backends/cpu/kernels/signal/<op>.cpp      # CPU kernels (only if composite won't do)
backends/cuda/kernels/signal/<op>.cu      # CUDA kernels (only if composite won't do)
tests/cpu/test_signal_<module>.cpp        # CPU tests
tests/cuda/test_signal_<module>.cpp       # CUDA tests
```

### Implementation Strategy: Composite First (with Backend Kernels)
Most cuSignal functions combine existing Insight7 primitives (FFT, math, linalg). All 14 signal
submodules now have dedicated backend kernels registered with the `signal_` prefix convention.
The general approach is:

1. **Frontend**: Uses composite Insight7 API (FFT, math, linalg) for high-level logic
2. **Backend kernels**: For performance-critical paths, sequential algorithms, and data generation
3. **Only write dedicated kernels when:**
   - Algorithm has sequential data dependency (e.g., IIR recursive filter `sosfilt`)
   - Performance-critical hot path that composite ops can't match
   - cuSignal already has a `.cu` file for the operation

**Signal kernel conventions (2026-06-01)**:
- All signal kernels use `signal_` prefix: `signal_<op>_kernel_cpu`, `signal_<op>_kernel_gpu`
- dtype support: CPU (F64, F32), CUDA (F64, F32, F16, BF16)
- OpenMP parallelism for CPU when `numel > 1000`
- 256 threads/block for CUDA

**Insight7 primitives available for composition:**
| Category | Functions |
|----------|-----------|
| FFT | `fft::rfft`, `fft::irfft`, `fft::fft`, `fft::ifft`, `fft::fftfreq`, `fft::fftshift`, `fft::next_fast_len` |
| Math | `sin`, `cos`, `exp`, `log`, `sqrt`, `abs`, `square`, `negative` |
| Compare | `greater_than`, `less_than`, `greater_equal`, `less_equal`, `equal` |
| Logical | `logical_and`, `logical_or`, `logical_not` |
| Binary | `add`, `sub`, `mul`, `div`, `pow`, `maximum`, `minimum` |
| Reduction | `sum`, `mean`, `max`, `min`, `cumsum`, `prod`, `argmax`, `argmin` |
| Creation | `zeros`, `ones`, `full`, `arange`, `linspace`, `zeros_like`, `full_like` |
| Manipulation | `concat`, `stack`, `pad`, `flip`, `roll`, `reshape`, `squeeze`, `unsqueeze` |
| Indexing | `slice`, `take`, `where`, `scatter`, `gather` |
| Linalg | `matmul`, `solve`, `inv`, `svd`, `norm` |
| Existing signal | `convolve`, `unwrap`, `sinc` |

### Python Dynamic Typing → C++ Overloading
Python's duck typing and optional args map to C++ patterns:

```python
# Python: string enum parameter
get_window('hann', 256)
chirp(t, f0, t1, f1, method='linear')
```

```cpp
// C++: std::string parameter or enum class
Array get_window(const std::string &window, int64_t Nx, bool fftbins = true);

enum class ChirpMethod { Linear, Quadratic, Logarithmic, Hyperbolic };
Array chirp(const Array &t, double f0, double t1, double f1,
            ChirpMethod method = ChirpMethod::Linear);
```

```python
# Python: str or tuple parameter
get_window(('kaiser', 5.0), 256)
```

```cpp
// C++: overload with extra param
Array get_window(const std::string &window, double param, int64_t Nx,
                 bool fftbins = true);
```

## Module Dependency Order (CRITICAL)

Port in this order to avoid circular dependencies:

```
Layer 0 (no deps):       windows, waveforms, bsplines
Layer 1 (deps on L0):    filter_design (→windows), convolution (uses fft)
Layer 2 (deps on L1):    filtering (→convolution,filter_design,windows),
                          spectral_analysis (→windows,filtering for detrend)
Layer 3 (deps on L2):    wavelets (→convolve), resample (→filter_design,windows),
                          acoustics (→fft), demod, peak_finding
Layer 4 (deps on L2+3):  estimation (standalone algo), radartools (→windows,convolve),
                          io (standalone)
```

## Step-by-Step Workflow for Each Function

### 1. Read cusignal Source
```bash
# Find the Python implementation
grep -rn "def function_name" ~/plum/cusignal/python/cusignal/<module>/

# Check for CUDA kernel
ls ~/plum/cusignal/cpp/src/<module>/
```

Key files to read:
- `python/cusignal/<module>/<module>.py` — Python API + logic
- `python/cusignal/<module>/__init__.py` — exports
- `python/cusignal/<module>/_<module>_cuda.py` — CUDA kernel wrappers (if exists)
- `cpp/src/<module>/_<module>.cu` — raw CUDA kernels (if exists)

### 2. Write Frontend Declaration
In `include/insight/ops/signal/<module>.h`:
```cpp
#pragma once
#include "insight/core/array.h"
#include <string>

namespace ins {
namespace signal {

/**
 * @brief Brief description matching scipy.signal docs.
 *
 * @param x Input array
 * @param param Parameter description
 * @return Output array description
 */
Array function_name(const Array &x, double param = 1.0);

} // namespace signal
} // namespace ins
```

### 3. Write Frontend Implementation
In `src/ops/signal/<module>.cpp`:
```cpp
#include "insight/ops/signal/<module>.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/fft.h"
// ... other needed includes

namespace ins {
namespace signal {

Array function_name(const Array &x, double param) {
  INS_CHECK(x.defined(), "function_name: input is undefined");
  INS_CHECK(is_floating_point(x.dtype()),
            "function_name: only floating point types supported");

  // Compose using existing primitives
  Array result = /* ... */;
  return result;
}

} // namespace signal
} // namespace ins
```

### 4. Update Master Include
In `include/insight/ops/signal.h`, add:
```cpp
#include "insight/ops/signal/<module>.h"
```

### 5. Write Tests
In `tests/cpu/test_signal_<module>.cpp`:
```cpp
#include "insight/insight.h"
#include <cmath>
#include <gtest/gtest.h>

using namespace ins;
using namespace ins::signal;

class Signal<Module>TestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init();
    set_device(CPUPlace());
  }
};

TEST_F(Signal<Module>TestCPU, FunctionBasic) {
  // Test against known values from scipy.signal
  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  Array x = to_array(input);
  Array result = signal::function_name(x);

  Array cpu_result = result.to(CPUPlace());
  const float *data = cpu_result.data<float>();
  EXPECT_NEAR(data[0], expected_val, 1e-5);
}
```

CUDA tests mirror CPU tests with `GPUPlace(0)` and `_gpu` helpers.

### 6. For Dedicated Kernels (Phase 2+)
Only when composite won't work. Follow existing skills:
- CPU: `add-cpu-op` skill
- CUDA: `add-cuda-op` skill
- Adapt cuSignal's `.cu` files: change CuPy `ElementwiseKernel` to raw CUDA `__global__` kernels, replace `cp.RawModule` with Insight7's kernel registration

## CUDA Kernel Adaptation Pattern

cuSignal kernels use CuPy's `ElementwiseKernel` or `RawKernel`. To port:

```python
# cuSignal Python (CuPy ElementwiseKernel)
_cupy_correlate_1d = cp.ElementwiseKernel(
    'T sig, T filt', 'T out',
    'out = sig * filt',  # or more complex logic
    '_cupy_correlate_1d')
```

```cpp
// Insight7 CUDA kernel
template<typename T>
__global__ void correlate_1d_kernel(const T* sig, const T* filt, T* out,
                                     int64_t numel) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    out[i] = sig[i] * filt[i];  // or more complex logic
}

extern "C" C_Status correlate_1d_kernel_gpu(void** inputs, void** outputs) {
    // ... standard wrapper pattern
}
REGISTER_GPU_KERNEL(correlate_1d, INSIGHT_DTYPE_F32, correlate_1d_kernel_gpu);
```

For cuSignal's `.cu` fatbin files (e.g., `_convolution.cu`), the kernel bodies can be extracted and wrapped with Insight7's registration macros.

## Testing Against scipy.signal

cuSignal tests compare GPU vs CPU. Our tests should compare against scipy.signal reference values:

```python
# Generate reference values (run once, hardcode in C++ tests)
import scipy.signal as sps
import numpy as np
result = sps.hann(256)
print(result.tolist())
```

Then use those values in C++ `EXPECT_NEAR` assertions.

## Common Pitfalls

1. **FFT length mismatch**: cuSignal uses `next_fast_len` for FFT padding — Insight7 has `fft::next_fast_len`, use it
2. **Axis conventions**: cuSignal follows NumPy axis conventions (last axis = -1), Insight7 does too
3. **Complex type handling**: cuSignal uses CuPy complex, Insight7 uses `std::complex<float/double>` on CPU and `cuFloatComplex/cuDoubleComplex` on CUDA
4. **Window symmetry**: Insight7 `sym=True` = scipy `fftbins=False` (periodic). The conventions are swapped vs scipy's default.
5. **Boundary modes**: convolution modes ("full", "same", "valid") and boundary modes ("fill", "wrap", "symmetric") must match scipy exactly
6. **dtype preservation**: cuSignal often preserves input dtype, do the same
7. **`Shape` uses `.dim(i)` not `[i]`**: The `Shape` class has no `operator[]`. Always use `shape().dim(i)`.
8. **`linalg::` is not a namespace**: Functions like `matmul`, `inv`, `solve` are in `namespace ins` directly. From `namespace ins::signal`, call them as just `matmul(a, b)`.
9. **No `dtype_sizeof`**: Use `insight_dtype_size(static_cast<int32_t>(dtype))` from `insight/c_api/dtype.h`.
10. **Tests MUST init device**: All CPU test classes need `SetUpTestSuite()` with `ins::init({"cpu"}); set_device(CPUPlace());`. Without this, `get_device()` throws "Device interface not available".
11. **`to_array` needs `#include "insight/ops/creation.h"`**: The `to_array` and `zeros` functions are in this header.
12. **Julia `_dtype_code` helper**: For Julia bindings that need dtype codes, check if the helper exists in Insight.jl before creating new ones.
13. **Python `firwin` cutoff must be Sequence**: The Python binding for `firwin` expects `cutoff` as a `Sequence`, not a scalar float. Pass `[0.3]` not `0.3`.

## Refactoring Raw Pointers to Composite Ops (2026-05-31)

When rewriting signal functions that use `data<T>()` raw pointer access in the frontend:

### Strategy: Replace with framework composite operations

| Raw pointer pattern | Composite replacement |
|---------------------|----------------------|
| Manual matmul triple loop | `matmul(a, b)` |
| Manual transpose double loop | `permute(x, {0, 2, 1})` or `transpose(x)` |
| Gauss-Jordan inverse (30+ lines) | `inv(m)` or `simple_inv(m)` (see below) |
| Manual element-wise add/sub | `add(a, b)`, `sub(a, b)` |
| Manual diff along axis | `diff(x, 1, axis)` |
| Manual phase extraction `std::arg(z)` | `atan(im/re)` with quadrant correction via `where()` |
| Manual convolution loop | `convolve(x, h, "same")` or `fft`→`mul`→`ifft` |
| Manual FFT multiply-accumulate | `rfft`→`mul`→`irfft` |
| Manual cumsum-based windowing | `cumsum` + `slice` |
| Manual neighbor comparison | `take(flat, gather_indices)` for neighbor lookup |
| Scalar read `data<double>()[0]` | `slice(arr, {0}, {0}, {1}).item<double>()` |
| Complex sort by magnitude | `real()`, `imag()`, `square()`, `argsort()` on mag_sq |
| Setting single element | `put(arr, idx_arr, val_arr)` — returns NEW array |

### Critical API pitfalls

1. **`slice()` takes `Array&` not `const Array&`**: Must use mutable local variable:
   ```cpp
   Array data_mutable = data;  // Required for slice()
   Array center = slice(data_mutable, {axis}, {start}, {end});
   ```

2. **`put()` returns new array, doesn't modify in-place**:
   ```cpp
   result = put(result, idx_arr, val_arr);  // Must capture return value
   ```

3. **`item()` only works on scalar arrays** (numel == 1):
   ```cpp
   // WRONG: a.item<double>() if a has multiple elements
   // RIGHT:
   Array a0 = slice(a, {0}, {0}, {1});  // Extract first element
   double val = a0.item<double>();
   ```

4. **`argsort` doesn't support complex dtype**: Use magnitude-squared approach:
   ```cpp
   Array re = real(p), im = imag(p);
   Array mag_sq = add(square(re), square(im));
   Array idx = argsort(mag_sq, 0, false);
   Array sorted = take(p, idx);
   ```

5. **`inv()`/`solve()` need OpenBLAS**: Use `simple_inv()` fallback for small matrices:
   ```cpp
   // Gauss-Jordan inverse — no LAPACK dependency
   Array simple_inv(const Array &mat) {
     int n = mat.shape().dim(0);
     // Copy to vector, Gauss-Jordan elimination, return result
   }
   ```

6. **FFT in loops — avoid temporary array issues**: Use `to_array()` for fresh arrays:
   ```cpp
   // WRONG: slice() → reshape() → rfft() (view may be invalidated)
   // RIGHT: extract to vector, create fresh array
   std::vector<double> pulse_vec(samples);
   // ... fill pulse_vec ...
   Array pulse = to_array(pulse_vec, dtype, cpu);
   Array fft_pulse = fft::rfft(pulse, nfft);
   ```

7. **`nonzero()` returns `Array` (flat indices), not `vector<Array>`**: Convert flat to multi-dim:
   ```cpp
   Array flat_idx = nonzero(mask);
   Array axis_idx = mod(div(flat_idx, stride_arr), dim_arr);
   ```

### Python binding multi-file structure

When splitting signal bindings into multi-file submodule:
```
bindings/python/insight/signal/
├── __init__.py        # Re-exports all sub-modules
├── windows.py         # Wrapper functions with docstrings
├── waveforms.py
├── bsplines.py
├── filter_design.py
├── convolution.py
├── filtering.py
├── spectral.py
├── wavelets.py
├── acoustics.py
├── demod.py
├── peak_finding.py
├── radar.py
├── estimation.py
└── io.py
```

Each file pattern:
```python
"""Module docstring."""
from insight._insight import signal as _signal
__all__ = ["func1", "func2"]

def func1(args):
    """Google-style docstring."""
    return _signal.func1(args)
```

Note: `square` → `square_wf` in native binding (avoids conflict with unary `square`).
Main `__init__.py` imports from `.signal` submodule, result types from `._insight.signal`.

## Phase 2+3 Module Patterns

### Estimation (KalmanFilter class)
- Stateful class with batched matrix operations
- Uses 3D arrays: `[points, dim_x, dim_x]` for matrices, `[points, dim_x, 1]` for vectors
- **Critical**: batched matmul must only broadcast batch dimensions (see `fix-batched-matmul-3d` skill)
- predict/update use `matmul` for F*P, H*P, etc.
- Small matrix inverse (dim_z×dim_z) done inline with Gauss-Jordan

### Radar (pulse_compression, pulse_doppler, ca_cfar, mvdr, ambgfun)
- `pulse_compression`: FFT-based matched filter — iterate over pulses, rfft→multiply→irfft
- `pulse_doppler`: FFT along pulse dimension (axis 0)
- `ca_cfar`: 2D cumsum for O(1) cell averaging
- `mvdr`: R = X*X^T/N, w = R^{-1}*sv / (sv^T*R^{-1}*sv)
- All work on CPU with data transfer for GPU

### Peak Finding (argrelextrema, argrelmax, argrelmin)
- CPU implementation with stride-aware comparison
- `boolrelextrema` internal helper: for each element, compare with `order` neighbors on each side
- Returns `std::vector<Array>` (one index array per dimension)
- GPU: transfer to CPU, compute, transfer back

### Demod (fm_demod)
- Pure composite: `unwrap(angle(x))` then `diff`
- `angle(x) = atan2(imag(x), real(x))` computed manually for complex arrays
- `unwrap` and `diff` are existing Insight7 primitives

### I/O (read_bin, write_bin, pack_bin, unpack_bin, read_sigmf, write_sigmf)
- Pure CPU implementation (file I/O)
- `pack_bin`/`unpack_bin`: byte-level endianness conversion
- `read_sigmf`: parse `.sigmf-meta` JSON for dtype/endianness, then read+unpack `.sigmf-data`
- SigMF metadata parsing uses simple string matching (no json library dependency)

## Build Verification

```bash
cd ~/plum/Insight7/build
cmake .. -DINSIGHT_WITH_CUDA=ON
cmake --build . -j$(nproc)
ctest -R "Signal" --output-on-failure
```
