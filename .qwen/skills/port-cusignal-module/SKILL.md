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

### Implementation Strategy: Composite First
Most cuSignal functions combine existing Insight7 primitives (FFT, math, linalg). Only write dedicated kernels when:
1. Algorithm has sequential data dependency (e.g., IIR recursive filter `sosfilt`)
2. Performance-critical hot path that composite ops can't match
3. cuSignal already has a `.cu` file for the operation

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
4. **Window symmetry**: cuSignal's `sym=True` means periodic window, `sym=False` means symmetric — match this exactly
5. **Boundary modes**: convolution modes ("full", "same", "valid") and boundary modes ("fill", "wrap", "symmetric") must match scipy exactly
6. **dtype preservation**: cuSignal often preserves input dtype, do the same

## Build Verification

```bash
cd ~/plum/Insight7/build
cmake .. -DINSIGHT_WITH_CUDA=ON
cmake --build . -j$(nproc)
ctest -R "Signal" --output-on-failure
```
