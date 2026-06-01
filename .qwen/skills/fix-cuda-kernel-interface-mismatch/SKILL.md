---
name: fix-cuda-kernel-interface-mismatch
description: Fix CUDA kernels whose input/output interface doesn't match the CPU kernel, causing segfaults or wrong results.
source: auto-skill
extracted_at: '2026-05-29T12:00:00.000Z'
updated: '2026-06-01'
---

# Fix CUDA Kernel Interface Mismatch

The most common source of segfaults and wrong results in CUDA kernels is a mismatch between the kernel's `inputs[]`/`outputs[]` expectations and what the frontend actually passes.

## Signal Kernel Interface Patterns (2026-06-01)

All 14 signal submodules have backend kernels with `signal_` prefix convention.
Common signal kernel interface patterns:

### Window/waveform kernels (data generation, no input array)
```
inputs[0..N-1] = scalar parameters as 1-element arrays (e.g., alpha, width)
outputs[0]     = output array (pre-allocated by frontend)
```
Example: `signal_morlet` — inputs: [width_arr, center_arr, s_arr], output: [wavelet_arr]
On CUDA, copy scalar params to host with `cudaMemcpy` before launch.

### Filter/processing kernels
```
inputs[0] = coefficients/weights array
inputs[1] = input signal array
inputs[2..N] = scalar parameters (axis, mode, etc.)
outputs[0] = output signal array
```
Example: `lfilter` — inputs: [b, a, x], output: [y]

### Key rules for signal kernels:
- Kernel name must use `signal_` prefix in both `REGISTER_*_KERNEL` and `ops().launch()`
- dtype support: CPU (F64, F32), CUDA (F64, F32, F16, BF16)
- CUDA kernels: 256 threads/block
- CPU kernels: OpenMP when `numel > 1000`

## How to Diagnose

1. **Segfault or garbage values** in a CUDA test, but CPU test passes → likely interface mismatch.
2. Read the CPU kernel's docstring or code to find the exact `inputs[]` layout.
3. Read the CUDA kernel's wrapper function to see how it interprets `inputs[]`.
4. Compare — they must match exactly.

## Common Mismatch Patterns

### Pattern 1: Kernel expects pre-computed values, frontend passes raw parameters

**Example**: `cumsum`, `cumprod`, `cummax`, `cummin`

CPU kernel:
```
inputs[0] = output, inputs[1] = input, inputs[2] = int* axis
```
CPU kernel computes `batch_size` and `reduce_size` internally from `axis`.

CUDA kernel (WRONG):
```
inputs[0] = output, inputs[1] = input, inputs[2] = int64_t* batch_size, inputs[3] = int64_t* reduce_size
```
CUDA kernel expects pre-computed values that the frontend never passes.

**Fix**: Rewrite the CUDA kernel to match the CPU interface — read `axis` from `inputs[2]` and compute `batch_size`/`reduce_size` internally.

### Pattern 2: Kernel expects more inputs than frontend provides

**Example**: `argsort`, `topk`

CPU kernel:
```
inputs[0] = output, inputs[1] = input, inputs[2] = bool* descending
```
Frontend transposes the array so axis is last, then passes 3 inputs.

CUDA kernel (WRONG):
```
inputs[0] = output, inputs[1] = input, inputs[2] = int* axis, inputs[3] = bool* descending
```
CUDA kernel expects 4 inputs (including axis), but frontend only passes 3.

**Fix**: Remove the extra parameter from the CUDA kernel. The frontend already handles axis transposition.

### Pattern 3: Kernel reads output arrays from inputs[] instead of outputs[]

**Example**: `unique`

CPU kernel reads flags from `inputs[1..3]` and outputs from `outputs[0..N]`.

CUDA kernel (WRONG) reads outputs from `inputs[0..1]`, causing null pointer.

**Fix**: Match the CPU convention — flags from `inputs[]`, data from `outputs[]`.

## Systematic Fix Procedure

For each CUDA kernel that fails:

1. **Read the CPU kernel** (`backends/cpu/kernels/<module>/<op>.cpp`):
   - Find the docstring listing `inputs[]` and `outputs[]` layout
   - Note the exact parameter types and order

2. **Read the frontend** (`src/ops/<module>.cpp`):
   - Find the `ops().launch()` call
   - Note what's passed in the inputs vector

3. **Read the CUDA kernel** (`backends/cuda/kernels/<module>/<op>.cu`):
   - Check how it reads `inputs[]` and `outputs[]`
   - Compare with CPU kernel

4. **Fix the CUDA kernel** to match the CPU kernel's interface exactly:
   - Same number of inputs/outputs
   - Same types at each index
   - Same computation logic (or compute missing values internally)

## Example Fix: cumsum.cu

**Before** (broken):
```cpp
C_Status cumsum_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *prepared = static_cast<InsightArray *>(inputs[1]);
  int64_t batch_size = *static_cast<int64_t *>(inputs[2]);  // WRONG: frontend passes axis
  int64_t reduce_size = *static_cast<int64_t *>(inputs[3]); // WRONG: doesn't exist
  ...
```

**After** (fixed):
```cpp
C_Status cumsum_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *x = static_cast<InsightArray *>(inputs[1]);
  int axis = *static_cast<int *>(inputs[2]);  // Match CPU: read axis
  // Compute batch_size and reduce_size from axis
  int64_t axis_size = x->dims[axis];
  int64_t inner_size = 1;
  for (int d = axis + 1; d < ndim; ++d) inner_size *= x->dims[d];
  ...
```
