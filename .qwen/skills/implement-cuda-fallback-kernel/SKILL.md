---
name: implement-cuda-fallback-kernel
description: Implementing CUDA kernels that return C_FALLBACK for automatic GPU→CPU data transfer
source: auto-skill
extracted_at: '2026-05-29T23:30:00.000Z'
---

# Implementing CUDA Kernels with C_FALLBACK

When a CUDA operation is too complex to implement natively (e.g., requires complex row-major↔column-major conversion, needs complex types, or is not performance-critical), register a GPU kernel that returns `C_FALLBACK`. The framework automatically transfers data to CPU, runs the CPU kernel, and transfers results back.

## 1. Register a Fallback Kernel

```cpp
// backends/cuda/kernels/<module>/fallback.cu
#include "common.cuh"  // or "../../registry/cuda_registry.h"

static C_Status myop_kernel_gpu(void **inputs, void **outputs) {
  return C_FALLBACK;
}
REGISTER_GPU_KERNEL(myop, INSIGHT_DTYPE_F32, myop_kernel_gpu);
REGISTER_GPU_KERNEL(myop, INSIGHT_DTYPE_F64, myop_kernel_gpu);
```

The CPU kernel must exist and be registered for the same op name and dtypes.

## 2. How the Fallback Works

`insight_kernel_launch()` in `src/core/kernel.cpp`:

1. Calls GPU kernel → returns `C_FALLBACK`
2. Scans `inputs[]`/`outputs[]` for NULL to count entries
3. Looks up CPU kernel for same op+dtype
4. Identifies all `InsightArray*` in inputs/outputs that are on GPU (with dedup)
5. Allocates CPU memory, copies GPU→CPU for each array
6. Calls CPU kernel with modified data pointers
7. Copies output arrays CPU→GPU
8. Restores original GPU pointers and device info

## 3. Critical: Dedup and is_output Flag

The same `InsightArray*` can appear in both `inputs[]` and `outputs[]` (e.g., `result.layout_ptr()` is passed as both). The dedup logic MUST update `is_output` when a duplicate is found:

```cpp
// CORRECT — update is_output on dedup
for (size_t si = 0; si < seen.size(); ++si) {
  if (seen[si] == arr) {
    if (is_output)
      transfers[si].is_output = true;  // ← CRITICAL
    return;
  }
}

// WRONG — silently drops the output flag
for (auto *s : seen) {
  if (s == arr) return;  // ← is_output stays false, output never copied back!
}
```

Without this fix, output arrays are never copied back from CPU → GPU, producing all zeros.

## 4. Critical: Scalar Pointer Detection

Non-array pointers (e.g., `&ord` for a double parameter) must NOT be treated as `InsightArray*`. Validate with multiple checks:

```cpp
if (arr->dtype <= INSIGHT_DTYPE_UNKNOWN || arr->dtype >= INSIGHT_DTYPE_COUNT)
  return;  // not an InsightArray
if (arr->device_type != INSIGHT_DEVICE_GPU)
  return;
if (arr->ndim < 0 || arr->ndim > INSIGHT_MAX_NDIM)
  return;  // stack garbage unlikely to have valid ndim
if (arr->numel < 0)
  return;
```

A raw `double*` on the stack has arbitrary bit patterns in the `dtype`/`ndim`/`numel` fields that could accidentally pass a single check.

## 5. Row-Major ↔ Column-Major for cuBLAS/cuSOLVER

Insight7 uses row-major. cuBLAS/cuSOLVER use column-major.

### cuBLAS (no explicit transpose needed)
Row-major `A(m,k) @ B(k,n) = C(m,n)` is equivalent to cuBLAS column-major:
```cpp
cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
            B_data, n, A_data, k, &beta, C_data, n);
```
Swap m,n and use `CUBLAS_OP_N` for both.

### cuSOLVER (explicit transpose required)
```cpp
// Transpose row-major → column-major
linalg_transpose_to_colmajor(blas, A_col, A_row, m, n, dtype);
// Call cuSOLVER on A_col
cusolverDnDgetrf(solver, m, n, A_col, m, ...);
// Transpose result back
linalg_transpose_from_colmajor(blas, out_row, A_col, m, n, dtype);
```

Helper functions in `backends/cuda/kernels/linalg/common.cuh`:
- `linalg_transpose_to_colmajor(blas, dst, src, m, n, dtype)`
- `linalg_transpose_from_colmajor(blas, dst, src, m, n, dtype)`

## 6. CUDA Test Array Creation

**NEVER** create arrays with default place when current device is GPU — `std::memcpy` can't write to GPU memory:

```cpp
// WRONG — Array created on GPU, memcpy to GPU memory = segfault
Array result(Shape({rows, cols}), DType::F64);  // GPU if current device is GPU
std::memcpy(result.data<double>(), ...);  // CRASH

// CORRECT — create on CPU, then transfer
Array result(Shape({rows, cols}), DType::F64, CPUPlace());
std::memcpy(result.data<double>(), ...);
return result.to(GPUPlace(0));
```
