---
name: fix-cuda-kernel-runtime
description: Fix CUDA kernel runtime bugs discovered through test alignment (missing dtypes, wrong dims, incorrect parameter reading).
source: auto-skill
extracted_at: '2026-05-29T10:24:53.502Z'
updated: '2026-06-06'
---

# Fix CUDA Kernel Runtime Bugs

When aligning CUDA tests with CPU tests, several categories of runtime bugs commonly appear. This skill covers the systematic approach to finding and fixing them.

## 0. Signal Kernel Patterns (2026-06-01)

All 14 signal submodules have backend kernels with these conventions:
- **Naming**: `signal_` prefix on all kernel names (e.g., `signal_morlet`, `signal_lombscargle`)
- **CPU dtype**: F64, F32 — with OpenMP `#pragma omp parallel for if (numel > 1000)`
- **CUDA dtype**: F64, F32, F16, BF16 — 256 threads/block
- **Kernel files**: `backends/cpu/kernels/signal/<module>/<op>.cpp` and `backends/cuda/kernels/signal/<module>/<op>.cu`
- **Registration**: `REGISTER_CPU_KERNEL(signal_<op>, dtype, func)` / `REGISTER_GPU_KERNEL(signal_<op>, dtype, func)`
- **Window/waveform kernels**: These generate data (no input array). Output is `outputs[0]`. Scalar params come via `inputs[]` as 1-element arrays.

When fixing signal CUDA kernel bugs, verify the `signal_` prefix is used consistently
in both `REGISTER_GPU_KERNEL` name and the frontend's `ops().launch("signal_<op>", ...)` call.

## 1. Missing Dtype Registration

**Symptom**: Test fails with `kernel not found for operator '<op>', device_type=1, dtype=<N>`.

**Cause**: The CUDA kernel's `switch(dtype)` block and `REGISTER_GPU_KERNEL` macros don't cover all dtypes that the CPU version supports. Common missing dtypes: `BOOL`, `U8`, `I8`.

### 1a. Complex type (C32/C64) template + `<<<>>>` parsing ambiguity

**Special case**: `cuFloatComplex` and `cuDoubleComplex` cannot be used as template parameters in `__global__` template functions that are then launched with `<<<blocks, threads>>>` syntax. nvcc's parser gets confused by `flip_kernel<cuFloatComplex><<<blocks, threads>>>`.

**Symptom**: Build fails with `identifier "cuFloatComplex" is undefined` even though `#include <cuComplex.h>` is present and the same type works fine in non-template contexts (e.g. `static_cast<cuFloatComplex*>(...)`).

**Fix**: Use a dedicated non-template `__global__` kernel function for complex types instead of template instantiation.

```cpp
// ❌ WRONG — template + <<<>>> parsing ambiguity with cuFloatComplex
template <typename T>
__global__ void flip_kernel(const T *src, T *dst, ...) { ... }

// In switch block:
case INSIGHT_DTYPE_C32:
    flip_kernel<cuFloatComplex><<<blocks, threads>>>(...); // PARSER ERROR
    break;

// ✅ CORRECT — dedicated kernel function
__global__ void flip_c64_kernel(const cuFloatComplex *src, cuFloatComplex *dst, ...) {
    // same logic as template, but with explicit types
}

// In switch block:
case INSIGHT_DTYPE_C32:
    flip_c64_kernel<<<blocks, threads>>>(...); // OK
    break;
```

**Also required**: Add `#include <cuComplex.h>` to the `.cu` file. The header is NOT pulled in transitively by `cuda_runtime.h` on all CUDA versions — always include it explicitly when using `cuFloatComplex`/`cuDoubleComplex`.

## 1c. Reduction Kernel Complex Type (sum specific)

**Symptom**: Build fails with `no suitable constructor exists to convert from "int" to "float2"` or `no operator "+=" matches these operands` for `cuFloatComplex`/`cuDoubleComplex` in reduction kernels.

**Cause**: `cuFloatComplex` (which is `float2`) and `cuDoubleComplex` (which is `double2`) are plain structs, not full C++ classes. They do NOT support:
- Default construction from `int` (i.e. `T(0)` fails — `float2` has no int constructor)
- `+=` operator (`cuFloatComplex += cuFloatComplex` is not defined)

These operations work fine for arithmetic types (`float`, `double`, `int32_t`, etc.) but fail for CUDA complex types.

**Fix**: Write dedicated non-template kernel functions using `make_cuFloatComplex(0.0f, 0.0f)`/`make_cuDoubleComplex(0.0, 0.0)` for initialization and `cuCaddf()`/`cuCadd()` for addition.

```cpp
// ❌ WRONG — template with T(0) and += fails for cuFloatComplex/cuDoubleComplex
template <typename T>
__global__ void sum_kernel(T *dst, const T *src, int64_t n, int64_t reduce) {
    T sum = T(0);            // FAILS: float2 has no int constructor
    for (...) sum += src[j]; // FAILS: no operator += for float2
}

// ✅ CORRECT — dedicated kernel with CUDA complex intrinsics
__global__ void sum_c64_kernel(cuFloatComplex *dst, const cuFloatComplex *src,
                               int64_t n, int64_t reduce) {
    cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
    for (...) sum = cuCaddf(sum, src[j]);  // use cuCaddf for float complex
    // ... shared memory reduction with cuCaddf ...
}

__global__ void sum_c128_kernel(cuDoubleComplex *dst, const cuDoubleComplex *src,
                                int64_t n, int64_t reduce) {
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    for (...) sum = cuCadd(sum, src[j]);  // use cuCadd for double complex
    // ... shared memory reduction with cuCadd ...
}
```

**Shared memory reduction for complex types**: The shared memory reduction loop must also use `cuCaddf()`/`cuCadd()` instead of `+=`:
```cpp
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] = cuCaddf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
}
```

**Shared memory size calculation**: Use `sizeof(cuFloatComplex)` or `sizeof(cuDoubleComplex)` instead of `sizeof(T)` for the dynamic shared memory allocation:
```cpp
sum_c64_kernel<<<blocks, threads, threads * sizeof(cuFloatComplex)>>>(...);
sum_c128_kernel<<<blocks, threads, threads * sizeof(cuDoubleComplex)>>>(...);
```

## 1b. Premature GPU Transfer Back (CmplxSort pattern)

**Symptom**: CUDA test segfaults on `data<T>()` access, but CPU test passes.

**Cause**: A function transfers input to CPU, does computation there, then transfers result back to GPU with `sorted.to(p.place())`. The test then calls `sorted.data<double>()` which dereferences a device pointer on the host.

**Fix**: Remove the transfer-back. If computation is CPU-only, return CPU result:
```cpp
// ❌ WRONG — transfers back to GPU, test dereferences device pointer
Array sorted = take(p_cpu, sort_idx);
if (p.place().kind() != DeviceKind::CPU)
  sorted = sorted.to(p.place());
return sorted;

// ✅ CORRECT — return CPU result
Array sorted = take(p_cpu, sort_idx);
return sorted;
```

**Why**: The caller (test or frontend) handles device placement. The function should return results on the device where they were computed.

**Fix**:
1. Check CPU kernel registrations: `grep "REGISTER_CPU_KERNEL" backends/cpu/kernels/<module>/<op>.cpp`
2. Check CUDA kernel registrations: `grep "REGISTER_GPU_KERNEL" backends/cuda/kernels/<module>/<op>.cu`
3. Add missing dtype cases to the `switch` block and `REGISTER_GPU_KERNEL` lines.
4. For `BOOL`, the kernel template should use `bool` as the C++ type.

**Example** (`count_nonzero.cu` missing BOOL):
```cpp
// Add before existing cases:
case INSIGHT_DTYPE_BOOL:
    count_nonzero_kernel<bool><<<blocks, threads>>>(...);
    break;
// Add registration:
REGISTER_GPU_KERNEL(count_nonzero, INSIGHT_DTYPE_BOOL, count_nonzero_kernel_gpu);
```

## 2. Wrong Array for Shape/Dims

**Symptom**: Test produces wrong values or garbage data, especially when input arrays have different shapes.

**Cause**: The kernel copies dimensions/strides from the wrong `InsightArray`. For example, using `values->dims` when `values` has fewer dimensions than `indices`.

**Fix**:
1. Identify which array determines the iteration shape (usually the one with the most dimensions, or the output array).
2. Use that array's `dims` and `strides` for `cudaMemcpy` to device memory.

**Example** (`put_along_axis.cu`):
```cpp
// Bad: values might be 1D while indices is 2D
cudaMemcpy(d_dims, values->dims, ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

// Good: use indices shape for iteration
cudaMemcpy(d_dims, indices->dims, ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
```

## 3. Incorrect Kernel Parameter Reading

**Symptom**: Kernel behaves as if parameters are always the same value, ignoring frontend arguments.

**Cause**: The kernel wrapper doesn't read all parameters from the `inputs[]` array, or reads from the wrong index.

**Fix**:
1. Check the frontend's `ops().launch()` call to see what parameters are passed and in what order.
2. Verify the kernel wrapper reads from the correct `inputs[N]` indices.
3. Use `*static_cast<Type*>(inputs[N])` to read each parameter.

**Example** (`take.cu`):
```cpp
// Frontend passes: {result, x, indices, &normalized_axis, &has_axis}
// inputs[0] = result, inputs[1] = x, inputs[2] = indices
// inputs[3] = &normalized_axis, inputs[4] = &has_axis
bool has_axis = *static_cast<bool *>(inputs[4]);
```

## Debugging Workflow

When a CUDA test fails but the CPU test passes:

1. **Read the error message carefully** — it usually tells you exactly what's wrong.
2. **Compare CPU and CUDA kernel registrations** — check for missing dtypes.
3. **Check the kernel's `cudaMemcpy` calls** — verify dims/strides come from the right array.
4. **Check `inputs[]` indexing** — verify the kernel reads parameters in the same order the frontend passes them.
5. **Don't hardcode values for testing** — this creates a false sense of correctness and breaks other tests.
