---
name: fix-cuda-kernel-runtime
description: Fix CUDA kernel runtime bugs discovered through test alignment (missing dtypes, wrong dims, incorrect parameter reading).
source: auto-skill
extracted_at: '2026-05-29T10:24:53.502Z'
---

# Fix CUDA Kernel Runtime Bugs

When aligning CUDA tests with CPU tests, several categories of runtime bugs commonly appear. This skill covers the systematic approach to finding and fixing them.

## 1. Missing Dtype Registration

**Symptom**: Test fails with `kernel not found for operator '<op>', device_type=1, dtype=<N>`.

**Cause**: The CUDA kernel's `switch(dtype)` block and `REGISTER_GPU_KERNEL` macros don't cover all dtypes that the CPU version supports. Common missing dtypes: `BOOL`, `U8`, `I8`.

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
