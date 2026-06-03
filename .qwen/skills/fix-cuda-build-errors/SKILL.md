---
name: fix-cuda-build-errors
description: Common fixes for CUDA kernel build errors (atomicAdd, void*, missing headers, half-precision).
source: auto-skill
extracted_at: '2026-06-03T04:29:25.212Z'
---

# Fix Common CUDA Build Errors

## 1. `atomicAdd` for Double Precision
**Error**: `no instance of overloaded function "atomicAdd" matches the argument list (double *, const double)` or `function "atomicAdd" has already been defined`.

**Cause**: `atomicAdd` for `double` is natively supported only on Compute Capability 6.0+. On sm_52/sm_61 it's missing. You **cannot** globally redefine `atomicAdd(double*, double)` because it conflicts with the CUDA toolkit's built-in definition.

**Solution**: Define a named helper `atomicAddDouble` in `backends/cuda/kernels/common/atomic_helpers.cuh` and call it explicitly for double types:

```cpp
// atomic_helpers.cuh
#pragma once
#include <cuda_runtime.h>

__device__ __forceinline__ double atomicAddDouble(double *address, double val) {
  unsigned long long int *address_as_ull =
      reinterpret_cast<unsigned long long int *>(address);
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
```

Then in kernels, use `if constexpr` to dispatch:
```cpp
if constexpr (std::is_same_v<W, double>) {
    atomicAddDouble(&dst[bin], weights[idx]);
} else {
    atomicAdd(&dst[bin], weights[idx]);
}
```

**Important:** Do NOT use `#if __CUDA_ARCH__ < 600` to redefine `atomicAdd` — it causes "already been defined" errors. Always use a distinct function name.

Also check `scatter_reduce.cu` and any other kernel that uses `atomicAdd` with a template type that could be `double`.

## 2. Half-Precision (`__half` / `__nv_bfloat16`) Compilation Errors (CUDA 11.8)

### Ambiguous conversion for `__half`
**Error**: `more than one conversion function from "const __half" to a built-in type applies`

**Cause**: CUDA 11.8's `__half` has multiple implicit conversion operators. Using `+`/`-`/`*`/`/` with template `kernel<__half>` triggers ambiguity.

**Solution**: Define device helpers in `common.cuh` and use explicit kernel specializations:
```cpp
// common.cuh — CUDA 11.8 only has __hadd for __half
__device__ __forceinline__ __half hadd(__half a, __half b) { return __hadd(a, b); }
__device__ __forceinline__ __half hsub(__half a, __half b) {
  return __float2half(__half2float(a) - __half2float(b));
}
__device__ __forceinline__ __half hmul(__half a, __half b) {
  return __float2half(__half2float(a) * __half2float(b));
}
__device__ __forceinline__ __half hdiv(__half a, __half b) {
  return __float2half(__half2float(a) / __half2float(b));
}
```

Then add explicit specializations instead of template instantiation:
```cpp
__global__ void add_f16_kernel(const __half *a, const __half *b, __half *out, ...) {
  out[i] = hadd(a[i], b[i]);  // NOT a[i] + b[i]
}
// In wrapper: case INSIGHT_DTYPE_F16: add_f16_kernel<<<...>>>(...);
```

### Undefined `__half` / `__nv_bfloat16`
**Error**: `identifier "__half" is undefined`

**Cause**: Missing `#include <cuda_fp16.h>` / `<cuda_bf16.h>`.

**Solution**: Add to `common.cuh` (all elementwise files include it):
```cpp
#include <cuda_bf16.h>
#include <cuda_fp16.h>
```

### `__nv_bfloat16` has no native arithmetic on CUDA 11.8
**Error**: `identifier "__hsub" is undefined` (for bfloat16)

**Cause**: CUDA 11.8 has NO native bfloat16 arithmetic intrinsics.

**Solution**: Always cast through float:
```cpp
__device__ __forceinline__ __nv_bfloat16 hadd(__nv_bfloat16 a, __nv_bfloat16 b) {
  return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
}
```

### Missing `#include <vector>` in `.cu` files
**Error**: `namespace "std" has no member "vector"`

**Cause**: CUDA `.cu` files don't implicitly include STL headers.

**Solution**: Always add `#include <vector>` when using `std::vector` in `.cu` files.

## 3. `void` type incompatible with `void*`
**Error**: `argument of type "void" is incompatible with parameter of type "void *"` or `a value of type "void" cannot be used to initialize an entity of type "void *"`.

**Cause**: Usually happens when calling `cudaMemcpy` or similar functions with the result of a function that returns `void`, or passing a `void` value where a pointer is expected. In `std::vector::data()`, this might happen if the compiler is confused or the include is missing.

**Solution**:
1. **Check Includes**: Ensure `<vector>` and `<cuda_runtime.h>` are included.
2. **Explicit Casting**: Sometimes the compiler needs a hint.
   ```cpp
   cudaMemcpy(out->data, (void*)result.data(), size, cudaMemcpyHostToDevice);
   ```
3. **Use Raw Pointers**: If `std::vector` causes issues in `.cu` files, switch to raw `new/delete` or `cudaMallocHost` as a fallback.

## 4. Undefined `is_nan_device` or similar helpers
**Error**: `identifier "is_nan_device" is undefined`.

**Cause**: Helper functions defined in one file but used in another without a shared header.

**Solution**:
- Move the helper to a shared header (e.g., `backends/cuda/kernels/reduction/common.cuh`).
- Include this header in all files that need it.

## 5. Template Artifacts (Double Braces `{{`)
**Error**: `expected a declaration` or syntax errors in `.cu` files.

**Cause**: Code generated from templates (like Python's Jinja2 or similar) often leaves double braces `{{` instead of single braces `{`, which is invalid C++ syntax.

**Solution**:
- Search for `{{` and `}}` in your `.cu` files.
- Replace them with `{` and `}`.
- Common locations: `extern "C" {{`, `if (...) {{`, `switch (x) {{`.
