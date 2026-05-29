---
name: fix-cuda-build-errors
description: Common fixes for CUDA kernel build errors (atomicAdd, void*, missing headers).
source: auto-skill
extracted_at: '2026-05-29T07:30:49.130Z'
---

# Fix Common CUDA Build Errors

## 1. `atomicAdd` for Double Precision
**Error**: `no instance of overloaded function "atomicAdd" matches the argument list (double *, const double)` or `function "atomicAdd" has already been defined`.

**Cause**: `atomicAdd` for `double` is natively supported only on Compute Capability 6.0+. If you define it manually, you might conflict with built-in definitions or get errors if the architecture is too low.

**Solution**:
1. **Check Architecture**: Ensure `CMAKE_CUDA_ARCHITECTURES` is set to 60 or higher in `CMakeLists.txt`.
   ```cmake
   set(CMAKE_CUDA_ARCHITECTURES 60)
   ```
2. **Conditional Definition**: If you must support older architectures, use a conditional definition in a shared header (e.g., `backends/cuda/kernels/common/atomic_helpers.cuh`).
   ```cpp
   #if __CUDA_ARCH__ < 600
   __device__ double atomicAdd(double* address, double val) { ... }
   #endif
   ```
   *Note*: In the current Insight7 setup, simply setting the architecture to 60+ (where available) is preferred to avoid redefinition conflicts.

## 2. `void` type incompatible with `void*`
**Error**: `argument of type "void" is incompatible with parameter of type "void *"` or `a value of type "void" cannot be used to initialize an entity of type "void *"`.

**Cause**: Usually happens when calling `cudaMemcpy` or similar functions with the result of a function that returns `void`, or passing a `void` value where a pointer is expected. In `std::vector::data()`, this might happen if the compiler is confused or the include is missing.

**Solution**:
1. **Check Includes**: Ensure `<vector>` and `<cuda_runtime.h>` are included.
2. **Explicit Casting**: Sometimes the compiler needs a hint.
   ```cpp
   cudaMemcpy(out->data, (void*)result.data(), size, cudaMemcpyHostToDevice);
   ```
3. **Use Raw Pointers**: If `std::vector` causes issues in `.cu` files, switch to raw `new/delete` or `cudaMallocHost` as a fallback.

## 3. Undefined `is_nan_device` or similar helpers
**Error**: `identifier "is_nan_device" is undefined`.

**Cause**: Helper functions defined in one file but used in another without a shared header.

**Solution**:
- Move the helper to a shared header (e.g., `backends/cuda/kernels/reduction/common.cuh`).
- Include this header in all files that need it.

## 4. Template Artifacts (Double Braces `{{`)
**Error**: `expected a declaration` or syntax errors in `.cu` files.

**Cause**: Code generated from templates (like Python's Jinja2 or similar) often leaves double braces `{{` instead of single braces `{`, which is invalid C++ syntax.

**Solution**:
- Search for `{{` and `}}` in your `.cu` files.
- Replace them with `{` and `}`.
- Common locations: `extern "C" {{`, `if (...) {{`, `switch (x) {{`.
