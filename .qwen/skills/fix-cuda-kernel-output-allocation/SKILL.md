---
name: fix-cuda-kernel-output-allocation
description: Fix CUDA kernels that need to allocate their own output arrays (e.g., nonzero, unique) instead of receiving pre-allocated output.
source: auto-skill
extracted_at: '2026-05-29T12:05:00.000Z'
---

# Fix CUDA Kernel Output Allocation

Some operations have **dynamic output sizes** that depend on the input data (e.g., `nonzero` returns a different number of elements depending on how many non-zero values exist). These kernels must allocate their own output arrays.

## The Problem

Most kernels receive a pre-allocated output array from the frontend:
```cpp
Array out(x.shape(), x.dtype(), x.place());  // Frontend allocates
ops().launch("op", ..., {x.layout_ptr()}, {out.layout_ptr()});
```

But for dynamic-size operations, the frontend creates a **placeholder** array:
```cpp
Array result;  // Empty/undefined placeholder
ops().launch("nonzero", ..., {x.layout_ptr(), result.layout_ptr()}, {result.layout_ptr()});
```

The kernel must:
1. Determine the actual output size
2. Allocate GPU memory for the output
3. Set the `InsightArray` metadata (dims, strides, numel, dtype, etc.)
4. Copy results to the allocated memory

## Affected Kernels

- `nonzero` — output size = number of non-zero elements × ndim
- `unique` — output size = number of unique elements
- Any kernel where the output shape depends on data content

## Implementation Pattern

```cpp
static void allocate_gpu_output(InsightArray *out, int64_t ndim,
                                const int64_t *dims, int64_t numel,
                                int32_t dtype) {
  int32_t elem_size = /* compute from dtype */;
  size_t bytes = numel * elem_size;
  void *data = nullptr;
  if (bytes > 0) {
    cudaMalloc(&data, bytes);
  }
  out->data = data;
  out->ndim = ndim;
  for (int i = 0; i < ndim; ++i) out->dims[i] = dims[i];
  out->numel = numel;
  out->dtype = dtype;
  // Compute contiguous strides
  if (ndim > 0) {
    out->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i)
      out->strides[i] = out->strides[i + 1] * out->dims[i + 1];
  }
  out->offset = 0;
  out->is_view = 0;
  out->device_type = INSIGHT_DEVICE_GPU;  // Requires #include "insight/c_api/place.h"
  out->device_id = 0;
  if (!out->ref_count) {
    out->ref_count = new int32_t;
    if (out->ref_count) *out->ref_count = 1;
  }
}
```

## CPU Fallback Pattern

For complex operations, the simplest approach is a CPU fallback:
1. Copy input data from GPU to CPU (`cudaMemcpy` DeviceToHost)
2. Compute on CPU (count, filter, sort, etc.)
3. Allocate GPU output (`cudaMalloc`)
4. Copy results from CPU to GPU (`cudaMemcpy` HostToDevice)

This is what `nonzero`, `unique`, `argsort`, `topk`, `searchsorted` etc. do.

## Frontend Pattern — CRITICAL

The frontend **must** use default-constructed `Array()` (not `Array(Shape({0}), ...)`) as placeholders, and construct the result with `Array(layout_ptr())` (not assignment from a vector element).

**Wrong** (causes "Device-to-host copy failed"):
```cpp
Array unique_arr(Shape({0}), x.dtype(), x.place());  // ref_count initialized
std::vector<Array> outputs;
outputs.push_back(unique_arr);
ops().launch("unique", ..., inputs, output_ptrs);
result.unique = outputs[0];  // Assignment copies layout, but lifecycle issues
```

**Correct** (matches `nonzero` pattern):
```cpp
Array unique_arr;  // Default constructor: ref_count = nullptr
ops().launch("unique", ..., inputs, {unique_arr.layout_ptr()});
result.unique = Array(unique_arr.layout_ptr());  // Construct from layout pointer
```

The difference: default-constructed `Array()` has `ref_count = nullptr`, which lets `allocate_gpu_output` properly initialize it. The `Array(layout_ptr())` constructor copies the kernel-modified layout and takes ownership.

## Key Points

- Always include `#include "insight/c_api/place.h"` for `INSIGHT_DEVICE_GPU` constant
- Use raw pointers (`new T[]` / `delete[]`) instead of `std::vector` to avoid the `void` type error with `cudaMemcpy`
- Handle the empty output case (numel = 0) — still set metadata, just don't allocate data
- Clean up CPU temporaries with `delete[]`
- **Frontend must use default-constructed `Array()` as placeholders** — `Array(Shape({0}), ...)` causes lifecycle issues with `allocate_gpu_output`
- **Construct result with `Array(layout_ptr())`** — not assignment from vector element
