---
name: fix-contiguous-copy-noncontiguous-output
description: "contiguous_copy kernel writes to wrong positions when output is a non-contiguous view — fast path and non-contiguous path both ignore output strides"
source: auto-skill
extracted_at: '2026-06-06T13:43:56.469Z'
---

# Fix contiguous_copy kernel for non-contiguous output

## The Problem

The `contiguous_copy` kernel in `backends/cpu/kernels/manipulation/contiguous_copy.cpp` (and `.cu`) had two bugs that caused writes to wrong memory positions when the **output** (destination) was a non-contiguous view (e.g., a 2D slice of a larger array).

### Bug 1: Fast path only checks input contiguity

```cpp
// WRONG: only checks input
if (insight_array_is_contiguous(in)) {
    std::memcpy(out->data, in->data, total * elem_size);  // linear write to output!
}
```

When input is contiguous but output is a non-contiguous view (e.g., `arr["0:2,1:3"]` with strides [4,1] and offset 1), the fast path triggers and writes linearly to `out->data`, ignoring output strides and offset. Data ends up at positions 0,1,2,3 instead of 1,2,5,6.

### Bug 2: Non-contiguous path uses linear indexing for output

```cpp
// WRONG: output written at linear position
char *out_ptr = static_cast<char *>(out->data) + linear * elem_size;
```

The non-contiguous path correctly computes input offsets from input strides, but writes to `out->data + linear * elem_size` — linear indexing that ignores output strides.

## The Fix

### CPU kernel (`backends/cpu/kernels/manipulation/contiguous_copy.cpp`)

1. **Fast path**: Check BOTH input AND output contiguity before memcpy:

```cpp
if (insight_array_is_contiguous(in) && insight_array_is_contiguous(out)) {
    std::memcpy(out->data, in->data, total * elem_size);
    return C_SUCCESS;
}
```

2. **Non-contiguous path**: Compute output offset from output strides:

```cpp
for (int64_t linear = 0; linear < total; ++linear) {
    // Input offset from input strides
    int64_t in_offset = in->offset;
    int64_t tmp = linear;
    for (int d = in->ndim - 1; d >= 0; --d) {
        int64_t idx = tmp % in->dims[d];
        in_offset += idx * in->strides[d];
        tmp /= in->dims[d];
    }

    // Output offset from output strides (NOT linear!)
    int64_t out_offset = out->offset;
    tmp = linear;
    for (int d = out->ndim - 1; d >= 0; --d) {
        int64_t idx = tmp % out->dims[d];
        out_offset += idx * out->strides[d];
        tmp /= out->dims[d];
    }

    char *in_ptr = static_cast<char *>(in->data) + in_offset * elem_size;
    char *out_ptr = static_cast<char *>(out->data) + out_offset * elem_size;
    std::memcpy(out_ptr, in_ptr, elem_size);
}
```

### CUDA kernel (`backends/cuda/kernels/manipulation/contiguous_copy.cu`)

1. **Fast path**: Same fix — check both contiguity.

2. **Kernel signature**: Add output dims/strides/offset parameters:

```cpp
__global__ void contiguous_copy_kernel(
    char *dst, const char *src,
    const int64_t *in_dims, const int64_t *in_strides, int64_t in_offset,
    const int64_t *out_dims, const int64_t *out_strides, int64_t out_offset,
    int64_t total, int in_ndim, int out_ndim, size_t elem_size);
```

3. **Launch**: Allocate and copy output dims/strides to device, pass `out->offset`.

## ⚠️ Bug 3 (2026-06-06): Fast path ignores offset even for contiguous 1D views

The original fast path condition was:

```cpp
// OLD: ndim <= 1 bypasses offset check entirely
if (ndim <= 1 || (in->offset == 0 && out->offset == 0)) {
    // ... check contiguity ...
    std::memcpy(out->data, in->data, total * elem_size);  // IGNORES offset!
}
```

When `ndim == 1` (1D view) and `out->offset == 300`, the condition `ndim <= 1` is `true`, **bypassing the offset check entirely**. The memcpy writes to `out->data[0]` instead of `out->data[300]`.

Additionally, `insight_array_is_contiguous()` (used by the CUDA fast path) only checks strides, **not offset**. A 1D view `arr[300:]` on a contiguous parent returns `contiguous=true` but still has `offset=300`.

### Fix for both CPU and CUDA:

1. **Remove the `ndim <= 1 ||` shortcut** — always check contiguity properly regardless of ndim.
2. **Use `data + offset * elem_size` in memcpy** — even for "contiguous" arrays:

```cpp
// FIXED: always respect offset
std::memcpy(
    static_cast<char *>(out->data) + out->offset * elem_size,
    static_cast<char *>(in->data) + in->offset * elem_size,
    total * elem_size);
```

Same fix for CUDA fast path using `cudaMemcpy`.

### Key lesson

`insight_array_is_contiguous()` checks **strides only** — a 1D view with `offset=300, stride=1` is contiguous in layout, but its writable memory starts at `data[300]`, not `data[0]`. Any kernel that uses "contiguous" as a flag to bypass offset-based addressing is **wrong**.

### How to detect

Echo simulation bug: `delayed[ds:] = s_tx[0:N-ds]` writes to `data[0..N-ds-1]` instead of `data[ds..ds+N-ds-1]`:

```python
delayed = ins.zeros([1000], ins.complex128)
delayed[300:] = ins.ones([700], ins.complex128)
echo_np = delayed.numpy()
nonzero = np.where(np.abs(echo_np) > 1e-10)[0]
# Bug:  [0, 699]  — wrote to data[0..699]
# Fix:  [300, 999] — wrote to data[300..999]
```

## Related Skills

- `add-array-setitem` — uses contiguous_copy for copy_from_()
- `fix-strided-view-data-access` — similar symptom (wrong data) but different cause (test code using raw pointers)
- `implement-cuda-fallback-kernel` — pattern for CPU/GPU kernel parity
